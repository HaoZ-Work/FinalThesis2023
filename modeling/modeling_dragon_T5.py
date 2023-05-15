import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel,T5Config

from transformers import PretrainedConfig
from transformers.file_utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
)

from modeling import modeling_gnn
from utils import layers
from utils import utils


class T5GAT(T5EncoderModel):

    def __init__(self, config, args={}, k=5, n_ntype=4, n_etype=38, hidden_size=200, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):

        super().__init__(config)

        self.args = args
        self.k = k
        self.concept_dim = concept_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.info_exchange = info_exchange
        if k >= 1:
            self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
            self.gnn_layers = nn.ModuleList([modeling_gnn.GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])
            self.activation = layers.GELU()
            self.dropout_rate = dropout

            self.sent_dim = config.hidden_size
            self.sep_ie_layers = sep_ie_layers
            if sep_ie_layers:
                self.ie_layers = nn.ModuleList([layers.MLP(self.sent_dim + concept_dim, ie_dim, self.sent_dim + concept_dim, ie_layer_num, p_fc) for _ in range(k)])
            else:
                self.ie_layer = layers.MLP(self.sent_dim + concept_dim, ie_dim, self.sent_dim + concept_dim, ie_layer_num, p_fc)
            if self.args.residual_ie == 2:
                self.ie_LayerNorm = nn.LayerNorm(self.sent_dim + concept_dim)

    def forward(self, hidden_states, attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type, _node_type,
                _node_feature_extra, special_nodes_mask, output_attentions=False, output_hidden_states=True):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, seq_len]
        head_mask: list of shape [num_hidden_layers]

        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """
        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()

        # T5 Encoder
        for i, layer_module in enumerate(self.encoder.block):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)



            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training=self.training)

                # Exchange info between LM and GNN hidden states (Modality interaction)
                if self.info_exchange == True or (
                        self.info_exchange == "every-other-layer" and (i - self.num_hidden_layers + self.k) % 2 == 0):
                    X = _X.view(bs, -1, _X.size(1))  # [bs, max_num_nodes, node_dim]
                    context_node_lm_feats = hidden_states[:, 0, :]  # [bs, sent_dim]
                    context_node_gnn_feats = X[:, 0, :]  # [bs, node_dim]
                    context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
                    if self.sep_ie_layers:
                        _context_node_feats = self.ie_layers[gnn_layer_index](context_node_feats)
                    else:
                        _context_node_feats = self.ie_layer(context_node_feats)
                    if self.args.residual_ie == 1:
                        context_node_feats = context_node_feats + _context_node_feats
                    elif self.args.residual_ie == 2:
                        context_node_feats = self.ie_LayerNorm(context_node_feats + _context_node_feats)
                    else:
                        context_node_feats = _context_node_feats
                    context_node_lm_feats, context_node_gnn_feats = torch.split(context_node_feats,
                                                                                [context_node_lm_feats.size(1),
                                                                                 context_node_gnn_feats.size(1)], dim=1)
                    hidden_states[:, 0, :] = context_node_lm_feats
                    X[:, 0, :] = context_node_gnn_feats
                    _X = X.view_as(_X)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, _X # last-layer hidden state, (all hidden states), (all attentions)

    # def get_fake_inputs(self, device="cuda:0"):
    #     bs = 20
    #     seq_len = 100
    #     input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
    #     attention_mask = torch.ones([bs, seq_len], dtype=torch.long).to(device)
    #     head_mask = [None] * self.num_hidden_layers
    #
    #     n_node = 200
    #     _X = torch.zeros([bs * n_node, self.concept_dim]).to(device)
    #     n_edges = 3
    #     edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
    #     edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
    #     _node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
    #     _node_type[:, 0] = 3
    #     _node_type = _node_type.view(-1)
    #     _node_feature_extra = torch.zeros([bs * n_node, self.concept_dim]).to(device)
    #     special_nodes_mask = torch.zeros([bs, n_node], dtype=torch.long).to(device)
    #     special_tokens_mask = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
    #     return input_ids, attention_mask, special_tokens_mask  , head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra,special_nodes_mask
    #
    #
    # def check_outputs(self, outputs, _X):
    #     bs = 20
    #     seq_len = 100
    #     assert outputs[0].size() == (bs, seq_len, self.sent_dim)
    #     n_node = 200
    #     assert _X.size() == (bs * n_node, self.concept_dim)
    def get_fake_inputs(self, device="cuda:0"):
        bs = 20
        seq_len = 100
        hidden_states = torch.zeros([bs, seq_len, self.sent_dim]).to(device)
        attention_mask = torch.zeros([bs, 1, 1, seq_len]).to(device)
        head_mask = [None] * self.num_hidden_layers

        n_node = 200
        _X = torch.zeros([bs * n_node, self.concept_dim]).to(device)
        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        _node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        _node_type[:, 0] = 3
        _node_type = _node_type.view(-1)
        _node_feature_extra = torch.zeros([bs * n_node, self.concept_dim]).to(device)
        return hidden_states, attention_mask, [], head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra, []

    def check_outputs(self, outputs, _X):
        bs = 20
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert _X.size() == (bs * n_node, self.concept_dim)


def test_T5GAT(device):
    config, _ = T5Config.from_pretrained(
        "t5-small",
        cache_dir=None, return_unused_kwargs=True,
        force_download=False,
        output_hidden_states=True
    )

    class Args:
        def __init__(self):
            self.residual_ie = 2
            self.fp16 = False
            self.update_ie = False

    test_args = Args()
    model = T5GAT(config, args=test_args, sep_ie_layers=True).to(device)
    print(model)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    model.check_outputs(*outputs)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    utils.print_cuda_info()
    # free_gpus = utils.select_free_gpus()
    # device = torch.device("cuda:{}".format(free_gpus[0]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_T5GAT(device)