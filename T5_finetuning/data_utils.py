from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
from transformers import Adafactor

DATA_BASE_PATH = '../data/csqa-debug/'
# DATA_BASE_PATH = '../data/csqa/'

csqa_debug_dev_path = DATA_BASE_PATH+'dev_rand_split.jsonl'


# Load inhouse_split_qids.txt
with open(DATA_BASE_PATH+"inhouse_split_qids.txt", "r") as file:
    train_qids = set(line.strip() for line in file)

# Load and split train_rand_split.jsonl
train_data = []
test_data = []

with open(DATA_BASE_PATH+"train_rand_split.jsonl", "r") as file:
    for line in file:
        item = json.loads(line)
        if item["id"] in train_qids:
            train_data.append(item)
        else:
            test_data.append(item)

# Now train_data contains the training set and test_data contains the test set



dev_data = []

with open(csqa_debug_dev_path, 'r') as file:
    for line in file:
        dev_data.append(json.loads(line))



# initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# task-specific parameters
source_max_length = 512
target_max_length = 128
batch_size = 5



class CSQADataset(Dataset):
    def __init__(self, data, tokenizer, source_max_length, target_max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]["stem"]
        correct_choice_text = next(choice["text"] for choice in item["question"]["choices"] if choice["label"] == item["answerKey"])

        encoding = self.tokenizer(
            f"question: {question}",
            truncation=True,
            max_length=self.source_max_length,
            padding="max_length",
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            correct_choice_text,
            truncation=True,
            max_length=self.target_max_length,
            padding="max_length",
            return_tensors="pt"
        )

        labels = target_encoding.input_ids
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoding.input_ids.flatten(),
            "attention_mask": encoding.attention_mask.flatten(),
            "labels": labels.flatten()
        }


class DataLoaderCreator:
    def __init__(self, source_max_length=512, target_max_length=128, batch_size=8):
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.batch_size = batch_size
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def create_dataloaders(self, data_type='csqa-debug'):
        if data_type == 'csqa':
            DATA_BASE_PATH = '../data/csqa/'
        else:
            DATA_BASE_PATH = '../data/csqa-debug/'

        csqa_debug_dev_path = DATA_BASE_PATH+'dev_rand_split.jsonl'

        # Load inhouse_split_qids.txt
        with open(DATA_BASE_PATH+"inhouse_split_qids.txt", "r") as file:
            train_qids = set(line.strip() for line in file)

        # Load and split train_rand_split.jsonl
        train_data = []
        test_data = []

        with open(DATA_BASE_PATH+"train_rand_split.jsonl", "r") as file:
            for line in file:
                item = json.loads(line)
                if item["id"] in train_qids:
                    train_data.append(item)
                else:
                    test_data.append(item)

        dev_data = []

        with open(csqa_debug_dev_path, 'r') as file:
            for line in file:
                dev_data.append(json.loads(line))

        # Initialize datasets
        train_dataset = CSQADataset(train_data, self.tokenizer, self.source_max_length, self.target_max_length)
        dev_dataset = CSQADataset(dev_data, self.tokenizer, self.source_max_length, self.target_max_length)
        test_dataset = CSQADataset(test_data, self.tokenizer, self.source_max_length, self.target_max_length)

        # Initialize dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, dev_data_loader, test_dataloader




