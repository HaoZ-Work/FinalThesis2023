import re
import string
from dotenv import load_dotenv
import os
import torch
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer
from data_utils import DataLoaderCreator
import argparse
from tqdm import tqdm
import wandb
from torch.optim import Adam  # import Adam optimizer
from torch.optim.lr_scheduler import StepLR



import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_accuracy(model, tokenizer, batch_or_dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    pad_token_id = tokenizer.pad_token_id  # get the ID for the <pad> token

    def handle_batch(batch):
        nonlocal correct_predictions, total_predictions

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # replace -100 in the labels with pad_token_id
        labels = labels.where(labels != -100, torch.tensor(pad_token_id, device=device))

        # forward pass
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        # convert model output tensors to strings
        predicted_texts = [normalize_answer(tokenizer.decode(t, skip_special_tokens=True)) for t in outputs]
        actual_texts = [normalize_answer(tokenizer.decode(t, skip_special_tokens=True)) for t in labels]

        # compare predictions to actuals
        correct_predictions += sum([actual == predicted for actual, predicted in zip(actual_texts, predicted_texts)])
        total_predictions += len(actual_texts)

    if isinstance(batch_or_dataloader, torch.utils.data.DataLoader):
        # If it's a dataloader, iterate over batches
        with torch.no_grad():
            for batch in batch_or_dataloader:
                handle_batch(batch)
    else:
        # If it's a single batch, just handle it
        handle_batch(batch_or_dataloader)

    # calculate accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy



def train_model(model, optimizer, scheduler, train_dataloader, dev_dataloader, config, tokenizer):
    device = model.device
    args = config
    # create directory for checkpoints
    ckpt_dir_name = f"T5_finetuning/checkpoints/bs_{args.batch_size}_opt_{optimizer.__class__.__name__}_epochs_{args.epochs}"
    os.makedirs(ckpt_dir_name, exist_ok=True)

    # Initialize best validation loss to infinity
    best_val_loss = float("inf")
    best_model_path = None  # Keep track of best model path

    # training loop
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        model.train()
        correct_predictions = 0
        total_predictions = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # get the loss
            loss = outputs.loss

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            # step the scheduler
            scheduler.step()

            step += 1

            # clear the gradients
            optimizer.zero_grad()

            # calculate train accuracy
            train_accuracy = calculate_accuracy(model, tokenizer, batch, device)
            correct_predictions += train_accuracy * len(input_ids)
            total_predictions += len(input_ids)

            # log the loss and accuracy
            wandb.log({"Train Loss": loss.item(), "Train Accuracy": train_accuracy})

            # Call calculate_accuracy every 100 steps for validation set
            if step % 100 == 0:
                accuracy = calculate_accuracy(model, tokenizer, dev_dataloader, device)
                print(f"Validation Accuracy at step {step}: {accuracy}")
                wandb.log({"Validation Accuracy": accuracy})

        # logging final accuracy after each epoch
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Training Accuracy after epoch {epoch + 1}: {epoch_accuracy}")
        wandb.log({"Epoch Train Accuracy": epoch_accuracy})

        # Evaluate on dev set
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # accumulate loss
                total_loss += outputs.loss.item()

        val_loss = total_loss / len(dev_dataloader)
        print(f"  Validation Loss: {val_loss}")
        wandb.log({"Validation Loss": val_loss})

        # If this model is the best so far, save it
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path is not None:  # If a best model exists, remove it
                os.remove(best_model_path)
            best_model_path = f"{ckpt_dir_name}/model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), best_model_path)

    if best_model_path is not None:
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model.")

def main(config=None):
    # Start a new run
    set_seed(42)

    if config != None:
        run_name = f"{config['mode']}_bs{config['batch_size']}_lr{config['lr']}_e{config['epochs']}"

        run = wandb.init(project='FinalThesis2023', entity='haoz', name=run_name, config=config)
    else:

        run = wandb.init(project='FinalThesis2023', entity='haoz', config=config)
        config = run.config
        run_name = f"{config['mode']}_bs{config['batch_size']}_lr{config['lr']}_e{config['epochs']}"
        wandb.run.name = run_name
        wandb.run.save()


    # Retrieve hyperparameters from W&B
    args = argparse.Namespace(**run.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model_vocab_size = model.config.vocab_size

    # move model to the device
    model = model.to(device)

    # initialize optimizer
    # optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

    # Initialize dataloaders using DataLoaderCreator
    data_loader_creator = DataLoaderCreator(tokenizer,source_max_length=args.source_max_length, target_max_length=args.target_max_length, batch_size=args.batch_size)
    train_dataloader, dev_dataloader, test_dataloader = data_loader_creator.create_dataloaders(data_type=args.data_type)

    # Watch the model
    wandb.watch(model, log_freq=100)

    # Train the model
    train_model(model, optimizer,scheduler, train_dataloader, dev_dataloader, args, tokenizer)



if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="'sweeps' for hyperparameter search, 'train' for training")
    parser.add_argument("--machine", type=str, required=True, help="'slurm' for SLURM, 'local' for local machine")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--data_type", type=str, default="csqa-debug")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--source_max_length", type=int, default=512)
    parser.add_argument("--target_max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    print(f"Running mode: {args.mode}")
    print(f"Running on machine: {args.machine}")
    print(f"The home path is{os.environ['HOME']}")

    if args.machine == 'slurm':
        os.environ['HOME'] = '/ukp-storage-1/zhang'
    elif args.machine == 'local':
        pass
    else:
        print(f"Unknown machine {args.machine}")
        exit(1)

    # sweep configuration
    sweep_name = f"T5HS_{args.model_name}_{args.machine}_{args.data_type}_lr_decay"
    # final sweep configuration
    sweep_config = {
        "name": sweep_name,
        "method": "grid",  # or "bayes"
        "metric": {
            "name": "Validation Loss",
            "goal": "minimize",
        },
        "parameters": {
            "batch_size": {"values": [2,4,8,16,32]},
            "source_max_length": {"values": [512]},
            "target_max_length": {"values": [128]},
            "epochs": {"values": [10,]},
            "lr": {"values": [1e-5,1e-4]},  # add learning rate parameter for sweep

            "model_name": {"value": args.model_name},
            "data_type": {"value": args.data_type},
            "mode": {"value": args.mode},
        },
    }

    # debug sweep configuration
    # sweep_config = {
    #     "name": sweep_name,
    #     "method": "grid",  # or "bayes"
    #     "metric": {
    #         "name": "Validation Loss",
    #         "goal": "minimize",
    #     },
    #     "parameters": {
    #         "batch_size": {"values": [2]},
    #         "source_max_length": {"values": [512]},
    #         "target_max_length": {"values": [128]},
    #         "epochs": {"values": [10,]},
    #         "lr": {"values": [1e-4, 1e-5,]},  # add learning rate parameter for sweep
    #
    #         "model_name": {"value": args.model_name},
    #         "data_type": {"value": args.data_type},
    #         "mode": {"value": args.mode},
    #     },
    # }

    load_dotenv()
    api_key = os.getenv("WANDB_API")

    wandb.login(key=api_key)


    if args.mode == 'sweeps':
        sweep_id = wandb.sweep(sweep_config, project='FinalThesis2023', entity='haoz')
        wandb.agent(sweep_id, function=main)
    elif args.mode == 'train':
        # Fetch hyperparameters from args
        config = {
            "mode": args.mode,
            "batch_size": args.batch_size,
            "source_max_length": args.source_max_length,
            "target_max_length": args.target_max_length,
            "epochs": args.epochs,
            "lr": args.lr,  # add learning rate in config
            "model_name": args.model_name,
            "data_type": args.data_type,
        }
        main(config)
    else:
        print(f"Unknown mode {args.mode}")

