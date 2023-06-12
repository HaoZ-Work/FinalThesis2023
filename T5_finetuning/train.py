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

def calculate_accuracy(model, tokenizer, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    pad_token_id = tokenizer.pad_token_id  # get the ID for the <pad> token


    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating Accuracy"):
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
            # print("Predicted Texts: ", predicted_texts, "\nActual Texts: ", actual_texts)
            # compare predictions to actuals
            correct_predictions += sum([actual == predicted for actual, predicted in zip(actual_texts, predicted_texts)])
            total_predictions += len(actual_texts)

    # calculate accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy


def train_model(model, optimizer, train_dataloader, dev_dataloader, config, tokenizer):
    device = model.device
    args = config
    # create directory for checkpoints
    ckpt_dir_name = f"checkpoints/bs_{args.batch_size}_opt_{optimizer.__class__.__name__}_epochs_{args.epochs}"
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

            step += 1

            # clear the gradients
            optimizer.zero_grad()

            # log the loss
            wandb.log({"Train Loss": loss.item()})

            # Call calculate_accuracy every 100 steps
            if step % 100 == 0:
                accuracy = calculate_accuracy(model, tokenizer, dev_dataloader, device)
                print(f"Validation Accuracy at step {step}: {accuracy}")
                wandb.log({"Validation Accuracy": accuracy})

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
        run_name = f"{config['mode']}_bs{config['batch_size']}_e{config['epochs']}"

        run = wandb.init(project='FinalThesis2023', entity='haoz', name=run_name, config=config)
    else:

        run = wandb.init(project='FinalThesis2023', entity='haoz', config=config)
        config = run.config
        run_name = f"{config['mode']}_bs{config['batch_size']}_sml{config['source_max_length']}_tml{config['target_max_length']}_e{config['epochs']}"
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
    optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True)

    # Initialize dataloaders using DataLoaderCreator
    data_loader_creator = DataLoaderCreator(source_max_length=args.source_max_length, target_max_length=args.target_max_length, batch_size=args.batch_size)
    train_dataloader, dev_dataloader, test_dataloader = data_loader_creator.create_dataloaders(data_type=args.data_type)

    # Watch the model
    wandb.watch(model, log_freq=100)

    # Train the model
    train_model(model, optimizer, train_dataloader, dev_dataloader, args, tokenizer)



if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="'sweeps' for hyperparameter search, 'train' for training")
    parser.add_argument("--machine", type=str, required=True, help="'slurm' for SLURM, 'local' for local machine")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--data_type", type=str, default="csqa-debug")
    parser.add_argument("--batch_size", type=int, default=8)
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
    sweep_name = f"T5HS_{args.machine}_{args.data_type}"
    sweep_config = {
        "name": sweep_name,
        "method": "grid",  # or "bayes"
        "metric": {
            "name": "Validation Loss",
            "goal": "minimize",
        },
        "parameters": {
            "batch_size": {"values": [5]},
            "source_max_length": {"values": [512]},
            "target_max_length": {"values": [128]},
            "epochs": {"values": [5]},
            "model_name": {"value": args.model_name},
            "data_type": {"value": args.data_type},
            "mode": {"value": args.mode},
        },
    }

    load_dotenv()
    api_key = os.getenv("WANDB_API")

    wandb.login(key=api_key)


    if args.mode == 'sweeps':
        sweep_id = wandb.sweep(sweep_config, project='FinalThesis2023', entity='haoz')
        wandb.agent(sweep_id, function=main, count=1)
    elif args.mode == 'train':
        # Fetch hyperparameters from args
        config = {
            "mode": args.mode,
            "batch_size": args.batch_size,
            "source_max_length": args.source_max_length,
            "target_max_length": args.target_max_length,
            "epochs": args.epochs,
            "model_name": args.model_name,
            "data_type": args.data_type,
        }
        main(config)
    else:
        print(f"Unknown mode {args.mode}")

