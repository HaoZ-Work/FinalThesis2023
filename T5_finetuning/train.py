import os
import torch
from torch.utils.data import DataLoader
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer
from data_utils import DataLoaderCreator
import argparse
from tqdm import tqdm
import wandb
from dotenv import load_dotenv


def main(args):

    # Start a new run
    load_dotenv()
    api_key = os.getenv("WANDB_API")

    wandb.login(key=api_key)
    wandb.init(project='FinalThesis2023', entity='haoz', name='T5_finetuning_csqa')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # move model to the device
    model = model.to(device)

    # initialize optimizer
    optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True)

    # Initialize dataloaders using DataLoaderCreator
    data_loader_creator = DataLoaderCreator(source_max_length=args.source_max_length, target_max_length=args.target_max_length, batch_size=args.batch_size)
    train_dataloader, dev_dataloader, test_dataloader = data_loader_creator.create_dataloaders(data_type=args.data_type)

    # Watch the model
    wandb.watch(model, log_freq=100)

    # create directory for checkpoints
    ckpt_dir_name = f"checkpoints/bs_{args.batch_size}_opt_{optimizer.__class__.__name__}_epochs_{args.epochs}"
    os.makedirs(ckpt_dir_name, exist_ok=True)

    # Initialize best validation loss to infinity
    best_val_loss = float("inf")
    best_model_path = None  # Keep track of best model path

    # training loop
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

            # clear the gradients
            optimizer.zero_grad()

            # log the loss
            wandb.log({"Train Loss": loss.item()})

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
            best_model_path = f"{ckpt_dir_name}/model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), best_model_path)

    if best_model_path is not None:
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model.")

    # Evaluate on test set
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # accumulate loss
            total_loss += outputs.loss.item()

    print(f"Test Loss: {total_loss / len(test_dataloader)}")
    wandb.log({"Test Loss": total_loss / len(test_dataloader)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--data_type", type=str, default="csqa-debug")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--source_max_length", type=int, default=512)
    parser.add_argument("--target_max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    main(args)
