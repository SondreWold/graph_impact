import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, set_seed, AutoModelForSequenceClassification, AutoModelForMultipleChoice, Trainer, TrainingArguments
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss, BCELoss
import logging
import argparse
from tqdm import tqdm
import os
import wandb
from modeling import SequenceModel, MCQA
from dataset import ExplaGraphs,CopaDataset
import os
import random
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'mps'
#torch.backends.cuda.matmul.allow_tf32 = True #(RTX3090 and A100 GPU only)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate graph impact on stance predicition")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use during training.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay to use during training.") 
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="The pretrained model to use")
    parser.add_argument("--task", type=str, default="copa", help="The task to train on")
    parser.add_argument("--epochs", type=int, default=3, help="The number of epochs.")
    parser.add_argument("--debug", action='store_true', help="Trigger debug mode")
    parser.add_argument("--use_graphs", action='store_true', help="Trigger graph mode")
    parser.add_argument("--test", action='store_true', help="Trigger test eval")
    parser.add_argument("--pgg", action='store_true', help="Trigger PathGenerator mode")
    parser.add_argument("--pgl", action='store_true', help="Trigger PathGeneratorLinked mode")
    parser.add_argument("--el", action='store_true', help="Trigger Entity Linker graph mode")
    parser.add_argument("--rg", action='store_true', help="Trigger string matching and retrieve mode")
    parser.add_argument("--lr", type=float, default=3e-5, help="The learning rate).")
    parser.add_argument("--seed", type=int, default=42, help="The rng seed")
    parser.add_argument("--gradient_clip", action='store_true', help="The gradient clip")
    parser.add_argument("--beta", type=float, default=1, help="The adam momentum")
    parser.add_argument("--patience", type=int, default=2, help="The patience value")
    parser.add_argument("--dropout", type=float, default=0.2, help="The dropout value")

    args = parser.parse_args()
    return args


def main(args):
    print("==========================================================================================")
    logging.info(f"Initialised training on task: {args.task.upper()}, debug={args.debug}")
    print("==========================================================================================")
    print("\n")
    logging.info(f"Use graphs={args.use_graphs}, use GPT-2 generated graph with gold head and tail={args.pgg}, use GPT-2 generated graph with linked head and tail={args.pgl}, use retrieved graphs={args.rg}, , use linked graphs={args.el}")


    model_name = args.model_name

    if args.task == "expla":
        logging.info(f"Init train dataset")
        train = ExplaGraphs(model_name, split="train", use_graphs=args.use_graphs, use_pgg=args.pgg, use_pgl=args.pgl, use_rg=args.rg, use_el=args.el)
        logging.info(f"Init validation dataset")
        val = ExplaGraphs(model_name, split="val", use_graphs=args.use_graphs, use_pgg=args.pgg, use_pgl=args.pgl, use_rg=args.rg, use_el=args.el)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
        #model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model = SequenceModel(model_name, dropout=args.dropout).to(device)
    if args.task == "copa":
        logging.info(f"Init train dataset")
        train = CopaDataset(model_name, split="train", use_graphs=args.use_graphs, use_pgg=args.pgg, use_pgl=args.pgl, use_rg=args.rg, use_el=args.el)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        logging.info(f"Init validation dataset")
        val = CopaDataset(model_name, split="val", use_graphs=args.use_graphs, use_pgg=args.pgg, use_pgl=args.pgl, use_rg=args.rg, use_el=args.el)
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
        #model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
        model = MCQA(model_name, dropout=args.dropout).to(device)

    if not args.debug:
        config = {
            "task": args.task,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "uses_graphs": args.use_graphs,
            "uses_generated": args.pgg,
            "uses_retrieved": args.rg,
            "uses_linked": args.el,
            "model": model_name,
            "test": args.test,
            "beta2": args.beta,
        }

        wandb.init(project="graph_quality", config=config, entity="sondrewo")

    decoded_sample = train.get_decoded_sample(10)
    logging.info(f"Decoded sentence: {decoded_sample}")


    criterion = CrossEntropyLoss()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)
    #scheduler = get_linear_schedule_with_warmup(optimizer, 0.06*steps, steps)
    
    patience = args.patience
    best_acc = 0.0
    best_val_loss = 9999999
    losses = []
    for epoch in range(args.epochs):
        logging.info(f"Staring training at epoch {epoch}")
        model.train()
        train_loss = 0.0
        for i, (input_ids, attention_masks, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            y = torch.LongTensor(y)
            input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
            out = model(input_ids, attention_masks)
            loss = criterion(out, y)
            train_loss += loss.item()
            loss.backward()
            if args.gradient_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if args.debug:
                break

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            n = 0
            for i, (input_ids, attention_masks, y) in enumerate(tqdm(val_loader)):
                y = torch.LongTensor(y)
                input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_masks)
                y_hat = torch.argmax(out, dim=-1)
                correct += (y_hat == y).sum()
                loss = criterion(out, y)
                val_loss += loss.item()
                n += len(y)
                if args.debug:
                    break

            accuracy = correct / n
            if not args.debug:
                wandb.log({"accuracy": accuracy})

        t_l = train_loss / len(train_loader)
        v_l = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch}, avg. train loss: {t_l} avg. val loss: {v_l}. Val. accuracy: {accuracy}")
        if not args.debug:
            wandb.log({"train_loss_epoch": t_l})
            wandb.log({"val_loss": v_l})
        if accuracy > best_acc:
            path = "models/copa/best_model.pt" if args.task == "copa" else "models/expla/best_model.pt"
            patience = args.patience #reset patience
            best_acc = accuracy
            torch.save({
            'model_state_dict': model.state_dict(),
            }, path)
        else:
            patience -= 1
            if patience == 0:
                logging.info(f"Early stopping at epoch {epoch} at val loss {v_l}")
                break
        
    
    if args.test:
        if args.task == "expla":
            test = ExplaGraphs(model_name, split="test",  use_graphs=args.use_graphs, use_pgg=args.pgg, use_pgl=args.pgl, use_rg=args.rg, use_el=args.el)
            test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
            checkpoint = torch.load("./models/expla/best_model.pt")
        if args.task == "copa":
            test = CopaDataset(model_name, split="test",  use_graphs=args.use_graphs, use_pgg=args.pgg, use_pgl=args.pgl, use_rg=args.rg, use_el=args.el)
            test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
            checkpoint = torch.load("./models/copa/best_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            correct = 0
            n = 0
            for i, (input_ids, attention_masks, y) in enumerate(tqdm(test_loader)):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                y = torch.LongTensor(y)
                y = y.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_masks)
                y_hat = torch.argmax(out, dim=-1)
                correct += (y_hat == y).float().sum()
                n += len(y)
            
            test_accuracy = correct / n
        
        logging.info(f" Test. accuracy: {test_accuracy}")
        if not args.debug:
            wandb.log({"test_score": test_accuracy})
        
    

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.pgg and args.rg:
        logging.info("Cant use RG and PG simoultaniously")
        exit()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(args)
    
