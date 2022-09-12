import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, set_seed, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
import logging
import argparse
from tqdm import tqdm
import os
import wandb
from modeling import SequenceModel
from dataset import ExplaGraphs
import os
import random
import numpy as np


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#torch.backends.cuda.matmul.allow_tf32 = True #(RTX3090 and A100 GPU only)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate graph impact on stance predicition")

    parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="The batch size to use during training.",
    )
    
    parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="The weight decay to use during training.",
    )

    parser.add_argument(
    "--model",
    type=str,
    default="bert-base-uncased",
    help="The pretrained model to use",
    )


    parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="The number of epochs.",
    )

    parser.add_argument(
    "--debug",
    action='store_true',
    help="Trigger debug mode",
    )

    parser.add_argument(
    "--use_graphs",
    action='store_true',
    help="Trigger graph mode",
    )

    parser.add_argument(
    "--test",
    action='store_true',
    help="Trigger test eval",
    )

    parser.add_argument(
    "--pg",
    action='store_true',
    help="Trigger PG mode",
    )

    parser.add_argument(
    "--rg",
    action='store_true',
    help="Trigger rand g mode",
    )
    
    parser.add_argument(
    "--lr",
    type=float,
    default=3e-5,
    help="The learning rate).",
    )

    parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The rng seed",
    )

    args = parser.parse_args()

    return args


def main(w_config=None):
    args = boo
    with wandb.init(config=w_config):
        config = wandb.config
        logging.info(f"Initialised")
        model_name = args["model_name"]
        train = ExplaGraphs(model_name, split="train", use_graphs=args["use_graphs"], use_pg=args["pg"], use_rg=args["rg"])
        val = ExplaGraphs(model_name, split="val", use_graphs=args["use_graphs"], use_pg=args["pg"], use_rg=args["rg"])
        train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

        criterion = CrossEntropyLoss()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        steps = args["epochs"] * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=steps)

        if args["debug"]:
            logging.info("Debug mode activated.")
            og, dec = train.get_decoded_sample(0)
            logging.info(f"Sample from dataset. Original was: ---- {og} ---- , decoded was ---- {dec} ---- ")
        
        patience = 2
        best_acc = 0.0
        losses = []
        for epoch in range(args["epochs"]):
            logging.info(f"Staring training at epoch {epoch}")
            model.train()
            train_loss = 0.0
            for i, (input_ids, attention_masks, y) in enumerate(tqdm(train_loader)):
                y = torch.LongTensor(y)
                optimizer.zero_grad()
                input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
                y_hat = model(input_ids, attention_masks).logits
                loss = criterion(y_hat, y)
                train_loss += loss.item()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                if args["debug"]:
                    break
                else:
                    if i % 10 == 0:
                        wandb.log({"train_loss_batch": sum(losses)/len(losses)})

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                n = 0
                for i, (input_ids, attention_masks, y) in enumerate(tqdm(val_loader)):
                    input_ids = input_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    y = torch.LongTensor(y)
                    y = y.to(device)
                    out = model(input_ids=input_ids, attention_mask=attention_masks).logits
                    y_hat = nn.Softmax(out)
                    y_hat = (torch.argmax(out, dim=1))
                    correct += (y_hat == y).float().sum()
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    n += 1*config.batch_size
                    if args["debug"]:
                        break

                accuracy = correct / n

            t_l = train_loss / len(train_loader)
            v_l = val_loss / len(val_loader)
            logging.info(f"Epoch {epoch}, avg. train loss: {t_l} avg. val loss: {v_l}. Val. accuracy: {accuracy}")
            if not args["debug"]:
                wandb.log({"train_loss_epoch": t_l})
                wandb.log({"val_loss": v_l})
                wandb.log({"accuracy": accuracy})
            if accuracy > best_acc:
                patience = 2 #reset patience
                best_acc = accuracy
                torch.save({
                'model_state_dict': model.state_dict(),
                }, "./models/best_model.pt")
            else:
                patience -= 1
                if patience == 0:
                    logging.info(f"Early stopping at epoch {epoch} with accuracy {accuracy}")
                    break
            
        
        if args["test"]:
            test = ExplaGraphs(model_name, split="test",  use_graphs=args["use_graphs"], use_pg=args["pg"], use_rg=args["rg"])
            test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=True)
            checkpoint = torch.load("./models/best_model.pt")
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
                    out = model(input_ids=input_ids, attention_mask=attention_masks).logits
                    y_hat = nn.Softmax(out)
                    y_hat = (torch.argmax(out, dim=1))
                    correct += (y_hat == y).float().sum()
                    n += 1*config.batch_size
                
                test_accuracy = correct / n
            
            logging.info(f" Test. accuracy: {test_accuracy}")
            if not args["debug"]:
                wandb.log({"test_score": test_accuracy})
        
    

if __name__ == "__main__":
    dart = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    


    sweep_config = {
    'method': 'random',
    'program': 'train.py'
    }

    parameters_dict = {
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0.1e-6,
        'max': 0.01
      },
    'weight_decay': {
          'values': [0.001, 0.002, 0.0003]
        },
    'batch_size': {
        'values': [8, 16, 32]
        },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="graph_impact")



    if dart.pg and dart.rg:
        logging.info("Cant use RG and PG simoultaniously")
        exit()

    set_seed(dart.seed)
    torch.manual_seed(dart.seed)
    torch.cuda.manual_seed(dart.seed)
    torch.cuda.manual_seed_all(dart.seed)
    np.random.seed(dart.seed)
    random.seed(dart.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    boo = {
        "learning_rate": dart.lr,
        "epochs": dart.epochs,
        "batch_size": dart.batch_size,
        "weight_decay": dart.weight_decay,
        "model_name": dart.model,
        "use_graphs": dart.use_graphs,
        "pg": dart.pg,
        "rg": dart.rg,
        "seed": dart.seed,
        "debug": dart.debug,
        "test": dart.test
    }


    wandb.agent(sweep_id, main, count=10)
    #main(args,sweep_config)
    
