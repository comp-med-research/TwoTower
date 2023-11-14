#Packages
import torch
import torch.nn as nn
from torch import optim
import argparse
import math
import os
from tqdm import tqdm
import torch.nn.functional as F

#import model
from model import TwoTower, QueryEmbedder, DocEmbedder

#import data 
from datasets import queries_docs

#import wandb
import wandb

def my_collate_fn(batch):
    return tuple(zip(*batch))

def validate_model(model, valid_dl, loss_func):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        h = model.query_embedder.init_hidden()
        for i, (query, docs, labels) in tqdm(enumerate(valid_dl)):

            # Forward pass ➡
            outputs = model(query, docs,h)
            val_loss += loss_func(outputs.squeeze(1), labels.float())*labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
                
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)


#----------------------------
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity',    type=str, default='hali94') 
    parser.add_argument('--issue',   type=str, default='two_tower') #required=True
    parser.add_argument('--group',     type=str, default='DPP') 
    parser.add_argument('--name',      type=str, required=True) 
    args = parser.parse_args()
    
    
    # 'config-defaults.yaml' is automatically loaded into wandb.config
    wandb.init(entity=args.entity, project=args.issue, group=args.group, name=args.name)
    
    # load config file
    config = wandb.config
    
    # set seed
    torch.manual_seed(config.seed)
    
    #define dataset
    train_data = queries_docs(**dict(config))
    val_data = queries_docs(dataset_type="val",**dict(config))
    
    train_dl = torch.utils.data.DataLoader(dataset=train_data, 
                                         batch_size=config.batch_size, 
                                         shuffle=True,drop_last = True)#collate_fn=my_collate_fn
    
    val_dl = torch.utils.data.DataLoader(dataset=val_data, 
                                         batch_size=config.batch_size, 
                                         shuffle=False,drop_last = True)#collate_fn=my_collate_fn
    
    # Define model 
    query_embedder = QueryEmbedder(**dict(config))
    doc_embedder = DocEmbedder(**dict(config))
    model = TwoTower(query_embedder=query_embedder,doc_embedder=doc_embedder,**dict(config))

    # Define steps per epoch
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # Make the loss and optimizer
    loss_func = nn.MSELoss()
    optimizer = optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    # Training
    example_ct = 0
    step_ct = 0
    model.train()
    for epoch in range(config.max_epochs):
        h = model.query_embedder.init_hidden()
        for step, (queries, docs, labels) in tqdm(enumerate(train_dl)):
            h = tuple([e.data for e in h])
            model.zero_grad()
            
#             print(f"queries shape is {queries.shape}")
#             print(f"docs shape is {docs.shape}")
#             print(f"labels shape is {labels.shape}")

            outputs = model(queries,docs,h) 
            train_loss = loss_func(outputs.squeeze(1), labels.float())
#             optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            example_ct += len(queries)
            metrics = {"train/train_loss": train_loss, 
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                       "train/example_ct": example_ct}
            
            if step + 1 < n_steps_per_epoch:
                # Log train metrics to wandb 
                wandb.log(metrics)
                
            step_ct += 1

        val_loss, accuracy = validate_model(model, val_dl, loss_func)

        #Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")
        
    
    
    #make directory for checkpoint if needed
    path = f'trained_weights/{args.issue}-{args.name}/'
    # Check whether the specified path exists or not
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
        
    #save model after training is finished    
    torch.save(model.state_dict(), path+"model.pth")


if __name__ == '__main__':
    main()
    
    

