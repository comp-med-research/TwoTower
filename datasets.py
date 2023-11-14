#Packages
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
import pandas as pd


class queries_docs(Dataset):
    
    def __init__(self, dataset_type="train", **kwargs):
        
        self.kwargs = kwargs
        self.dataset_type = dataset_type
        
        # Load SentencePiece tokenizer
        self.tokenizer_path = self.kwargs["tokenizer_path"]
        self.tokenize = spm.SentencePieceProcessor()
        self.tokenize.load(self.tokenizer_path)
        
        #location of csv file
        if self.dataset_type == "train":
            self.loc = self.kwargs["train_data_location"]
        elif self.dataset_type == "val":
            self.loc = self.kwargs["val_data_location"]
        else:
            self.loc = self.kwargs["test_data_location"]
            
        self.df = pd.read_csv(self.loc)
        
        
    def __len__(self):
        
        return len(self.df)
        
        
    def __getitem__(self, idx):
        
        #load data
        query = self.df["query"][idx]
        doc = self.df["examples"][idx]
        label = self.df["labels"][idx]
        
        #tokenize
        query = self.tokenize.encode(query)
        doc = self.tokenize.encode(doc)
        
        #add padding
        """<pad> token value is 32000"""
        query += [32000] * (500 - len(query))
        doc += [32000] * (500 - len(doc))
         
        #convert to torch tensor
        query = torch.tensor(query)
        doc = torch.tensor(doc)
        label = torch.tensor(label)
        
        return query, doc, label 
    
    
    
    
        
    