#Packages
import torch
import torch.nn as nn

class QueryEmbedder(nn.Module):
    
    def __init__(self, **kwargs):
        super(QueryEmbedder,self).__init__()
        self.kwargs = kwargs
        self.embed = nn.Embedding(self.kwargs["vocab_size"],self.kwargs["embedding_size"])
        
        self.lstm = nn.LSTM(input_size = self.kwargs["embedding_size"], hidden_size = self.kwargs["hidden_size"], 
                            num_layers = self.kwargs["n_layers"],
                            dropout = self.kwargs["dropout_prob"], batch_first=True)
        self.dropout = nn.Dropout(self.kwargs["dropout_prob"])
        self.output_size = self.kwargs["output_size"]
        self.fc = nn.Linear(self.kwargs["hidden_size"], self.kwargs["output_size"])
        
    def forward(self, x, hidden):
        x = self.embed(x)
        _, (x,_) = self.lstm(x, hidden)
        x = self.fc(x)
        
        return x
    
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.kwargs["n_layers"], self.kwargs["batch_size"], self.kwargs["hidden_size"]).zero_(),
                      weight.new(self.kwargs["n_layers"], self.kwargs["batch_size"], self.kwargs["hidden_size"]).zero_())
        return hidden
        
        
class DocEmbedder(nn.Module):
    
    def __init__(self, **kwargs):
        super(DocEmbedder,self).__init__()
        self.kwargs = kwargs
        self.embed = nn.Embedding(self.kwargs["vocab_size"],self.kwargs["embedding_size"])
        self.lstm = nn.LSTM(input_size = self.kwargs["embedding_size"], hidden_size = self.kwargs["hidden_size"], 
                            num_layers = self.kwargs["n_layers"],
                            dropout = self.kwargs["dropout_prob"], batch_first=True)
        self.dropout = nn.Dropout(self.kwargs["dropout_prob"])
        self.output_size = self.kwargs["output_size"]
        self.fc = nn.Linear(self.kwargs["hidden_size"], self.kwargs["output_size"])
        
    def forward(self, x, hidden):
        
        x = self.embed(x)
        _, (x,_)= self.lstm(x,hidden)
        x = self.fc(x)
        return x
    
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.kwargs["n_layers"], self.kwargs["batch_size"], self.kwargs["hidden_size"]).zero_(),
                      weight.new(self.kwargs["n_layers"], self.kwargs["batch_size"], self.kwargs["hidden_size"]).zero_())
        return hidden
       
    
#Two-TowerModel
class TwoTower(nn.Module):
    
    def __init__(self, query_embedder, doc_embedder, **kwargs):
        super(TwoTower,self).__init__()
        self.kwargs = kwargs
        self.query_embedder = query_embedder
        self.doc_embedder = doc_embedder
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.sigmoid = nn.Sigmoid()
    
        
    def forward(self, query, doc, h):
        """input: query and document embeddings 
           output: logit 
        """
        query = self.query_embedder(query, h)
        doc = self.doc_embedder(doc, h) 
        sim = self.cos(query,doc)
        output = self.sigmoid(sim)
        
        return output
        
        