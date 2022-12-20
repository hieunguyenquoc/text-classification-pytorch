import torch
import torch.nn as nn
from load_pretrain_embedded import load_pretrain_embedded

class TextClassification(nn.Module):
    def __init__(self,args):
        super(TextClassification,self).__init__()

        #define hyperparameter
        self.batch_size = args.batch_size 
        self.hidden_dim = args.hidden_dim
        self.LTSM_layer = args.lstm_layers
        self.input_size = args.max_words

        #load glove pretrain embedding
        pretrain_model_embedded = load_pretrain_embedded()
        self.dropout = nn.Dropout(0.5)
        
        # self.embedding = nn.Embedding(self.input_size,self.hidden_dim,padding_idx=0) 
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_model_embedded).float())
        ''' input_size : size of vocab
            hidden_dim : dimension of output vecto
            '''      
        self.lstm = nn.LSTM(input_size = self.hidden_dim, hidden_size = self.hidden_dim, num_layers = self.LTSM_layer, batch_first = True)
        ''' input_size : size of embedding vecto
            hidden_size : size of hidden state and cell statep (output)
            num_layers : number of lstm layer
           '''
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim*2)
        self.fc2 = nn.Linear(in_features=self.hidden_dim*2, out_features=1)
        '''
            in_features : size of input
            out_features : size of output
        '''
    def forward(self,x):

        #define hidden state and cell state as 0 at the beginning
        h = torch.zeros((self.LTSM_layer, x.size(0),self.hidden_dim))
        c = torch.zeros((self.LTSM_layer, x.size(0),self.hidden_dim))

        #Initialization fo hidden and cell states

        #define pipeline for each token
        out = self.embedding(x)

        #pass throught lstm
        out, (hidden, cell) = self.lstm(out, (h,c))

        #call dropout
        out = self.dropout(out)

        #go throught linear layer
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)

        out = torch.sigmoid(self.fc2(out))
        # print(out.size())
        return out
            
            



