

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchtext.vocab import FastText
import spacy 



class ToxicClassifier(nn.Module):
     def __init__(self, max_seq_len=32, emb_dim=300, hidden=64):
          super(ToxicClassifier, self).__init__()
          self.input_layer   = nn.Linear(max_seq_len*emb_dim, hidden)
          self.first_hidden  = nn.Linear(hidden, hidden)
          self.second_hidden = nn.Linear(hidden, hidden)
          self.third_hidden  = nn.Linear(hidden, hidden)

          self.output = nn.Linear(hidden, 6)
          
          self.sigmoid    = nn.Sigmoid()


     def forward(self, inputs):
          x = F.relu(self.input_layer(inputs.squeeze(1).float()))

          x = F.relu(self.first_hidden(x))
          x = F.relu(self.first_hidden(x))
          x = F.relu(self.second_hidden(x))
          x = F.relu(self.third_hidden(x))

          output = self.output(x)

          return output



def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def front_padding(list_of_indexes, max_seq_len, padding_index=0):
    new_out = (max_seq_len - len(list_of_indexes))*[padding_index] + list_of_indexes
    return new_out[:max_seq_len] 


nlp = spacy.load("en_core_web_sm") 
def preprocessing(sentence):
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return tokens


model = ToxicClassifier()

model.eval()

max_seq_length = 50
emb_dim = 300 

fasttext = FastText("simple") 
