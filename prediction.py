# from torchtext.vocab import FastText

import torch 
from helper_functions import ToxicClassifier, token_encoder, encoder, front_padding, preprocessing, nlp, fasttext
import numpy as np 

max_seq_length, emb_dim = 64, 300
model = ToxicClassifier()

#################  LOADING MODEL  ##############################################################

train_modeled = torch.load('trained_models/trained_all_model')
model_state = train_modeled['model_state']
model = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=64)
model.load_state_dict(model_state)


###################################################   CLI   #####################################################################

# import argparse 
# parser = argparse.ArgumentParser(description='Comment')
# parser.add_argument('comments',  type=str, help='type your comment')
# args = parser.parse_args()
# comment = args.comments


#################################################################################################################################


# comment = 'Do something about him. He constantly blames me of vandalism, even though I have not done such.  See the current version of the Battle of Budapest article.'
# comment = 'The WaPo article says, in effect, that a Fed hate crime prosecution would require a racial motivation, but if Brown had been white shooting at him when fleeing would still have been an intentional attempt to deprive him of his civil rights, per Garner. Are you saying there is no Federal statute that allows the DOJ to prosecute him for this, despite the Constitutional violation?'
comment = "He said he's WORKING ON IT, The Rouge Penguin, have some patience, Don't be an asshole."


features = front_padding(encoder(preprocessing(comment), fasttext), max_seq_length) 
# print(features)

# print(fasttext.vectors)
embeddings = [fasttext.vectors[el] for el in features]

inputs = torch.stack(  embeddings )


model.eval()
with torch.no_grad():
    prediction = model.forward(inputs.flatten().unsqueeze(1))
    probability_test = torch.sigmoid(prediction)
    classes = probability_test > 0.5

pred = np.array(classes)
# print(pred)
    # print(classes)
    

toxic, severe_toxic, obscene, threat, insult, identity_hate, space = 'TOXIC', 'SEVERE_TOXIC', 'OBSCENE', 'THREAT', 'INSULT', 'IDENTITY_HATE', '---------------'

print('')
print('')
print(f'{toxic:15}|{severe_toxic:15}|{obscene:15}|{threat:15}|{insult:15}|{identity_hate}')
print(f"{space:15}|{space:15}|{  space:15}|{  space:15}|{  space:15}|{  space:15}")
print(f"{str(pred[0]):15}|{str(pred[1]):15}|{  str(pred[2]):15}|{  str(pred[3]):15}|{  str(pred[4]):15}|{  str(pred[5]):15}")



