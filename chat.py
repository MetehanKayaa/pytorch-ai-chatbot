import random
import json
import torch
import torch_directml
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import os


device = torch_directml.device()
# ---- BU KISMI GÜNCELLE ----
# chat.py dosyasının olduğu klasörün tam adresini bulur
script_dir = os.path.dirname(__file__) 

intents_path = os.path.join(script_dir, 'intents.json')
file_path = os.path.join(script_dir, 'data.pth')

with open(intents_path, 'r') as f:
    intents = json.load(f)

# weights_only kısmını False yapıyoruz ki hata vermeden tüm verileri yüklesin
data = torch.load(file_path, weights_only=False)
# ----------------------------

FILE='data.pth'
data=torch.load(FILE)
input_size=data['input_size']
hidden_size=data['hidden_size']
output_size=data['output_size']
all_words=data['all_words']
tags=data['tags']
model_state=data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name='Sam'
def get_response(msg):
    sentence=tokenize(msg)
    X=bag_of_words(sentence,all_words)
    X=X.reshape(1,X.shape[0])
    # Parantez içine X'i koyduk, float yaptık ve DirectML (AMD) cihazına yolladık:
    X = torch.from_numpy(X).to(dtype=torch.float).to(device)
    
    output=model(X)
    _,predicted=torch.max(output,dim=1)
    tag=tags[(predicted.item())]
    
    
    
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    
    if prob.item()>0.75:
    
        for intent in intents['intents']:
            if tag==intent['tag']:
                return random.choice(intent['responses'])
    
    
  
    return 'I do not understand'


        
   








