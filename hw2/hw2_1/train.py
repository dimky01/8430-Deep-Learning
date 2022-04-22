from model import RNNMODELS, ENCODER, DECODER, ATTENTION
from dataloader import VideoDataset, Vocabulary, Collate
from torch.utils.data import DataLoader
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#load train dataset
train_dataset = VideoDataset('training_data/feat/','training_label.json')
trainloader = DataLoader(dataset=train_dataset, batch_size=64, num_workers=0, shuffle=True, collate_fn=Collate(pad_idx=0))

learning_rate = 3e-4
epochs=2
input_size=4096
hidden_size=512
vocabs = train_dataset.vocab
vocab_size=len(vocabs.wtoi)
embed_dim=256
LOAD_MODEL = False

if LOAD_MODEL == True:
    model = torch.load('savedModels/model0.h5')
    model.to(device)
else:
    model = RNNMODELS(input_size=input_size, hidden_size=hidden_size, vocab_size=vocab_size, embed_dim=embed_dim)
    model.to(device)
    
criterion = nn.CrossEntropyLoss(ignore_index=vocabs.wtoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)


modelloss=[]
#def train(e):
model.train()
step=0
bestloss = 20
for epoch in range(0, epochs):
    trainloss = 0
    step = 0 
    for idx, features, captions in trainloader:
        if step%10==0:
            print(f'Epoch: {epoch+1}  Step: {step+1}')
        step+=1
        features = features.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()
        outputs, seq_pred = model(features, captions, 'train')
        #ground_truths = torch.transpose(captions,0,1)
        targets = captions[:, 1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        trainloss += loss.item()
        #print("Training loss", loss)x
        #step += 1
        #loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
    avgloss = trainloss/len(trainloader)
    modelloss.append(avgloss)
    if epoch%1==0:
        print('Epoch: {}    TotalLoss: {}'.format(epoch, avgloss))
    if avgloss < bestloss:
        bestloss = avgloss
        torch.save(model, f"Models/{'Model1'}.h5")


fig = plt.figure(figsize=(7, 7), dpi=80)
plt.subplot(2,1,2)
plt.plot(np.arange(0,epochs), modelloss, "b")
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.tight_layout()
fig.savefig('testplot.png')