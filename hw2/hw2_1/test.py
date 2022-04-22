from model import RNNMODELS, ENCODER, DECODER, ATTENTION
from dataloader import VideoDataset, TestDataset, Vocabulary, Collate
from torch.utils.data import DataLoader
import torch
from bleu_eval import BLEU
import json
import pandas as pd
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_dataset = TestDataset(f'{sys.argv[1]}/feat/')
testloader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, shuffle=False)

#create vocabulary dictionary from training captions
df = pd.json_normalize(json.load(open('training_label.json')), meta=['id'], record_path=['caption'])
df.columns = ['caption', 'id']
caption = df['caption']
vocabs = Vocabulary(caption)
vocabs.build_vocab()


if not torch.cuda.is_available():
    model = torch.load('Models/model2.h5', map_location=device)
else:
    model = torch.load('Models/model2.h5')
    
ids=[]
#for i in tqdm(range(1)):
for idx, features in testloader:
    model.eval()
    with torch.no_grad():
        #features, hidden = model.encoder(features.to(device))
        features, state = model.encoder(features.to(device))
        logprobs,caps = model.decoder.inference(encoder_hidden=None, encoder_output=features, vocab=vocabs)
        caption = ' '.join([vocabs.itow[i] for i in caps[0].tolist()])
        caption = caption.split('<EOS>')[0]
        if idx not in ids:
            ids.append(idx)
            with open(sys.argv[2], 'a') as f:
                f.write(idx[0]+','+caption+'\n')



test = json.load(open(f'{sys.argv[1]}/testing_label.json','r'))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))