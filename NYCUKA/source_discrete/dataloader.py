from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
#from eda import *
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import pandas as pd
from functools import partial
import torch
import csv

aug_eda = naf.Sequential([
        naf.Sometimes([naw.RandomWordAug(action="swap")]),
        naf.Sometimes([naw.RandomWordAug(action="delete")]),
        naf.Sometimes([naw.SynonymAug(aug_src='wordnet')])
    ])

emotion_list = ['disgust/sad', 'fear/sad', 'anger/sad', 'happiness/fear', 'fear/anger', 'disgust/anger', 'disgust/anger/sad', 'surprise/anger/sad', 'surprise/disgust', 'surprise/disgust/anger', 'no emotion',
                'disgust', 'sad', 'fear', 'anger', 'happiness', 'surprise']
# mapping to index
emotion_to_index = {emotion: idx for idx, emotion in enumerate(emotion_list)}

#new_data is the whole data
def collate_fn(batch, new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")):
    #print(batch) # utterance context
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    llm_tokenizer.pad_token = llm_tokenizer.unk_token
    #print(tokenizer)

    aug_=[]
    aug_batch_neg = []
    #print("tmp")
    for ele in batch:
        if isinstance(ele['Q'], list):
            while ele['Q']:
                item = ele['Q'].pop(0) #until str
                #print(item)
                if isinstance(item, str):
                    item = aug_eda.augment(item,1)
                    aug_.append(item)
                    break
                else:
                    item = aug_eda.augment(item,1)
                    aug_.extend(item)
        else:
            aug_.append(ele['Q'])
            
        current_keys = list(new_data.keys())
        current_keys.remove(ele['context'])
        
        for key in current_keys:
            # print(key)
            # print(new_data)

            neg_data = new_data[key][0]['Q'] #0 can change to rand
            if isinstance(neg_data, list):
                while neg_data:
                    item = neg_data.pop(0) #until str
                    #print(item)
                    if isinstance(item, str):
                        item = aug_eda.augment(item,1) 
                        aug_batch_neg.append(item)
                        break
                    else:
                        item = aug_eda.augment(item,1) 
                        aug_batch_neg.extend(item)
                break 
                    # do once now 
            else :
                aug_batch_neg.append(neg_data)

    ## Tokenizing             
    emotion_batch = torch.tensor([emotion_to_index[ele['context']] for ele in batch])
    text_batch = tokenizer([ele['Q'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    #prompt_batch = tokenizer([ele['prompt'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    emo = tokenizer([ele['context'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    llm_text_batch = llm_tokenizer([ele['Q'] for ele in batch], return_tensors='pt',max_length=128,truncation=True, padding=True)
    llm_target_batch = llm_tokenizer([ele['A'] for ele in batch], return_tensors='pt',max_length=128,truncation=True, padding=True)
    
    aug_batch = tokenizer(aug_, return_tensors='pt', max_length=128, truncation=True, padding=True)
    aug_batch_neg = tokenizer(aug_batch_neg, return_tensors='pt', max_length=128, truncation=True, padding=True)
    #print(aug_batch_neg.input_ids.shape,aug_batch.input_ids.shape)
    
    return emotion_batch, text_batch, aug_batch, aug_batch_neg,llm_text_batch, llm_target_batch

def collate_fn_val(batch, new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")):
    #print(batch) # utterance context
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    llm_tokenizer.pad_token = llm_tokenizer.unk_token
    #print(tokenizer)

    ## Tokenizing             
    emotion_batch = torch.tensor([emotion_to_index[ele['context']] for ele in batch])
    # emo = tokenizer([ele['context'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    # emotion_batch = torch.tensor([ele['context'] for ele in batch])
    text_batch = tokenizer([ele['Q'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    llm_text_batch = llm_tokenizer([ele['Q'] for ele in batch], return_tensors='pt',max_length=128,truncation=True, padding=True)
    llm_target_batch = ([ele['A']] for ele in batch)
    input_batch = ([ele['Q']] for ele in batch)
    emo_label = ([ele['context']] for ele in batch)
    
    return emotion_batch, emo_label, text_batch, llm_text_batch, llm_target_batch, input_batch

def data_loader(a,new_data, batch_size = 16):
    return DataLoader(a, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, new_data=new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def data_loader_val(b, new_data_val, batch_size = 16):
    return DataLoader(b, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn_val, new_data=new_data_val, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def predata():
    dataset = pd.read_csv('data/train.csv')
    dataset_val = pd.read_csv('data/valid.csv')
    dataset_test = pd.read_csv('data/test.csv')
    #,dialogue_id,turns,text,emotion,act
    idx = None
    q = None
    new_data = {}
    new_data_val = {}
    new_data_test = {}
    #print(dataset)
    for i, ele in tqdm(dataset.iterrows(), total=len(dataset), ncols=0):
        #print(idx,ele['dialogue_id'])
        if idx == ele['dialogue_id']:
            one_data={'Q': q, 'A': ele['text'], 'context': ele['emotion']}
            if ele['emotion'] in new_data:
                new_data[ele['emotion']].append(one_data)
                # print('ya')
            else:
                new_data[ele['emotion']]=[ one_data ]
            q = ele['text']
        else:
            idx = ele['dialogue_id']
            q = ele['text']
            continue

    idx = None
    q = None    
    ##val    
    for i, ele in tqdm(dataset_val.iterrows(), total=len(dataset_val), ncols=0):
        #print(idx,ele['dialogue_id'])
        if idx == ele['dialogue_id']:
            one_data={'Q': q, 'A': ele['text'], 'context': ele['emotion']}
            if ele['emotion'] in new_data_val:
                new_data_val[ele['emotion']].append(one_data)
                # print('ya')
            else:
                new_data_val[ele['emotion']]=[ one_data ]
            q = ele['text']
        else:
            idx = ele['dialogue_id']
            q = ele['text']
            continue
    idx = None
    q = None  
    ##test
    for i, ele in tqdm(dataset_test.iterrows(), total=len(dataset_test), ncols=0):
        #print(idx,ele['dialogue_id'])
        if idx == ele['dialogue_id']:
            one_data={'Q': q, 'A': ele['text'], 'context': ele['emotion']}
            if ele['emotion'] in new_data_test:
                new_data_test[ele['emotion']].append(one_data)
                # print('ya')
            else:
                new_data_test[ele['emotion']]=[ one_data ]
            q = ele['text']
        else:
            idx = ele['dialogue_id']
            q = ele['text']
            continue
    
    # dict[list[dict]]
    a=[] #train
    b=[] #val
    c=[] #test
    # a=[1,2,3]
    # a.extend([4,5,6])
    for k, v in new_data.items():
        a.extend(v)
    for k, v in new_data_val.items():
        b.extend(v)
    for k, v in new_data_test.items():
        c.extend(v)
    #print(new_data)
    print(a[22]['A']) #sample neg context
    print(b[20]['context'])  #sample neg context
    print(c[30]['context'])
    
    # return a, new_data, b, new_data_val 
    return c, new_data_test

if __name__ == "__main__": 
    a, new_data, b, new_data_val, c, new_data_test   = predata()
    