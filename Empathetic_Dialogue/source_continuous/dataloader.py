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

aug_eda = naf.Sequential([
        naf.Sometimes([naw.RandomWordAug(action="swap")]),
        naf.Sometimes([naw.RandomWordAug(action="delete")]),
        naf.Sometimes([naw.SynonymAug(aug_src='wordnet')])
    ])

emotion_list = ['sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful', 'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed', 'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised', 'nostalgic', 'confident', 'furious', 'disappointed', 'caring', 'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful', 'content', 'impressed', 'apprehensive', 'devastated']
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
            neg_data = new_data[key][1]['Q'] #1 can change to rand
            #print(neg_data)
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
    emo = tokenizer([ele['context'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    llm_text_batch = llm_tokenizer([ele['Q'] for ele in batch], return_tensors='pt',max_length=128,truncation=True, padding=True)
    llm_target_batch = llm_tokenizer([ele['A'] for ele in batch], return_tensors='pt',max_length=128,truncation=True, padding=True)
    
    aug_batch = tokenizer(aug_, return_tensors='pt', max_length=128, truncation=True, padding=True)
    aug_batch_neg = tokenizer(aug_batch_neg, return_tensors='pt', max_length=128, truncation=True, padding=True)
    #print(aug_batch_neg.input_ids.shape,aug_batch.input_ids.shape)
    
    return emo, text_batch, llm_text_batch, llm_target_batch

def collate_fn_val(batch, new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")):
    #print(batch) # utterance context
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    llm_tokenizer.pad_token = llm_tokenizer.unk_token
    #print(tokenizer)

    ## Tokenizing             
    # emotion_batch = torch.tensor([emotion_to_index[ele['context']] for ele in batch])
    emo = tokenizer([ele['context'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    text_batch = tokenizer([ele['Q'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    llm_text_batch = llm_tokenizer([ele['Q'] for ele in batch], return_tensors='pt',max_length=128,truncation=True, padding=True)
    llm_target_batch = ([ele['A']] for ele in batch)
    input_batch = ([ele['Q']] for ele in batch)
    emo_label = ([ele['context']] for ele in batch)
    prompt_batch = ([ele['prompt']] for ele in batch)
    
    return prompt_batch, emo, text_batch, llm_text_batch, llm_target_batch, input_batch, emo_label

def data_loader(a,new_data, batch_size = 16):
    return DataLoader(a, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, new_data=new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def data_loader_val(b, new_data_val, batch_size = 16):
    return DataLoader(b, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn_val, new_data=new_data_val, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def predata():
    dataset = load_dataset("empathetic_dialogues",split='train')
    dataset_val = load_dataset("empathetic_dialogues",split='validation')
    dataset_test = load_dataset("empathetic_dialogues",split='test')

    idx = None
    q = None
    new_data = {}
    new_data_val = {}
    new_data_test = {}
    idx = None
    q = None
    new_data = {}
    new_data_val = {}
    for ele in tqdm(dataset, ncols=0):
        if idx == ele['conv_id']:
            one_data={'Q': q, 'A': ele['utterance'], 'context': ele['context'], 'prompt': ele['prompt']}
            if ele['context'] in new_data:
                new_data[ele['context']].append(one_data)
            else:
                new_data[ele['context']]=[ one_data ]
            q = ele['utterance']
        else:
            idx = ele['conv_id']
            q = ele['utterance']
            continue
    idx = None
    q = None    
    ##val    
    for ele in tqdm(dataset_val, ncols=0):
        if idx == ele['conv_id']:
            one_data={'Q': q, 'A': ele['utterance'], 'context': ele['context'], 'prompt': ele['prompt']}
            if ele['context'] in new_data_val:
                new_data_val[ele['context']].append(one_data)
            else:
                new_data_val[ele['context']]=[ one_data ]
            q = ele['utterance']
        else:
            idx = ele['conv_id']
            q = ele['utterance']
            continue
    idx = None
    q = None  
    #print(dataset_test)
    ##test
    for ele in tqdm(dataset_test, ncols=0):
        if idx == ele['conv_id']:
            one_data={'Q': q, 'A': ele['utterance'], 'context': ele['context'], 'prompt': ele['prompt']}
            if ele['context'] in new_data_test:
                new_data_test[ele['context']].append(one_data)
            else:
                new_data_test[ele['context']]=[ one_data ]
            q = ele['utterance']
        else:
            idx = ele['conv_id']
            q = ele['utterance']
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
    print(a[2]['context']) #sample neg context
    print(b[20]['context']) #sample neg context
    print(c[30]['prompt'])
    return a, new_data, b, new_data_val, c, new_data_test
    #new_data[neg][rand]

#a, new_data = predata()

if __name__ == "__main__": 
    a, new_data, b, new_data_val   = predata()
    batch_size = 16
    loader = data_loader(a, batch_size)
        
    for emotion_batch, text_batch ,aug_batch, aug_batch_neg, llm_text, llm_target in tqdm(loader): ## text positive negative

        break