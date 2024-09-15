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
emotion_to_index = {emotion: idx for idx, emotion in enumerate(emotion_list)}

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

def data_loader_val(b, new_data_val, batch_size = 16):
    return DataLoader(b, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn_val, new_data=new_data_val, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def predata():
    # dataset = load_dataset("empathetic_dialogues",split='train')
    # dataset_val = load_dataset("empathetic_dialogues",split='validation')
    dataset_test = load_dataset("empathetic_dialogues",split='test')

    idx = None
    q = None
    new_data = {}
    new_data_val = {}
    new_data_test = {}
    
    idx = None
    q = None  
    #print(dataset_test)
    ##test
    for ele in tqdm(dataset_test, ncols=0):
        if idx == ele['conv_id']:
            one_data={'Q': q, 'A': ele['utterance'], 'context': ele['context']}
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
    
    for k, v in new_data_test.items():
        c.extend(v)
    # print(a[2]['context']) #sample neg context
    # print(b[20]['context']) #sample neg context
    print(c[30]['context'])
    return c, new_data_test
