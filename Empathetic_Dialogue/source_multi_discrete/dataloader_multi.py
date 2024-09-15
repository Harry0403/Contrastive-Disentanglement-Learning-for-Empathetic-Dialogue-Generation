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

def collate_fn_val(batch, new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")):
    #print(batch) # utterance context
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    llm_tokenizer.pad_token = llm_tokenizer.unk_token
    #print(tokenizer)
    # orig_batch = []
    role = ["user", "assistant"]
    messages=[]
    chat_template = []
    chat_template_llama2 = []
    i = 0
    for ele in batch:
        for item in ele['Q']:
            messages.append({"role": role[i%2], "content": item})
            i+=1
        chat_tmp = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_template.append(chat_tmp)
        chat_template_llama2.append(chat_tmp[:3]+"<<SYS>>\nKeep the sentences briefly.\n<</SYS>>"+chat_tmp[3:])

    ## Tokenizing             
    emotion_batch = torch.tensor([emotion_to_index[ele['context']] for ele in batch]) #this is for discrete classifer
    text_batch = tokenizer(chat_template, return_tensors='pt', max_length=128, truncation=True, padding=True)
    # emo = tokenizer([ele['context'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True) # this is for continuous classifer
    # text_batch = tokenizer([ele['Q'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    llm_text_batch = llm_tokenizer(chat_template, return_tensors='pt', max_length=128, truncation=True, padding=True)
    llm_target_batch = ([ele['A']] for ele in batch)
    
    input_batch = (chat_template)
    emo_label = ([ele['context']] for ele in batch)
    prompt_batch = ([ele['prompt']] for ele in batch)
    
    return prompt_batch, emotion_batch, text_batch, llm_text_batch, llm_target_batch, input_batch, emo_label

def data_loader(a,new_data, batch_size = 16):
    return DataLoader(a, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, new_data=new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def data_loader_val(b, new_data_val, batch_size = 16):
    return DataLoader(b, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn_val, new_data=new_data_val, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def predata():
    
    dataset_test = load_dataset("empathetic_dialogues",split='test')

   
    idx = None
    q = None    
    new_data_test = {}
    prev_context = None  
    prev_conv_id = None  
    # ccc = 0

    for ele in tqdm(dataset_test, ncols=0):
        if idx == ele['conv_id']:
            q.append(ele['utterance'])

            one_data = {
                'Q': q[:-1],
                'A': q[-1],
                'context': ele['context'],
                'prompt': ele['prompt'],
                'utterance_idx': ele['utterance_idx']
            }

            
            if prev_conv_id == ele['conv_id'] and prev_context in new_data_test:
                new_data_test[prev_context].pop()

    
            if ele['context'] in new_data_test:
                new_data_test[ele['context']].append(one_data)
            else:
                new_data_test[ele['context']] = [one_data]

            prev_context = ele['context']  
            prev_conv_id = ele['conv_id']  

        else:
            idx = ele['conv_id']
            q = [ele['utterance']]

        
    c=[] #test
    
    for k, v in new_data_test.items():
        c.extend(v)
    # print(a[2]['context'])
    print(c[20]['context']) 
    # print(new_data.keys())
    return c, new_data_test
    #new_data[neg][rand]


if __name__ == "__main__": 
    a, new_data, b, new_data_val   = predata()
    batch_size = 16
    loader = data_loader(a,new_data, batch_size)
        
    for emo, text_batch, prompt_batch, aug_batch, aug_batch_neg, llm_text_batch, llm_target_batch in tqdm(loader): ## text positive negative
        
        break
