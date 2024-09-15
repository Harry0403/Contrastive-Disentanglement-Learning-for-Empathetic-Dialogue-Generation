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

    aug_ = []  # List to store augmented Qs
    aug_batch_neg = []  # List to store augmented negative samples
    orig_batch = []  # List to store original Qs
    q_counter = 0

    role = ["user", "assistant"]
    messages=[]
    i = 0
    chat_template = []
    for ele in batch:
        orig_qs = []
        for item in ele['Q']:
            if isinstance(item, str):
                # Augment the item for positive pair
                aug_item = aug_eda.augment(item, 1)[0]
                aug_.append(aug_item)
                orig_qs.append(item)
                
                q_counter += 1
            else:
                aug_item = aug_eda.augment(item, 1)[0]
                aug_.append(aug_item)
                orig_qs.append(item)
                
                q_counter += 1
            messages.append({"role": role[i%2], "content": item})
            i+=1
            # print(item)
        orig_batch.append(orig_qs)
        chat_template.append(llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        # print(chat_template)
        
        # Prepare negative samples from different emotion's first Q
        current_keys = list(new_data.keys())  # Get all emotion keys
        current_keys.remove(ele['context'])  # Remove current context to avoid selecting same emotion
        
        for key in current_keys:
            neg_data = new_data[key][0]['Q'][0]  # Select the first Q of the first record in different emotion
            if isinstance(neg_data, str):
                aug_neg = aug_eda.augment(neg_data, 1)[0]
                aug_batch_neg.append(aug_neg)
            else:
                aug_neg = aug_eda.augment(neg_data, 1)[0]
                aug_batch_neg.append(aug_neg)
            break  # Stop after one negative sample is generated

    
    text_batch = tokenizer(chat_template, return_tensors='pt', max_length=128, truncation=True, padding=True) # input

    # Tokenizing
    
    # emo = tokenizer([ele['context'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    emotion_batch = torch.tensor([emotion_to_index[ele['context']] for ele in batch]) #this is for discrete classifer
    llm_text_batch = llm_tokenizer([q for qs in orig_batch for q in qs], return_tensors='pt', max_length=128, truncation=True, padding=True)
    llm_target_batch = llm_tokenizer(text = [ele['A']+llm_tokenizer.eos_token for ele in batch], text_target = [ele['A']+llm_tokenizer.eos_token for ele in batch] , return_tensors='pt', max_length=128, truncation=True, padding=True)

    # tokenized_chat = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    aug_batch = tokenizer(aug_, return_tensors='pt', max_length=128, truncation=True, padding=True)
    aug_batch_neg = tokenizer(aug_batch_neg, return_tensors='pt', max_length=128, truncation=True, padding=True)

    return emotion_batch, text_batch, aug_batch, aug_batch_neg, llm_text_batch, llm_target_batch

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
    # chat_template_llama2 = []
    i = 0
    for ele in batch:
        for item in ele['Q']:
            messages.append({"role": role[i%2], "content": item})
            i+=1
        chat_tmp = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_template.append(chat_tmp)
        # chat_template_llama2.append(chat_tmp[:3]+"<<SYS>>\nKeep the sentences briefly.\n<</SYS>>"+chat_tmp[3:])
    # print(chat_template)
    ## Tokenizing             
    emotion_batch = torch.tensor([emotion_to_index[ele['context']] for ele in batch]) #this is for discrete classifer
    text_batch = tokenizer(chat_template, return_tensors='pt', max_length=128, truncation=True, padding=True)
    # emo = tokenizer([ele['context'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True) # this is for continuous classifer
    
    llm_target_batch = ([ele['A']] for ele in batch)

    return emotion_batch, text_batch, llm_target_batch

def collate_fn_test(batch, new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
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
    llm_text_batch = llm_tokenizer(chat_template_llama2, return_tensors='pt', max_length=128, truncation=True, padding=True)
    # llm_target_batch = llm_tokenizer(chat_template_llama2, return_tensors="pt", max_length=128, truncation=True, padding=True)
    
    input_batch = (chat_template)
    emo_label = ([ele['context']] for ele in batch)
    # prompt_batch = ([ele['prompt']] for ele in batch)
    llm_target_batch = ([ele['A']] for ele in batch)
    
    return emotion_batch, text_batch, llm_text_batch, llm_target_batch, input_batch, emo_label

def data_loader(a,new_data, batch_size = 16):
    return DataLoader(a, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, new_data=new_data, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def data_loader_val(b, new_data_val, batch_size = 16):
    return DataLoader(b, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_val, new_data=new_data_val, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def data_loader_test(b, new_data_val, batch_size = 16):
    return DataLoader(b, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_test, new_data=new_data_val, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ,
               llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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") ))

def predata_test():
    dataset_test = pd.read_csv('data/test.csv')

    idx = None
    q = None
    new_data = {}
    new_data_val = {}
    # ccc = 0
    # idxx = 0
    for i, ele in tqdm(dataset_test.iterrows(), total=len(dataset_test), ncols=0):

        if idx == ele['dialogue_id']:
            q.append(ele['text'])
            multi_data = {
                'Q': q[:-1],
                'A': q[-1],
                'context': ele['emotion'],
                # 'prompt': ele['prompt'],
                'utterance_idx': ele['turns']
            }            
            if ele['emotion'] in new_data:
                new_data[ele['emotion']].append(multi_data)
                # if ele['utterance_idx'] != "2":
                #     new_data[ele['context']].append(one_data)   
            else:
                new_data[ele['emotion']] = [multi_data]            
        else:
            idx = ele['dialogue_id']
            q = [ele['text']]
            # idxx = 0
    


    a=[] #train
    for k, v in new_data.items():
        a.extend(v)
    print(a[2]['context']) #sample neg context

    return a, new_data

def predata():
    dataset = pd.read_csv('data/train.csv')
    dataset_val = pd.read_csv('data/valid.csv')
    # dataset_test = pd.read_csv('data/test.csv')

    idx = None
    q = None
    new_data = {}
    new_data_val = {}
    # ccc = 0
    # idxx = 0
    for i, ele in tqdm(dataset.iterrows(), total=len(dataset), ncols=0):

        if idx == ele['dialogue_id']:
            q.append(ele['text'])
            multi_data = {
                'Q': q[:-1],
                'A': q[-1],
                'context': ele['emotion'],
                # 'prompt': ele['prompt'],
                'utterance_idx': ele['turns']
            }            
            if ele['emotion'] in new_data:
                new_data[ele['emotion']].append(multi_data)
                # if ele['utterance_idx'] != "2":
                #     new_data[ele['context']].append(one_data)   
            else:
                new_data[ele['emotion']] = [multi_data]            
        else:
            idx = ele['dialogue_id']
            q = [ele['text']]
            # idxx = 0
        # ccc+=1
        # if ccc == 3:
           
        #     print(new_data)
        #     exit()     

    idx = None
    q = None    
    ##val    
    for i, ele in tqdm(dataset_val.iterrows(), total=len(dataset_val), ncols=0):
        if idx == ele['dialogue_id']:
            q.append(ele['text'])
            one_data = {
                'Q': q[:-1],
                'A': q[-1],
                'context': ele['emotion'],
                # 'prompt': ele['prompt'],
                'utterance_idx': ele['turns']
            }
            if ele['emotion'] in new_data_val:
                new_data_val[ele['emotion']].append(one_data)
            else:
                new_data_val[ele['emotion']] = [one_data]
        else:
            idx = ele['dialogue_id']
            q = [ele['text']]
    
    # dict[list[dict]]
    a=[] #train
    b=[] #val
    # a=[1,2,3]
    # a.extend([4,5,6])
    for k, v in new_data.items():
        a.extend(v)
    for k, v in new_data_val.items():
        b.extend(v)
    print(a[2]['context']) #sample neg context
    print(b[20]['context']) #sample neg context
    # print(new_data.keys())
    return a, new_data, b, new_data_val
    #new_data[neg][rand]

#a, new_data = predata()

if __name__ == "__main__": 
    # print(transformers.__version__)
    a, new_data, b, new_data_val   = predata()
    batch_size = 16
    loader = data_loader(a,new_data, batch_size)
 