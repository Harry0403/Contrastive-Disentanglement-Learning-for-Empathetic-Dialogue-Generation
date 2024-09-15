import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, BertTokenizer, BertModel
from tqdm import tqdm
# from dataloader import data_loader, predata
from .encoder import ContrastiveLearningModel
# from encoder_ConClassifier import ContrastiveLearningModel
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ContrastiveLearningModel()
        self.decoder = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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", device_map="cuda:0", torch_dtype = torch.bfloat16)
        #self.decoder = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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", torch_dtype = torch.float16)
        self.llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")
        self.decoder.requires_grad_(False)
        self.MLP = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3072),
            nn.ReLU(),
            nn.Linear(3072, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)
        )
        self.generate_ori={
            'top_k':90,
            'top_p':1.0,
            'temperature':3.0
        }
        

    def forward(self, text_batch, prompt_batch, aug_batch, neg_batch,llm_text_batch, llm_target_batch, emotion_batch=None):
        # encode
        loss_encoder, hidden_states_clean, predictions_va, emo_va = self.encoder(input_tokens = text_batch, input_prompt=prompt_batch, input_plus_tokens=aug_batch, input_minus_tokens=neg_batch, emo=emotion_batch)
        hidden_states_clean = self.MLP(hidden_states_clean)

        emb_target = self.decoder.model.embed_tokens(llm_target_batch.input_ids)
        emb_input = self.decoder.model.embed_tokens(llm_text_batch.input_ids)
        input_embeds = torch.cat([hidden_states_clean.to(torch.bfloat16),emb_input, emb_target], dim=1)
        input_embeds_tmp = torch.cat([hidden_states_clean.to(torch.bfloat16),emb_input], dim=1)
    
        tmp_ids = torch.ones([emb_target.shape[0], input_embeds_tmp.shape[1]], dtype=torch.long, device=device) * -100
        label_ids = torch.cat([tmp_ids, llm_target_batch.input_ids], dim=1)
        label_ids[:,-llm_target_batch.input_ids.shape[1]:][llm_target_batch.attention_mask==0] = -100

        output = self.decoder(inputs_embeds=input_embeds, labels=label_ids, use_cache=True)

        return loss_encoder, output.loss, predictions_va, emo_va
    
    def generate(self, text_batch, emotion_batch=None):
        hidden_states_clean, context_x , emotion_x , x_emo = self.encoder(text_batch)

        hidden_states_clean = self.MLP(hidden_states_clean) #b,n,4096
        input_embeds = hidden_states_clean.to(torch.bfloat16)

        generated_tokens = self.decoder.generate(inputs_embeds=input_embeds,
                                                 max_length=64,
                                                 num_return_sequences=1,
                                                 do_sample=True)
        return generated_tokens, context_x , emotion_x , x_emo

if __name__ == "__main__":
    a, new_data, b, new_data_val = predata()
    