import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, BertTokenizer, BertModel
from tqdm import tqdm
from encoder_multi import ContrastiveLearningModel  #for continous
# from encoder import ContrastiveLearningModel #for discrete
import torch
import torch.nn as nn

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
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)
        )

    def forward(self, text_batch, prompt_batch, aug_batch, neg_batch, llm_target_batch, emotion_batch=None):
        # encode
        # loss_encoder, hidden_states_clean,  emo_va = self.encoder(input_tokens = text_batch, input_prompt=prompt_batch, input_plus_tokens=aug_batch, input_minus_tokens=neg_batch, emo=emotion_batch)
        loss_encoder, hidden_states_clean,  predictions_va, emo_va = self.encoder(input_tokens = text_batch, input_prompt=prompt_batch, input_plus_tokens=aug_batch, input_minus_tokens=neg_batch, emo=emotion_batch)
        hidden_states_clean = self.MLP(hidden_states_clean)

        emb_target = self.decoder.model.embed_tokens(llm_target_batch.input_ids)
        input_embeds = torch.cat([hidden_states_clean.to(torch.bfloat16), emb_target], dim=1)
        # input_embeds_tmp = torch.cat([hidden_states_clean.to(torch.bfloat16),emb_input], dim=1)
        input_embeds_tmp = torch.cat([hidden_states_clean.to(torch.bfloat16)], dim=1)
    
        tmp_ids = torch.ones([emb_target.shape[0], input_embeds_tmp.shape[1]], dtype=torch.long, device=device) * -100
        label_ids = torch.cat([tmp_ids, llm_target_batch.input_ids], dim=1)
        label_ids[:,-llm_target_batch.input_ids.shape[1]:][llm_target_batch.attention_mask==0] = -100

        output = self.decoder(inputs_embeds=input_embeds, labels=label_ids, use_cache=True)

        # return loss_encoder, output.loss,  emo_va
        return loss_encoder, output.loss,  predictions_va, emo_va
    
    def generate(self, text_batch, emotion_batch=None):
        # hidden_states_clean,  emo_va = self.encoder(text_batch, emo=emotion_batch)
        hidden_states_clean,  predictions_va, emo_va = self.encoder(text_batch, emo=emotion_batch)

        hidden_states_clean = self.MLP(hidden_states_clean) #b,n,4096
        input_embeds = hidden_states_clean.to(torch.bfloat16)
        emb_target = self.decoder.model.embed_tokens(torch.tensor([[self.llama_tokenizer.bos_token_id]],device=device))
        # print(hidden_states_clean.shape, emb_target.shape)
        input_embeds = torch.cat([hidden_states_clean.to(torch.bfloat16), emb_target], dim=1)

        generated_tokens = self.decoder.generate(inputs_embeds=input_embeds,
                                                 max_length=64,
                                                 num_return_sequences=1,
                                                 do_sample=True)
        return generated_tokens,   predictions_va, emo_va

if __name__ == "__main__":
    a, new_data, b, new_data_val = predata()
    batch_size = 8
    loader = data_loader(a, new_data, batch_size)
    # Instantiate InfoNCE loss
    model = MyModel()

    cnt = 0
    # Compute loss
    model.encoder.load_state_dict(torch.load("500_model_encoder.pth",map_location=device))
    model.MLP.load_state_dict(torch.load("500_model_mlp.pth",map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        cnt = 0
        bar = tqdm(loader, ncols=110, smoothing=0.05)
        for emotion_batch, text_batch, aug_batch, neg_batch, llm_text_batch, llm_target_batch in bar: ## text positive negative 
            #chaning device
            text_batch = text_batch.to(device)
            aug_batch = aug_batch.to(device)
            neg_batch = neg_batch.to(device)
            #llm_text_batch = llm_text_batch.to(device)
            llm_target_batch = llm_target_batch.to(device)
            #validating step       
            Infoloss, generated_tokens = model.generation(text_batch, aug_batch, neg_batch)

            total_loss += Infoloss.item()
            cnt += 1
            if cnt == 5 :
                #print(f'Generated Tokens: {generated_tokens}')
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")
                print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])
                break
        avg_loss = total_loss / cnt
