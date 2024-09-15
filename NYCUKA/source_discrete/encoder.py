import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
# from dataloader import data_loader, predata
#from disentanglement import Disentangle
#from pytorch_metric_learning.losses import InfoNCE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self, input_dim, num_emotions):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(input_dim // 4, num_emotions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class ContrastiveLearningModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", temperature=0.07):
        super(ContrastiveLearningModel, self).__init__()

        self.model = AutoModelForMaskedLM.from_pretrained(model_name, device_map="cuda:1")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.info_nce_loss = InfoNCE()

        # Fully connected layers for content and emotion
        self.fc_content = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.fc_emotion = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        
        # Emotion classifier
        self.classifier = Classifier(self.model.config.hidden_size, num_emotions=32)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.squeeze(-1)  # [batch_size, seq_len, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled_output
    def cosine_similarity_loss(self, x, y):
        cos_sim = F.cosine_similarity(x, y)
        return 1 - cos_sim.mean()

    def forward(self, input_tokens, input_plus_tokens=None, input_minus_tokens=None, emo=None):
        # Get hidden states for input
        outputs_x = self.model(**input_tokens, output_hidden_states=True)
        hidden_states_x = outputs_x.hidden_states[-1]

        # Separate hidden states into context and emotion
        context_x = self.fc_content(hidden_states_x)
        emotion_x = self.fc_emotion(hidden_states_x)
        pooled_context_x = self.mean_pooling(context_x, input_tokens['attention_mask'])
        pooled_emotion_x = self.mean_pooling(emotion_x, input_tokens['attention_mask'])

        if input_plus_tokens is not None and input_minus_tokens is not None and emo is not None:
            # prompt_x = self.model(**input_prompt, output_hidden_states=True)
            # hidden_states_prompt = prompt_x.hidden_states[-1]
            # pooled_prompt_x = self.mean_pooling(hidden_states_prompt, input_prompt['attention_mask'])
            # Get hidden states for positive and negative samples
            outputs_plus = self.model(**input_plus_tokens, output_hidden_states=True)
            outputs_minus = self.model(**input_minus_tokens, output_hidden_states=True)

            context_plus = self.fc_content(outputs_plus.hidden_states[-1])
            context_minus = self.fc_content(outputs_minus.hidden_states[-1])

            emotion_plus = self.fc_emotion(outputs_plus.hidden_states[-1])
            emotion_minus = self.fc_emotion(outputs_minus.hidden_states[-1])

            # Calculate InfoNCE loss with pooled context features
           

            # Calculate mean pooling for emotional words
            pooled_emotion_plus = self.mean_pooling(emotion_plus, input_plus_tokens['attention_mask'])
            pooled_emotion_minus = self.mean_pooling(emotion_minus, input_minus_tokens['attention_mask'])

            # Calculate InfoNCE losses
            # loss_c = self.cosine_similarity_loss(pooled_context_x, pooled_prompt_x)
            loss_e = self.info_nce_loss(pooled_emotion_x, pooled_emotion_plus, pooled_emotion_minus)

            # Combine losses
            contrastive_loss = loss_e
            
            # Emotion prediction and classification loss
            emotion_logits = self.classifier(pooled_emotion_x)
            if emo.dtype != torch.long:
                emo = emo.long()
            classifier_loss = F.cross_entropy(emotion_logits, emo)

            # # Predict x_emo
            x_emo = torch.argmax(emotion_logits, dim=1)

            # Total loss
            total_loss = classifier_loss

            return total_loss, hidden_states_x, x_emo
        else:
            # Inference case
            emotion_logits = self.classifier(pooled_emotion_x)
            x_emo = torch.argmax(emotion_logits, dim=1)
            return hidden_states_x, x_emo

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, query, positive_key, negative_keys):
        # Normalize the embeddings
        query = F.normalize(query, p=2, dim=-1)
        positive_key = F.normalize(positive_key, p=2, dim=-1)
        negative_keys = F.normalize(negative_keys, p=2, dim=-1)

        # Compute cosine similarity between query and positive key
        similarity_pos = torch.matmul(query, positive_key.t()) / self.temperature

        # Compute cosine similarity between query and negative keys
        similarity_neg = torch.matmul(query, negative_keys.t()) / self.temperature

        # Compute the log probabilities
        logits = torch.cat([similarity_pos, similarity_neg], dim=1)

        # Compute the loss
        loss = -torch.mean(torch.log(torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)))

        return loss
       
if __name__ == "__main__":
    a, new_data, b, new_data_val = predata()
    