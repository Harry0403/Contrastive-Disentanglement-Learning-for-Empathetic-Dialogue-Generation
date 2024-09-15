import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
#from dataloader import data_loader, predata
#from disentanglement import Disentangle
#from pytorch_metric_learning.losses import InfoNCE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .VAD_classifier import BertRegressor

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ContrastiveLearningModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", temperature=0.07):
        super(ContrastiveLearningModel, self).__init__()

        self.model = AutoModelForMaskedLM.from_pretrained(model_name, device_map="cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = temperature
        self.info_nce_loss = InfoNCE()

        # Fully connected layers for content and emotion
        self.fc_content = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.fc_emotion = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        
        # Emotion classifier
        #self.classifier = Classifier(self.model.config.hidden_size, num_emotions=32)
        self.classifier = BertRegressor()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.squeeze(-1)  # [batch_size, seq_len, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled_output
    
    
    def cross_entropy_loss(self, x, y):
        log_probs = F.log_softmax(x, dim=-1)
        return -torch.mean(torch.sum(y * log_probs, dim=-1))
    
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
            

            # Get hidden states for positive and negative samples
            outputs_plus = self.model(**input_plus_tokens, output_hidden_states=True)
            outputs_minus = self.model(**input_minus_tokens, output_hidden_states=True)

            # context_plus = self.fc_content(outputs_plus.hidden_states[-1])
            # context_minus = self.fc_content(outputs_minus.hidden_states[-1])

            emotion_plus = self.fc_emotion(outputs_plus.hidden_states[-1])
            emotion_minus = self.fc_emotion(outputs_minus.hidden_states[-1])

            # # Calculate InfoNCE loss with pooled context features
            # pooled_context_plus = self.mean_pooling(context_plus, input_plus_tokens['attention_mask'])
            # pooled_context_minus = self.mean_pooling(context_minus, input_minus_tokens['attention_mask'])

            # Calculate mean pooling for emotional words
            pooled_emotion_plus = self.mean_pooling(emotion_plus, input_plus_tokens['attention_mask'])
            pooled_emotion_minus = self.mean_pooling(emotion_minus, input_minus_tokens['attention_mask'])

            # Calculate InfoNCE losses
           
            loss_e = self.info_nce_loss(pooled_emotion_x, pooled_emotion_plus, pooled_emotion_minus)

            # Combine losses
            contrastive_loss = 0.8*loss_e
            
            # Emotion prediction and classification loss
            predictions_va = self.classifier.generate(pooled_emotion_x)
            
            emo_va = self.classifier(emo)

            # Calculate MSE loss between predicted VA values and true VA values
            V_batch = emo_va[:, 0].unsqueeze(1)  # Extract valence values
            A_batch = emo_va[:, 1].unsqueeze(1)  # Extract arousal values
            D_batch = emo_va[:, 2].unsqueeze(1)
            loss_V = F.mse_loss(predictions_va[:, 0].unsqueeze(1), V_batch)
            loss_A = F.mse_loss(predictions_va[:, 1].unsqueeze(1), A_batch)
            loss_D = F.mse_loss(predictions_va[:, 2].unsqueeze(1), D_batch)
            classifier_loss = (loss_V + loss_A + loss_D) / 3

            # Total loss
            total_loss = 0.3*contrastive_loss + 0.7*classifier_loss

            return total_loss, outputs_x.hidden_states[-1], pooled_context_x, pooled_emotion_x, predictions_va, emo_va
        else:
            # Inference case
            predictions_va = self.classifier.generate(pooled_emotion_x)
            emo_va = self.classifier(emo)
            return hidden_states_x, pooled_context_x, pooled_emotion_x, predictions_va, emotion_x, emo_va

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
    
def plot_predictions(valences, arousals, preds_v, preds_a):
    plt.figure(figsize=(10, 10))
    plt.scatter(valences, arousals, color='blue', label='True Values')
    plt.scatter(preds_v, preds_a, color='red', alpha=0.5, label='Predictions')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.legend()
    plt.title('True vs Predicted Valence and Arousal')
    plt.savefig(f'En_V_A.png')
       
if __name__ == "__main__":
    a, new_data, b, new_data_val = predata()
    batch_size = 16
    