import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, BertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Define a custom dataset
def collate_fn(batch, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') ):
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    ## Tokenizing             
    text_batch = tokenizer([ele['text'] for ele in batch], return_tensors='pt', max_length=128, truncation=True, padding=True)
    V_batch = torch.tensor([ele['V'] for ele in batch], dtype=torch.float).unsqueeze(1)
    A_batch = torch.tensor([ele['A'] for ele in batch], dtype=torch.float).unsqueeze(1)
    #print(text_batch)

    return text_batch, V_batch, A_batch

# Define the model
class BertRegressor(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(BertRegressor, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 3)  # Predicting Valence and Arousal and Dominance
        self.loss_fn = nn.MSELoss()

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, inputs, V_batch=None, A_batch=None ,D_batch =None):
        outputs = self.bert(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden state
        pooled_output = self.mean_pooling(hidden_states, inputs.attention_mask)
        output = self.drop(pooled_output)
        predictions = self.fc(output)

        if V_batch is not None and A_batch is not None and D_batch is not None:
            # Compute loss if targets are provided
            V_predictions = predictions[:, 0].unsqueeze(1)
            A_predictions = predictions[:, 1].unsqueeze(1)
            D_predictions = predictions[:, 2].unsqueeze(1)

            loss_V = self.loss_fn(V_predictions, V_batch)
            loss_A = self.loss_fn(A_predictions, A_batch)
            loss_D = self.loss_fn(D_predictions, D_batch)
            loss = (loss_V + loss_A + loss_D) / 3
            return loss
        
        return predictions
    
    def generate(self, inputs):# after pooling then input here
        output = self.drop(inputs)
        predictions = self.fc(output)
      
        return predictions

# Training function
def train_model(model, data_loader, optimizer, device, epoch=3):
    model = model.train()
    cnt = 0
    losses = []
    bar = tqdm(data_loader, ncols=110, smoothing=0.05)
    for text_batch, V_batch, A_batch in bar:
        #print(text_batch)
        text_batch = text_batch.to(device)
        V_batch = V_batch.to(device)
        A_batch = A_batch.to(device) 
        
        loss = model(text_batch,V_batch, A_batch)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cnt += 1 
        if cnt == 10:
            break

    return np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    valences = []
    arousals = []
    preds_v = []
    preds_a = []
    with torch.no_grad():
        bar = tqdm(data_loader, ncols=110, smoothing=0.05, desc='Evaluation')
        for text_batch, V_batch, A_batch in bar:
            text_batch = text_batch.to(device)
            V_batch = V_batch.to(device)
            A_batch = A_batch.to(device)
            #print(type(text_batch))

            predictions = model(text_batch)
          
            preds_v.extend(predictions[:, 0].cpu().numpy())
            preds_a.extend(predictions[:, 1].cpu().numpy())
            valences.extend(V_batch.cpu().numpy())
            arousals.extend(A_batch.cpu().numpy())
    return valences, arousals, preds_v, preds_a


# Predict and plot results
def plot_predictions(valences, arousals, preds_v, preds_a):
    plt.figure(figsize=(10, 10))
    plt.scatter(valences, arousals, color='blue', label='True Values')
    plt.scatter(preds_v, preds_a, color='red', alpha=0.5, label='Predictions')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.legend()
    plt.title('True vs Predicted Valence and Arousal')
    plt.savefig(f'V_A.png')
    
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("emobank/emobank.csv")
    #print(df['split'].value_counts())
    train_set = df[df['split'] == 'train'].to_dict('records')
    val_set = df[df['split'] == 'test'].to_dict('records')

    
    # Load tokenizer and data
    MAX_LEN = 128
    BATCH_SIZE = 16
    
    train_data_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = BertRegressor().to(device)

    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    # Training loop
    EPOCHS = 3
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_model(model, train_data_loader,  optimizer, device)
        print(f"Train loss {train_loss}")

        valences, arousals, preds_v, preds_a = eval_model(model, val_data_loader, device)
        print("val")
        
    torch.save(model.state_dict(), 'tmpe.bin')
    plot_predictions(valences[-10:-1], arousals[-10:-1], preds_v[-10:-1], preds_a[-10:-1])
 

