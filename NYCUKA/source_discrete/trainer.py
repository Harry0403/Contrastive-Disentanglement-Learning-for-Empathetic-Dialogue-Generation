import torch
from tqdm import tqdm
from dataloader import data_loader, predata ,data_loader_val
from datasets import load_dataset
from decoder import MyModel
import sacrebleu
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_bleu_score(references, hypotheses):
    return sentence_bleu([references.split()], hypotheses.split(), smoothing_function=SmoothingFunction().method4,)

def get_dist(responses):
    """
    Calculate the distinct-1 and distinct-2 scores for a list of tokenized responses.

    Args:
    responses (list of list of str): The tokenized responses.

    Returns:
    tuple: A tuple containing distinct-1 and distinct-2 scores.
    """
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.

    for response in responses:
        ugs = response
        bgs = [ugs[i] + ugs[i + 1] for i in range(len(ugs) - 1)]
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)

    n = len(responses)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams) + 1e-16)
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams) + 1e-16)
    avg_len /= n

    return mi_dist1, mi_dist2

def calculate_perplexity(model, tokenizer, sentence, device):
    inputs = tokenizer(sentence, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    return torch.exp(loss).item()

def calculate_acc(x_emo, emotion_batch):
    correct_predictions = 0
    total_predictions = 0
    correct_predictions += (x_emo == emotion_batch).sum().item()
    #print(x_emo, emotion_batch)
    total_predictions += emotion_batch.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

def trainer(dataloader, loader_val, model, epoch=3):
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.8, 0.95], weight_decay=0.0001)

    for epo in range(epoch):
        cnt = 0
        loss_temp = 1000
        total_loss = 0
        context_features = []
        emotion_features = []
        correct_predictions = 0
        total_predictions = 0
        pre_V_batch = []
        pre_A_batch = []
        pre_D_batch = []
        V_batch = []
        A_batch = []
        D_batch = []
        label = []
        
        bar = tqdm(dataloader, ncols=120, smoothing=0.05, desc=f'epoch:{epo+1:02d}/{epoch:02d}')
        for emotion_batch, text_batch, aug_batch, neg_batch,llm_text_batch, llm_target_batch in bar: ## text positive negative 
            #chaning device
            text_batch = text_batch.to(device)
            aug_batch = aug_batch.to(device)
            neg_batch = neg_batch.to(device)            
            llm_target_batch = llm_target_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            # prompt_batch = prompt_batch.to(device)
            llm_text_batch = llm_text_batch.to(device)

            #trianing step
            optimizer.zero_grad()      
            Infoloss, llm_loss, x_emo = model(text_batch, aug_batch, neg_batch,llm_text_batch, llm_target_batch, emotion_batch)
            #Infoloss, hidden_states_clean = model.encoder(text_batch, aug_batch, neg_batch) #Encoder
            
            
            loss = 0.8*Infoloss + 0.7*llm_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  

            current_acc = calculate_acc(x_emo, emotion_batch)
            bar.set_postfix_str(f'Total_L: {loss:.4f}, Info:{Infoloss:4f}, Acc:{current_acc:4f}')
            cnt += 1
            if cnt == 3000 :                
                torch.save(model.encoder.state_dict(), "ceg_dis_encoder.pth")
                torch.save(model.MLP.state_dict(), "ceg_dis_mlp.pth")
                # break
        # num = -15
        # plot_predictions(label[num:-1], pre_V_batch[num:-1], pre_A_batch[num:-1], pre_D_batch[num:-1], V_batch[num:-1], A_batch[num:-1], D_batch[num:-1])
        mean_loss = total_loss / cnt
        
        #val
        torch.save(model.encoder.state_dict(), "ceg_dis_encoder.pth")
        torch.save(model.MLP.state_dict(), "ceg_dis_mlp.pth")
        validation(loader_val, model) 
        # context_features = np.concatenate(context_features, axis=0)
        # emotion_features = np.concatenate(emotion_features, axis=0)
        # visualize_tsne(context_features, emotion_features, epo)    

        #bar.set_postfix_str(f'Training Loss: {loss:.4f}, Validation Loss :{val_loss:.4f}')
        print(f'Training Loss: {mean_loss:.4f}')

def validation(dataloader, model):
    # global rouge_temp 
    model.eval()
    total_loss = 0.0
    cnt = 0
    references = []
    my_score_bleu = []
    my_score_rougel = []
    my_score_dist1 = []
    my_score_dist2 = []
    my_perplexity = []
    acc = []
    correct_predictions = 0
    total_predictions = 0

    bar = tqdm(dataloader, ncols=110, smoothing=0.05, desc=f'validation:')
    with torch.no_grad():
        for emotion_batch, emo_label, text_batch, llm_text_batch, llm_target_batch, input_batch in bar: 
            text_batch = text_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            #llm_target_batch = llm_target_batch.to(device)

            my_token, x_emo = model.generate(text_batch)
            my_response = model.llama_tokenizer.batch_decode(my_token,skip_special_tokens=True)
            # disentan_response = model.llama_tokenizer.batch_decode(disentangle_tokens,skip_special_tokens=True)
            # print(disentan_response)
            # exit()
            references.extend(llm_target_batch)
            reference_str = ' '.join(references[cnt])
            my_response_str = ' '.join(my_response)

            my_score_bleu.append(calculate_bleu_score(reference_str, my_response_str))
            current_rouge = rougeScore(my_response_str, reference_str)['rougeL_fmeasure'].tolist()
            my_score_rougel.append(current_rouge)
            distinct_1_score, distinct_2_score = get_dist( my_response_str.split())
            my_score_dist1.append(distinct_1_score)
            my_score_dist2.append(distinct_2_score)
            my_perplexity.append(calculate_perplexity(model.decoder, model.llama_tokenizer, my_response_str, device))

            accuracy = calculate_acc(x_emo, emotion_batch)
            acc.append(accuracy)

            # correct_predictions += (x_emo == emotion_batch).sum().item()
            # total_predictions += emotion_batch.size(0)
            # accuracy = correct_predictions / total_predictions
           

            bar.set_postfix_str(f'rougel:{current_rouge:4f},accuracy:{accuracy:4f}')
            cnt += 1
            if cnt == 400 :                
                break
   
    # Calculate rougeL scores
    print(f"BLEU score for MyModel: {np.mean(my_score_bleu)}")
    print(f"rougeL score for MyModel: {np.mean(my_score_rougel)}")
    print(f"Distinct-1 score for MyModel: {np.mean(my_score_dist1)}")
    print(f"Distinct-2 score for MyModel: {np.mean(my_score_dist2)}")
    print(f"Perplexity for MyModel: {np.mean(my_perplexity)}")
    print(f"Acc for MyModel: {np.mean(acc)}")
    
    #print(f"Acc for MyModel: {np.mean(accuracy)}")
    model.train()

def visualize_tsne(context_features, emotion_features, epo):
    tsne = TSNE(n_components=2, random_state=42)
    combined_features = np.vstack((context_features, emotion_features))
    tsne_results = tsne.fit_transform(combined_features)

    context_tsne = tsne_results[:context_features.shape[0]]
    emotion_tsne = tsne_results[context_features.shape[0]:]

    plt.figure(figsize=(10, 5))
    plt.scatter(context_tsne[:, 0], context_tsne[:, 1], label='Context Features', alpha=0.5)
    plt.scatter(emotion_tsne[:, 0], emotion_tsne[:, 1], label='Emotion Features', alpha=0.5)
    plt.legend()
    #plt.title("t-SNE of Context and Emotion Features")
    plt.savefig(f'{epo}_clg.png')

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
    batch_size = 4
    loader = data_loader(a, new_data, batch_size)
    loader_val = data_loader_val(b, new_data_val, batch_size)
    
    model = MyModel()
    # model.encoder.load_state_dict(torch.load("3755_model_encoder.pth",map_location=device))
    # model.MLP.load_state_dict(torch.load("3755_model_mlp.pth",map_location=device))
    model.to(device)
    # train+val
    trainer(loader, loader_val, model)
    print("training done")
