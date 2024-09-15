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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()

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
    if n == 0 :
        n = 1
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

correct_predictions = 0
total_predictions = 0

def calculate_acc(predictions_va, emo_va):
    global correct_predictions, total_predictions

    threshold = 0.01

    correct_V = (np.abs(predictions_va[:, 0].detach().cpu().numpy() - emo_va[:, 0].detach().cpu().numpy()) < threshold)
    correct_A = (np.abs(predictions_va[:, 1].detach().cpu().numpy() - emo_va[:, 1].detach().cpu().numpy()) < threshold)
    correct_D = (np.abs(predictions_va[:, 2].detach().cpu().numpy() - emo_va[:, 2].detach().cpu().numpy()) < threshold)

    correct_predictions += np.sum(correct_V & correct_A & correct_D)
    total_predictions += emo_va.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        opt_loss = 100
        
        bar = tqdm(dataloader, ncols=120, smoothing=0.05, desc=f'epoch:{epo+1:02d}/{epoch:02d}')
        for emotion_label, emotion_batch, text_batch, prompt_batch, aug_batch, neg_batch,llm_text_batch, llm_target_batch in bar: ## text positive negative 
            #chaning device
            text_batch = text_batch.to(device)
            aug_batch = aug_batch.to(device)
            neg_batch = neg_batch.to(device)            
            llm_target_batch = llm_target_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            prompt_batch = prompt_batch.to(device)
            llm_text_batch = llm_text_batch.to(device)

            #trianing step
            optimizer.zero_grad()      
            Infoloss, llm_loss, predictions_va, emo_va = model(text_batch, prompt_batch, aug_batch, neg_batch,llm_text_batch, llm_target_batch, emotion_batch)
            #Infoloss, hidden_states_clean = model.encoder(text_batch, aug_batch, neg_batch) #Encoder
            # context_features.append(context_x.detach().cpu().numpy())
            # emotion_features.append(emotion_x.detach().cpu().numpy())

            # pre_V_batch.extend(predictions_va[:, 0].unsqueeze(1).detach().cpu().numpy())
            # pre_A_batch.extend(predictions_va[:, 1].unsqueeze(1).detach().cpu().numpy())
            # pre_D_batch.extend(predictions_va[:, 2].unsqueeze(1).detach().cpu().numpy())
            # V_batch.extend(emo_va[:, 0].unsqueeze(1).detach().cpu().numpy())
            # A_batch.extend(emo_va[:, 1].unsqueeze(1).detach().cpu().numpy())
            # D_batch.extend(emo_va[:, 2].unsqueeze(1).detach().cpu().numpy())
            # label.extend(emotion_label)
            
            loss = 0.9*Infoloss + 0.9*llm_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  

            bar.set_postfix_str(f'Total_L: {loss:.4f}, Info:{Infoloss:4f}, PPL:{torch.exp(loss).item():4f}')
            cnt += 1
            if cnt%15 == 0 :
                if opt_loss > loss:  
                    torch.save(model.encoder.state_dict(), "ceclg_c_v9_encoder.pth")
                    torch.save(model.MLP.state_dict(), "ceclg_c_v9_mlp.pth")   
                    opt_loss = loss  
                    # print("save")         
                    # break
        # num = -15
        # plot_predictions(label[num:-1], pre_V_batch[num:-1], pre_A_batch[num:-1], pre_D_batch[num:-1], V_batch[num:-1], A_batch[num:-1], D_batch[num:-1])
        # plot_predictions(label[num:-1], preds_v = pre_V_batch[num:-1], preds_a = pre_A_batch[num:-1], preds_d = None, valences = V_batch[num:-1], arousals = A_batch[num:-1], dominances = None)
        # plot_predictions(label[num:-1], preds_v = pre_V_batch[num:-1], preds_a = None, preds_d = pre_D_batch[num:-1], valences = V_batch[num:-1], arousals = None, dominances = D_batch[num:-1])
        # plot_predictions(label[num:-1], preds_v = None, preds_a = pre_A_batch[num:-1], preds_d = pre_A_batch[num:-1], valences = None, arousals = A_batch[num:-1], dominances = D_batch[num:-1])
        mean_loss = total_loss / cnt
        
        #val
        if opt_loss > loss:  
            torch.save(model.encoder.state_dict(), "ceclg_c_v9_encoder.pth")
            torch.save(model.MLP.state_dict(), "ceclg_c_v9_mlp.pth")   
            opt_loss = loss
            print("savepth")
        validation(loader_val, model) 
        # context_features = np.concatenate(context_features, axis=0)
        # emotion_features = np.concatenate(emotion_features, axis=0)
        # visualize_tsne(context_features, emotion_features, epo)    

        #bar.set_postfix_str(f'Training Loss: {loss:.4f}, Validation Loss :{val_loss:.4f}')
        print(f'Training Loss: {mean_loss:.4f}') #, Validation Loss :{val_loss:.4f}, perplexity: {perplexity:4f}')


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
    acc_my = []
    correct_predictions = 0
    total_predictions = 0

    bar = tqdm(dataloader, ncols=110, smoothing=0.05, desc=f'validation:')
    with torch.no_grad():
        for emotion_batch, text_batch, llm_text_batch, llm_target_batch in bar: 
            text_batch = text_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            #llm_target_batch = llm_target_batch.to(device)

            my_token, predictions_va, emo_va = model.generate(text_batch, emotion_batch)
            my_response = model.llama_tokenizer.batch_decode(my_token,skip_special_tokens=True)
            # disentan_response = model.llama_tokenizer.batch_decode(disentangle_tokens,skip_special_tokens=True)
            # print(disentan_response)
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

            acc=calculate_acc(predictions_va, emo_va)
            acc_my.append(acc) 

            # correct_predictions += (x_emo == emotion_batch).sum().item()
            # total_predictions += emotion_batch.size(0)
            # accuracy = correct_predictions / total_predictions
            # if current_rouge > rouge_temp:
            #     torch.save(model.encoder.state_dict(), "SimD_continuous_model_encoder.pth")
            #     torch.save(model.MLP.state_dict(), "SimD_continuous_model_mlp.pth")
            #     rouge_temp = current_rouge

            bar.set_postfix_str(f'rougel:{current_rouge:4f},acc:{acc:4f}')
            cnt += 1
            if cnt == 40 :                
                break
    # Calculate rougeL scores
    print(f"BLEU score for MyModel: {np.mean(my_score_bleu)}")
    print(f"rougeL score for MyModel: {np.mean(my_score_rougel)}")
    print(f"Distinct-1 score for MyModel: {np.mean(my_score_dist1)}")
    print(f"Distinct-2 score for MyModel: {np.mean(my_score_dist2)}")
    print(f"Perplexity for MyModel: {np.mean(my_perplexity)}")
    print(f"Acc for MyModel: {np.mean(acc_my)}")
    
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
    plt.savefig(f'{epo}_disen.png')

def plot_predictions(labels, valences, arousals, dominances, preds_v, preds_a, preds_d):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Unique labels and corresponding colors
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('hsv', len(unique_labels))

    # Dictionary to track if a label has been plotted
    plotted_labels = set()

    for i, label in enumerate(labels):
        if label not in plotted_labels:
            # Plot true values with circle marker
            ax.scatter(valences[i], arousals[i], dominances[i], color=colors(unique_labels.index(label)), label=f'{label}', marker='o')
            # Plot predicted values with triangle marker
            ax.scatter(preds_v[i], preds_a[i], preds_d[i], color=colors(unique_labels.index(label)), marker='^')
            plotted_labels.add(label)

    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.legend(loc='center left', bbox_to_anchor=(-0.12, 0.5), ncol=1)
    plt.savefig('En_V_A_D.png')

def plot_predictions(labels, preds_v=None, preds_a=None, preds_d=None, valences=None, arousals=None, dominances=None):
    flat_labels = [item[0] for item in labels]
    fig, ax = plt.subplots(figsize=(10, 10))
    unique_labels = list(set(flat_labels))
    
    colors = plt.cm.get_cmap('hsv', len(unique_labels))

    plotted_labels = set()

    if preds_v is not None and preds_a is not None and valences is not None and arousals is not None:
        # Valence-Arousal plot
        x_pred, y_pred = preds_v, preds_a
        x_true, y_true = valences, arousals
        xlabel, ylabel = 'Valence', 'Arousal'
    elif preds_v is not None and preds_d is not None and valences is not None and dominances is not None:
        # Valence-Dominance plot
        x_pred, y_pred = preds_v, preds_d
        x_true, y_true = valences, dominances
        xlabel, ylabel = 'Valence', 'Dominance'
    elif preds_a is not None and preds_d is not None and arousals is not None and dominances is not None:
        # Arousal-Dominance plot
        x_pred, y_pred = preds_a, preds_d
        x_true, y_true = arousals, dominances
        xlabel, ylabel = 'Arousal', 'Dominance'
    else:
        raise ValueError("Insufficient or incorrect input dimensions provided.")

    for i, label in enumerate(flat_labels):
        color = colors(unique_labels.index(label))
        if label not in plotted_labels:
            # Plot one true and one predicted marker for each unique label
            ax.scatter(x_true[i], y_true[i], color=color, label=f'{label} (true)', marker='o', s=100)
            ax.scatter(x_pred[i], y_pred[i], color=color, label=f'{label} (pred)', marker='^', s=100)
            plotted_labels.add(label)
        else:
            # Plot remaining points without markers
            ax.scatter(x_true[i], y_true[i], color=color, marker='o', s=100)
            ax.scatter(x_pred[i], y_pred[i], color=color, marker='^', s=100)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=12)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f'En_{xlabel[0]}_{ylabel[0]}.png')
        

if __name__ == "__main__":
    a, new_data, b, new_data_val = predata()
    batch_size = 4
    loader = data_loader(a, new_data, batch_size)
    loader_val = data_loader_val(b, new_data_val, batch_size)
    
    model = MyModel()
    # model.encoder.load_state_dict(torch.load("ceclg_c_v4_encoder.pth",map_location=device))
    # model.MLP.load_state_dict(torch.load("ceclg_c_v4_mlp.pth",map_location=device))
    # model.encoder.classifier.load_state_dict(torch.load("3_tmpe.bin",map_location=device))
    model.encoder.classifier.load_state_dict(torch.load("2_tmpe.bin",map_location=device))
    model.to(device)
    # train+val
    trainer(loader, loader_val, model)
    print("training done")
