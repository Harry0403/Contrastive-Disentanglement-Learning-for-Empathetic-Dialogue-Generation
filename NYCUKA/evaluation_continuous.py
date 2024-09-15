import torch
from tqdm import tqdm
from source_continuous.dataloader import predata, data_loader_val
from datasets import load_dataset
from source_continuous.decoder import MyModel
# import sacrebleu
# from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
import numpy as np
import nltk
from itertools import chain
import csv
import matplotlib.pyplot as plt


def calculate_bleu_score(references, hypotheses):
    return sentence_bleu([references.split()], hypotheses.split(), smoothing_function=SmoothingFunction().method4,)

def calculate_perplexity(model, tokenizer, sentence, device):
    inputs = tokenizer(sentence, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    return torch.exp(loss).item()

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
    if n == 0:
        n=1
        print(responses)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams) + 1e-16)
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams) + 1e-16)
    avg_len /= n

    return mi_dist1, mi_dist2

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

    plt.savefig(f'ceclg_c_400_{xlabel[0]}_{ylabel[0]}.png')

def plot_predictions_3(labels, preds_v, preds_a, preds_d, valences, arousals, dominances):
    flat_labels = [item[0] for item in labels]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = list(set(flat_labels))
    colors = plt.cm.get_cmap('hsv', len(unique_labels))

    plotted_labels = set()

    for i, label in enumerate(flat_labels):
        color = colors(unique_labels.index(label))
        if label not in plotted_labels:
            # Plot one true and one predicted marker for each unique label
            ax.scatter(valences[i], arousals[i], dominances[i], color=color, label=f'{label} (true)', marker='o', s=100)
            ax.scatter(preds_v[i], preds_a[i], preds_d[i], color=color, label=f'{label} (pred)', marker='^', s=100)
            plotted_labels.add(label)
        else:
            # Plot remaining points without markers
            ax.scatter(valences[i], arousals[i], dominances[i], color=color, marker='o', s=100)
            ax.scatter(preds_v[i], preds_a[i], preds_d[i], color=color, marker='^', s=100)

    ax.set_xlabel('Valence', fontsize=16)
    ax.set_ylabel('Arousal', fontsize=16)
    ax.set_zlabel('Dominance', fontsize=16)

    ax.legend(loc='center left', bbox_to_anchor=(-0.12, 0.5), ncol=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('ceclg_c_400_V_A_D.png')



emotion_list = ['sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful', 'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed', 'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised', 'nostalgic', 'confident', 'furious', 'disappointed', 'caring', 'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful', 'content', 'impressed', 'apprehensive', 'devastated']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Initialize MyModel
    my_model = MyModel()

    my_model.encoder.load_state_dict(torch.load("My_model_pth/ceclg_c_encoder.pth",map_location=device))
    my_model.MLP.load_state_dict(torch.load("My_model_pth/ceclg_c_mlp.pth",map_location=device))
    my_model.to(device)
    #my_model.encoder.classifier.load_state_dict(torch.load("source_continuous/2classifier.bin",map_location=device))

    # Set models to evaluation mode
    my_model.eval()

    a, new_data, b, new_data_val, c, new_data_test = predata()
    loader_val = data_loader_val(c, new_data_test, batch_size=1)
    references = []
    inputs = []
    emotion_labels = []
    prompt_batchs =[]
    # my_model_responses = []
    # llama_responses = []
    
    #score
    my_score_bleu = []
    llama_score_bleu = []
    my_score_rougel = []
    llama_score_rougel = []
    my_score_dist1 = []
    llama_score_dist1 = []
    my_score_dist2 = []
    llama_score_dist2 = []
    my_perplexity = []
    llama_perplexity = []
    acc_my = []

    #VAD
    pre_V_batch = []
    pre_A_batch = []
    pre_D_batch = []
    V_batch = []
    A_batch = []
    D_batch = []
    label = []

    cnt = 0
    bar = tqdm(loader_val, ncols=110, smoothing=0.05)
    with open('continuous_response/testing.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Prompt', 'Input', 'Golden', 'MyResponse', 'LLamaResponse', 'Label','rougel_my','rougel_llama'])
        # csvwriter.writerow(['--------------'])
        for prompt_batch, emotion_batch, text_batch, llm_text_batch, llm_target_batch, input_batch, emo_label in bar: 
            text_batch = text_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            llm_text_batch = llm_text_batch.to(device)

            generated_tokens, predictions_va,emo_va = my_model.generate(text_batch, emotion_batch)
            my_response = my_model.llama_tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            # Get LLama predictions
            with torch.no_grad():
                #print(text_batch)
                llama_token = my_model.decoder.generate(llm_text_batch.input_ids, max_length=60)
                llama_response = my_model.llama_tokenizer.batch_decode(llama_token, skip_special_tokens=True)

            # from datasets
            inputs.extend(input_batch)
            references.extend(llm_target_batch)
            emotion_labels.extend(emo_label)
            prompt_batchs.extend(prompt_batch)

            # Prepare my_response and llama_response for metric
            reference_str = ' '.join(references[cnt])
            my_response_str = ' '.join(my_response)
            llama_response_str = ' '.join(llama_response)

            ## BLEU
            my_score_bleu.append(calculate_bleu_score(reference_str, my_response_str))
            llama_score_bleu.append(calculate_bleu_score(reference_str, llama_response_str))

            ## RougeL
            rougel_my = rougeScore(my_response_str, reference_str)['rougeL_fmeasure'].tolist()
            rougel_llama = rougeScore(llama_response_str, reference_str)['rougeL_fmeasure'].tolist()
            my_score_rougel.append(rougel_my)
            llama_score_rougel.append(rougel_llama)

            ## Dist-1,2
            distinct_1_score, distinct_2_score = get_dist( my_response_str.split())
            my_score_dist1.append(distinct_1_score)
            my_score_dist2.append(distinct_2_score)
            distinct_1_score, distinct_2_score = get_dist(llama_response_str.split())
            llama_score_dist1.append(distinct_1_score)
            llama_score_dist2.append(distinct_2_score)

            ## Acc
            my_perplexity.append(calculate_perplexity(my_model.decoder, my_model.llama_tokenizer, my_response_str, device))
            llama_perplexity.append(calculate_perplexity(my_model.decoder, my_model.llama_tokenizer, llama_response_str, device))   
            acc_my.append(calculate_acc(predictions_va, emo_va))     

            bar.set_postfix_str(f'RougeL:{np.mean(my_score_rougel):4}, acc:{np.mean(acc_my)}')

            pre_V_batch.extend(predictions_va[:, 0].unsqueeze(1).detach().cpu().numpy())
            pre_A_batch.extend(predictions_va[:, 1].unsqueeze(1).detach().cpu().numpy())
            pre_D_batch.extend(predictions_va[:, 2].unsqueeze(1).detach().cpu().numpy())
            V_batch.extend(emo_va[:, 0].unsqueeze(1).detach().cpu().numpy())
            A_batch.extend(emo_va[:, 1].unsqueeze(1).detach().cpu().numpy())
            D_batch.extend(emo_va[:, 1].unsqueeze(1).detach().cpu().numpy())
            #label.extend(emo_label)

            ## Example
            csvwriter.writerow([prompt_batchs[cnt],inputs[cnt], reference_str, my_response_str, llama_response_str, emotion_labels[cnt] ,rougel_my,rougel_llama])

            if cnt%40 == 0:
                print(f"BLEU score for MyModel: {np.mean(my_score_bleu)}")
                print(f"BLEU score for LLama: {np.mean(llama_score_bleu)}")
                print(f"rougel score for MyModel: {np.mean(my_score_rougel)}")
                print(f"rougel score for LLama: {np.mean(llama_score_rougel)}")
                print(f"Distinct-1 score for MyModel: {np.mean(my_score_dist1)}")
                print(f"Distinct-1 score for LLama: {np.mean(llama_score_dist1)}")
                print(f"Distinct-2 score for MyModel: {np.mean(my_score_dist2)}")
                print(f"Distinct-2 score for LLama: {np.mean(llama_score_dist2)}")
                print(f"Perplexity for MyModel: {np.mean(my_perplexity)}")
                print(f"Perplexity for LLama: {np.mean(llama_perplexity)}")
                print(f"Acc for MyModel: {np.mean(acc_my)}")
                num = -400
                #print(emotion_labels[num:-1][2])
                plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = pre_A_batch[num:-1], preds_d = None, valences = V_batch[num:-1], arousals = A_batch[num:-1], dominances = None)
                plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = None, preds_d = pre_D_batch[num:-1], valences = V_batch[num:-1], arousals = None, dominances = D_batch[num:-1])
                plot_predictions(emotion_labels[num:-1], preds_v = None, preds_a = pre_A_batch[num:-1], preds_d = pre_A_batch[num:-1], valences = None, arousals = A_batch[num:-1], dominances = D_batch[num:-1])

            cnt += 1
            if cnt == 400:
                break

            # csvwriter.writerow(['\n--------------'])

    print(f"BLEU score for MyModel: {np.mean(my_score_bleu)}")
    print(f"BLEU score for LLama: {np.mean(llama_score_bleu)}")
    print(f"rougel score for MyModel: {np.mean(my_score_rougel)}")
    print(f"rougel score for LLama: {np.mean(llama_score_rougel)}")
    print(f"Distinct-1 score for MyModel: {np.mean(my_score_dist1)}")
    print(f"Distinct-1 score for LLama: {np.mean(llama_score_dist1)}")
    print(f"Distinct-2 score for MyModel: {np.mean(my_score_dist2)}")
    print(f"Distinct-2 score for LLama: {np.mean(llama_score_dist2)}")
    print(f"Perplexity for MyModel: {np.mean(my_perplexity)}")
    print(f"Perplexity for LLama: {np.mean(llama_perplexity)}")
    print(f"Acc for MyModel: {np.mean(acc_my)}")

    num = -400
    print(emotion_labels[num:-1][2])
    plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = pre_A_batch[num:-1], preds_d = None, valences = V_batch[num:-1], arousals = A_batch[num:-1], dominances = None)
    plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = None, preds_d = pre_D_batch[num:-1], valences = V_batch[num:-1], arousals = None, dominances = D_batch[num:-1])
    plot_predictions(emotion_labels[num:-1], preds_v = None, preds_a = pre_A_batch[num:-1], preds_d = pre_A_batch[num:-1], valences = None, arousals = A_batch[num:-1], dominances = D_batch[num:-1])
    plot_predictions_3(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = pre_A_batch[num:-1], preds_d = pre_A_batch[num:-1], valences = V_batch[num:-1], arousals = A_batch[num:-1], dominances = D_batch[num:-1])
    print("done")