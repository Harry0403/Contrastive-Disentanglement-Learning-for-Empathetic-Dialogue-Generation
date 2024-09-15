import torch
from tqdm import tqdm
from source_multi_turn.dataloader_multi import predata_test, data_loader_test
from datasets import load_dataset
from source_multi_turn.decoder_multi import MyModel
# import sacrebleu
# from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore(rouge_keys=('rouge1', 'rouge2','rougeL'))
import numpy as np
# import nltk
# from itertools import chain
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def calculate_bleu_score(references, hypotheses):
    # Ensure that references and hypotheses are not empty
    if not references or not hypotheses:
        return 0.0  # Return a BLEU score of 0 if either string is empty
    
    # Split the input strings into tokens
    reference_tokens = references.split()
    hypothesis_tokens = hypotheses.split()
    
    # Ensure that the tokens lists are not empty
    if not reference_tokens or not hypothesis_tokens:
        return 0.0  # Return a BLEU score of 0 if either token list is empty

    # Calculate BLEU score with smoothing
    try:
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=SmoothingFunction().method4)
    except ValueError:
        # Return 0.0 if there's a math domain error
        return 0.0

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

def calculate_acc(x_emo, emotion_batch):
    correct_predictions = 0
    total_predictions = 0
    correct_predictions += (x_emo == emotion_batch).sum().item()
    #print(x_emo, emotion_batch)
    total_predictions += emotion_batch.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

# def calculate_acc(predictions_va, emo_va):
#     global correct_predictions, total_predictions

#     # pre_V_batch.extend(predictions_va[:, 0].unsqueeze(1).detach().cpu().numpy())
#     # pre_A_batch.extend(predictions_va[:, 1].unsqueeze(1).detach().cpu().numpy())
#     # V_batch.extend(emo_va[:, 0].unsqueeze(1).detach().cpu().numpy())
#     # A_batch.extend(emo_va[:, 1].unsqueeze(1).detach().cpu().numpy())

#     # Assuming that the correct prediction is when the prediction is close enough to the target
#     # You can adjust the threshold as needed
#     threshold = 0.015
#     correct_V = (np.abs(predictions_va[:, 0].detach().cpu().numpy() - emo_va[:, 0].detach().cpu().numpy()) < threshold)
#     correct_A = (np.abs(predictions_va[:, 1].detach().cpu().numpy() - emo_va[:, 1].detach().cpu().numpy()) < threshold)
#     # correct_D = (np.abs(predictions_va[:, 2].detach().cpu().numpy() - emo_va[:, 2].detach().cpu().numpy()) < threshold)

#     correct_predictions += np.sum(correct_V & correct_A )
#     total_predictions += emo_va.size(0)

#     accuracy = correct_predictions / total_predictions
#     return accuracy

def plot_predictions(labels, preds_v=None, preds_a=None, preds_d=None, valences=None, arousals=None, dominances=None):
    flat_labels = [item[0] for item in labels]
    fig, ax = plt.subplots(figsize=(10, 10))
    unique_labels = list(set(flat_labels))
    
    # 使用不重複的顏色
    colors = ListedColormap(plt.cm.get_cmap('tab10').colors[:len(unique_labels)])
    
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

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=14)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(f'multi_{xlabel[0]}_{ylabel[0]}.png')

def plot_predictions_3(labels, preds_v, preds_a, preds_d, valences, arousals, dominances):
    flat_labels = [item[0] for item in labels]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = list(set(flat_labels))
    
    # 使用不重複的顏色
    colors = ListedColormap(plt.cm.get_cmap('tab10').colors[:len(unique_labels)])

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

    ax.set_xlabel('Valence', fontsize=16, labelpad=20)  # Adjust the labelpad value
    ax.set_ylabel('Arousal', fontsize=16, labelpad=20)  # Adjust the labelpad value
    ax.set_zlabel('Dominance', fontsize=16, labelpad=20)  # Adjust the labelpad value

    ax.legend(loc='best', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.zaxis.set_tick_params(labelsize=16)

    plt.savefig('multi.png')


emotion_list = ['sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful', 'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed', 'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised', 'nostalgic', 'confident', 'furious', 'disappointed', 'caring', 'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful', 'content', 'impressed', 'apprehensive', 'devastated']
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Initialize MyModel
    my_model = MyModel()

    my_model.encoder.load_state_dict(torch.load("My_model_pth_multi/ceg_d_multi_encoder.pth",map_location=device))
    my_model.MLP.load_state_dict(torch.load("My_model_pth_multi/ceg_d_multi_mlp.pth",map_location=device))
    
    my_model.to(device)
    #my_model.encoder.classifier.load_state_dict(torch.load("source_continuous/2classifier.bin",map_location=device))

    # Set models to evaluation mode
    my_model.eval()

    # a, new_data, b, new_data_val, c, new_data_test = predata()
    c, new_data_test = predata_test()
    loader_val = data_loader_test(c, new_data_test, batch_size=1)
    references = []
    inputs = []
    emotion_labels = []
    # prompt_batchs =[]
    # my_model_responses = []
    # llama_responses = []
    
    #score
    my_score_r1 = []
    my_score_r2 = []
    my_score_rL = []
    llama_score_r1 = []
    llama_score_r2 = []
    llama_score_rL = []
    
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
    with open('multi_dis/ceg_d_multi.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Input', 'Golden', 'MyResponse', 'LLamaResponse', 'Label','rougel_my','rougel_llama'])
        # csvwriter.writerow(['--------------'])
        for emotion_batch, text_batch, llm_text_batch, llm_target_batch, input_batch, emo_label in bar: 
            text_batch = text_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            llm_text_batch = llm_text_batch.to(device)

            generated_tokens, predictions_va,emo_va = my_model.generate(text_batch, emotion_batch)
            my_response = my_model.llama_tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            # Get LLama predictions
            with torch.no_grad():
                #print(text_batch)
                llama_token = my_model.decoder.generate(llm_text_batch.input_ids, max_length=256)
                llama_response = my_model.llama_tokenizer.batch_decode(llama_token, skip_special_tokens=True)

            # from datasets
            inputs.extend(input_batch)
            references.extend(llm_target_batch)
            emotion_labels.extend(emo_label)

            # Prepare my_response and llama_response for metric
            reference_str = ' '.join(references[cnt])
            my_response_str = ' '.join(my_response)
            llama_response_str = ' '.join(llama_response)

            ## BLEU
            my_score_bleu.append(calculate_bleu_score(reference_str, my_response_str))
            llama_score_bleu.append(calculate_bleu_score(reference_str, llama_response_str))

            ## RougeL
            rougel_my = rougeScore(my_response_str, reference_str)
            r1_my = rougel_my['rouge1_fmeasure']
            r2_my = rougel_my['rouge2_fmeasure']
            rL_my = rougel_my['rougeL_fmeasure']
            rougel_llama = rougeScore(llama_response_str, reference_str)
            r1_llama = rougel_llama['rouge1_fmeasure']
            r2_llama = rougel_llama['rouge2_fmeasure']
            rL_llama = rougel_llama['rougeL_fmeasure']

            my_score_r1.append(r1_my)
            my_score_r2.append(r2_my)
            my_score_rL.append(rL_my)
            llama_score_r1.append(r1_llama)
            llama_score_r2.append(r2_llama)
            llama_score_rL.append(rL_llama)

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
            acc_my.append(calculate_acc(predictions_va, emotion_batch))     

            bar.set_postfix_str(f'R1:{np.mean(my_score_r1):4}, acc:{np.mean(acc_my)}')

            ## Example
            csvwriter.writerow([inputs[cnt], reference_str, my_response_str, llama_response_str, emotion_labels[cnt] ,rougel_my,rougel_llama])

            if cnt%30 == 0:
                print(f"BLEU score for MyModel: {np.mean(my_score_bleu)}")
                print(f"BLEU score for LLama: {np.mean(llama_score_bleu)}")
                print(f"rougel score for MyModel: {np.mean(my_score_r1)}")
                print(f"rougel score for LLama: {np.mean(llama_score_r1)}")
                print(f"rouge2 score for MyModel: {np.mean(my_score_r2)}")
                print(f"rouge2 score for LLama: {np.mean(llama_score_r2)}")
                print(f"rougeL score for MyModel: {np.mean(my_score_rL)}")
                print(f"rougeL score for LLama: {np.mean(llama_score_rL)}")

                print("---------------------------------------")
                print(f"Distinct-1 score for MyModel: {np.mean(my_score_dist1)}")
                print(f"Distinct-1 score for LLama: {np.mean(llama_score_dist1)}")
                print(f"Distinct-2 score for MyModel: {np.mean(my_score_dist2)}")
                print(f"Distinct-2 score for LLama: {np.mean(llama_score_dist2)}")
                print(f"Perplexity for MyModel: {np.mean(my_perplexity)}")
                print(f"Perplexity for LLama: {np.mean(llama_perplexity)}")
                print(f"Acc for MyModel: {np.mean(acc_my)}")
                

            # csvwriter.writerow(['\n--------------'])

    print(f"BLEU score for MyModel: {np.mean(my_score_bleu)}")
    print(f"BLEU score for LLama: {np.mean(llama_score_bleu)}")
    print(f"rougel score for MyModel: {np.mean(my_score_r1)}")
    print(f"rougel score for LLama: {np.mean(llama_score_r1)}")
    print(f"rouge2 score for MyModel: {np.mean(my_score_r2)}")
    print(f"rouge2 score for LLama: {np.mean(llama_score_r2)}")
    print(f"rougeL score for MyModel: {np.mean(my_score_rL)}")
    print(f"rougeL score for LLama: {np.mean(llama_score_rL)}")
    print("---------------------------------------")
    print(f"Distinct-1 score for MyModel: {np.mean(my_score_dist1)}")
    print(f"Distinct-1 score for LLama: {np.mean(llama_score_dist1)}")
    print(f"Distinct-2 score for MyModel: {np.mean(my_score_dist2)}")
    print(f"Distinct-2 score for LLama: {np.mean(llama_score_dist2)}")
    print(f"Perplexity for MyModel: {np.mean(my_perplexity)}")
    print(f"Perplexity for LLama: {np.mean(llama_perplexity)}")
    print(f"Acc for MyModel: {np.mean(acc_my)}")

    print("done")