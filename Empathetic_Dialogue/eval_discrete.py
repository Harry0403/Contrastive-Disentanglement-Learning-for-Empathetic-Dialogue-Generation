import torch
from tqdm import tqdm
from source_discrete.dataloader import predata, data_loader_val
from datasets import load_dataset
from source_discrete.decoder import MyModel
# import sacrebleu
# from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
import numpy as np
import nltk
from itertools import chain
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

def calculate_acc(x_emo, emotion_batch):
    correct_predictions = 0
    total_predictions = 0
    correct_predictions += (x_emo == emotion_batch).sum().item()
    #print(x_emo, emotion_batch)
    total_predictions += emotion_batch.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy



def plot_confusion_matrix(true_labels, predicted_labels, emotion_list):
    """
    Plot confusion matrix for the emotion classification task.

    Args:
    true_labels (list): List of true emotion labels.
    predicted_labels (list): List of predicted emotion labels by the model.
    emotion_list (list): List of all possible emotion labels.

    Returns:
    None
    """
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=emotion_list)
    
    # Normalize confusion matrix by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=emotion_list)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    # plt.title('Normalized Confusion Matrix')
    # plt.show()
    plt.savefig('Confusion_Matrix.png')

# Example usage within your code:
# plot_confusion_matrix(true_labels, predicted_labels, emotion_list)


emotion_list = ['sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful', 'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed', 'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised', 'nostalgic', 'confident', 'furious', 'disappointed', 'caring', 'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful', 'content', 'impressed', 'apprehensive', 'devastated']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Initialize MyModel
    my_model = MyModel()

    # my_model.encoder.load_state_dict(torch.load("My_model_pth/SimD_continuous_model_encoder.pth",map_location=device))
    # my_model.MLP.load_state_dict(torch.load("My_model_pth/SimD_continuous_model_mlp.pth",map_location=device))
    # my_model.encoder.classifier.load_state_dict(torch.load("My_model_pth/3_tmpe.bin",map_location=device))
    my_model.encoder.load_state_dict(torch.load("My_model_pth/ceclg_dis_encoder.pth",map_location=device))
    my_model.MLP.load_state_dict(torch.load("My_model_pth/ceclg_dis_mlp.pth",map_location=device))
    my_model.to(device)
    #my_model.encoder.classifier.load_state_dict(torch.load("source_continuous/2classifier.bin",map_location=device))

    # Set models to evaluation mode
    my_model.eval()

    c, new_data_test = predata()
    loader_val = data_loader_val(c, new_data_test, batch_size=1)
    references = []
    inputs = []
    emotion_labels = []
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
    true_labels = []
    predicted_labels = []

    cnt = 0
    bar = tqdm(loader_val, ncols=110, smoothing=0.05)
    with open('discrete_response/ceclg_d_testtesinginging.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Input', 'Golden', 'MyResponse', 'LLamaResponse', 'Label','rougel_my','rougel_llama'])
        # csvwriter.writerow(['--------------'])
        for emotion_batch, emo_label, text_batch, llm_text_batch, llm_target_batch, input_batch in bar: 
            text_batch = text_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            llm_text_batch = llm_text_batch.to(device)

            generated_tokens, _,_,emo_x = my_model.generate(text_batch)
            print(emo_x,emotion_batch)
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

            # Prepare my_response and llama_response for metric
            reference_str = ' '.join(references[cnt])
            my_response_str = ' '.join(my_response)
            llama_response_str = ' '.join(llama_response)

            ##labels
            true_labels.append(emotion_batch)
            predicted_labels.append(emo_x)

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
            acc_my.append(calculate_acc(emo_x,emotion_batch))     

            bar.set_postfix_str(f'RougeL:{np.mean(my_score_rougel):4}, acc:{np.mean(acc_my)}')

            ## Example
            csvwriter.writerow([inputs[cnt], reference_str, my_response_str, llama_response_str, emotion_labels[cnt] ,rougel_my,rougel_llama])

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
        
                #print(emotion_labels[num:-1][2])
                # plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = pre_A_batch[num:-1], preds_d = None, valences = V_batch[num:-1], arousals = A_batch[num:-1], dominances = None)
                # plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = None, preds_d = pre_D_batch[num:-1], valences = V_batch[num:-1], arousals = None, dominances = D_batch[num:-1])
                # plot_predictions(emotion_labels[num:-1], preds_v = None, preds_a = pre_A_batch[num:-1], preds_d = pre_A_batch[num:-1], valences = None, arousals = A_batch[num:-1], dominances = D_batch[num:-1])

            cnt += 1
            if cnt == 4:
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

    plot_confusion_matrix(true_labels, predicted_labels, emotion_list)

    num = -15
    # print(emotion_labels[num:-1][2])
    # plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = pre_A_batch[num:-1], preds_d = None, valences = V_batch[num:-1], arousals = A_batch[num:-1], dominances = None)
    # plot_predictions(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = None, preds_d = pre_D_batch[num:-1], valences = V_batch[num:-1], arousals = None, dominances = D_batch[num:-1])
    # plot_predictions(emotion_labels[num:-1], preds_v = None, preds_a = pre_A_batch[num:-1], preds_d = pre_A_batch[num:-1], valences = None, arousals = A_batch[num:-1], dominances = D_batch[num:-1])
    # plot_predictions_3(emotion_labels[num:-1], preds_v = pre_V_batch[num:-1], preds_a = pre_A_batch[num:-1], preds_d = pre_A_batch[num:-1], valences = V_batch[num:-1], arousals = A_batch[num:-1], dominances = D_batch[num:-1])
    print("done")