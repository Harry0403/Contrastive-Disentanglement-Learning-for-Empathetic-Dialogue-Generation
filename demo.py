import torch
import gradio as gr
from empathetic_work.source_multi.decoder_multi import MyModel
import re

# Load the custom model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()
model.encoder.load_state_dict(torch.load("/home/Work/Empathetic_Dialogue/My_model_pth/ceclg_c_multi_encoder.pth", map_location=device))
model.MLP.load_state_dict(torch.load("/home/Work/Empathetic_Dialogue/My_model_pth/ceclg_c_multi_mlp.pth", map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = model.llama_tokenizer

def greet(inputs, emotion="nervous"):
    role = ["user", "assistant"]
    messages = []
    chat_template = []
    i = 0
    for item in inputs:
        messages.append({"role": role[i % 2], "content": item})
        i += 1

    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)+("[INST]" if len(inputs)%2==0 else "")
    chat_template.append(chat)

    # Tokenize the formatted string
    inputs_tokenized = model.encoder.tokenizer(chat_template, return_tensors="pt").to(device)
    emo_tokenized = model.encoder.tokenizer(emotion, return_tensors='pt', max_length=128, truncation=True, padding=True).to(device)
    
    # print(messages)
    chat_template_llama = chat[:3]+"<<SYS>>\nKeep the sentences briefly.\n<</SYS>>"+chat[3:]
    print(chat_template_llama)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.unk_token
    llm_text_batch = tokenizer([chat_template_llama], return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)

    # Generate output using the model
    with torch.no_grad():
        generated_tokens, predictions_va, emo_va = model.generate(inputs_tokenized, emo_tokenized)
        my_response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        llama_token = model.decoder.generate(llm_text_batch.input_ids, max_new_tokens=64)
        llama_response = model.llama_tokenizer.decode(llama_token[0,llm_text_batch.input_ids.shape[1]:], skip_special_tokens=True)

    return re.sub("_comma_",", ", my_response[0]), llama_response

# Gradio interface
def format_inputs(input_text, emotion):
    # Split the multiline input into a list of strings
    input_list = input_text.strip().split("\n")
    return input_list, emotion

# Set up Gradio inputs: the first is for the list of strings, the second is for the emotion
input_textbox = gr.Textbox(lines=5, placeholder="Enter multiple sentences, each on a new line")
emotion_textbox = gr.Textbox(lines=1, placeholder="Enter an emotion (e.g., 'nervous')")

# Define the interface using two inputs and two outputs
demo = gr.Interface(
    fn=lambda input_text, emotion: greet(*format_inputs(input_text, emotion)),
    inputs=[input_textbox, emotion_textbox],
    outputs=[gr.Textbox(label="My Proposed Response"), gr.Textbox(label="Llama2 Response")]
)

demo.launch(share=True)
# i am nervous now.
# you are going to have an interview?
# I do not think I will be able to get a job.
# No, you should open your mind.
# I guess I just need to be more confident.


# I ate an entire cake last night.
# oh, how do your parents think?


#i like to eat ice cream.
#Oh, that too sweet.

#i have a meeting later.
#you can take a rest before the meeting.
#i don't like sleeping during the day.