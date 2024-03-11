import torch
import gradio as gr # ! pip install gradio

from tokenizer import LlamaTokenizer
from model import Transformer, ModelArgs
from inference import sample

# load model
model_path = './base.pt'
device = torch.device('cpu')

config = ModelArgs()
tokenizer = LlamaTokenizer()

model = Transformer(config).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
    
def chat(number):
    number = str(number)
    x = tokenizer.encode(' '.join(number), add_special_tokens=True)
    x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0) # shape: [1, t]
    
    logit = model(x, mask=None)
    result = sample(logit)[0]
    result = tokenizer.decode([result.item()])
    
    return result[0]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    with gr.Row():
        clear = gr.ClearButton([msg, chatbot])
        submit = gr.Button('submit')

    def respond(message, chat_history):
        bot_message = chat(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])

demo.launch()