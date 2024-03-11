from typing import Optional
from tqdm import tqdm
import torch

from tokenizer import LlamaTokenizer
from model import Transformer, ModelArgs

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

@ torch.inference_mode()
def inference(numbers: list, device = 'cpu'):
    wrong = 0
    for number in tqdm(numbers):
        number = str(number)
        x = tokenizer.encode(' '.join(number), add_special_tokens=True)
        x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0) # shape: [1, t]
        
        logit = model(x, mask=None)
        result = sample(logit)[0]
        result = tokenizer.decode([result.item()])
        
        true_result = '[EVEN]' if int(number) % 2 == 0 else '[ODD]'
        if result[0] != true_result:
            wrong += 1
            # print(f'{number}: {result}')
        print(f'{number}: {result}')
            
    print(f'accuracy: {(len(numbers) - wrong) / len(numbers)}')

if __name__ == '__main__':
    model_path = './base.pt'
    device = torch.device('cpu')
    
    config = ModelArgs()
    tokenizer = LlamaTokenizer()
    
    model = Transformer(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Total param: {n_params / 1e6}')
    
    numbers = range(100000) # a list contains the numbers that you want to inference
    inference(numbers, device=device)