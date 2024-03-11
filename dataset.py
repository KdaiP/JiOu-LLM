import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import LlamaTokenizer

class LlamaDataset(Dataset):
    def __init__(self, range_min=1, range_max=18, num_samples=100000000):
        self.range_min = range_min
        self.range_max = range_max
        self.num_samples = num_samples

        self.data, self.labels = self._generate_dataset()
        self.tokenizer = LlamaTokenizer()

    def _generate_dataset(self):
        data = generate_random_numbers(self.num_samples, self.range_min, self.range_max,)
        labels = data % 2 == 0 # 假设even_token(偶数)对应 1, odd_token(奇数) 对应 0
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        num, label = self.data[idx], self.labels[idx]
        num_str = ' '.join(str(num)) # output example: '1 1 4 5 1 4'
        
        input_sequence = self.tokenizer.encode(num_str, add_special_tokens=True)
        
        label_token = self.tokenizer.special_tokens['even_token'] if label == 1 else self.tokenizer.special_tokens['odd_token']
        label_sequence = self.tokenizer.encode(label_token, add_special_tokens=False)
        
        sequence = input_sequence + label_sequence
        return torch.LongTensor(sequence)
    
def collate_fn(data):
    """将一个batch中不同位数的随机数补零到相同长度"""
    data_length = torch.LongTensor([x.size(0) for x in data])
    data = torch.nested.to_padded_tensor(torch.nested.nested_tensor(data), padding=0)
    return data, data_length

def generate_random_numbers(num_count: int, min_digits: int, max_digits: int) -> np.ndarray:
    """生成不同位数的随机数"""
    digits = np.random.randint(min_digits, max_digits + 1, size=num_count)
    min_values = 10**(digits - 1)
    max_values = 10**digits
    return np.unique(np.random.randint(min_values, max_values))

if __name__ ==  '__main__':
    # 测试Dataset
    dataset = LlamaDataset()

    # 获取第一个样本
    sample_data = dataset[0]

    print("Encoded data:", sample_data)

    # 使用DataLoader进行批处理
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)

    for data in dataloader:
        print("Batch data shape:", data.shape)
        break  # 只打印第一批数据的形状