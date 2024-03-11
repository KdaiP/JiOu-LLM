from typing import List, Dict, Optional

class LlamaTokenizer:
    def __init__(self, vocab: Optional[List[str]] = None):
        default_vocab: List[str] = ['[PAD]', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    '[SEP]', '[ODD]', '[EVEN]', '[UNK]'] # PAD一定要放在第一位，因为collate_fn默认补零。
        self.vocab: List[str] = vocab if vocab is not None else default_vocab
        self.special_tokens: Dict[str, str] = {
            'unk_token': '[UNK]',
            'pad_token': '[PAD]',
            'sep_token': '[SEP]',
            'odd_token': '[ODD]',
            'even_token': '[EVEN]'
        }

        assert all(token in self.vocab for token in self.special_tokens.values()), "Some special tokens are missing in the vocab."

        self.token_to_id: Dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: token for token, idx in self.token_to_id.items()}

    def tokenize(self, text: str) -> List[str]:
        """按空格进行分割"""
        return text.split()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(token, self.token_to_id[self.special_tokens['unk_token']]) for token in tokens]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.special_tokens['sep_token']] + tokens + [self.special_tokens['sep_token']]
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids
    
    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(id_, self.id_to_token[self.token_to_id[self.special_tokens['unk_token']]]) for id_ in ids]
    
    def add_tokens(self, new_tokens: List[str]) -> None:
        for token in new_tokens:
            if token not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab.append(token)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token

    

if __name__ == "__main__":
    tokenizer = LlamaTokenizer()
    encoded_text = tokenizer.encode("1 1 4 5 1 4", add_special_tokens=True)
    decoded_text = tokenizer.decode(encoded_text, )
    print(encoded_text)
    print(decoded_text)