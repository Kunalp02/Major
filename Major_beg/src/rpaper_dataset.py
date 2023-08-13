import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer

class ResearchPaperDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        abstract = self.data.loc[idx, 'Abstract']
        summary = self.data.loc[idx, 'Summarization']

        input_ids = self.tokenizer.encode(abstract, max_length=self.max_length, truncation=True)
        target_ids = self.tokenizer.encode(summary, max_length=self.max_length, truncation=True)

        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids),
        }
