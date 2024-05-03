from pathlib import Path
from typing import Any, Callable

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from text_process import remove_emoji, remove_rus_stopwords_func


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
    

    
        CLASSES = list(df['category'].unique())

        labels = dict(zip(CLASSES, range(len(CLASSES))))
       
        self.labels = [labels[label] for label in df['category']]
   
            
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
      
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])
        return batch_texts, batch_y
       


_collate_fn_t = Callable[[list[tuple[Tensor, Any]]], Any]


class Datamodule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,  
    ):
        super().__init__()
        self.small_classes = cfg['preprocessing']['small_classes']
        self.datadir = cfg['data_path']
        self.batch_size = cfg['batch_size']
        self.num_workers = cfg['num_workers']

        self.test_size = cfg['test_size']
        self.val_size = cfg['val_size']

        self.tokenizer = BertTokenizer.from_pretrained(cfg['model']['tokenizer_name'])

    def prepare_data(self) -> None:
        df = pd.read_csv(self.datadir)

        df = df[~df['Категория'].isin(self.small_classes)]    

        df = df[['Категория', 'Комментарий']].dropna()

        df['Комментарий'] = df['Комментарий'].apply(lambda text: remove_rus_stopwords_func(text))

        df['Комментарий'] = df['Комментарий'].apply(lambda text: remove_emoji(text))

        df = df[df.Комментарий.apply(lambda x: len(x.split())) > 1]

        df.drop_duplicates(inplace=True, subset=['Комментарий'])
        
        rename = {
            'Категория': 'category',
            'Комментарий': 'text'
        }
        df = df.rename(columns=rename)
        
        self.train, self.test = train_test_split(df, test_size=self.test_size, random_state=1337)
        self.train, self.val = train_test_split(self.train, test_size=self.val_size, random_state=1337)
    
    @property
    def collate_fn(self) -> _collate_fn_t | None:
        return lambda batch: tuple(zip(*batch))
        
    def setup(self, stage: str) -> None:

        if stage == "fit":
            self.train_dataset = CustomDataset(
                self.train,
                self.tokenizer
            )
            self.val_dataset = CustomDataset(
                self.val,
                self.tokenizer
            )
   
        elif stage == "validate":
            self.val_dataset = CustomDataset(
                self.val,
                self.tokenizer
            )
        elif stage == "test":
            self.test_dataset = CustomDataset(
                self.test,
                self.tokenizer
            )
        else:
            raise NotImplementedError


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
     
        )