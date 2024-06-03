import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Union

import hydra
import lightning as L
import nltk
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import torch
import torch.nn as nn

from lightning.pytorch.utilities.types import (EVAL_DATALOADERS, STEP_OUTPUT,
                                               TRAIN_DATALOADERS)
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             multilabel_confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          get_cosine_schedule_with_warmup)

from datamodule import Datamodule
from lit import Lit

def predict(logits: torch.Tensor) -> np.ndarray:
    """Helper function for predictions calculating.

    Args:
        logits (torch.Tensor): model's raw output

    Returns:
        np.ndarray: array with predicted class id.
    """
    s = torch.nn.Softmax()
    probs = s(torch.tensor(logits))
    return np.argmax(probs)

def get_score(datamodule, lit_module):
    
    datamodule.setup(stage="test")

    preds_logits = torch.tensor([])
    targets = torch.tensor([])

    with torch.no_grad():
        for val_input, val_label in tqdm(datamodule.test_dataloader()):
            mask = val_input['attention_mask']
            input_id = val_input['input_ids'].squeeze(1)
            output = lit_module.model(input_id, mask)[0]
            preds_logits = torch.cat((preds_logits, output))
            targets = torch.cat((targets, val_label.long().cpu()))
    preds = np.apply_along_axis(predict, 1, preds_logits)
    return  f1_score(targets, preds, average='macro')    



@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = Datamodule(cfg)
    
    # Create lit module
    lit_module = Lit(
        cfg
    )

    # Create trainer
    trainer = L.Trainer(
        accelerator=cfg['training']['accelerator'],
        max_epochs=cfg['training']['num_train_epochs'], 
    )

    # Fit
    print('START FITTING')
    trainer.fit(
        model=lit_module,
        datamodule=datamodule,
    )

    # Inference
    score = get_score(datamodule, lit_module)
    print(f"SCORE: {score}")
main()







