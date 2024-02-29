import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchmetrics import F1Score, Accuracy
from transformers import BertTokenizer, BertForSequenceClassification, get_cosine_schedule_with_warmup, AdamW
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
from matplotlib import pyplot as plt
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    multilabel_confusion_matrix,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
)

import re

import nltk
from nltk.corpus import stopwords
import string
import sys
from src.config import compose_config
from hydra.utils import instantiate
import logging
import hydra

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union


log = logging.getLogger(__name__)


import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

def remove_emoji(inputString):
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F" # emoticons
    u"\U0001F300-\U0001F5FF" # symbols & pictographs
    u"\U0001F680-\U0001F6FF" # transport & map symbols
    u"\U0001F1E0-\U0001F1FF" # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u'\U00010000-\U0010ffff'
    u"\u200d"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\u3030"
    u"\ufe0f"
    u"\u2069"
    u"\u2066"
    u"\u200c"
    u"\u2068"
    u"\u2067"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', inputString)


nltk.download("stopwords")
def remove_rus_stopwords_func(text):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without Stop Words
    ''' 
    

   
    # check in lowercase 
    t = [token for token in text.split() if not token in set(stopwords.words("russian"))]
    text = ' '.join(t)    
    return text


def process_data(df, cfg_prep):
    # df = df[(df['Категория'] != "Качество материалов") & (df['Категория'] != "Интерфейс платформы") & (df['Категория'] != "Общение с куратором")]
    df = df[~df['Категория'].isin(cfg_prep['small_classes'])]

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
    return df



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, cfg_model):
       
        CLASSES = list(df['category'].unique())

        labels = dict(zip(CLASSES, range(len(CLASSES))))

        self.labels = [labels[label] for label in df['category']]
   
            
        self.texts = [tokenizer(text, 
                               padding=cfg_model["padding"], max_length = cfg_model["max_length"], truncation=True,
                                return_tensors="pt") for text in df['text']]
        
        self.indexes = df.index.values

    def classes(self):
        return self.labels

    def __len__(self):
      
        return len(self.labels)
       

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    
    def get_batch_oid(self, idx):
        return np.array(self.oid[idx])

    def get_batch_texts(self, idx):
        
        return self.texts[idx]
    
    def get_batch_indexes(self, idx):
        return self.indexes[idx]

    def __getitem__(self, idx):
        
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_indexes = self.get_batch_indexes(idx)
        return batch_indexes, batch_texts, batch_y

class FocalLoss(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target!=self.ignore_index) 
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:

        m = nn.Softmax()
        x = m(x)
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

class BertClassifier(nn.Module):
    def __init__(self, data, CLASSES, cfg_train, cfg_model, cgf_global):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(cfg_model['pretrained_model_name_or_path'])
        self.tokenizer = BertTokenizer.from_pretrained(cfg_model['tokenizer_name'])
        self.data = data
        self.device = torch.device('cuda')
        self.max_len = cfg_model['max_length']
        self.epochs = cfg_train['num_train_epochs']
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.CLASSES = list(data['category'].unique())
        n_classes = len(self.CLASSES)
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes).cuda()
        self.model = self.model.cuda()

        self.cfg_train = cfg_train
        self.cfg_model = cfg_model
        self.cgf_global = cgf_global


       
            
    def preparation(self):
        self.df_train, self.df_val = train_test_split(self.data, test_size=self.cgf_global['val_size'], random_state=1337)
        
        self.train = CustomDataset(self.df_train, self.tokenizer, self.cfg_model)
        self.val = CustomDataset(self.df_val, self.tokenizer, self.cfg_model)

        c = Counter(self.df_train.category)
        self.weights = {cat: 1/c.get(cat) for cat in self.CLASSES}
        weights = torch.tensor([1/c.get(cat) for cat in self.CLASSES]).to('cuda')
        target_proportions = {cat: 1 for cat in self.CLASSES}
        sample_weights = [target_proportions[i]*self.weights[i] for i in self.df_train.category.values]

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(self.train))
        self.train_dataloader = torch.utils.data.DataLoader(self.train, batch_size=self.cfg_train['per_device_train_batch_size'])
        self.val_dataloader = torch.utils.data.DataLoader(self.val, batch_size=self.cfg_train['per_device_val_batch_size'])
    
       
        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg_train['learning_rate'], correct_bias=False)
        self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.train_dataloader) * self.epochs
            )
        
        self.loss_fn = FocalLoss(gamma=2, weights=weights).cuda()
            
    def fit(self):
        self.model = self.model.train()
        
        for epoch_num in range(self.epochs):
            total_acc_train = 0
            total_loss_train = 0
            for _, train_input, train_label in tqdm(self.train_dataloader):
                train_label = train_label.cuda()
                mask = train_input['attention_mask'].cuda()
                input_id = train_input['input_ids'].squeeze(1).cuda()
              
                output = self.model(input_id.cuda(), mask.cuda())

                batch_loss = self.loss_fn(output[0], train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output[0].argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            total_acc_val, total_loss_val = self.eval()
           
            print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(self.df_train): .3f} \
            | Train Accuracy: {total_acc_train / len(self.df_train): .3f} \
            | Val Loss: {total_loss_val / len(self.df_val): .3f} \
            | Val Accuracy: {total_acc_val / len(self.df_val): .3f}')

            model_id = self.cgf_global['MODEL_ID']
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(self.model, f'checkpoint/model_{model_id}_epoch_{epoch_num}.pt')
        os.makedirs('models', exist_ok=True)
        
        torch.save(self.model, f"models/model_{model_id}.pt")

        return total_acc_train, total_loss_train
    
    def eval(self):
        self.model = self.model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for _, val_input, val_label in tqdm(self.val_dataloader):
                val_label = val_label.cuda()
                mask = val_input['attention_mask'].cuda()
                input_id = val_input['input_ids'].squeeze(1).cuda() 

                output = self.model(input_id.to('cuda'), mask.to('cuda'))

                batch_loss = self.loss_fn(output[0], val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output[0].argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
            
        return total_acc_val, total_loss_val

def pred_by_logits(logits: torch.Tensor) -> np.ndarray:
    """Helper function for predictions calculating.

    Args:
        logits (torch.Tensor): model's raw output

    Returns:
        np.ndarray: array with predicted class id.
    """
    s = torch.nn.Softmax()
    probs = s(torch.tensor(logits))
    return np.argmax(probs)

def predict(model, test_dataloader):
    preds_logits = torch.tensor([])
    targets = torch.tensor([])

    with torch.no_grad():
        for _, val_input, val_label in tqdm(test_dataloader):
            mask = val_input['attention_mask'].cuda()
            input_id = val_input['input_ids'].squeeze(1).cuda()
            output = model(input_id, mask)[0].cpu()
            preds_logits = torch.cat((preds_logits, output))
            targets = torch.cat((targets, val_label.long().cpu()))
    
    
    return np.apply_along_axis(pred_by_logits, 1, preds_logits), targets
    
    




@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(cfg: DictConfig):
    # log.info(OmegaConf.to_yaml(cfg, resolve=True))
    df = pd.read_csv(r"C:\Users\vsevo\MKN\skillbox_nlp-vsevolod-lavrov\data\practice_cleaned.csv")
    df = process_data(df, cfg["preprocessing"])
    train, test = train_test_split(df, test_size=cfg["test_size"], random_state=1337)
    train = train.head(5)
    test = test.head(5)
    CLASSES = list(train['category'].unique())
    bert_tiny = BertClassifier(train, CLASSES, cfg['training'], cfg['model'], cfg)
    tokenizer = BertTokenizer.from_pretrained(cfg['model']['tokenizer_name'])
    bert_tiny.preparation()
    bert_tiny.fit()
    test_dataset = CustomDataset(test, tokenizer, cfg['model'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['per_device_test_batch_size'] )
    preds, targets = predict(bert_tiny.model, test_dataloader)
    cr = classification_report(targets, preds, target_names=CLASSES, output_dict=True)
    cr = pd.DataFrame(cr).T
    cm = confusion_matrix(targets, preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix =cm, display_labels=CLASSES)
    # log.info(f"Achieved f1-score (macro): {cr['f1-score']['macro avg']}")
#     tokenizer = AutoTokenizer.from_pretrained(
#         cfg["preprocessing"]["tokenizer_name"], token=cfg["secrets"]["hf_token"]
#     )
#     label2id = t["Категория"].unique().sort().to_list()
#     label2id = dict(zip(label2id, range(len(label2id))))
#     id2label = {v: k for k, v in label2id.items()}
#     train_ds, test_ds = make_dataset(
#         t, tokenizer, label2id, cfg["preprocessing"], cfg["test_size"]
#     )
#     trainer = make_training_pipeline(
#         tokenizer, train_ds, test_ds, cfg["model"], cfg["training"], id2label, label2id
#     )
#     trainer.train()
#     preds = trainer.predict(test_ds)
#     pred_labels = np.apply_along_axis(predict, 1, preds[0])
#     pred_labels = [id2label[x] for x in pred_labels]
#     gt_labels = [id2label[x] for x in preds[1]]
#     cr = classification_report(gt_labels, pred_labels, output_dict=True)
#     cr = pd.DataFrame(cr).T
#     cm = confusion_matrix(gt_labels, pred_labels, labels=list(label2id.keys()))
    # x = list(label2id.keys())
    # y = list(reversed(label2id.keys()))
    # fig = ff.create_annotated_heatmap(np.flipud(cm), x=x, y=y, colorscale="Viridis")
    # fig.update_layout(title_text="Confusion matrix")
    # fig.add_annotation(
    #     dict(
    #         x=0.5,
    #         y=-0.15,
    #         showarrow=False,
    #         text="Predicted value",
    #         xref="paper",
    #         yref="paper",
    #     )
    # )

#     fig.add_annotation(
#         dict(
#             x=-0.16,
#             y=0.5,
#             showarrow=False,
#             text="Real value",
#             textangle=-90,
#             xref="paper",
#             yref="paper",
#         )
#     )

#     fig["data"][0]["showscale"] = True

#     log.info(f"Achieved f1-score (macro): {cr['f1-score']['macro avg']}")
#     cr.to_csv(f"reports/hydra_run_example_{cfg['preprocessing']['max_length']}.csv")
#     fig.write_html(
#         f"reports/hydra_run_example_{cfg['preprocessing']['max_length']}.html"
#     )


if __name__ == "__main__":
    main()


        
