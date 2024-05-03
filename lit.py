from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch import Tensor
import lightning as L
from torchmetrics import F1Score, Accuracy
from sklearn.metrics import f1_score

class Lit(L.LightningModule):
    def __init__(self, cfg, n_classes=4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForSequenceClassification.from_pretrained(cfg['model']['model_name'])
        self.tokenizer = BertTokenizer.from_pretrained(cfg['model']['tokenizer_name']) 
        self.max_len = cfg['model']['max_length']
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model = self.model
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = cfg['training']['learning_rate']
        self.n_classes = n_classes

    def training_step(
        self, batch: tuple[list[Tensor], list[dict[str, Tensor]]], batch_idx: int
    ):
        self.model.train()

        train_input, train_label = batch
       
       
        mask = train_input['attention_mask']
        input_id = train_input['input_ids'].squeeze(1)
        output = self.model(input_id, mask)

        loss = self.loss(output[0], train_label.long())

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ):
       
        train_input, train_label = batch
      
        
        
        
        mask = train_input['attention_mask']
        input_id = train_input['input_ids'].squeeze(1)
        output = self.model(input_id, mask)


        f1 = F1Score(task="multiclass", num_classes=self.n_classes, average='macro').to('cuda')
        
        score = f1(output[0], train_label).to('cuda')

        self.log('val_f1_score', score)
        
        return {
            "f1_score": score,
        }
    
    def test_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ):
   
        train_input, train_label = batch
      
        mask = train_input['attention_mask']
        input_id = train_input['input_ids'].squeeze(1)
        output = self.model(input_id, mask)
       
        score = f1_score(train_label, output, average='macro')        

        self.log('test_f1_score', score)
        
        return {
            "f1_score": score,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[5, 10, 15]
            )
        }