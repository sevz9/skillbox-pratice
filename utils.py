import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

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