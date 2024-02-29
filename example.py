import logging
from functools import partial
from typing import Any

import hydra
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import polars as pl
import torch
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EarlyStoppingCallback, EvalPrediction, Trainer

log = logging.getLogger(__name__)

frame = pl.read_csv("data/practice_cleaned.csv")


def preprocess_frame(frame: pl.DataFrame, small_classes: list[str]) -> pl.DataFrame:
    """Filter out empty comments and comments from small categs.

    Args:
        frame (pl.DataFrame): input raw frame.
        small_classes (list[str]): list with classes
        irrelevant for classification task.

    Returns:
        pl.DataFrame: clear processed frame.
    """
    original_shape = frame.shape
    frame = frame.filter(~pl.col("Категория").is_in(small_classes))
    frame = frame.filter(~pl.col("Категория").is_null())
    frame = frame.filter(~(pl.col("Комментарий").is_null()))
    log.info(f"Empty comments & Category filtering: {original_shape} -> {frame.shape}")
    return frame


def preprocess_sample(
    sample: dict[str, Any], tokenizer: AutoTokenizer, padding: str, max_length: int
) -> dict[str, Any]:
    """Encode input raw string to sequence of tokens.
    Also add corresponding labels.

    Args:
        sample (dict[str, Any]): dataset sample w/ <text-label> pair
        tokenizer (AutoTokenizer): model tokenizer
        padding (str): padding type
        max_length (int): text truncation bound.

    Returns:
        dict[str, Any]: transformed sample with tokenized text and labels.
    """
    text = sample["text"]
    # каждый сэмпл паддится до самой длинной посл-ти в этом батче (padding="max_length")
    # макс. длина посл-ти 512 (max_length=512),
    # все, что длиннее, обрезается (truncation=True)
    encoding = tokenizer(
        text,
        padding=padding,
        truncation=True,
        max_length=max_length,
    )
    encoding["labels"] = sample["labels"]
    return encoding


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """Calculate metrics used on validation step.

    Args:
        p (EvalPrediction): container with predictions and
        ground-truth labels

    Returns:
        dict[str, float]: dictionary with computed labels
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(p.label_ids, preds, average="macro")
    acc = accuracy_score(p.label_ids, preds)
    res = {"f1": f1, "accuracy": acc}
    return res


def make_dataset(
    frame: pl.DataFrame,
    tokenizer: AutoTokenizer,
    label2id: dict[str, int],
    prepr_conf: dict[str, Any],
    test_size: float = None,
) -> tuple[Dataset, Dataset]:
    """Create huggingface datasets used in training process.

    Args:
        frame (pl.DataFrame): input frame with text data
        tokenizer (AutoTokenizer): model tokenizer
        label2id (dict[str, int]): mapping from category text names
        to digital ids.
        prepr_conf (dict[str, Any]): config with preprocessing parameters
        test_size (float, optional): test split share. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: train & test splits, tokenized, vectorized and batched.
    """
    # переименуем столбцы для целостности с api hf-datasets
    clear_frame = frame.select(
        pl.col("Комментарий").alias("text"), pl.col("Категория").alias("labels")
    )

    # перейдем от строковых названий к численным меткам
    clear_frame = clear_frame.with_columns(pl.col("labels").map_dict(label2id))

    # каррированная функция с фиксированным токенизатором
    # для дальнейшего исп-я в Dataset.map()
    part_prepr = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        padding=prepr_conf["padding"],
        max_length=prepr_conf["max_length"],
    )

    train_df, test_df = train_test_split(
        clear_frame,
        test_size=test_size,
        random_state=42,
        stratify=clear_frame["labels"],
    )
    train_dataset = Dataset.from_pandas(train_df.to_pandas(), split="train")
    test_dataset = Dataset.from_pandas(test_df.to_pandas(), split="test")
    encoded_train = train_dataset.map(
        part_prepr, batched=True, remove_columns=train_dataset.column_names
    )
    encoded_test = test_dataset.map(
        part_prepr, batched=True, remove_columns=test_dataset.column_names
    )
    encoded_train.set_format("torch")
    encoded_test.set_format("torch")
    return encoded_train, encoded_test


def make_training_pipeline(
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model_conf: dict[str, Any],
    train_conf: dict[str, Any],
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> Trainer:
    """Training process wrapper.

    Args:
        tokenizer (AutoTokenizer): model tokenizer
        train_dataset (Dataset): train dataset split
        eval_dataset (Dataset): test dataset split
        model_conf (dict[str, Any]): config with model parameters
        train_conf (dict[str, Any]): config with params for training process
        id2label (dict[int, str]): numeric class label -> string representation map
        label2id (dict[str, int]): string class label -> numeric label map
    Returns:
        Trainer: hf training pipeline abstraction class.
    """

    args = instantiate(train_conf)

    model = instantiate(
        model_conf, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer


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


@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    t = preprocess_frame(frame, cfg["small_classes"])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["preprocessing"]["tokenizer_name"], token=cfg["secrets"]["hf_token"]
    )
    label2id = t["Категория"].unique().sort().to_list()
    label2id = dict(zip(label2id, range(len(label2id))))
    id2label = {v: k for k, v in label2id.items()}
    train_ds, test_ds = make_dataset(
        t, tokenizer, label2id, cfg["preprocessing"], cfg["test_size"]
    )
    trainer = make_training_pipeline(
        tokenizer, train_ds, test_ds, cfg["model"], cfg["training"], id2label, label2id
    )
    trainer.train()
    preds = trainer.predict(test_ds)
    pred_labels = np.apply_along_axis(predict, 1, preds[0])
    pred_labels = [id2label[x] for x in pred_labels]
    gt_labels = [id2label[x] for x in preds[1]]
    cr = classification_report(gt_labels, pred_labels, output_dict=True)
    cr = pd.DataFrame(cr).T
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(label2id.keys()))
    x = list(label2id.keys())
    y = list(reversed(label2id.keys()))
    fig = ff.create_annotated_heatmap(np.flipud(cm), x=x, y=y, colorscale="Viridis")
    fig.update_layout(title_text="Confusion matrix")
    fig.add_annotation(
        dict(
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    fig.add_annotation(
        dict(
            x=-0.16,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    fig["data"][0]["showscale"] = True

    log.info(f"Achieved f1-score (macro): {cr['f1-score']['macro avg']}")
    cr.to_csv(f"reports/hydra_run_example_{cfg['preprocessing']['max_length']}.csv")
    fig.write_html(
        f"reports/hydra_run_example_{cfg['preprocessing']['max_length']}.html"
    )


if __name__ == "__main__":
    main()