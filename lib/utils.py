import ast
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from google.colab import files
from torch.nn.modules.loss import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
)

# token_subgroup_single
model_type = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type)
num_labels = 3
label_format = "maj_multi"


def id_array_to_labels(id_array):
    """Converts an array of indices to a bit array
    e.g. the array [0, 2, 3, 4, 5] is converted to [1. 0. 1. 1. 1. 1. 0. 0. 0.]"""
    labels = np.zeros(9)
    labels[id_array] = 1
    return labels.astype(float)


def tokenize_and_align_labels(example, single_label=True):

    # Tokenize the sentence pair
    tokenized_inputs = tokenizer(
        example["sentence1_tokenized"],
        example["sentence2_tokenized"],
        padding="max_length",
        max_length=90,
        truncation=True,
        is_split_into_words=True,
    )

    label_array_1 = example["s1_token_labs"]  # Label array for the first sentence
    label_array_2 = example["s2_token_labs"]  # Label array for the second sentence
    word_ids = tokenized_inputs.word_ids(batch_index=0)

    label_ids = []
    sentence_switch = False  # Flag to indicate when to switch from the first to the second sentence's labels
    previous_word_id = None

    if single_label:
        pad_value = -100
    else:
        pad_value = [-100.0] * num_labels

    for index, word_id in enumerate(word_ids):
        if word_id is None and not sentence_switch:
            # First [CLS] or [SEP] token encountered
            label_ids.append(pad_value)
            if index > 0:
                # First [SEP] token encountered
                sentence_switch = True  # Switch to the second sentence's labels
        elif word_id is None:
            # Second [SEP] token or [CLS] token at the end
            label_ids.append(pad_value)
        else:
            # Normal token, choose appropriate label array
            current_label_array = label_array_2 if sentence_switch else label_array_1
            label_ids.append(
                current_label_array[word_id]
                if single_label
                else current_label_array[word_id].tolist()
            )

        previous_word_id = word_id

    tokenized_inputs["labels"] = label_ids

    return tokenized_inputs


def apply_tokenization(train_df, test_df, val_df, single_label=True):
    "Tokenize sentences and save as new column in dfs"

    train_df["tokenized_sentences"] = train_df.apply(
        tokenize_and_align_labels, single_label=single_label, axis=1
    )
    test_df["tokenized_sentences"] = test_df.apply(
        tokenize_and_align_labels, single_label=single_label, axis=1
    )
    val_df["tokenized_sentences"] = val_df.apply(
        tokenize_and_align_labels, single_label=single_label, axis=1
    )

    # Convert tokenized sentences to tensors. Those will be the inputs to our (PyTorch) model
    train_df["inputs"] = train_df["tokenized_sentences"].apply(
        lambda x: x.convert_to_tensors("pt")
    )
    test_df["inputs"] = test_df["tokenized_sentences"].apply(
        lambda x: x.convert_to_tensors("pt")
    )
    val_df["inputs"] = val_df["tokenized_sentences"].apply(
        lambda x: x.convert_to_tensors("pt")
    )


def compute_metrics(p: EvalPrediction):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    y_pred = [
        p
        for prediction, label in zip(predictions, labels)
        for p, l in zip(prediction, label)
        if l != -100
    ]
    y_true = [l for label in labels for l in label if l != -100]

    non_zero_labels = list(range(1, num_labels))

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    accuracy = f1_score(y_true, y_pred, average="micro", labels=non_zero_labels)
    inflated_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
    results = {
        "f1": f1_micro_average,
        "accuracy": accuracy,
        "0accuracy": inflated_accuracy,
        "precision": precision,
        "recall": recall,
    }
    return results


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 8),
        "warmup_steps": trial.suggest_int("warmup_steps", 10, 300),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.05, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32, 64, 128]
        ),
        "seed": trial.set_user_attr("seed", 3),
    }


def model_init():
    return BertForTokenClassification.from_pretrained(model_type, num_labels=num_labels)


def get_accuracy(input):
    return input["eval_accuracy"]


# Defining evaluation metrics
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def test_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    y_pred = [
        p
        for prediction, label in zip(predictions, labels)
        for p, l in zip(prediction, label)
        if l != -100
    ]
    y_true = [l for label in labels for l in label if l != -100]
    labs = list(range(1, num_labels))
    overall_f1_macro = f1_score(
        y_true=y_true, y_pred=y_pred, average="macro", labels=labs
    ).tolist()
    overall_f1_micro = f1_score(
        y_true=y_true, y_pred=y_pred, average="micro", labels=labs
    ).tolist()
    f1_micro_average = f1_score(
        y_true=y_true, y_pred=y_pred, average=None, labels=labs
    ).tolist()
    precision_overall = precision_score(
        y_true=y_true, y_pred=y_pred, average="micro", labels=labs
    )
    recall_overall = recall_score(
        y_true=y_true, y_pred=y_pred, average="micro", labels=labs
    )

    # accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true=y_true, y_pred=y_pred, average=None, labels=labs
    ).tolist()
    recall = recall_score(
        y_true=y_true, y_pred=y_pred, average=None, labels=labs
    ).tolist()

    results = {
        "F1": f1_micro_average,
        "Overal F1 Macro": overall_f1_macro,
        "Overall Accuracy": overall_f1_micro,
        #'accuracy': accuracy,
        "Precision": precision,
        "Recall": recall,
        "Precision Overall": precision_overall,
        "Recall Overall": recall_overall,
    }

    return results


def multilabel_test_metrics(predictions, labels, thresholds=[0.5] * num_labels):
    thresholds = torch.Tensor(thresholds)
    # First, apply sigmoid on predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # Flatten probs and labels
    # Originally of dims [batch_size, sequence_length, num_labels] to [batch_size * sequence_length, num_labels]
    flat_probs = probs.view(-1, probs.shape[-1])
    flat_labels = labels.reshape(-1, labels.shape[-1])

    # Filter rows where all labels are -100
    mask = ~(flat_labels == -100).all(axis=1)
    filtered_probs = flat_probs[mask]
    filtered_labels = flat_labels[mask]

    # Generate predictions using threshold
    y_pred = np.zeros(filtered_probs.shape)
    y_pred[np.where(filtered_probs > thresholds)] = 1

    y_true = filtered_labels

    # Compute overall metrics
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    precision_overall = precision_score(y_true=y_true, y_pred=y_pred, average="micro")
    recall_overall = recall_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    hamming = hamming_loss(y_true, y_pred)

    # Compute class-wise Precision, Recall, F1 Score
    precision_classwise = precision_score(y_true, y_pred, average=None).tolist()
    recall_classwise = recall_score(y_true, y_pred, average=None).tolist()
    f1_classwise = f1_score(y_true, y_pred, average=None).tolist()

    # Samples
    f1_samples = f1_score(y_true, y_pred, average="samples").tolist()
    accuracy = accuracy_score(y_true, y_pred)

    # Return metrics in a dictionary
    metrics = {
        "f1": f1_micro_average,
        "roc_auc": roc_auc,
        "hamming_loss": hamming,
        "precision_per_class": precision_classwise,
        "recall_per_class": recall_classwise,
        "f1_per_class": f1_classwise,
        "f1_samples": f1_samples,
        "accuracy": accuracy,
        "precision_overall": precision_overall,
        "recall_overall": recall_overall,
    }
    return metrics


def subset_labels(df, label_format):
    "Returns a subset of df containing only labels according to label_format"
    assert label_format in [
        "sub_single",
        "maj_single",
        "sub_multi",
        "maj_multi",
    ], "Invalid label_format"
    new_df = df[
        [
            "sentence1",
            "sentence2",
            "sentence1_tokenized",
            "sentence2_tokenized",
            "collapsed_labels",
            f"s1_token_labs_{label_format}",
            f"s2_token_labs_{label_format}",
        ]
    ]
    new_df.rename(
        columns={
            f"s1_token_labs_{label_format}": "s1_token_labs",
            f"s2_token_labs_{label_format}": "s2_token_labs",
        },
        inplace=True,
    )
    return new_df


def show_test_result(trainer, test_df):
    test_result = trainer.predict(test["inputs"].values)

    # Print default metrics collected during prediction
    for item, value in test_result.metrics.items():
        print(f"{item}: {value}")

    predictions = torch.Tensor(test_result.predictions)
    labels = torch.Tensor(test_result.label_ids)

    # Compute class-wise metrics
    # thresholds = [0.15, 0.5, 0.5, 0.5, 0.5, 0.19, 0.5, 0.5]
    results = test_metrics((predictions, labels))
    df = pd.DataFrame.from_dict(results)
    # df.drop(columns = ['f1', 'roc_auc', 'hamming_loss'], inplace=True)
    if num_labels == 4:
        indices = [["1. Addition/Deletion", "2. Change of Order", "3. Substitution"]]
    else:
        indices = [
            [
                "1. Add/Del - Function Word",
                "2. Add/Del - Content Word",
                "3. Change of Order",
                "4. Substitution - Synonym",
                "5. Substitution - Contextual Synonym",
                "6. Substitution - Morphological",
                "7. Substitution - Spelling and Format",
                "8. Add/Del - Punctuation",
            ]
        ]
    df.index = indices
    return df


def show_multilabel_test_result(trainer, test_df):
    test_result = trainer.predict(test["inputs"].values)

    # Print default metrics collected during prediction
    for item, value in test_result.metrics.items():
        print(f"{item}: {value}")

    predictions = torch.Tensor(test_result.predictions)
    labels = torch.Tensor(test_result.label_ids)

    # Compute class-wise metrics
    # thresholds = [0.15, 0.5, 0.5, 0.5, 0.5, 0.19, 0.5, 0.5]
    results = multilabel_test_metrics(predictions, labels)
    df = pd.DataFrame.from_dict(results)
    df.drop(columns=["f1", "roc_auc", "hamming_loss"], inplace=True)
    if num_labels == 3:
        indices = [["1. Addition/Deletion", "2. Change of Order", "3. Substitution"]]
    else:
        indices = [
            [
                "1. Add/Del - Function Word",
                "2. Add/Del - Content Word",
                "3. Change of Order",
                "4. Substitution - Synonym",
                "5. Substitution - Contextual Synonym",
                "6. Substitution - Morphological",
                "7. Substitution - Spelling and Format",
                "8. Add/Del - Punctuation",
            ]
        ]
    df.index = indices
    return df


class MultiLabelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            # logging.info(f"Using multi-label classification with class weights", class_weights)
        self.loss_fct = BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # this simultaneously accesses predictions for tokens that aren't CLS or PAD
        # and flattens the logits or labels
        flat_outputs = outputs.logits[labels != -100]
        flat_labels = labels[labels != -100]

        try:
            loss = self.loss_fct(flat_outputs, flat_labels.float())
        except AttributeError:  # DataParallel
            loss = self.loss_fct(flat_outputs, flat_labels.float())

        return (loss, outputs) if return_outputs else loss


def multilabel_metrics(predictions, labels, threshold=0.5):
    # First, apply sigmoid on predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # Flatten probs and labels
    # Originally of dims [batch_size, sequence_length, num_labels] to [batch_size * sequence_length, num_labels]
    flat_probs = probs.view(-1, probs.shape[-1])
    flat_labels = labels.reshape(-1, labels.shape[-1])

    # Filter rows where all labels are -100
    mask = ~(flat_labels == -100).all(axis=1)
    filtered_probs = flat_probs[mask]
    filtered_labels = flat_labels[mask]

    # Generate predictions using threshold
    y_pred = np.zeros(filtered_probs.shape)
    y_pred[np.where(filtered_probs > threshold)] = 1

    # Now we can compute metrics:
    y_true = filtered_labels
    # print(y_true)
    # print(y_pred)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    hamming = hamming_loss(y_true, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "hamming loss": hamming}
    return metrics


def compute_multilabel_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multilabel_metrics(predictions=preds, labels=p.label_ids)
    return result
