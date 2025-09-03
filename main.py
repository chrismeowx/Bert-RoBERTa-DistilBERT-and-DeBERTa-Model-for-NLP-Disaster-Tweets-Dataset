import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from scipy.special import softmax
import re
from google.colab import drive

drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/train.csv')

data

data = data.drop(["id", "keyword", "location"], axis=1, inplace=False)
data = data.dropna()

print(data["target"].value_counts())

# preprocess samain

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

data["text"] = data["text"].apply(preprocess)

def tokenize(texts, labels, model, max_len=128):
  tokenizer = AutoTokenizer.from_pretrained(model)

  encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
  encodings["labels"] = labels
  dataset = Dataset.from_dict(encodings)
  return dataset

data["target"] = data["target"].astype(int)

x = list(data["text"])
y = list(data["target"])

print(len(x))
print(len(y))

def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

def train(model_name, train_dataset, test_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_strategy="epoch",
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    return trainer

def compare():
    model = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "deberta": "microsoft/deberta-base"
    }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    for model_label, model_name in model.items():
        print(f"\nTraining with {model_label.upper()}...")

        train_dataset = tokenize(x_train, y_train, model_name)
        test_dataset = tokenize(x_test, y_test, model_name)

        trainer = train(model_name, train_dataset, test_dataset)

        results = trainer.evaluate()
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        report = classification_report(y_true, y_pred, output_dict=True)

        print(report)

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_true, predictions.predictions[:, 1])
        auc = roc_auc_score(y_true, predictions.predictions[:, 1])

        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend()
        plt.show()

    return results

results = compare()