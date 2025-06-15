import json
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from kobert_tokenizer import KoBERTTokenizer
from transformers import EarlyStoppingCallback
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer

def load_multiple_jsonl(paths):
    combined = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            combined.extend(json.loads(line) for line in f)
    return Dataset.from_list(combined)

data_paths = [
    "data/preference_labeled_1400.jsonl",
    "data/implicit_preference_balanced_1000.jsonl"
]

# 데이터 로드 및 분할
dataset = load_multiple_jsonl(data_paths)
dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 분할 데이터 저장 (선택)
train_dataset.to_json("train_split.jsonl", orient="records", lines=True, force_ascii=False)
test_dataset.to_json("test_split.jsonl", orient="records", lines=True, force_ascii=False)

# 모델 및 토크나이저 설정
model_name = "skt/kobert-base-v1"
config = BertConfig.from_pretrained(
    model_name,
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)
model = BertForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = KoBERTTokenizer.from_pretrained(model_name)

# 전처리 함수
def tokenize(example):
    encoded = tokenizer.encode_plus(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded.get("token_type_ids", [0] * 128),
        "label": example["label"]
    }

# 데이터 토크나이징
tokenized_train = train_dataset.map(tokenize, batched=False)
tokenized_test = test_dataset.map(tokenize, batched=False)

# 평가 지표
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# 학습 설정
training_args = TrainingArguments(
    output_dir="./kobert-importance",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.1,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    disable_tqdm=False
)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 학습 실행
trainer.train()

# 모델 저장
model.save_pretrained("./kobert-importance")
tokenizer.save_pretrained("./kobert-importance")
torch.save(model.state_dict(), "./kobert_importance.pth")
