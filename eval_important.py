import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

# 1. 모델과 토크나이저 로드
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load("./kobert_importance.pth", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 평가 데이터 로드 및 전처리
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return Dataset.from_list([json.loads(line) for line in f])

# HuggingFace Dataset 로드 방식
test_dataset = load_dataset("json", data_files="test_split.jsonl", split="train")

# 토크나이즈 함수
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

tokenized_test = test_dataset.map(tokenize, batched=False)

# 배치 구성 함수
def collate_fn(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
        "token_type_ids": torch.tensor([x["token_type_ids"] for x in batch], dtype=torch.long),
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long)
    }

test_loader = DataLoader(tokenized_test, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 3. 평가 루프
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"]
        )
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# 4. 성능 지표 및 혼동 행렬 출력
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="binary")
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Important", "Important"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
