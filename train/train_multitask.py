import torch
from torch.utils.data import DataLoader
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd

from dataset import MultiTaskDDRelDataset
from model import KoBertMultiTaskModel

csv_path = "ddrel_data_label_train.csv"
model_name = "skt/kobert-base-v1"
batch_size = 16
epochs = 10
lr = 2e-5

full_df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(full_df, test_size=0.1, random_state=42)

# KoBERT 전용 토크나이저 사용
tokenizer = KoBERTTokenizer.from_pretrained(model_name)

train_dataset = MultiTaskDDRelDataset(train_df, tokenizer)
val_dataset = MultiTaskDDRelDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = KoBertMultiTaskModel(model_name)
model.cuda()

optimizer = AdamW(model.parameters(), lr=lr)
reg_loss_fn = torch.nn.MSELoss()
cls_loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        score = batch["score"].cuda()
        awkward = batch["awkward"].cuda()

        pred_score, pred_awkward = model(input_ids, mask)

        reg_loss = reg_loss_fn(pred_score, score)
        cls_loss = cls_loss_fn(pred_awkward, awkward)
        loss = reg_loss + 0.5 * cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"Step {step} | Batch Loss: {loss.item():.4f}")
            print("→ Predicted scores:", pred_score[:5].detach().cpu().numpy())
            print("→ True scores:", score[:5].detach().cpu().numpy())
            print("→ Awkward logits:", pred_awkward[:5].detach().cpu().numpy())
            print("→ True awkward:", awkward[:5].detach().cpu().numpy())
            print("------")

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "../kobert_multitask_trained.pth")
print("모델 저장 완료: kobert_multitask_trained.pth")