# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class KoBertMultiTaskModel(nn.Module):
    def __init__(self, model_name="skt/kobert-base-v1"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        # self.regression_head = nn.Linear(hidden_size, 1)
        # self.regression_head = nn.Sequential(
        #     nn.Linear(hidden_size, 1),
        #     nn.Sigmoid()
        # )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.classification_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]
        # score_pred = self.regression_head(pooled_output).squeeze(-1)  # [batch_size]
        score_pred = self.regression_head(pooled_output).squeeze(-1) * 5  # 0~1 → 0~5
        awkward_pred = self.classification_head(pooled_output)        # [batch_size, 2]
        return score_pred, awkward_pred
