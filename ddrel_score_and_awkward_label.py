import json
import os
import pandas as pd

folder_path = r"C:\Users\user\Desktop\project\chat-gift-recommender\data\refined_ddrel"
output_csv = os.path.join(folder_path, "ddrel_data_label_new.csv")

texts = []
labels = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    context = obj.get("context") or obj.get("컨텍스트")
                    label = obj.get("label")
                    if context and label is not None:
                        text = " ".join(context)
                        texts.append(text)
                        labels.append(int(label))
                except json.JSONDecodeError:
                    continue  # 깨진 라인 무시

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

# label → score 매핑
label_to_score = {
    1: 5.0,
    3: 4.5,
    4: 3.9,
    5: 4.8,
    6: 4.0,
    7: 4.2,
    8: 3.0,
    10: 2.5,
    11: 1.5,
    13: 0.5
}

df["score"] = df["label"].map(label_to_score)

# 어색함 라벨: score ≤ 3.0 → 1
df["awkward"] = df["score"].apply(lambda s: 1 if s <= 3.0 else 0)

df = df[["text", "label", "score", "awkward"]]

df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"label 저장 완료: {output_csv}")

