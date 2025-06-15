import os, pandas as pd, torch
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

input_dir = "category_files"
output_dir = "cached_embeddings"
os.makedirs(output_dir, exist_ok=True)

def read_csv_flexible(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

# category_files 내 모든 .csv 파일 처리
for filename in os.listdir(input_dir):
    if not filename.endswith(".csv"):
        continue

    safe_key = os.path.splitext(filename)[0]  # 예: "beauty_3"
    pt_path = os.path.join(output_dir, f"{safe_key}.pt")
    if os.path.exists(pt_path):
        print(f"[▶] Skipping {safe_key}, already cached.")
        continue

    csv_path = os.path.join(input_dir, filename)
    try:
        df = read_csv_flexible(csv_path)
    except Exception as e:
        print(f"[!] Failed to read {csv_path} → {e}")
        continue

    product_list = [
        {"name": row["상품명"], "keywords": row["keywords"], "category": row["대분류"]}
        for _, row in df.iterrows()
        if pd.notna(row["keywords"]) and pd.notna(row["상품명"])
    ]
    for p in product_list:
        p["embedding"] = embedding_model.encode(p["keywords"], convert_to_tensor=True)

    torch.save(product_list, pt_path)
    print(f"[✓] Saved {len(product_list):>5} items → {pt_path}")
