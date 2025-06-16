import os, pandas as pd, torch
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

input_dir = "category_files"
output_dir = "cached_embeddings_2"
os.makedirs(output_dir, exist_ok=True)

def read_csv_flexible(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

for filename in os.listdir(input_dir):
    if not filename.endswith(".csv"):
        continue

    safe_key = os.path.splitext(filename)[0]
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

    product_list = []
    for _, row in df.iterrows():
        if pd.isna(row.get("keywords")) or pd.isna(row.get("상품명")):
            continue

        product = {
            "name": row["상품명"],
            "keywords": row["keywords"],
            "price": row.get("가격", ""),
            "category": row.get("대분류", ""),
            "brand": row.get("브랜드", ""),
            "image_url": row.get("image_url", ""),
            "product_url": row.get("product_url", "")
        }
        product["embedding"] = embedding_model.encode(product["keywords"], convert_to_tensor=True)
        product_list.append(product)

    torch.save(product_list, pt_path)
    print(f"[✓] Saved {len(product_list):>5} items → {pt_path}")
