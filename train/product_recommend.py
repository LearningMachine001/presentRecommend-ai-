from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# ✅ 모델 로드 (한국어 전용 문장 임베딩)
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# ✅ CSV에서 상품 불러오기
df = pd.read_csv("beauty.csv")  # C열: 상품명, H열: keywords
product_list = [
    {"name": row["상품명"], "keywords": row["keywords"]}
    for _, row in df.iterrows()
    if pd.notna(row["keywords"]) and pd.notna(row["상품명"])
]

# ✅ 각 상품의 키워드 문장을 임베딩
product_embeddings = [
    {
        "name": p["상품명"],
        "embedding": embedding_model.encode(p["keywords"], convert_to_tensor=True)
    }
    for p in product_list
]

# ✅ 키워드 리스트를 받아 의미 유사도 기반 추천 수행
def recommend_products_from_keywords(keywords):
    query = " ".join(keywords[:5])  # 상위 키워드 최대 5개 사용
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = [
        (prod["상품명"], util.cos_sim(query_embedding, prod["embedding"]).item())
        for prod in product_embeddings
    ]
    return sorted(scores, key=lambda x: x[1], reverse=True)

# ✅ 테스트 예시
if __name__ == "__main__":
    test_keywords = ["에스쁘아", "틴트","코랄"]
    recommendations = recommend_products_from_keywords(test_keywords)

    print("🎁 추천 선물 TOP 3:")
    for name, score in recommendations[:10]:
        print(f"- {name} ({score:.2f})")
