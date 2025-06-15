import re
from collections import defaultdict
from keybert import KeyBERT
from konlpy.tag import Okt
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification
import torch

# ✅ KoBERT 모델 로드 (로컬 파인튜닝 모델)
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load("./kobert_importance.pth", map_location="cpu"))
model.eval()

kw_model = KeyBERT(model="distiluse-base-multilingual-cased-v1")
okt = Okt()

# 불용어 사전 준비
stopwords_path = "stopwords-ko.txt"
with open(stopwords_path, encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())

date_patterns = [
    r"(\d{4})년 (\d{1,2})월 (\d{1,2})일",  # ex: 2025년 5월 13일
]

def extract_date_key(text):
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            y, m, d = match.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None

def is_valid_keyword(kw):
    return not re.search(r"(다|어|지|음)$", kw)

def extract_kakao_dialogues(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    data_by_date = defaultdict(list)
    for line in lines:
        date_key = extract_date_key(line)
        if date_key:
            current_date = date_key
        elif re.search(r"[오전|오후]+\s*\d{1,2}:\d{2},\s*[^:]+:", line):
            msg = re.sub(r"^\d{4}\. \d{1,2}\. \d{1,2}\. [오전|오후]+\s*\d{1,2}:\d{2},\s*[^:]+:\s*", "", line).strip()
            if len(msg) > 0:
                data_by_date[current_date].append(msg)
    return data_by_date

# def extract_sentences_from_kakao_txt(file_path):
#     with open(file_path, encoding="utf-8") as f:
#         lines = f.readlines()
#     dialogue_lines = [
#         re.sub(r"\d{4}\. \d{1,2}\. \d{1,2}\. [오전|오후]+\s*\d{1,2}:\d{2},\s*[^:]+:\s*", "", line).strip()
#         for line in lines
#         if re.search(r"\d{4}\. \d{1,2}\. \d{1,2}\.", line) is None and ":" in line
#     ]
#     return [line for line in dialogue_lines if len(line) > 1]

# ✅ 의미 있는 문장 필터링
def is_valid_conversation(msg):
    if not re.search(r"[가-힣]", msg):
        return False
    if re.search(r"https?://|총\s*금액", msg):
        return False
    return True

# # ✅ 불용어 제거 및 키워드 추출
# def extract_keywords(sentences):
#     keyword_scores = defaultdict(float)
#     for sentence in sentences:
#         okt_nouns = set(okt.nouns(sentence))
#         filtered_nouns = {n for n in okt_nouns if n not in stopwords and len(n) > 1}
#
#         keybert_keywords = kw_model.extract_keywords(
#             sentence, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=5
#         )
#
#         for kw, score in keybert_keywords:
#             tokens = kw.split()
#             if all(token in filtered_nouns for token in tokens) and is_valid_keyword(kw):
#                 multiplier = 2.5 if len(tokens) > 1 else 2.0
#                 keyword_scores[kw] += score * multiplier
#
#         for noun in filtered_nouns:
#             keyword_scores[noun] += 0.2
#
#     for parent_kw in list(keyword_scores.keys()):
#         for sub_kw in keyword_scores:
#             if sub_kw != parent_kw and sub_kw in parent_kw:
#                 keyword_scores[parent_kw] += 0.5 * keyword_scores[sub_kw]
#
#     return sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

def classify_interest(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return int(torch.argmax(probs, dim=1))  # 1 = 관심, 0 = 무관심

def extract_interest_weighted_keywords(sentences):
    keyword_scores = defaultdict(float)
    for sentence in sentences:
        label = classify_interest(sentence)  # 0 or 1
        nouns = {n for n in okt.nouns(sentence) if n not in stopwords and len(n) > 1}
        keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=5)

        if label == 1:
            print(f"\n💬 문장: {sentence}")
            print(f"🧠 관심도 분류: 관심 있음 (1)")

        for kw, score in keywords:
            tokens = kw.split()
            if all(token in nouns for token in tokens):
                multiplier = 2.5 if len(tokens) > 1 else 2.0
                final_score = score * (multiplier if label == 1 else 0.5)
                keyword_scores[kw] += final_score
                print(f"  🔑 키워드: {kw:15} | base: {score:.2f} → 적용 점수: {final_score:.2f}")

        for noun in nouns:
            add_score = 0.3 if label == 1 else 0.1
            keyword_scores[noun] += add_score
            print(f"  ➕ 명사 가중치: {noun:15} → {add_score:.2f}")

    filtered_keywords = [
        (kw, sc) for kw, sc in keyword_scores.items()
        if all(not re.search(r"(다|어|지|음|야)$", token) for token in kw.split())
    ]
    return sorted(filtered_keywords, key=lambda x: x[1], reverse=True)

file_path = "Talk_2025.5.13 16_38-1.txt"
data_by_date = extract_kakao_dialogues(file_path)

print(f"✅ 날짜 블록 수: {len(data_by_date)}\n")

for date, messages in sorted(data_by_date.items()):
    filtered_msgs = [msg for msg in messages if is_valid_conversation(msg)]
    print(f"▶ {date} / 대화 수: {len(messages)}")
    for m in messages:
        if not is_valid_conversation(m):
            print(f"❌ 제외된 문장: {m}")
    print(f"  🔍 유효 대화 수: {len(filtered_msgs)}")

    if len(filtered_msgs) == 0:
        print("  ⚠️ 건너뜀: 의미 있는 대화 없음\n")
        continue

    keywords = extract_interest_weighted_keywords(filtered_msgs)
    print(f"\n📅 {date} 키워드:")
    for kw, score in keywords[:10]:
        print(f"- {kw}: {score:.2f}")
    print()
