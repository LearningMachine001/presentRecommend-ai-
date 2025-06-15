import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer

# 1. 모델 정의
class KoBertExtendedModel(nn.Module):
    def __init__(self, model_name="skt/kobert-base-v1", num_subjects=20):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.score_head = nn.Linear(768, 1)
        self.awkward_head = nn.Linear(768, 2)
        self.subject_head = nn.Linear(768, num_subjects)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        score = self.score_head(pooled_output)
        awkward = self.awkward_head(pooled_output)
        subject = self.subject_head(pooled_output)
        return score, awkward, subject

# 2. subject 인덱스 → 이름 매핑
subject_id2name = {
    0: "미용", 1: "스포츠/레저", 2: "교육", 3: "가족", 5: "영화/만화",
    6: "교통", 7: "여행", 8: "회사/아르바이트", 9: "건강", 10: "연애/결혼",
    11: "게임", 12: "계절/날씨", 13: "방송/연예", 14: "사회이슈",
    15: "주거와 생활", 16: "반려동물", 17: "군대", 18: "식음료"
}

# 3. subject_id → 대분류 매핑 (11개)
subject_to_main_category = {
    0: "뷰티", 1: "레저/스포츠", 2: "리빙/도서", 3: "디지털/가전", 5: "패션",
    6: "디지털/가전", 7: "레저/스포츠", 8: "리빙/도서", 9: "식품", 10: "패션",
    11: "디지털/가전", 12: "식품", 13: "패션", 14: "리빙/도서",
    15: "리빙/도서", 16: "유아동/반려", 17: "식품", 18: "식품"
}

# 4. 예측 함수
def predict_all(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        score, awkward_logits, subject_logits = model(input_ids, attention_mask)
        score_value = torch.sigmoid(score).item() * 5
        awkward_class = torch.argmax(awkward_logits, dim=1).item()
        subject_id = torch.argmax(subject_logits, dim=1).item()
        subject_name = subject_id2name.get(subject_id, "알 수 없음")
        main_category = subject_to_main_category.get(subject_id, "없음")

    return round(score_value, 2), awkward_class, subject_id, subject_name, main_category

# 5. 실행
if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)

    model = KoBertExtendedModel()
    model.load_state_dict(torch.load("../kobert_extended_with_subject.pth", map_location=device), strict=False)
    model.to(device)
    model.eval()

    while True:
        user_input = input("\n문장을 입력하세요 (예: A: 뭐 갖고 싶어? [SEP] B: 블루투스 이어폰!):\n종료하려면 엔터만 입력하세요: ").strip()
        if not user_input:
            print("종료합니다.")
            break

        score, awkward, subject_id, subject_name, main_category = predict_all(user_input, model, tokenizer, device)

        print(f"\n예측된 친밀도 점수 (0~5): {score}")
        print(f"어색함 여부 (0: 자연스러움, 1: 어색함): {awkward}")
        print(f"예측된 주제: {subject_name} (ID: {subject_id})")
        print(f"추천 대분류 카테고리: {main_category}")



# 출력 예시 (기억용)
#
# 문장을 입력하세요 (예: A: 뭐 갖고 싶어? [SEP] B: 블루투스 이어폰!):
# 종료하려면 엔터만 입력하세요: A: 아빠 요즘 골프 자주 나가시더라. [SEP] B: 진짜 완전 푹 빠지셨어.
#
# 예측된 친밀도 점수 (0~5): 2.47
# 어색함 여부 (0: 자연스러움, 1: 어색함): 0
# 예측된 주제: 가족 (ID: 3)
# 추천 대분류 카테고리: 디지털/가전
