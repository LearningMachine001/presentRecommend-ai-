# presentRecommend-ai

| 학습 데이터 | 이용 모델 | 목적 | 파인 튜닝 후 최종 모델 |
|-------------|-----------|------------------------------------------------|------------------------------|
| DDREL 영화 대사 데이터 번역본 | KoBERT | 관계 예측 및 관계마다 친밀도 점수 매핑 | `kobert_extended_with_subject.pth` |
| AI Hub 대화 데이터 | KoBERT | 대화의 주제를 파악하여 카카오톡 선물하기 CSV의 카테고리와 매핑 | `kobert_extended_with_subject.pth` |
| 명시적/암묵적 선호도 데이터 (GPT로 생성) | KoBERT | 대화의 선호도가 파악되면 해당 문장의 키워드에 높은 가중치 부여 | `kobert_importance.pth` |


### 용량이 커서 업로드하지 못한 파일
- 임베딩 벡터로 변환한 카카오톡 선물 item 파일 (.pt)
- 모델 파일
  - kobert_extended_with_subject.pth 
  - kobert_importance.pth
 
### 작동 예시
![image](https://github.com/user-attachments/assets/d25a481a-cdfc-4334-88eb-457d5288a167)
![image](https://github.com/user-attachments/assets/21b82b33-045e-4066-a6dc-260d2544fc22)
