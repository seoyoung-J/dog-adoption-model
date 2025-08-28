# dog-adoption-model
**유기견 이미지 기반 예측 모델 개발**  

--- 

### 프로젝트 개요 
- **문제 정의** 
    - 보호소 운영 비효율성: 장기간 보호로 인한 공간 및 관리 비용 증가 
    - 낮은 유기견 입양률: 보호소에 머무는 기간이 길수록 입양 가능성 감소 
    - 입양 성공에 영향을 미치는 요인을 파악해 보호소 운영 효율화 및 입양률 향상 방안 마련 필요  
- **프로젝트 목표**
    - 유기견 이미지 데이터에서 시각적 속성(피처)을 추출하는 모델 개발 
    - 추출된 이미지 기반 피처와 메타 데이터를 결합하여 입양 성공 여부를 예측하는 모델 개발
- **기간/인원**: 2025.01.20 ~ 2025.02.07 / (4명)  
- **담당 역할**
    - 데이터 수집 및 라벨링(Label Studio)
    - 이미지 데이터 전처리
    - CNN 기반 딥러닝 모델 개발 (입모양 분류)
    - 머신러닝 모델 개발 및 튜닝 
- **사용 기술**
    - 데이터 수집/라벨링: Selenium, Label Studio 
    - 전처리/분석: Python, NumPy, Pandas, Matplotlib  
    - 머신러닝/딥러닝: scikit-learn, TensorFlow(Keras, ResNet50V2) 
    - 이미지 처리: OpenCV  
---  
### 데이터 수집 
- **수집 경로:** 동물보호관리시스템 
- **데이터 구성**: 
    - **딥러닝(CNN 학습용)**: 입양대기 500장, 입양완료 500장  
      → 입모양·귀모양·배경·악세서리 분류 모델 학습  
    - **머신러닝(분류기)**: 입양대기 100장, 입양완료 100장  
      → 이미지 피처 + 메타데이터(품종, 나이, 성별, 체중, 관할기관, 중성화 여부)  

---  

### 데이터 전처리 
- **1) 이미지 데이터 (입 모양)**: 
    - **라벨링**: Label Studio 활용 (Mouth Open=1, 그 외=0)  
    - **처리 과정**: UUUID 제거 → RGB 변환 → 224×224 리사이즈 → float32(0~1) 정규화  
    - **산출물**: 
        - X_mouth.npy(이미지 배열)
        - y_mouth.npy(라벨 배열),  
        - files_used.npy(파일명 리스트)   
    - **데이터 분할**: StratifiedShuffleSplit활용, seed 고정, 70/15/15 분할 (train/val/test) 
    - **클래스 불균형 보정**: Train set에 한해 1:1 오버샘플링 적용
    - **데이터 증강**: Keras augmentation layer 활용  
        - Horizontal Flip, Rotation(±5%), Random Zoom(±10%), Random Contrast(±10%)   

- **2) 메타 데이터 (CSV)**
    - **범주형 피처 (라벨 인코딩)**:  
        - 품종: 믹스견(0) / 외국견(1) / 한국견(2)  
        - 성별: 수컷(0) / 암컷(1) / 미상(2)  
        - 관할기관: 수도권(0), 전라권(1), 경상권(2), 충청권(3), 강원(4), 제주(5)  
        - 중성화 여부: 아니오(0), 예(1), 미상(2)  
    - **수치형 피처 (구간화 후 라벨 인코딩)**:  
        - 나이: 0–2세(0), 3–5세(1), 6세+(2)  
        - 체중: 0–4kg(0), 5–14kg(1), 15kg+(2)
    - **최종 피처**: 
    breed, gender, region, neuter_status, age, weight, color, background, ear_shape, accessory, mouth_shape, adoption_status  

--- 

### 모델링 
#### 1) CNN 기반 입 모양 피처 분류 모델 개발   
- **데이터**: 입 모양 라벨링 데이터 (총 1000건)  
- **실험 모델**: CNN Scratch / ResNet50V2 Transfer / ResNet50V2 Focal / ResNet50V2 + AdamW  
    - **평가**: Validation set에서 F1-score 최대화 기준으로 임계값(threshold) 선택  
    - **최종 선정 CNN 모델**: esNet50V2 + AdamW  
    - **성능 (test)**: PR-AUC 0.843, ROC-AUC 0.892, F1 0.765
- **선택 모델 구조** 
    - Backbone: ResNet50V2 (ImageNet pretrained, include_top=False, frozen)
    - Head: GAP → Dropout(0.4) → Dense(1, sigmoid)
    - Optimizer: AdamW (learning_rate=3e-4 → fine-tuning 단계 1e-5)
    - Loss: Binary Crossentropy
    - Metrics: Accuracy, ROC-AUC, PR-AUC

- **팀원 담당 모델**:
    - 악세서리: ResNet50V2 (Binary, 불균형 클래스 증강 적용)  
    - 배경: ResNet50V2 (다중 분류)  
    - 귀모양: ResNet50V2 (다중 분류)  

#### 2) 머신러닝 기반 입양 성공 분류 모델 개발  
- **데이터**: 이미지 기반 피처(수기 라벨링) + 메타데이터 (총 200건)
- **모델 후보**: RandomForest, XGBoost, LightGBM, KNN
- **베이스라인**: RandomForest (Accuracy 0.725, F1-Score 0.732)  
- **최종 선정 ML 모델**: RF + XGB Soft Voting (α=0.81, thr=0.53, recall ≥ 0.75 조건)
    - **성능 (Test)**: 
        - Accuracy 0.775 (+0.05)
        - F1-Score 0.769 (+0.037)
        - Recall 0.75 유지  
        - ROC-AUC: 0.778

---  

### 참고자료 
- 상세 분석 과정은 [Notion](https://www.notion.so/2089627a91c7804989aed3d5f4d39fda?v=8d37600a577e46f1bcec447df314b4a6&source=copy_link)에 정리되어 있습니다.  
