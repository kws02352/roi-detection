# 🔍 카드/신분증 ROI Detection 모델

> 모바일 환경에서 신분증/카드를 자동 인식하는 ROI Detection 모델 설계·개발·상용화

---

## 📌 프로젝트 개요

모바일 환경에서 신분증/카드를 자동 인식하는 ROI Detection 모델을 단독으로 설계·개발·상용화한 프로젝트.  
초기 Computer Vision 기반 방식에서 출발해 딥러닝 기반 아키텍처로 전환하고, 이후 클래스 확장과 지속적인 성능 고도화까지 담당했다.

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

---

## 🏗 Architecture

### 1-Stage Detection 구조

**Input Image → Backbone (ViT 기반) → 3채널 출력**

- 0채널 (박스 마스크): erode/dilate → 4개 꼭지점 좌표 추출
- 1채널 (타이틀 영역): 좌상단 판별 → 회전 정보 획득
- 2채널 (꼭지점 영역): 가장 밝은 점 → 정밀 4점 추출

### 기존 CV 방식 대비 개선점

| 항목 | 기존 (OpenCV) | 개선 (딥러닝) |
|---|---|---|
| 회전 정보 | 불가 → 0/90/180/270도 4회 추론 | 단일 추론으로 회전 정보 동시 획득 |
| 방향 판별 | 별도 처리 필요 | 타이틀 채널로 통합 처리 |
| 추론 속도 | 기준 | 20% 향상 |

---

## 📊 성과

- **정확도 98.9%** 달성, 금융권 주요 고객사 도입 확대
- **ROI Score 0.9635** 달성 (클래스 8→11 확장 후 기존 최고 성능 초과)
- **추론 속도 20% 향상**, 모바일 배포 제약 조건 충족 (ONNX 100ms 이하 / ~1,100KB)

---

## 🔬 핵심 개선 사항

### 데이터 분포 분석 기반 Augmentation 재설계
16만여 장 전수 조사로 Train-Eval 간 분포 불일치 발견 및 해소

| 지표 | 문제 |
|---|---|
| Contrast | Train 대비가 Eval 대비 과도하게 낮음 |
| Saturation | Train 채도가 Eval 대비 2배 낮음 |
| Blur | Train이 Eval보다 2.6배 선명 |
| Area Ratio | Train 카드 크기가 지나치게 작음 |

### Loss 구조 전면 재설계
기존 MSE + Cross Entropy → OHEM + Focal Weighting + Dice + IoU + DT Boundary Loss + Loss Weight Scheduler