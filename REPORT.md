## 각 단계별 문제점 및 해결방법
### 1. Dataset
- 요구사항 : Flickr8k 데이터셋을 사용하여 `[image, text]` 형태의 학습 데이터를 구성해야 한다.
- 데이터 구성
    - Flickr8k는 한 이미지에 여러 개의 caption이 연결된 1:N 구조를 가진다.
    - 따라서 이미지 단위가 아니라 `(image, caption)` 쌍을 하나의 학습 샘플로 사용하도록 구성하였다.
    - train / validation / test split 파일을 별도로 읽어 기본 분할을 유지하도록 설계하였다.

### 2. Embedding Align
- 요구사항 : CLIP의 이미지 embedding을 LLM embedding 공간으로 정렬하여, 이미지 정보와 텍스트 정보가 같은 표현 공간에서 비교될 수 있어야 한다.
- 구현
    - CLIP은 `apple/MobileCLIP-B`, LLM은 `HuggingFaceTB/SmolLM2-135M-Instruct`를 사용하였다.
    - CLIP image encoder의 출력 차원과 LLM hidden size가 다르기 때문에, 두 공간을 연결하는 `ClipToLlmProjector`를 추가하였다.
    - Projector는 `Linear` &rightarrow; `GELU` &rightarrow; `Linear` 구조의 MLP로 구현하였고, hidden size는 `llm_dim * multi_rate`로 설정하였다.
    - 이미지는 `image` &rightarrow; `CLIP` &rightarrow; `projector` &rightarrow; `aligned embedding` 흐름으로 인코딩하고, 텍스트는 tokenizer와 LLM input embedding layer를 이용해 문장 임베딩으로 변환하였다.
- 학습
    - alignment 단계에서는 CLIP과 LLM 본체를 모두 freeze하고 projector만 학습하였다.
    - 텍스트 임베딩은 attention mask를 이용한 mean pooling으로 문장 단위 표현으로 만들었다.
    - 이미지 임베딩, 텍스트 임베딩을 정규화한 뒤 배치 내 정답 쌍이 서로 가장 가까워지도록 contrastive loss를 적용하였다.
    - image-to-text, text-to-image 양방향 cross entropy를 평균내는 방식으로 학습하여 두 표현 공간이 함께 정렬되도록 하였다.

### 3. Caption Train
- 요구사항 : Alignment된 모델을 기반으로 이미지 입력만으로 자연어 caption을 생성할 수 있어야 한다.
- 구현
    - caption 학습 단계에서는 alignment 단계에서 저장한 projector 가중치를 불러와 사용하였다.
    - 전체 흐름은 `image` &rightarrow; `CLIP` &rightarrow; `projector` &rightarrow; `LLM` &rightarrow; `text` 구조로 구성하였다.
    - CLIP이 추출한 이미지 임베딩을 projector를 통해 LLM 입력 공간으로 변환한 뒤, 이를 하나의 prefix token처럼 caption 토큰 앞에 붙여서 `inputs_embeds`를 생성하였다.
    - attention mask와 label도 같은 방식으로 확장하여, 이미지 토큰 위치는 loss 계산에서 제외하고 텍스트 토큰만 예측하도록 구성하였다.
    - 학습 데이터는 동일하게 Flickr8k의 `(image, caption)` 쌍을 사용하였다.
- 학습
    - caption 단계에서는 CLIP은 freeze하고, projector와 LLM은 함께 학습하였다.
    - 즉 alignment 단계에서 맞춘 시각 임베딩을 기반으로, 실제 문장 생성 성능이 좋아지도록 LLM을 추가로 fine-tuning하였다.
    - 학습 목표는 이미지 prefix가 주어진 상태에서 정답 caption의 다음 토큰을 예측하는 loss이다.

### 4. Evaluation
- 요구사항 : Caption 모델을 적절한 지표로 평가
- 구현
    - Flickr8k의 test split을 사용하여 생성된 caption과 정답 caption들을 비교하도록 구성하였다.
    - 평가 시 각 테스트 이미지에 대해 모델이 caption을 생성하고, 해당 이미지에 연결된 여러 개의 reference caption과 비교하였다.
    - 단일 이미지에 대한 caption 생성 기능과, 전체 test set에 대한 일괄 평가 기능을 모두 구현하였다.
    - 결과는 `eval_results.json` 파일로 저장되며, 이미지별 예측 결과와 전체 평가 지표를 함께 기록하도록 하였다.
- 평가 지표
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4를 계산하여 생성 문장이 reference 문장과 얼마나 유사한 n-gram 구성을 가지는지 측정하였다.
    - BLEU에서는 reference 문장이 여러 개일 경우, 각 n-gram에 대해 reference들에서의 최대 등장 횟수를 기준으로 hypothesis의 n-gram 개수를 clipping하여 precision을 계산한다.
    - ROUGE-L을 함께 계산하여, 예측 문장과 정답 문장 사이의 longest common subsequence 기반 유사도를 평가하였다.
    - ROUGE-L의 경우 reference 문장이 여러 개일 때 각 reference와의 점수 중 최대값을 해당 샘플의 점수로 채택하였다.
    - 한 이미지에 여러 reference caption이 존재하므로, reference 집합 전체를 기준으로 점수를 계산하도록 구현하였다.
- 평가 방법
    - 학습이 완료된 `projector`와 `LLM` 가중치를 불러와 test split 이미지에 대해 caption을 생성하였다.
    - 생성된 caption은 토큰 단위로 분리한 뒤, reference caption들과 비교하여 corpus-level BLEU와 평균 ROUGE-L 점수를 계산하였다.
    - 최대 생성 길이는 `max_new_tokens=32`로 제한하여, 지나치게 긴 문장 생성으로 인한 평가 왜곡을 줄이도록 하였다.

### 5. 데이터 시각화
- 요구사항 : 데이터에 대한 EDA
- 구현
    - `visualize_data.py`를 작성하여 Flickr8k 데이터셋의 기본 통계와 예시 데이터를 시각화하도록 구성하였다.
    - 데이터 분할 파일을 직접 읽어 train / dev / test split의 이미지 개수를 집계하고, 이를 막대 그래프로 저장하였다.
    - 전체 caption을 대상으로 문장 길이(단어 수) 분포를 계산하여 histogram으로 시각화하였다.
    - caption에서 자주 등장하는 단어의 빈도를 집계하여 상위 단어를 막대 그래프로 정리하였다. 이때 `a`, `the`, `in` 같은 기본 관사는 제외하여 실제 의미 있는 단어 분포가 보이도록 하였다.
    - train split의 샘플 이미지와 대표 caption을 함께 배치한 grid 이미지를 생성하여, 데이터가 어떤 장면과 문장으로 구성되어 있는지 직관적으로 확인할 수 있도록 하였다.
- 시각화 결과
    - split 크기 시각화를 통해 train 데이터가 가장 크고, dev / test가 별도로 분리되어 있어 과제 요구사항에 맞는 학습 및 평가 구성이 가능함을 확인하였다.
    - caption 길이 분포를 통해 대부분의 문장이 비교적 짧은 설명형 문장으로 이루어져 있음을 확인하였고, 이에 따라 caption 생성 시 최대 길이를 과도하게 크게 잡지 않도록 참고하였다.
    - 상위 단어 분석 결과 `dog`, `man`, `woman`, `boy`, `running` 등 사람, 동물, 행동을 나타내는 단어가 자주 등장하여 Flickr8k가 일상 장면 중심의 이미지 캡셔닝 데이터셋임을 확인할 수 있었다.
    - 샘플 이미지 grid를 통해 하나의 이미지에 대해 자연어 설명이 직접 연결되는 데이터 구조를 확인하였고, 모델이 학습해야 하는 입력-출력 관계를 시각적으로 점검할 수 있었다.
- 산출물
    - 시각화 결과는 `report_visuals/` 디렉토리에 저장되며, `split_sizes.svg`, `caption_length_histogram.svg`, `top_caption_words.svg`, `sample_images_grid.png` 파일로 생성된다.

### 6. TODO List
- 모델 확장성
    - 현재 코드는 특정 모델만 학습이 가능한 코드이다.
    - 확장성을 고려해 Interface 기반 각 모델의 구현체를 구현하도록 구조 변경
    - HuggingFace로 로드시키는 모델이 아니더라도 다양하게 학습 가능하도록 설계
- 파이프라인 간소화
    - 현재 코드는 `train_align.py` -> `train_caption.py` -> `test.py` 로 실행
    - 초기 설계에서는 각자 학습을 자유롭게 하게 하기 위해서 분리
    - `train_align.py`, `train_caption.py` 가 합쳐서 학습이 가능하도록 간소화 필요
- 평가 지표 확장
    - 현재 BLEU, ROUGE-L 중심 평가를 수행하였으나, CIDEr, METEOR 등 captioning에 자주 사용되는 지표를 추가할 필요가 있다.
    - 다양한 지표를 함께 비교하여 생성 문장의 품질을 더 정밀하게 평가하도록 개선
- 실험 재현성 강화
    - seed, 하이퍼파라미터, 데이터 split, 모델 버전을 자동 저장하도록 개선 필요
    - config 기반 실행 구조로 정리하여 동일 조건의 재현 실험이 가능하도록 개선
- 체크포인트 관리 개선
    - 체크포인트 에서 이어서 학습 또는 해당 모델 기준 다른 파라미터로 학습 등의 기능 필요
- 에러 분석 및 결과 시각화 강화
    - 생성 caption 실패 사례를 별도로 저장하여 qualitative analysis가 가능하도록 개선
    - loss curve, metric curve, sample prediction 시각화까지 확장 필요
- 추론 인터페이스 개선
    - 단일 이미지 입력으로 caption을 생성할 수 있는 간단한 inference script 또는 demo 형태로 확장 가능