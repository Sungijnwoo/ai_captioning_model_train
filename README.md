## 개요

CLIP으로 이미지 특징을 추출하고, LLM으로 텍스트를 생성하며, 두 모델의 임베딩을 정렬(alignment)한 뒤 `[image, text]` 데이터로 캡셔닝을 fine-tuning하는 프로젝트입니다.

## 과제 개요

| 단계 | 내용 |
|------|------|
| **1. Embedding Alignment** | CLIP·LLM 임베딩을 동일 공간에 맞춤 (Hugging Face Transformers 활용) |
| **2. Captioning** | 정렬된 표현을 바탕으로 이미지 입력 → 자연어 캡션 생성 모델 학습 |
| **3. Evaluation** | 캡셔닝에 적합한 지표로 평가 (외부 데이터 사용 시 데이터 출처 명시) |

### 사용 모델·데이터

- **CLIP:** [`apple/MobileCLIP-B`](https://huggingface.co/apple/MobileCLIP-B) (MIT)
- **LLM:** [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (Apache-2.0)
- **데이터셋:** Flickr8k [다운로드 링크](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)

## 개발 환경
[uv](https://docs.astral.sh/uv/getting-started/installation/)로 동기화합니다.
```bash
uv sync
```

## 디렉터리 구조 (요약)

```text
├── config/                 # align / caption / test YAML
├── dataset/                # Flickr8k 데이터로더
├── domain/dto/             # 설정 DTO (Align, Caption, Test)
├── model/                  # CLIP→LLM 프로젝터 등
├── trainer/                # AlignTrainer, CaptionTrainer, CaptionTester
├── util/                   # 로깅, 모델 로드·다운로드
├── train_align.py          # 1단계: alignment 학습
├── train_caption.py        # 2단계: 캡션 학습
├── test.py                 # 3단계: 평가
├── data/                   # Flickr8k (직접 준비)
├── weight/                 # CLIP 가중치 캐시 (자동 다운로드 시 생성)
└── output/                 # 학습 산출물 (align / caption)
```

## 데이터 준비
```bash
mkdir data && cd data
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
unzip Flickr8k_Dataset.zip
unzip Flickr8k_text.zip
```

## 로컬 실행 방법

### 1) Alignment 학습

프로젝터 등 alignment 가중치가 `output/align/<타임스탬프>/` 아래에 저장됩니다.

```bash
uv run train_align.py --config config/align_config.yaml
```

### 2) Captioning 학습

alignment 실행 후 생성된 디렉터리를 가리키도록 `config/caption_config.yaml`의 `project_model_dir`을 설정합니다.

```yaml
project_model_dir: "output/align/<실제_생성된_폴더명>"
```

이후 캡션 학습을 실행합니다.

```bash
uv run train_caption.py --config config/caption_config.yaml
```

### 3) 평가(추론·지표)

`config/test_config.yaml`에서 학습 산출물 디렉터리를 지정합니다.

```yaml
model_dir: output/caption/<실제_생성된_폴더명>
```

```bash
uv run test.py --config config/test_config.yaml
```

