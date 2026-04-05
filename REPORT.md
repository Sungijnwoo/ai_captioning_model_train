## 각 단계별 문제점 및 해결방법
### 1. Dataset 
- 요구사항 : flickr8k 사용
- Dataset 
    - 1:n (image, caption) 으로 매칭 
    - 한 이미지당 여러개 caption 을 가지고 있음
    - Return 값 : (image, caption)
### 2. Embedding Align
- 요구사항 : CLIP의 embedding을 LLM embedding 공간으로 align
- 구현
    - clip 모델의 output shape &rightarrow; LLM 모델의 input shape 으로 align
    - ClipToLlmProjector 를 이용해 간단한 Linear &rightarrow; GELU &rightarrow; Linear 모델 구축
    - multi_rate 를 활용한 Projector Hidden Size 결정
- 학습
    - clip, llm 모델 모든 레이어 freeze
    - projector 모델만 학습
    - image &rightarrow; clip_model &rightarrow; embedding &rightarrow; projector &rightarrow; align_embedding
    - text &rightarrow; llm_tokenizer &rightarrow; text_embedding
    - align_embedding, text_embedding의 유사도 비교값 이용해 Loss 계산
### 3. Caption Train
- 요구사항 : Embdding Align 모델을 이용하여 Caption
- 구현
    - clip, projector, llm 모델을 이용해 caption 모델 구축
    - clip &rightarrow; projector &rightarrow; llm &rightarrow; text
    - 같은 Flickr8k Dataset 이용 (image, caption)
