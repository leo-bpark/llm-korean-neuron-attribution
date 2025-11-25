# LLM Korean Neuron Attribution Tool

LLM의 MLP 뉴런 기여도를 측정하고 입력 토큰에 시각화하는 도구입니다.

## 주요 기능

- **모델 로드**: HuggingFace Transformers 모델 로드 지원
- **뉴런 선택**: 특정 레이어와 뉴런 선택
- **Attribution 계산**: 
  - Activation Patching (빠른 방법)
  - Integrated Gradients (더 정확한 방법)
- **시각화**: 토큰별 기여도를 색상으로 표시

## 설치

```bash
pip install -r requirements.txt
```

또는 개발 모드로 설치:

```bash
pip install -e .
```

## 사용법

### 서버 실행

```bash
python run_server.py
```

서버가 시작되면 브라우저에서 `http://localhost:8000`을 열어주세요.

### 웹 인터페이스 사용

1. **모델 로드**: HuggingFace 모델 ID를 입력하고 "모델 로드" 클릭
   - 예: `gpt2`, `microsoft/DialoGPT-small`, `kakaobrain/kogpt`
2. **입력 텍스트**: 분석할 텍스트 입력
3. **뉴런 선택**: 
   - Layer Index 입력 후 "뉴런 정보 불러오기" 클릭
   - Neuron Index 선택
4. **Attribution 계산**: 계산 방법 선택 후 "Attribution 계산" 클릭
5. **시각화**: 토큰별 기여도가 색상으로 표시됩니다 (진한 파란색 = 높은 기여도)

## API 엔드포인트

### POST /api/load_model
모델을 로드합니다.

```json
{
  "model_name": "gpt2"
}
```

### POST /api/get_neuron_info
특정 레이어의 뉴런 정보를 가져옵니다.

```json
{
  "model_name": "gpt2",
  "layer_idx": 0
}
```

### POST /api/compute_attribution
Attribution을 계산합니다.

```json
{
  "model_name": "gpt2",
  "input_text": "안녕하세요",
  "layer_idx": 0,
  "neuron_idx": 100,
  "method": "activation_patching"
}
```

## 아키텍처

- `LKN/attribution.py`: Attribution 계산 로직 (Integrated Gradients, Activation Patching)
- `LKN/server.py`: FastAPI 서버 및 API 엔드포인트
- `static/index.html`: 웹 인터페이스
- `run_server.py`: 서버 실행 스크립트

## 지원 모델

- GPT-2 스타일 모델
- GPT-NeoX 스타일 모델
- LLaMA 스타일 모델
- 기타 Transformer 기반 Causal LM

## 주의사항

- GPU 사용 시 더 빠른 계산이 가능합니다
- 큰 모델의 경우 모델 로드에 시간이 걸릴 수 있습니다
- Integrated Gradients는 Activation Patching보다 느리지만 더 정확합니다

## 라이선스

Apache License 2.0
