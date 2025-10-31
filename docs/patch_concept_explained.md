# Patch 개념 완전 정복 🎯

## 1. Patch란 무엇인가?

**패치(Patch)**는 연속된 시계열 데이터를 **일정한 크기의 청크(chunk)로 나누는** 기법입니다.

### 비유로 이해하기

```
시계열 데이터를 "영화 필름"으로 생각해보세요:

[원본 필름]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  (연속된 365개의 프레임)

[Patchify 적용 - patch_size=7]
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐...  (7일짜리 패치들로 분할)
 7 days  7 days  7 days  7 days  7 days

각 패치 = 하나의 "토큰(token)"으로 취급됩니다!
```

## 2. 실제 데이터로 이해하기

### 예시 1: 일간 주가 데이터

```python
# 원본 시계열 데이터
원본 데이터 = [100, 102, 101, 105, 103, 104, 106, 108, 107, 109, 110, 111, 112, 115, 114, 116]
길이 = 16 timesteps (16일)

# patch_size=4로 Patchify 적용
패치 0: [100, 102, 101, 105]  # 1~4일
패치 1: [103, 104, 106, 108]  # 5~8일
패치 2: [107, 109, 110, 111]  # 9~12일
패치 3: [112, 115, 114, 116]  # 13~16일

결과 shape:
  원본: (16,)              # 16개 timesteps
  패치: (4, 4)             # 4개 패치, 각 4 timesteps
```

### 예시 2: 다변량 시계열 (주가 OHLCV)

```python
# 5개 변수 (Open, High, Low, Close, Volume)
# 각 변수마다 16 timesteps

원본 shape: (5 variables, 16 timesteps)

┌─ Variable 0 (Open):   [100, 102, 101, 105, 103, 104, 106, 108, 107, 109, 110, 111, 112, 115, 114, 116]
├─ Variable 1 (High):   [102, 104, 103, 107, 105, 106, 108, 110, 109, 111, 112, 113, 114, 117, 116, 118]
├─ Variable 2 (Low):    [ 98, 100,  99, 103, 101, 102, 104, 106, 105, 107, 108, 109, 110, 113, 112, 114]
├─ Variable 3 (Close):  [101, 103, 102, 106, 104, 105, 107, 109, 108, 110, 111, 112, 113, 116, 115, 117]
└─ Variable 4 (Volume): [1.5, 1.7, 1.4, 2.1, 1.8, 1.9, 2.0, 2.2, 2.1, 2.3, 2.4, 2.5, 2.6, 2.9, 2.8, 3.0]

# patch_size=4로 Patchify 적용
패치 후 shape: (5 variables, 4 patches, 4 timesteps)

Variable 0:
  패치 0: [100, 102, 101, 105]
  패치 1: [103, 104, 106, 108]
  패치 2: [107, 109, 110, 111]
  패치 3: [112, 115, 114, 116]

Variable 1:
  패치 0: [102, 104, 103, 107]
  패치 1: [105, 106, 108, 110]
  ...
```

## 3. 코드로 보는 Patchify 과정

### Step-by-Step 변환

```python
import numpy as np
from einops import rearrange

# 원본 데이터: 1일 간격으로 16일치
data = np.array([100, 102, 101, 105, 103, 104, 106, 108,
                 107, 109, 110, 111, 112, 115, 114, 116])

print("원본 shape:", data.shape)  # (16,)

# Step 1: patch_size=4로 재배열
patch_size = 4
patched = rearrange(data, '(time patch) -> time patch', patch=patch_size)

print("패치 후 shape:", patched.shape)  # (4, 4)
print("\n패치별 데이터:")
print("패치 0:", patched[0])  # [100, 102, 101, 105]
print("패치 1:", patched[1])  # [103, 104, 106, 108]
print("패치 2:", patched[2])  # [107, 109, 110, 111]
print("패치 3:", patched[3])  # [112, 115, 114, 116]
```

### Uni2TS의 실제 Patchify 구현

```python
# src/uni2ts/transform/patch.py:155
def _patchify_arr(arr, patch_size, max_patch_size=128):
    """
    arr: shape (var, time*patch)
    return: shape (var, time, max_patch)
    """
    # 예: (5, 160) with patch_size=16
    assert arr.shape[-1] % patch_size == 0  # 160 % 16 == 0 ✓

    # Step 1: 패치로 재배열
    arr = rearrange(arr, '... (time patch) -> ... time patch', patch=patch_size)
    # (5, 160) → (5, 10, 16)  # 10개 패치, 각 16 timesteps

    # Step 2: max_patch_size까지 패딩
    pad_width = [(0, 0) for _ in range(arr.ndim)]
    pad_width[-1] = (0, max_patch_size - patch_size)  # (0, 128-16) = (0, 112)
    arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    # (5, 10, 16) → (5, 10, 128)  # 마지막에 112개 0 패딩

    return arr
```

## 4. 왜 Patchify를 사용하는가?

### 4.1 계산 효율성 향상 ⚡

```
패치 없이 처리할 때:
┌──────────────────────────────────────────┐
│ 시퀀스 길이 = 365 timesteps              │
│ Self-Attention 계산: O(365²) = 133,225  │
└──────────────────────────────────────────┘

패치 사용 (patch_size=16):
┌──────────────────────────────────────────┐
│ 시퀀스 길이 = 23 patches                 │
│ Self-Attention 계산: O(23²) = 529       │
│ 약 252배 빠름! 🚀                         │
└──────────────────────────────────────────┘
```

**계산량 비교**:
```
원본 길이: 1024 timesteps
- 패치 없음: O(1024²) = 1,048,576 operations
- patch_size=16: O(64²) = 4,096 operations (약 256배 감소)
- patch_size=32: O(32²) = 1,024 operations (약 1024배 감소)
```

### 4.2 Long-Range Dependency 학습 🎯

```
패치 없이:
각 timestep이 하나의 토큰
→ 멀리 있는 시점과의 관계 학습 어려움

┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│1│2│3│4│5│6│7│8│9│10│11│12│13│14│15│16│  (16개 토큰)
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
  ↕️ 각 step 간 관계만 학습

패치 사용 (patch_size=4):
각 패치가 하나의 토큰
→ 더 넓은 시간 범위를 한 번에 포착

┌───────┬───────┬───────┬───────┐
│ [1-4] │ [5-8] │[9-12] │[13-16]│  (4개 토큰)
└───────┴───────┴───────┴───────┘
  ↕️ 4일 단위로 패턴 학습
```

### 4.3 메모리 효율성 💾

```
긴 시계열 처리 시:

패치 없음 (2048 timesteps):
- KV cache: 2048 × d_model × num_layers
- GPU 메모리 초과 가능성 높음 ❌

패치 사용 (patch_size=32):
- 시퀀스 길이: 2048 / 32 = 64 tokens
- KV cache: 64 × d_model × num_layers
- 32배 메모리 절약! ✅
```

### 4.4 Vision Transformer와의 유사성 🖼️

```
Vision Transformer (ViT):
이미지 (224×224 pixels) → 패치 (16×16 patches) = 196 tokens

Time Series Transformer (Uni2TS):
시계열 (1024 timesteps) → 패치 (patch_size=16) = 64 tokens

동일한 원리! 고차원 입력을 토큰으로 변환
```

## 5. Uni2TS의 Patch Size 결정 방식

### 5.1 주파수별 기본 Patch Size

```python
# src/uni2ts/transform/patch.py:59-70
DEFAULT_RANGES = {
    "S": (64, 128),   # 초 단위: 512s ~ 4096s (8.5분 ~ 68분)
    "T": (32, 128),   # 분 단위: 64분 ~ 512분 (1시간 ~ 8.5시간)
    "H": (32, 64),    # 시간 단위: 128시간 (5.3일)
    "D": (16, 32),    # 일 단위: 16일 ~ 32일
    "B": (16, 32),    # 영업일: 16일 ~ 32일
    "W": (16, 32),    # 주 단위: 16주 ~ 32주
    "M": (8, 32),     # 월 단위: 8개월 ~ 32개월
    "Q": (1, 8),      # 분기: 1분기 ~ 8분기
    "Y": (1, 8),      # 연도: 1년 ~ 8년
}
```

**예시**:
```
일간 데이터 (D):
- patch_size=16 → 각 패치는 약 2주 데이터
- patch_size=32 → 각 패치는 약 1개월 데이터

시간별 데이터 (H):
- patch_size=32 → 각 패치는 약 1.3일 데이터
- patch_size=64 → 각 패치는 약 2.7일 데이터
```

### 5.2 동적 Patch Size 선택

```python
# GetPatchSize transform 동작
def get_patch_size(length, min_time_patches, constraints):
    # 최소 패치 개수 보장
    patch_size_ceil = length // min_time_patches

    # 예: length=365, min_time_patches=16
    # → patch_size_ceil = 22

    # 제약 조건과 최대 크기를 모두 만족하는 후보 선택
    candidates = [8, 16]  # 32는 22보다 크므로 제외

    # 랜덤하게 선택 (augmentation 효과)
    patch_size = np.random.choice(candidates)

    return patch_size
```

## 6. 실제 데이터 흐름 예시

### 주가 예측 시나리오

```python
# 설정
input_data = 365일치 일간 주가 (1년)
variables = 5개 (OHLCV)
patch_size = 16
prediction_length = 30일

# Step 1: 원본 데이터
shape: (5 variables, 365 timesteps)

┌─ Open:   [100.1, 100.5, 100.3, ..., 120.7]  (365 values)
├─ High:   [101.2, 101.8, 101.5, ..., 121.5]
├─ Low:    [ 99.8, 100.1,  99.9, ..., 119.8]
├─ Close:  [100.4, 100.7, 100.5, ..., 120.3]
└─ Volume: [1.5M,  1.7M,  1.4M,  ..., 2.8M]

# Step 2: Crop to multiple of patch_size
365 → 352 (352 = 16 × 22)
shape: (5, 352)

# Step 3: Patchify
shape: (5, 22, 16)
        ↓   ↓   ↓
        │   │   └─ 각 패치는 16일
        │   └───── 22개 패치 (약 11개월)
        └───────── 5개 변수

# Step 4: Pad to max_patch_size=128
shape: (5, 22, 128)
        ↑   ↑    ↑
        │   │    └─ 마지막 112개는 0으로 패딩
        │   └───── 22개 패치
        └───────── 5개 변수

# Step 5: Flatten to tokens
shape: (5 × 22, 128) = (110 tokens, 128)

각 토큰:
- 토큰 0: Variable 0, 패치 0 (1~16일)
- 토큰 1: Variable 0, 패치 1 (17~32일)
- ...
- 토큰 22: Variable 1, 패치 0 (1~16일)
- ...
- 토큰 109: Variable 4, 패치 21 (337~352일)
```

### Embedding 후

```python
# MultiInSizeLinear: (110 tokens, 128) → (110 tokens, 768)

각 패치 표현:
┌──────────────────────────────────────┐
│ Token 0: [0.12, -0.45, 0.78, ..., 0.34]  │  768차원 벡터
├──────────────────────────────────────┤
│ Token 1: [0.23, -0.12, 0.56, ..., 0.45]  │
├──────────────────────────────────────┤
│   ...                                │
└──────────────────────────────────────┘

이제 Transformer가 패치 단위로 attention 계산!
```

## 7. Patch와 Attention의 관계

### Self-Attention on Patches

```python
# Transformer Encoder에서 패치들 간 관계 학습

Query: 패치 10 (Variable 0, 161~176일)
Key:   모든 패치 (0~109)
Value: 모든 패치 (0~109)

Attention 계산:
패치 10이 어떤 다른 패치들과 관련 있는지?

Attention Weights (예시):
┌──────────┬──────────────────────────┐
│ 패치 9   │ ████████████████ 0.45   │  (같은 변수, 바로 이전)
│ 패치 8   │ ██████ 0.18             │  (같은 변수, 2개 전)
│ 패치 32  │ ████ 0.12               │  (다른 변수, 같은 시간대)
│ 패치 54  │ ███ 0.09                │
│ 기타     │ ██ 0.16                 │
└──────────┴──────────────────────────┘

결과: 패치 10의 새로운 표현
= 0.45 × V[패치9] + 0.18 × V[패치8] + 0.12 × V[패치32] + ...
```

### Binary Attention Bias 효과

```python
같은 변수 내 패치들:
패치 0 (Var 0, Day 1-16)  ←→ 패치 1 (Var 0, Day 17-32)
         ↕️ same_var_embedding = +2.0 (boost)

다른 변수 간 패치들:
패치 0 (Var 0, Day 1-16)  ←→ 패치 22 (Var 1, Day 1-16)
         ↕️ diff_var_embedding = -1.0 (suppress)

효과:
- 같은 변수 내에서 시간적 패턴 강하게 학습
- 변수 간 간섭 최소화
```

## 8. Patch Size 선택의 영향

### Trade-off 분석

```
┌────────────────┬─────────────┬──────────────┬──────────────┐
│ Patch Size     │ 패치 개수   │ 계산 복잡도  │ 표현력       │
├────────────────┼─────────────┼──────────────┼──────────────┤
│ 8  (작음)      │ 45개        │ O(45²)=2025  │ 높음 (세밀)  │
│ 16 (중간)      │ 22개        │ O(22²)=484   │ 중간         │
│ 32 (큼)        │ 11개        │ O(11²)=121   │ 낮음 (거칠)  │
└────────────────┴─────────────┴──────────────┴──────────────┘

예: 365일 데이터 (각 패치로 나눈 후)
```

### 권장 사항

```
시계열 특성에 따라:

1. 고주파 데이터 (분 단위, 초 단위)
   → 큰 patch_size (64-128)
   → 노이즈 감소, 계산 효율성 중시

2. 저주파 데이터 (일 단위, 주 단위)
   → 작은 patch_size (8-32)
   → 세밀한 패턴 포착

3. 긴 시퀀스 (>2048 timesteps)
   → 큰 patch_size
   → 메모리 제약 해결

4. 짧은 시퀀스 (<512 timesteps)
   → 작은 patch_size
   → 충분한 토큰 개수 확보
```

## 9. 실제 Uni2TS 사용 예시

```python
# 예: 2년치 일간 주가 데이터 예측

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# 데이터 준비
df = pd.read_csv('stock_prices.csv')  # 730일 (2년)
ds = PandasDataset(dict(df))

# 모델 로드
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
    prediction_length=30,    # 30일 예측
    context_length=700,      # 700일 히스토리 사용
    patch_size=16,           # 16일 = 하나의 패치
    num_samples=100,
)

# 내부적으로 일어나는 일:
# 1. 700일 데이터 → 700/16 = 43개 패치 (마지막 12일 crop)
# 2. 각 패치 (16일) → embedding (768차원)
# 3. 43개 토큰 → Transformer Encoder
# 4. GQA로 패치 간 관계 학습
# 5. 예측: 30일 / 16 = 2개 패치 예측 (32일 예측 후 crop)
```

## 10. 핵심 요약

### Patch의 본질

```
🎯 Patch = 연속된 timestep들의 묶음

목적:
1. ⚡ 계산 효율성: 시퀀스 길이 단축 → Attention 계산량 감소
2. 🎯 Long-range: 넓은 범위를 하나의 토큰으로 → 장기 의존성 학습
3. 💾 메모리 효율: KV cache 크기 감소 → 긴 시퀀스 처리 가능
4. 🧩 Inductive Bias: 시계열의 국소적 패턴을 자연스럽게 포착

비유: 영화를 "프레임 단위"가 아닌 "씬 단위"로 이해하는 것
```

### 데이터 변환 흐름

```
원본 시계열:
[t₀, t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈, t₉, t₁₀, t₁₁, t₁₂, t₁₃, t₁₄, t₁₅]

↓ Patchify (patch_size=4)

패치들:
[Patch₀      ] [Patch₁      ] [Patch₂      ] [Patch₃      ]
[t₀,t₁,t₂,t₃] [t₄,t₅,t₆,t₇] [t₈,t₉,t₁₀,t₁₁] [t₁₂,t₁₃,t₁₄,t₁₅]

↓ Embedding

토큰 표현:
[Token₀] [Token₁] [Token₂] [Token₃]
[e₀...] [e₁...] [e₂...] [e₃...]  (각 768차원)

↓ Transformer

Self-Attention으로 토큰 간 관계 학습
→ 각 패치가 다른 패치들의 정보를 종합

↓ Output Projection

예측 결과 (분포 파라미터)
```

---

## 참고

- **Vision Transformer (ViT) 논문**: "An Image is Worth 16x16 Words" - 패치 개념의 원조
- **Moirai 논문**: Section 3.2 "Patchification" - 시계열에 패치 적용
- **코드**: `src/uni2ts/transform/patch.py` - 실제 구현
