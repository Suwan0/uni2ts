# 완전히 구체적인 데이터 흐름 설명 (실제 예시 포함)

## 🎯 목표: 2개의 시계열을 입력받아 모델 출력까지의 전체 과정

### 시나리오 설정

```python
# 두 개의 시계열 데이터
시계열 1: 512 timesteps (예: 512일간의 일일 주가)
시계열 2: 512 timesteps (예: 512일간의 다른 주식 주가)

설정:
- patch_size = 16
- max_patch_size = 128
- d_model = 768
```

---

## Step 0: 원본 데이터

### 시계열 1 (단변량 univariate)
```
Data: [100.1, 100.5, 100.3, 101.2, 101.5, ..., 150.7, 151.2]
Shape: (512,)  # 512개의 timesteps

변수 수: 1개
시간 길이: 512 timesteps
```

### 시계열 2 (단변량 univariate)
```
Data: [50.3, 50.8, 50.1, 51.5, 51.2, ..., 75.3, 75.8]
Shape: (512,)

변수 수: 1개
시간 길이: 512 timesteps
```

**현재 상태**:
```
Sample 1: numpy array (512,)
Sample 2: numpy array (512,)

두 개는 완전히 별개의 시계열!
```

---

## Step 1: Reshape to (var, time)

단변량 시계열을 (1, 512) 형태로 변환:

```python
# Sample 1
data_entry_1 = {
    'target': np.array([[100.1, 100.5, 100.3, ..., 151.2]]),  # shape: (1, 512)
    'freq': 'D',  # 일별 데이터
}

# Sample 2
data_entry_2 = {
    'target': np.array([[50.3, 50.8, 50.1, ..., 75.8]]),  # shape: (1, 512)
    'freq': 'D',
}
```

**핵심**:
- 첫 번째 차원 = 변수 수 (variable)
- 두 번째 차원 = 시간 (time)

---

## Step 2: Patchify - 시간을 패치로 나누기

### 동작 원리

```python
# src/uni2ts/transform/patch.py:155
def _patchify_arr(arr, patch_size=16):
    """
    arr: shape (var, time) = (1, 512)
    return: shape (var, num_patches, patch_size) = (1, 32, 16)
    """
    # Step 1: Reshape
    # (1, 512) → (1, 32, 16)
    arr = rearrange(arr, 'var (time patch) -> var time patch', patch=16)

    return arr
```

### Sample 1 변환

```
원본: shape (1, 512)
┌─────────────────────────────────────────────────┐
│ [100.1, 100.5, 100.3, ..., 151.2]              │ 512 timesteps
└─────────────────────────────────────────────────┘

↓ patch_size=16으로 재배열

Patchified: shape (1, 32, 16)
┌────────────────┐
│ 패치 0:         │ [100.1, 100.5, 100.3, 101.2, 101.5, ..., 102.8]  # timestep 0-15
│ 패치 1:         │ [103.1, 103.4, 103.8, 104.2, 104.5, ..., 105.3]  # timestep 16-31
│ 패치 2:         │ [106.1, 106.7, 107.2, 107.8, 108.1, ..., 109.5]  # timestep 32-47
│ ...            │
│ 패치 31:        │ [149.2, 149.8, 150.1, 150.5, 150.9, ..., 151.2]  # timestep 496-511
└────────────────┘

총 32개의 패치, 각 패치는 16개의 연속된 timesteps
```

**질문 답변**:
> "512개의 timestamps가 있다는 얘기는 예를 들어서 1년을 512개의 시간으로 나눠서 데이터가 존재한다고 하면 16개의 timestamps를 묶어서 하나의 패치로 보겠다는 이야기야?"

**답**: 정확합니다! ✅
- 512일 = 512개 timesteps
- 16일씩 묶어서 하나의 패치
- 32개의 패치가 생성됨

### Sample 2도 동일하게 변환

```
Sample 2: (1, 512) → (1, 32, 16)
```

**현재 상태**:
```
Sample 1: {
    'target': np.array, shape (1, 32, 16)
}

Sample 2: {
    'target': np.array, shape (1, 32, 16)
}

여전히 두 샘플은 분리되어 있음!
```

---

## Step 3: Pad to max_patch_size

### 왜 필요한가?

**이유**: **다양한 patch_size를 지원하기 위해!**

```
문제 상황:
- 어떤 샘플은 patch_size=8 사용
- 어떤 샘플은 patch_size=16 사용
- 어떤 샘플은 patch_size=32 사용

모델 입력:
- Linear layer는 고정된 입력 크기 필요!
- MultiInSizeLinear이 다양한 크기 처리하지만, 배치 처리 위해 통일 필요

해결책:
- 모든 패치를 max_patch_size=128로 패딩
- 실제 patch_size 정보는 별도 필드로 저장
```

### 동작 원리

```python
# src/uni2ts/transform/patch.py:156-159
def _patchify_arr(arr, patch_size=16, max_patch_size=128):
    # Step 1: Reshape (위에서 설명)
    arr = rearrange(arr, 'var (time patch) -> var time patch', patch=16)
    # (1, 32, 16)

    # Step 2: Pad to max_patch_size
    pad_width = [(0, 0), (0, 0), (0, max_patch_size - patch_size)]
    #            var      time    patch 차원
    #                              ↑ 16 → 128로 늘림 (112개 0 추가)

    arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    # (1, 32, 128)

    return arr
```

### Sample 1 변환

```
Patchified: (1, 32, 16)
┌────────────────┐
│ 패치 0:         │ [100.1, 100.5, 100.3, ..., 102.8] (16개)
└────────────────┘

↓ Pad to 128

Padded: (1, 32, 128)
┌─────────────────────────────────────────────────────────────┐
│ 패치 0: [100.1, 100.5, ..., 102.8, 0, 0, 0, ..., 0]        │ 16개 값 + 112개 0
│         ├──────── 16개 실제 값 ──────┤├── 112개 패딩 ──┤   │
│                                                              │
│ 패치 1: [103.1, 103.4, ..., 105.3, 0, 0, 0, ..., 0]        │
│ ...                                                          │
│ 패치 31: [149.2, 149.8, ..., 151.2, 0, 0, 0, ..., 0]       │
└─────────────────────────────────────────────────────────────┘

Shape: (1, 32, 128)
```

**현재 상태**:
```
Sample 1: {
    'target': np.array, shape (1, 32, 128)
    'patch_size': 16  # 실제 사용된 patch_size 기록!
}

Sample 2: {
    'target': np.array, shape (1, 32, 128)
    'patch_size': 16
}
```

---

## Step 4: Add Features - variate_id, time_id 추가

### 중요! 이것은 nn.Parameter가 아닙니다!

**질문 답변**:
> "add features는 원래 데이터에 없는 애들을 nn.parameter 이런식으로 patch개수 만큼의 parameter로 추가하는거야?"

**답**: 아닙니다! ❌
- **numpy array**로 추가되는 메타데이터입니다
- **학습 가능한 파라미터가 아님**
- 위치 정보를 제공하는 **인덱스 배열**입니다
- 나중에 embedding layer를 통과해서 학습 가능한 벡터로 변환됨

### 4.1 AddVariateIndex

각 변수(variate)에 고유 ID 부여:

```python
# src/uni2ts/transform/feature.py:55-71
def _generate_variate_id(arr):
    """
    arr: shape (var, time) = (1, 32)  # patchify 후
    return: shape (var, time) = (1, 32)
    """
    dim, time = arr.shape[:2]  # dim=1, time=32

    # 변수별 ID: 0, 1, 2, ..., dim-1
    variate_ids = np.arange(dim)  # [0]

    # 각 timestep마다 반복
    field_dim_id = repeat(variate_ids, 'var -> var time', time=time)
    # (1,) → (1, 32)

    return field_dim_id
```

#### Sample 1: variate_id 생성

```
원본: (1, 32, 128)
     └── 1개 변수

variate_id: (1, 32)
┌─────────────────────────────────────────┐
│ Variable 0: [0, 0, 0, 0, ..., 0]       │ 32개 모두 0
└─────────────────────────────────────────┘

의미:
- 패치 0의 variate_id = 0  (변수 0에 속함)
- 패치 1의 variate_id = 0  (변수 0에 속함)
- ...
- 패치 31의 variate_id = 0 (변수 0에 속함)
```

#### 다변량 예시 (OHLCV 5개 변수)

만약 5개 변수가 있다면:

```
원본: (5, 32, 128)
     └── 5개 변수 (Open, High, Low, Close, Volume)

variate_id: (5, 32)
┌─────────────────────────────────────────┐
│ Variable 0 (Open):  [0, 0, 0, ..., 0]  │ 32개 모두 0
│ Variable 1 (High):  [1, 1, 1, ..., 1]  │ 32개 모두 1
│ Variable 2 (Low):   [2, 2, 2, ..., 2]  │ 32개 모두 2
│ Variable 3 (Close): [3, 3, 3, ..., 3]  │ 32개 모두 3
│ Variable 4 (Volume):[4, 4, 4, ..., 4]  │ 32개 모두 4
└─────────────────────────────────────────┘

Flatten 후: (5×32 = 160 tokens)
variate_id = [0,0,0,...,0, 1,1,1,...,1, 2,2,2,...,2, 3,3,3,...,3, 4,4,4,...,4]
             └─ 32개 0 ─┘ └─ 32개 1 ─┘ └─ 32개 2 ─┘ └─ 32개 3 ─┘ └─ 32개 4 ─┘
```

### 4.2 AddTimeIndex

각 패치에 시간 순서 ID 부여:

```python
# src/uni2ts/transform/feature.py:98-104
def _generate_time_id(arr):
    """
    arr: shape (var, time) = (1, 32)
    return: shape (var, time) = (1, 32)
    """
    var, time = arr.shape[:2]  # var=1, time=32

    # 시간 순서 ID: 0, 1, 2, ..., time-1
    field_seq_id = np.arange(time)  # [0, 1, 2, ..., 31]

    # 각 변수마다 반복
    field_seq_id = repeat(field_seq_id, 'time -> var time', var=var)
    # (32,) → (1, 32)

    return field_seq_id
```

#### Sample 1: time_id 생성

```
time_id: (1, 32)
┌─────────────────────────────────────────┐
│ Variable 0: [0, 1, 2, 3, 4, ..., 31]   │
└─────────────────────────────────────────┘

의미:
- 패치 0의 time_id = 0  (첫 번째 패치)
- 패치 1의 time_id = 1  (두 번째 패치)
- ...
- 패치 31의 time_id = 31 (마지막 패치)
```

#### 다변량 예시

```
time_id: (5, 32)
┌─────────────────────────────────────────┐
│ Variable 0: [0, 1, 2, 3, ..., 31]      │
│ Variable 1: [0, 1, 2, 3, ..., 31]      │
│ Variable 2: [0, 1, 2, 3, ..., 31]      │
│ Variable 3: [0, 1, 2, 3, ..., 31]      │
│ Variable 4: [0, 1, 2, 3, ..., 31]      │
└─────────────────────────────────────────┘

모든 변수가 같은 시간 순서를 가짐!
```

### 4.3 AddSampleIndex (또는 Collation 시 생성)

같은 샘플 내 모든 토큰에 동일한 ID:

```python
# src/uni2ts/transform/feature.py:152-159
def _generate_sample_id(arr):
    """
    arr: shape (var, time) = (1, 32)
    return: shape (var, time) = (1, 32)
    """
    var, time = arr.shape[:2]

    # 모두 1로 채움 (같은 샘플)
    field_seq_id = np.ones(time, dtype=int)  # [1, 1, 1, ..., 1]
    field_seq_id = repeat(field_seq_id, 'time -> var time', var=var)
    # (32,) → (1, 32)

    return field_seq_id
```

#### Sample 1: sample_id

```
sample_id: (1, 32)
┌─────────────────────────────────────────┐
│ Variable 0: [1, 1, 1, 1, ..., 1]       │ 모두 1
└─────────────────────────────────────────┘

의미: 이 32개 패치는 모두 같은 샘플(Sample 1)에 속함
```

### 현재 상태 정리

```
Sample 1:
{
    'target': np.array, shape (1, 32, 128),      # 실제 값
    'variate_id': np.array, shape (1, 32),       # [0,0,0,...,0]
    'time_id': np.array, shape (1, 32),          # [0,1,2,...,31]
    'sample_id': np.array, shape (1, 32),        # [1,1,1,...,1]
    'patch_size': 16,
}

Sample 2:
{
    'target': np.array, shape (1, 32, 128),      # 실제 값
    'variate_id': np.array, shape (1, 32),       # [0,0,0,...,0]
    'time_id': np.array, shape (1, 32),          # [0,1,2,...,31]
    'sample_id': np.array, shape (1, 32),        # [1,1,1,...,1]  # Sample 1과 동일!
    'patch_size': 16,
}

아직 두 샘플은 분리된 상태!
```

---

## Step 5: Flatten (var, time) → (tokens)

각 샘플을 토큰 시퀀스로 변환:

```python
# Flatten: (var, time, patch_size) → (var*time, patch_size)

Sample 1:
target: (1, 32, 128) → (32, 128)  # 1*32 = 32 tokens
variate_id: (1, 32) → (32,)       # [0,0,0,...,0]
time_id: (1, 32) → (32,)          # [0,1,2,...,31]
sample_id: (1, 32) → (32,)        # [1,1,1,...,1]
```

**토큰 구성**:
```
Sample 1의 32개 토큰:

Token 0:  target=[100.1, 100.5, ..., 102.8, 0, ...], variate_id=0, time_id=0,  sample_id=1
Token 1:  target=[103.1, 103.4, ..., 105.3, 0, ...], variate_id=0, time_id=1,  sample_id=1
Token 2:  target=[106.1, 106.7, ..., 109.5, 0, ...], variate_id=0, time_id=2,  sample_id=1
...
Token 31: target=[149.2, 149.8, ..., 151.2, 0, ...], variate_id=0, time_id=31, sample_id=1

Sample 2의 32개 토큰:

Token 0:  target=[50.3, 50.8, ..., 51.5, 0, ...], variate_id=0, time_id=0,  sample_id=1
Token 1:  target=[52.1, 52.4, ..., 53.2, 0, ...], variate_id=0, time_id=1,  sample_id=1
...
Token 31: target=[74.8, 75.1, ..., 75.8, 0, ...], variate_id=0, time_id=31, sample_id=1
```

---

## Step 6: Collation - 배치로 합치기

### 두 가지 전략

#### 6.1 PadCollate - 단순 패딩

```python
# src/uni2ts/data/loader.py:55-101

max_length = 64  # 예시

Sample 1: 32 tokens
Sample 2: 32 tokens

↓ Pad to max_length=64

Batch:
┌──────────────────────────────────────────────┐
│ Sample 1: [token_0, token_1, ..., token_31, │
│            pad, pad, ..., pad]               │ 32 + 32 padding
├──────────────────────────────────────────────┤
│ Sample 2: [token_0, token_1, ..., token_31, │
│            pad, pad, ..., pad]               │ 32 + 32 padding
└──────────────────────────────────────────────┘

Shape: (2, 64, 128)
       ↑  ↑   ↑
       │  │   └── max_patch_size
       │  └────── max_length (padded)
       └───────── batch_size
```

**sample_id 생성**:
```python
# src/uni2ts/data/loader.py:92-100

sample_id = [
    [1, 1, 1, ..., 1, 0, 0, 0, ..., 0],  # Sample 1: 32개 1, 32개 0 (padding)
    [1, 1, 1, ..., 1, 0, 0, 0, ..., 0],  # Sample 2: 32개 1, 32개 0 (padding)
]

Shape: (2, 64)
```

**문제**: 32개의 패딩 토큰이 낭비됨 (계산에 사용 안됨)

#### 6.2 PackCollate - 효율적 패킹 (추천)

```python
# src/uni2ts/data/loader.py:103-210

max_length = 64

Sample 1: 32 tokens
Sample 2: 32 tokens

↓ Bin Packing: 32 + 32 = 64 (딱 맞음!)

Packed Batch:
┌──────────────────────────────────────────────┐
│ Bin 0: [Sample1_token_0, Sample1_token_1, ..., Sample1_token_31, │
│         Sample2_token_0, Sample2_token_1, ..., Sample2_token_31]  │
└──────────────────────────────────────────────┘

Shape: (1, 64, 128)
       ↑  ↑   ↑
       │  │   └── max_patch_size
       │  └────── max_length (꽉 참!)
       └───────── batch_size (1개 bin)
```

**sample_id 생성**:
```python
# src/uni2ts/data/loader.py:161-184

sample_id = [
    [1, 1, 1, ..., 1, 2, 2, 2, ..., 2],
    └── Sample 1 ──┘ └── Sample 2 ──┘
     (32개)           (32개)
]

Shape: (1, 64)
```

**핵심**: sample_id가 달라져서 두 샘플을 구분!
- Sample 1 토큰들: sample_id = 1
- Sample 2 토큰들: sample_id = 2

### 최종 Batch 텐서

**PadCollate 사용 시**:
```python
batch = {
    'target':      torch.Tensor, shape (2, 64, 128),
    'variate_id':  torch.Tensor, shape (2, 64),
    'time_id':     torch.Tensor, shape (2, 64),
    'sample_id':   torch.Tensor, shape (2, 64),
    'patch_size':  torch.Tensor, shape (2, 64),
    'observed_mask': torch.Tensor, shape (2, 64, 128),
    'prediction_mask': torch.Tensor, shape (2, 64),
}
```

**PackCollate 사용 시** (효율적!):
```python
batch = {
    'target':      torch.Tensor, shape (1, 64, 128),  # 1개 bin!
    'variate_id':  torch.Tensor, shape (1, 64),
    'time_id':     torch.Tensor, shape (1, 64),
    'sample_id':   torch.Tensor, shape (1, 64),       # [1,1,...,1,2,2,...,2]
    'patch_size':  torch.Tensor, shape (1, 64),
    'observed_mask': torch.Tensor, shape (1, 64, 128),
    'prediction_mask': torch.Tensor, shape (1, 64),
}
```

---

## Step 7: Model Forward - PackedStdScaler

### 샘플별, 변수별 정규화

```python
# src/uni2ts/module/packed_scaler.py

# 입력: batch['target'] = (1, 64, 128)
#       batch['sample_id'] = (1, 64) = [1,1,...,1,2,2,...,2]
#       batch['variate_id'] = (1, 64) = [0,0,...,0,0,0,...,0]

for sample_id in unique(batch['sample_id']):  # [1, 2]
    for variate_id in unique(batch['variate_id']):  # [0]
        # Sample 1, Variate 0의 토큰들만 선택
        mask = (batch['sample_id'] == sample_id) & (batch['variate_id'] == variate_id)
        # mask = [True, True, ..., True, False, False, ..., False]
        #        └─ Sample 1의 32개 ─┘ └─ Sample 2의 32개 ─┘

        values = batch['target'][mask]  # Sample 1의 32개 패치

        # 통계량 계산 (observed_mask 고려)
        loc = mean(values[observed_mask])   # 평균
        scale = std(values[observed_mask])  # 표준편차

        # 정규화
        batch['target'][mask] = (values - loc) / scale
```

**결과**:
- Sample 1: 평균=125.0, 표준편차=15.0으로 정규화
- Sample 2: 평균=62.5, 표준편차=7.5으로 정규화
- **각 샘플이 독립적으로 정규화됨!**

---

## Step 8: MultiInSizeLinear - Patch Embedding

### 동작

```python
# src/uni2ts/module/ts_embed.py

# 입력: (1, 64, 128)
# patch_size: (1, 64) = [16, 16, ..., 16]  # 모두 16

for each token:
    # patch_size=16인 weight 선택
    weight = self.weights[16]  # shape: (128, 768)

    # 실제 사용 부분만 선택
    mask = [1,1,1,...,1,0,0,...,0]  # 16개 1, 112개 0

    # Linear transformation
    output = input @ weight * mask
    # (128,) @ (128, 768) → (768,)

# 출력: (1, 64, 768)
```

**결과**: 각 패치가 768차원 벡터로 변환됨

---

## Step 9: Mask Fill - 예측 구간 마스킹

```python
# src/uni2ts/model/moirai/module.py

# prediction_mask: (1, 64) = [False, False, ..., False, True, True, ..., True]
#                             └─── history ───┘ └── prediction ──┘

# 학습 가능한 mask embedding
mask_embedding = nn.Parameter(torch.randn(1, d_model))  # (1, 768)

# prediction_mask=True인 위치를 mask_embedding으로 대체
embedded[prediction_mask] = mask_embedding
```

---

## Step 10: Transformer Encoder

### Attention Mask 생성

```python
# sample_id: (1, 64) = [1,1,...,1,2,2,...,2]

# Packed Attention Mask
packed_mask = (sample_id.unsqueeze(-1) == sample_id.unsqueeze(-2))
# (1, 64, 64)

# 예시:
#        토큰 0  1  2 ... 31 32 33 ... 63
# 토큰 0 [ True  True ... True False False ... False ]  Sample 1
# 토큰 1 [ True  True ... True False False ... False ]
# ...
# 토큰 31[ True  True ... True False False ... False ]
# 토큰 32[ False False ... False True True ... True ]  Sample 2
# 토큰 33[ False False ... False True True ... True ]
# ...

각 샘플 내에서만 attention 허용!
```

### GQA 계산

```python
# Layer 1
for layer in encoder.layers:
    # Self-Attention
    q = layer.gqa.q_proj(x)  # (1, 64, 768) → (1, 64, 768)
    k = layer.gqa.k_proj(x)  # (1, 64, 768) → (1, 64, 256)  # 4 groups
    v = layer.gqa.v_proj(x)  # (1, 64, 768) → (1, 64, 256)

    # Reshape to multi-head
    q = q.view(1, 64, 4, 3, 64)  # (batch, seq, groups, hpg, head_dim)
    k = k.view(1, 64, 4, 1, 64)
    v = v.view(1, 64, 4, 1, 64)

    # Expand K, V
    k = k.expand(1, 64, 4, 3, 64)
    v = v.expand(1, 64, 4, 3, 64)

    # Position Encoding
    # Time: RoPE based on time_id
    # Variate: Binary bias based on variate_id

    # Attention with packed_mask
    attn_out = scaled_dot_product_attention(q, k, v, attn_mask=packed_mask)

    # Feedforward
    x = x + attn_out
    x = x + ffn(x)

# 출력: (1, 64, 768)
```

**핵심**:
- Sample 1의 토큰들끼리만 attention
- Sample 2의 토큰들끼리만 attention
- **두 샘플은 완전히 독립적으로 처리됨!**

---

## Step 11: Output Projection

```python
# param_proj: (1, 64, 768) → (1, 64, num_params)

# 예: Normal distribution → num_params = 2 (mean, std)
output = param_proj(x)  # (1, 64, 2)

# Rescale (원래 스케일로 복원)
params[:, :, 0] = params[:, :, 0] * scale + loc  # mean
params[:, :, 1] = params[:, :, 1] * scale        # std

# Distribution 생성
dist = Normal(params[:, :, 0], params[:, :, 1])
```

---

## 전체 Shape 변화 요약 (PackCollate 사용)

```
Step 0: Raw Data
  Sample 1: (512,)
  Sample 2: (512,)

Step 1: Reshape
  Sample 1: (1, 512)
  Sample 2: (1, 512)

Step 2: Patchify
  Sample 1: (1, 32, 16)
  Sample 2: (1, 32, 16)

Step 3: Pad
  Sample 1: (1, 32, 128)
  Sample 2: (1, 32, 128)

Step 4: Add Features
  Sample 1: {target: (1,32,128), variate_id: (1,32), time_id: (1,32), sample_id: (1,32)}
  Sample 2: {target: (1,32,128), variate_id: (1,32), time_id: (1,32), sample_id: (1,32)}

Step 5: Flatten
  Sample 1: {target: (32,128), variate_id: (32,), time_id: (32,), sample_id: (32,)}
  Sample 2: {target: (32,128), variate_id: (32,), time_id: (32,), sample_id: (32,)}

Step 6: PackCollate (32 + 32 = 64)
  Batch: {
    target: (1, 64, 128),
    variate_id: (1, 64) = [0,0,...,0,0,0,...,0],
    time_id: (1, 64) = [0,1,...,31,0,1,...,31],
    sample_id: (1, 64) = [1,1,...,1,2,2,...,2],  ← 핵심!
  }

Step 7: PackedStdScaler
  target: (1, 64, 128) - 샘플별로 정규화

Step 8: MultiInSizeLinear
  (1, 64, 128) → (1, 64, 768)

Step 9: Mask Fill
  prediction_mask 위치를 mask_embedding으로 대체

Step 10: Transformer Encoder (24 layers)
  (1, 64, 768) → (1, 64, 768)
  - Packed attention mask로 샘플 분리
  - GQA로 효율적 attention
  - RoPE + Binary bias

Step 11: Output Projection
  (1, 64, 768) → (1, 64, num_params)
  - 분포 파라미터 생성
  - Rescale
  - Distribution 객체
```

---

## 핵심 포인트 정리

### 1. Patchify
```
512 timesteps → 32 patches (각 16 timesteps)
= 연속된 16개 timestep을 하나의 토큰으로
```

### 2. Pad to max
```
이유: 다양한 patch_size 지원
- 어떤 샘플: patch_size=8
- 어떤 샘플: patch_size=16
- 어떤 샘플: patch_size=32

→ 모두 max_patch_size=128로 통일
→ 실제 크기는 patch_size 필드에 기록
```

### 3. Add Features
```
variate_id: 어느 변수인지 (0, 1, 2, ...)
time_id: 몇 번째 패치인지 (0, 1, 2, ..., 31)
sample_id: 어느 샘플인지 (1, 2, 3, ...)

→ numpy array로 추가 (nn.Parameter 아님!)
→ 나중에 embedding layer 통과
```

### 4. sample_id의 역할
```
Collation 시 여러 샘플이 하나의 batch로 합쳐짐

sample_id로 구분:
- Sample 1의 토큰: sample_id=1
- Sample 2의 토큰: sample_id=2

Attention mask 생성:
- sample_id가 같은 토큰끼리만 attention
- 다른 샘플 간 간섭 방지!
```

### 5. 2개 시계열 처리
```
Transform 단계: 각각 독립적 처리
  Sample 1: (1, 512) → (1, 32, 128)
  Sample 2: (1, 512) → (1, 32, 128)

Collation 단계: 하나의 batch로 합침
  Batch: (1, 64, 128)
  sample_id: [1,1,...,1,2,2,...,2]

Model 단계: sample_id로 분리해서 처리
  - Scaling: 샘플별로 독립적
  - Attention: 샘플별로 독립적
  - Output: 각 샘플의 예측 생성
```

---

## 다변량 예시 (OHLCV 5개 변수)

```
원본: (5, 512)
  ↓ Patchify
(5, 32, 16)
  ↓ Pad
(5, 32, 128)
  ↓ Add Features
variate_id: (5, 32)
[[0,0,...,0],    # Open
 [1,1,...,1],    # High
 [2,2,...,2],    # Low
 [3,3,...,3],    # Close
 [4,4,...,4]]    # Volume

time_id: (5, 32)
[[0,1,2,...,31],
 [0,1,2,...,31],
 [0,1,2,...,31],
 [0,1,2,...,31],
 [0,1,2,...,31]]

  ↓ Flatten
(160, 128)  # 5×32 = 160 tokens
variate_id: (160,) = [0,0,...,0,1,1,...,1,2,2,...,2,3,3,...,3,4,4,...,4]
                      └─ 32 ─┘ └─ 32 ─┘ └─ 32 ─┘ └─ 32 ─┘ └─ 32 ─┘
time_id: (160,) = [0,1,...,31,0,1,...,31,0,1,...,31,0,1,...,31,0,1,...,31]
```

이제 완전히 이해되셨나요? 🎯
