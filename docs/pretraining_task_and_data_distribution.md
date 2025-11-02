# Moirai Pre-training: Task Distribution과 Data Distribution 설명

실제 코드를 기반으로 Moirai의 pre-training 개념을 설명합니다.

## 핵심 용어 정리

### Lookback vs Horizon

**Lookback (Context Window)**:
- 모델이 **과거 데이터를 보는 구간**
- 이 구간의 데이터는 모델이 "읽을 수" 있음
- 예측을 위한 입력으로 사용됨

**Horizon (Prediction Window)**:
- 모델이 **예측해야 하는 미래 구간**
- 이 구간의 실제 값은 모델에게 숨겨짐 (masked)
- 모델은 이 구간의 값을 예측해야 함

**시각화**:
```
전체 시계열 데이터:
[t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]
 └────── lookback ──────┘ └─── horizon ───┘
 (context=0~5, 6 steps)   (prediction=6~9, 4 steps)

모델 입력:
- t0~t5: 실제 값 사용 (observed)
- t6~t9: MASKED (learnable mask embedding으로 대체)

모델 목표:
- t6~t9의 값을 예측
```

---

## 1. Pre-training 목표

Moirai의 pre-training은 **mixture distribution의 log-likelihood를 최적화**하는 것입니다:

```python
# Equation (1) from paper
max_θ E_{(Y,Z)~p(D), (t,l,h)~p(T|D)} [ log p_θ(Y_{t+l:t+l+h} | Y_{t:t+l}, Z) ]

where:
- Y: time series data
- Z: metadata (frequency, domain, etc.)
- t: start time
- l: lookback length (context)
- h: horizon length (prediction)
- p(D): data distribution
- p(T|D): task distribution
```

**핵심**: 고정된 (context_length, prediction_length) 대신, **다양한 조합**에서 학습합니다.

---

## 2. Data Distribution p(D)

### 2.1 개념

LOTSA 데이터셋은 **dataset of datasets**입니다:
- 여러 domain (finance, weather, energy, traffic, etc.)
- 여러 frequency (hourly, daily, monthly, etc.)
- 각각을 **sub-dataset**이라고 부름

**Data distribution**은 두 단계로 분해됩니다:
```
p(D) = p(Y, Z | D_k) * p(D_k)

1. p(D_k): sub-dataset 선택 확률
2. p(Y, Z | D_k): 주어진 sub-dataset에서 time series 선택 확률
```

### 2.2 Time Series Distribution (주어진 sub-dataset 내)

**Proportional to length**:

```python
# src/uni2ts/data/dataset.py:72-73
elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
    self.probabilities = indexer.get_proportional_probabilities()
```

Sub-dataset D_k 내에서 time series i를 선택할 확률:

```
p(Y^(i), Z^(i) | D_k) = T_i * 1{i ∈ D_k} / Σ_{j∈D_k} T_j

where:
- T_i: time series i의 길이 (number of observations)
- 1{i ∈ D_k}: indicator function (i가 D_k에 속하면 1, 아니면 0)
```

**예시**:
```python
# Sub-dataset: Daily stock prices
Series 1: 252 days (1 year)
Series 2: 504 days (2 years)
Series 3: 126 days (6 months)

Total: 252 + 504 + 126 = 882 days

Probabilities:
p(Series 1) = 252 / 882 ≈ 0.286 (28.6%)
p(Series 2) = 504 / 882 ≈ 0.571 (57.1%)
p(Series 3) = 126 / 882 ≈ 0.143 (14.3%)

→ 긴 시계열이 더 자주 샘플링됨
```

### 2.3 Sub-dataset Distribution

**문제점**: Domain/frequency imbalance
- 일부 domain은 매우 많은 데이터 (예: finance)
- 일부 frequency는 매우 많음 (예: hourly)
- 단순히 proportional sampling하면 일부 sub-dataset만 학습됨

**해결책: Capping**

```python
# From paper
p(D_k) = ω_k / Σ_i ω_i

where:
ω_k = min(|D_k| / Σ_i |D_i|, ε)
ε = 0.001 (0.1%)
|D_k| = Σ_{i∈D_k} T_i  (total observations in sub-dataset k)
```

**의미**:
- 각 sub-dataset의 **최대 기여도를 0.1%로 제한**
- 매우 큰 sub-dataset도 전체의 0.1%만 차지
- 작은 sub-dataset도 충분히 학습됨

**예시**:
```python
# Before capping
Sub-dataset A (finance-daily): 1,000,000 observations → 80%
Sub-dataset B (weather-hourly): 200,000 observations → 16%
Sub-dataset C (energy-monthly): 50,000 observations → 4%

# After capping (ε = 0.001)
Sub-dataset A: min(0.80, 0.001) = 0.001 → 0.1%
Sub-dataset B: min(0.16, 0.001) = 0.001 → 0.1%
Sub-dataset C: min(0.04, 0.001) = 0.001 → 0.1%

# Re-normalize
Total weight = 0.001 + 0.001 + 0.001 = 0.003
p(D_A) = 0.001 / 0.003 ≈ 33.3%
p(D_B) = 0.001 / 0.003 ≈ 33.3%
p(D_C) = 0.001 / 0.003 ≈ 33.3%

→ 모든 sub-dataset이 균등하게 샘플링됨
```

---

## 3. Task Distribution p(T|D)

### 3.1 개념

**기존 forecasting 패러다임**:
```python
# Fixed setup
context_length = 512
prediction_length = 96

# 항상 동일한 설정으로 학습/평가
```

**Moirai의 접근**:
```python
# Random setup (매 샘플마다 다름)
window_length = random.uniform(min_seq_len, max_seq_len)
prediction_ratio = random.uniform(0.15, 0.5)
prediction_length = round(window_length * prediction_ratio)
context_length = window_length - prediction_length
```

**목표**: 다양한 (context_length, prediction_length) 조합에서 학습하여 **유연한 모델** 만들기

### 3.2 실제 구현

#### Step 1: Patch Size 선택

**src/uni2ts/transform/patch.py:85-120**:

```python
@dataclass
class GetPatchSize(Transformation):
    min_time_patches: int
    patch_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    patch_size_constraints: PatchSizeConstraints = DefaultPatchSizeConstraints()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]
        constraints = self.patch_size_constraints(freq)

        # 데이터 길이 기반으로 최대 patch size 결정
        target = data_entry["target"]
        length = target[0].shape[0]
        patch_size_ceil = length // self.min_time_patches

        # 유효한 patch size 후보 필터링
        patch_size_candidates = [
            patch_size
            for patch_size in self.patch_sizes
            if (patch_size in constraints) and (patch_size <= patch_size_ceil)
        ]

        # 랜덤 선택
        data_entry["patch_size"] = np.random.choice(patch_size_candidates)
        return data_entry
```

**예시**:
```python
# Daily data, length=512, min_time_patches=8
freq = "D"
constraints = (16, 32)  # Daily는 16~32 사용
patch_size_ceil = 512 // 8 = 64

candidates = [16, 32]  # 8은 constraints 미충족, 64는 constraints 초과
patch_size = np.random.choice([16, 32])  # 예: 16 선택
```

#### Step 2: Window Crop (Lookback + Horizon 결정)

**src/uni2ts/transform/crop.py:69-108**:

```python
@dataclass
class PatchCrop(Transformation):
    min_time_patches: int  # minimum number of patches
    max_patches: int       # maximum number of patches (max_seq_len)
    will_flatten: bool = False

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        patch_size = data_entry["patch_size"]
        field = data_entry["target"]
        time = field[0].shape[0]  # original time series length
        nvar = len(field)  # number of variates

        # Random offset so start is not always multiple of patch_size
        offset = np.random.randint(time % patch_size + 1) if self.offset else 0

        # Total available patches
        total_patches = (time - offset) // patch_size

        # Max patches (고려: flatten 시 variate 수)
        max_patches = min(self.max_patches // nvar, total_patches) if self.will_flatten else min(self.max_patches, total_patches)

        # Random window length (in patches)
        num_patches = np.random.randint(self.min_time_patches, max_patches + 1)

        # Random start position
        first = np.random.randint(total_patches - num_patches + 1)

        start = offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop
```

**구체적 예시**:
```python
# 설정
time = 512  # original length
patch_size = 16  # 앞서 선택됨
nvar = 5  # 5 variates
min_time_patches = 2
max_patches = 512  # max_seq_len
will_flatten = True

# Step 1: Offset
offset = np.random.randint(1)  # 512 % 16 = 0, so offset = 0

# Step 2: Total available patches
total_patches = (512 - 0) // 16 = 32 patches

# Step 3: Max patches (flatten 고려)
max_patches = min(512 // 5, 32) = min(102, 32) = 32 patches

# Step 4: Random window length
num_patches = np.random.randint(2, 33)  # 예: 20 patches

# Step 5: Random start
first = np.random.randint(32 - 20 + 1) = np.random.randint(13)  # 예: 5

# Step 6: Boundaries
start = 0 + 5 * 16 = 80
stop = 80 + 20 * 16 = 400

# 결과: time series의 [80:400] 구간 사용 (320 timesteps = 20 patches)
```

**중요**: 이 시점에는 아직 lookback/horizon이 구분되지 않음. 전체 window만 선택됨.

#### Step 3: Prediction Mask 생성 (Lookback vs Horizon 구분)

**src/uni2ts/transform/task.py:54-63**:

```python
@dataclass
class MaskedPrediction(Transformation):
    min_mask_ratio: float  # 0.15
    max_mask_ratio: float  # 0.5

    def _generate_prediction_mask(
        self, target: np.ndarray  # shape: (var, time, patch)
    ) -> np.ndarray:
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)

        # Random mask ratio (prediction proportion)
        mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        mask_length = max(1, round(time * mask_ratio))

        # Mask last mask_length positions
        prediction_mask[:, -mask_length:] = True
        return prediction_mask
```

**구체적 예시**:
```python
# 앞선 예시에서 이어짐
# target.shape = (5, 20, 128)  # 5 vars, 20 patches (after crop), 128 max_patch_size
var, time = 5, 20

# Random mask ratio
mask_ratio = np.random.uniform(0.15, 0.5)  # 예: 0.3
mask_length = max(1, round(20 * 0.3)) = 6 patches

# Prediction mask
prediction_mask = np.zeros((5, 20), dtype=bool)
prediction_mask[:, -6:] = True
# prediction_mask[:, 0:14] = False  (lookback, 14 patches = 224 timesteps)
# prediction_mask[:, 14:20] = True  (horizon, 6 patches = 96 timesteps)
```

**시각화**:
```
Original data: [80:400] (320 timesteps)
After patchify: 20 patches

Patches:    [0, 1, 2, 3, ..., 13, 14, 15, 16, 17, 18, 19]
Mask:       [F, F, F, F, ..., F,  T,  T,  T,  T,  T,  T]
            └──── lookback (14) ────┘ └── horizon (6) ──┘

Lookback: 14 patches * 16 = 224 timesteps
Horizon: 6 patches * 16 = 96 timesteps
```

### 3.3 최종 정리: Task Distribution 요약

```python
# From paper
p(T|D) defines:
- t: start time of window
- l: lookback length (context)
- h: horizon length (prediction)

# 실제 구현
1. Patch size 선택: patch_size ~ constraints & data length
2. Window length 선택: num_patches ~ Uniform(min_patches, max_patches)
3. Window start 선택: first ~ Uniform(0, total_patches - num_patches)
4. Prediction ratio 선택: mask_ratio ~ Uniform(0.15, 0.5)
5. Prediction length: mask_length = round(num_patches * mask_ratio)
6. Context length: context_length = num_patches - mask_length

# 제약조건
- Min sequence length per variate: min_time_patches = 2 (논문에서 언급)
- Max total sequence length: max_patches = 512 (max_seq_len)
- Prediction ratio: [0.15, 0.5]
```

---

## 4. Data Augmentation

### 4.1 Variate Subsampling (Multivariate → 작은 Multivariate)

**목적**: Multivariate time series의 일부 변수만 선택하여 다양성 증가

**src/uni2ts/transform/resample.py:59-66**:

```python
@dataclass
class SampleDimension(Transformation):
    max_dim: int  # maximum number of variates
    sampler: Sampler = get_sampler("uniform")

    def _process(
        self, data_entry: dict[str, Any], field: str, total_field_dim: int
    ) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]
        rand_idx = np.random.permutation(len(arr))
        field_max_dim = (self.max_dim * len(arr)) // total_field_dim
        n = self.sampler(min(len(arr), field_max_dim))
        return [arr[idx] for idx in rand_idx[:n]]
```

**예시**:
```python
# Original: 10 variates
target = [ts0, ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9]

# SampleDimension
max_dim = 128
n = np.random.randint(1, min(10, 128) + 1)  # 예: 5

# Random permutation and select
rand_idx = [3, 7, 1, 9, 2, 5, 0, 8, 4, 6]
selected = [ts3, ts7, ts1, ts9, ts2]  # 첫 5개 선택

# Result: 5 variates (원래 10에서 감소)
```

### 4.2 Constructing Multivariate from Univariate

**목적**: Univariate time series들을 합쳐서 Multivariate 생성

**src/uni2ts/data/dataset.py:157-162**:

```python
class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    def __init__(
        self,
        max_ts: int,  # maximum number of time series to combine
        combine_fields: tuple[str, ...],
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        ...

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        # Sample number of time series to combine
        n_series = self.sampler(min(self.num_ts, self.max_ts))

        # Select random time series (excluding current idx)
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)

        # Combine
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples
```

**Beta-binomial Sampler** (src/uni2ts/common/sampler.py:33-41):

```python
def beta_binomial_sampler(
    n: int, a: float = 1, b: float = 1
) -> int:
    p = np.random.beta(a, b)
    return np.random.binomial(n - 1, p) + 1
```

**Beta-binomial 분포**:
```python
# From paper
n = 128  (maximum number of variates)
a = 2
b = 5

# Mean
E[X] = n * a / (a + b) = 128 * 2 / 7 ≈ 36.57 ≈ 37

# Distribution shape
- a=2, b=5 → left-skewed (작은 값 선호)
- 대부분의 샘플은 10~60 variates 정도
- 가끔 100+ variates도 가능
```

**예시**:
```python
# Sub-dataset with univariate time series
Series 0: [t0, t1, t2, ..., t99]  (100 timesteps, 1 variate)
Series 1: [s0, s1, s2, ..., s99]  (100 timesteps, 1 variate)
Series 2: [r0, r1, r2, ..., r99]  (100 timesteps, 1 variate)
...

# MultiSampleTimeSeriesDataset
max_ts = 128
n_series = beta_binomial_sampler(128, a=2, b=5)  # 예: 25

# Randomly select 25 time series
selected_indices = [0, 5, 12, 23, ..., 99]  # 25개

# Combine into multivariate
combined_target = [
    Series[0],  # [t0, t1, ..., t99]
    Series[5],  # [u0, u1, ..., u99]
    Series[12], # [v0, v1, ..., v99]
    ...
    Series[99], # [z0, z1, ..., z99]
]  # shape: (25, 100) - 25 variates, 100 timesteps

# Result: Multivariate time series with 25 variates
```

**중요**: 이 방식은 **aligned time series**가 필요합니다 (같은 start/end date).

---

## 5. 전체 Pre-training Pipeline 흐름

### 5.1 Training Transform Pipeline

**src/uni2ts/model/moirai/pretrain.py:382-489**:

```python
def default_train_transform():
    return (
        # 1. Variate subsampling
        SampleDimension(
            max_dim=128,
            fields=("target",),
        )
        # 2. Patch size 선택
        + GetPatchSize(
            min_time_patches=2,
            patch_sizes=(8, 16, 32, 64, 128),
        )
        # 3. Window crop (lookback + horizon 전체)
        + PatchCrop(
            min_time_patches=2,
            max_patches=512,
            will_flatten=True,
        )
        # 4. Pack fields
        + PackFields(output_field="target", fields=("target",))
        # 5. Add observed mask (missing value 표시)
        + AddObservedMask(fields=("target",))
        # 6. Impute missing values
        + ImputeTimeSeries(imputation_method=DummyValueImputation(value=0.0))
        # 7. Patchify (non-overlapping patches)
        + Patchify(max_patch_size=128, fields=("target", "observed_mask"))
        # 8. Add variate_id
        + AddVariateIndex(
            variate_id_field="variate_id",
            max_dim=128,
            randomize=True,
        )
        # 9. Add time_id
        + AddTimeIndex(time_id_field="time_id")
        # 10. Generate prediction mask (lookback vs horizon 구분)
        + MaskedPrediction(
            min_mask_ratio=0.15,
            max_mask_ratio=0.5,
        )
        # 11. Flatten (variate × time → single sequence)
        + FlatPackCollection(field="variate_id")
        + FlatPackCollection(field="time_id")
        + FlatPackCollection(field="prediction_mask")
        + FlatPackFields(output_field="target")
        # 12. Sequencify patch_size
        + SequencifyField(field="patch_size", target_field="target")
        # 13. Select output fields
        + SelectFields(fields=seq_fields)
    )
```

### 5.2 구체적 예시: End-to-End

```python
# ========== Step 0: Raw Data ==========
# Sub-dataset: Daily stock prices, 5 stocks (univariate each)
Series 0: [10.2, 10.5, 10.3, ..., 15.7]  # 1000 days
Series 1: [20.1, 20.3, 20.2, ..., 25.4]  # 1000 days
Series 2: [30.5, 30.7, 30.6, ..., 35.9]  # 1000 days
Series 3: [40.8, 40.9, 40.7, ..., 45.2]  # 1000 days
Series 4: [50.3, 50.5, 50.4, ..., 55.8]  # 1000 days

# ========== Step 1: Sample Sub-dataset ==========
# p(D_k) with capping ε=0.001
selected_subdataset = "stock-daily"

# ========== Step 2: Sample Time Series (or Combine) ==========
# MultiSampleTimeSeriesDataset
n_series = beta_binomial_sampler(min(5, 128), a=2, b=5)  # 예: 3
selected_series = [Series 0, Series 2, Series 4]

# Combine into multivariate
target = [
    Series[0],  # (1000,)
    Series[2],  # (1000,)
    Series[4],  # (1000,)
]  # shape: (3, 1000)

# ========== Step 3: SampleDimension ==========
# 이미 3 variates이므로 변화 없음 (max_dim=128보다 작음)
# target.shape = (3, 1000)

# ========== Step 4: GetPatchSize ==========
freq = "D"
constraints = (16, 32)
length = 1000
min_time_patches = 2
patch_size_ceil = 1000 // 2 = 500

candidates = [16, 32]
patch_size = 16  # 랜덤 선택

# ========== Step 5: PatchCrop ==========
patch_size = 16
nvar = 3
max_patches = 512
will_flatten = True

total_patches = 1000 // 16 = 62
max_patches = min(512 // 3, 62) = min(170, 62) = 62
num_patches = np.random.randint(2, 63)  # 예: 32
first = np.random.randint(62 - 32 + 1)  # 예: 10

start = 10 * 16 = 160
stop = 160 + 32 * 16 = 672

# Crop
target = [
    Series[0][160:672],  # (512,)
    Series[2][160:672],  # (512,)
    Series[4][160:672],  # (512,)
]  # shape: (3, 512)

# ========== Step 6: Patchify ==========
# target.shape = (3, 512)
# Patchify with patch_size=16
target = rearrange(target, "var (time patch) -> var time patch", patch=16)
# target.shape = (3, 32, 16)

# Pad to max_patch_size=128
target = np.pad(target, [(0,0), (0,0), (0, 112)], constant_values=0)
# target.shape = (3, 32, 128)

# ========== Step 7: Add IDs ==========
# AddVariateIndex
variate_id = [0, 1, 2]  # shape: (3,) → broadcast to (3, 32)

# AddTimeIndex
time_id = [0, 1, 2, ..., 31]  # shape: (32,) → broadcast to (3, 32)

# ========== Step 8: MaskedPrediction ==========
mask_ratio = np.random.uniform(0.15, 0.5)  # 예: 0.25
mask_length = round(32 * 0.25) = 8

prediction_mask = np.zeros((3, 32), dtype=bool)
prediction_mask[:, -8:] = True
# prediction_mask.shape = (3, 32)
# prediction_mask[:, 0:24] = False  (lookback)
# prediction_mask[:, 24:32] = True  (horizon)

# ========== Step 9: Flatten ==========
# FlatPackFields
# (3, 32, 128) → (3*32, 128) = (96, 128)

target = rearrange(target, "var time patch -> (var time) patch")
# target.shape = (96, 128)

variate_id = rearrange(variate_id, "var time -> (var time)")
# variate_id = [0,0,0,...,0, 1,1,1,...,1, 2,2,2,...,2]
# variate_id.shape = (96,)

time_id = rearrange(time_id, "var time -> (var time)")
# time_id = [0,1,2,...,31, 0,1,2,...,31, 0,1,2,...,31]
# time_id.shape = (96,)

prediction_mask = rearrange(prediction_mask, "var time -> (var time)")
# prediction_mask.shape = (96,)

# ========== Step 10: Forward Pass ==========
# Input to model
batch = {
    "target": target,  # (96, 128)
    "observed_mask": observed_mask,  # (96, 128)
    "sample_id": sample_id,  # (96,) - 모두 1 (single sample)
    "time_id": time_id,  # (96,)
    "variate_id": variate_id,  # (96,)
    "prediction_mask": prediction_mask,  # (96,)
    "patch_size": patch_size,  # (96,) - 모두 16
}

# Model forward (MoiraiModule)
distr = model(**batch)

# Loss (only on prediction_mask=True positions)
loss = PackedNLLLoss()(
    pred=distr,
    target=target,
    prediction_mask=prediction_mask,
    observed_mask=observed_mask,
    sample_id=sample_id,
    variate_id=variate_id,
)

# Backpropagation
loss.backward()
```

### 5.3 핵심 요약

**Data Distribution**:
1. Sub-dataset 선택 (capping으로 균등화)
2. Time series 선택 (length에 비례) 또는 여러 개 combine
3. Variate subsampling (선택적)

**Task Distribution**:
1. Patch size 랜덤 선택 (frequency 기반 constraints)
2. Window length 랜덤 선택 (min_patches ~ max_patches)
3. Window start 랜덤 선택 (uniformly from available positions)
4. Prediction ratio 랜덤 선택 (0.15 ~ 0.5)
5. 마지막 mask_length patches를 horizon으로 설정

**결과**:
- 매 샘플마다 **다른 조합**의 (context_length, prediction_length)
- 매 샘플마다 **다른 patch_size**
- 매 샘플마다 **다른 number of variates**
- → **Versatile model** 학습됨

---

## 6. Pre-training vs Fine-tuning/Inference 차이

### Pre-training
```python
# Random everything
patch_size ~ constraints & data length
window_length ~ Uniform(min_patches, max_patches)
prediction_ratio ~ Uniform(0.15, 0.5)
num_variates ~ BetaBinomial(128, a=2, b=5)
```

### Fine-tuning
```python
# Fixed setup (but still flexible)
context_length = 512  # user-specified
prediction_length = 96  # user-specified
patch_size = 자동 선택 (frequency 기반)
```

### Inference
```python
# User-provided
context_data = last 512 timesteps
prediction_length = 96  # how far to forecast
patch_size = 자동 선택
```

---

## 7. 참고: 코드 위치

- **PatchCrop**: `src/uni2ts/transform/crop.py:31-108`
- **MaskedPrediction**: `src/uni2ts/transform/task.py:28-86`
- **SampleDimension**: `src/uni2ts/transform/resample.py:30-66`
- **MultiSampleTimeSeriesDataset**: `src/uni2ts/data/dataset.py:127-183`
- **Beta-binomial Sampler**: `src/uni2ts/common/sampler.py:33-41`
- **Train Transform Map**: `src/uni2ts/model/moirai/pretrain.py:358-489`
- **GetPatchSize**: `src/uni2ts/transform/patch.py:78-120`
- **TimeSeriesDataset**: `src/uni2ts/data/dataset.py:49-125`
