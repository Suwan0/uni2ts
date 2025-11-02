# Multi Patch Size와 Non-overlapping Patches 설명

실제 코드를 기반으로 multi patch size projection과 non-overlapping patch의 개념을 설명합니다.

## 1. Non-overlapping Patches란?

### 1.1 개념

Non-overlapping(겹치지 않는) patches는 **각 timestep이 정확히 하나의 patch에만 속한다**는 의미입니다.

**예시**: 512개의 timesteps를 patch_size=16으로 나누면:
```
Original: [t0, t1, t2, ..., t15, t16, t17, ..., t31, ..., t511]
          └─── patch 0 ───┘ └─── patch 1 ───┘      └─ patch 31 ─┘

Patch 0: [t0, t1, t2, ..., t15]
Patch 1: [t16, t17, t18, ..., t31]
...
Patch 31: [t496, t497, ..., t511]
```

각 timestep은 정확히 하나의 patch에만 속합니다. 예를 들어, t15는 patch 0에만 속하고, t16은 patch 1에만 속합니다.

**Overlapping이 있었다면** (참고용):
```
Patch 0: [t0, t1, t2, ..., t15]
Patch 1: [t8, t9, t10, ..., t23]  <- t8~t15가 patch 0과 중복
```

하지만 Moirai는 non-overlapping 방식을 사용합니다.

### 1.2 실제 구현 (src/uni2ts/transform/patch.py:151-159)

```python
def _patchify_arr(
    self, arr: Num[np.ndarray, "var time*patch"], patch_size: int
) -> Num[np.ndarray, "var time max_patch"]:
    assert arr.shape[-1] % patch_size == 0
    # Step 1: Reshape to create non-overlapping patches
    arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)
    # Step 2: Pad to max_patch_size
    pad_width = [(0, 0) for _ in range(arr.ndim)]
    pad_width[-1] = (0, self.max_patch_size - patch_size)
    arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
    return arr
```

**구체적 예시**:
```python
# Input
arr.shape = (1, 512)  # 1 variable, 512 timesteps
patch_size = 16

# Step 1: rearrange (non-overlapping split)
arr = rearrange(arr, "... (time patch) -> ... time patch", patch=16)
# arr.shape = (1, 32, 16)  # 32 patches, each with 16 consecutive timesteps

# Step 2: pad to max_patch_size=128
pad_width = [(0, 0), (0, 0), (0, 112)]  # 112 = 128 - 16
arr = np.pad(arr, pad_width, constant_values=0)
# arr.shape = (1, 32, 128)
# arr[0, 0, :] = [t0, t1, ..., t15, 0, 0, ..., 0]  <- 첫 16개만 실제 값, 나머지 112개는 padding
```

**핵심**: `(time patch)` → `time patch`로 reshape하면 자동으로 non-overlapping이 됩니다.

---

## 2. Multi Patch Size란?

### 2.1 개념

Moirai는 **하나의 모델이 여러 patch size를 지원**합니다:
- Supported patch sizes: `(8, 16, 32, 64, 128)`
- 각 데이터 샘플마다 **하나의 patch size가 랜덤 선택**됩니다
- 같은 배치 내 다른 샘플은 다른 patch size를 가질 수 있습니다

**왜 여러 patch size를 사용하나?**
1. **다양한 시간 해상도**:
   - Hourly data → 큰 patch size (32-64)
   - Daily data → 중간 patch size (16-32)
   - Monthly data → 작은 patch size (8-16)

2. **Robustness**:
   - 학습 시 여러 patch size로 학습하면 더 robust한 모델이 됩니다
   - 추론 시에도 유연하게 다양한 길이의 시계열을 처리할 수 있습니다

### 2.2 Patch Size 선택 과정 (src/uni2ts/transform/patch.py:85-120)

```python
@dataclass
class GetPatchSize(Transformation):
    min_time_patches: int
    target_field: str = "target"
    patch_sizes: tuple[int, ...] | list[int] | range = (8, 16, 32, 64, 128)
    patch_size_constraints: PatchSizeConstraints = DefaultPatchSizeConstraints()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]
        constraints = self.patch_size_constraints(freq)

        # 데이터 길이 기반으로 최대 patch size 결정
        target: list[UnivarTimeSeries] = data_entry[self.target_field]
        length = target[0].shape[0]
        patch_size_ceil = length // self.min_time_patches

        # 유효한 patch size 후보 필터링
        if isinstance(self.patch_sizes, (tuple, list)):
            patch_size_candidates = [
                patch_size
                for patch_size in self.patch_sizes
                if (patch_size in constraints) and (patch_size <= patch_size_ceil)
            ]

        # 랜덤 선택
        data_entry["patch_size"] = np.random.choice(patch_size_candidates)
        return data_entry
```

**DefaultPatchSizeConstraints** (src/uni2ts/transform/patch.py:57-74):
```python
class DefaultPatchSizeConstraints(PatchSizeConstraints):
    DEFAULT_RANGES = {
        "S": (64, 128),  # 512s = 8.53min, 4096s = 68.26min
        "T": (32, 128),  # 64min = 1.07h, 512min = 8.53h
        "H": (32, 64),   # 128h = 5.33days
        "D": (16, 32),   # Daily data
        "B": (16, 32),   # Business days
        "W": (16, 32),   # Weekly
        "M": (8, 32),    # Monthly
        "Q": (1, 8),     # Quarterly
        "Y": (1, 8),     # Yearly
        "A": (1, 8),     # Annual
    }
```

**예시 1**: Daily 데이터, 길이 512
```python
freq = "D"
constraints = (16, 32)  # Daily는 16~32 사용 가능
length = 512
min_time_patches = 8
patch_size_ceil = 512 // 8 = 64

patch_sizes = (8, 16, 32, 64, 128)
candidates = [16, 32]  # 8은 constraints 미충족, 64는 constraints 초과, 128은 constraints 초과
selected = np.random.choice([16, 32])  # 16 또는 32 중 랜덤
```

**예시 2**: Hourly 데이터, 길이 256
```python
freq = "H"
constraints = (32, 64)
length = 256
min_time_patches = 8
patch_size_ceil = 256 // 8 = 32

patch_sizes = (8, 16, 32, 64, 128)
candidates = [32]  # 32만 constraints와 patch_size_ceil 모두 만족
selected = 32
```

---

## 3. Multi Patch Size Projection Layers

### 3.1 문제: 여러 patch size를 어떻게 처리할까?

각 샘플이 다른 patch size를 가질 수 있는데, 모델은 어떻게 이를 처리할까요?

**Naive 방법** (비효율적):
```python
# 각 patch size마다 별도의 Linear layer
self.proj_8 = nn.Linear(8, 768)
self.proj_16 = nn.Linear(16, 768)
self.proj_32 = nn.Linear(32, 768)
self.proj_64 = nn.Linear(64, 768)
self.proj_128 = nn.Linear(128, 768)

# Forward
if patch_size == 8:
    return self.proj_8(x)
elif patch_size == 16:
    return self.proj_16(x)
...
```

**문제점**:
1. Batch 내에서 다른 patch size가 섞여 있으면 처리 불가
2. Conditional logic이 많아 비효율적

### 3.2 실제 구현: MultiInSizeLinear (src/uni2ts/module/ts_embed.py:37-102)

Moirai는 **단일 tensor + mask** 방식으로 모든 patch size를 동시에 처리합니다.

```python
class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],  # (8, 16, 32, 64, 128)
        out_features: int,                 # 768
        bias: bool = True,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        # 모든 patch size의 weight를 하나의 tensor에 저장
        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls))
            )
        )
        # Shape: (5, 768, 128)
        # 5 = number of patch sizes
        # 768 = output dimension
        # 128 = max patch size

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features))
            )
        # Shape: (5, 768)

        # Mask: 각 patch size에 대해 유효한 input dimension만 1로 표시
        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                "num_feats max_feat -> num_feats 1 max_feat",
            ),
        )
        # mask[0] = [1,1,1,1,1,1,1,1, 0,0,0,...,0]  # 첫 8개만 1 (patch_size=8)
        # mask[1] = [1,1,1,...,1, 0,0,...,0]        # 첫 16개만 1 (patch_size=16)
        # mask[2] = [1,1,1,...,1, 0,0,...,0]        # 첫 32개만 1 (patch_size=32)
        # mask[3] = [1,1,1,...,1, 0,0,...,0]        # 첫 64개만 1 (patch_size=64)
        # mask[4] = [1,1,1,...,1]                   # 전체 128개 1 (patch_size=128)
```

**Forward Pass**:
```python
def forward(
    self,
    x: Float[torch.Tensor, "*batch max_feat"],      # (batch, seq, 128)
    in_feat_size: Int[torch.Tensor, "*batch"],      # (batch, seq) - 각 위치의 patch_size
) -> Float[torch.Tensor, "*batch out_feat"]:        # (batch, seq, 768)
    out = 0

    # 모든 patch size에 대해 iteration
    for idx, feat_size in enumerate(self.in_features_ls):  # [8, 16, 32, 64, 128]
        weight = self.weight[idx] * self.mask[idx]  # (768, 128), 유효한 부분만 남김
        bias = self.bias[idx] if self.bias is not None else 0  # (768,)

        # 현재 patch_size가 feat_size와 일치하는 위치만 선택
        out = out + (
            torch.eq(in_feat_size, feat_size).unsqueeze(-1)  # (batch, seq, 1)
            * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)  # (batch, seq, 768)
        )

    return out
```

### 3.3 구체적 예시

**시나리오**: Batch size=2, seq_len=각 160 (5 variates × 32 patches)

```python
# Batch 구성
# Sample 1: patch_size=16 (160개 위치 모두)
# Sample 2: patch_size=32 (160개 위치 모두)

x.shape = (2, 160, 128)  # 모든 patch는 128로 padding됨
in_feat_size.shape = (2, 160)
# in_feat_size[0, :] = [16, 16, 16, ..., 16]  # Sample 1의 모든 위치
# in_feat_size[1, :] = [32, 32, 32, ..., 32]  # Sample 2의 모든 위치

# Forward pass
out = 0

# Iteration 0: feat_size=8
weight = self.weight[0] * self.mask[0]  # (768, 128), 첫 8개만 유효
torch.eq(in_feat_size, 8).shape = (2, 160)
# [[False, False, ..., False],  # Sample 1은 모두 16이므로 매칭 안됨
#  [False, False, ..., False]]  # Sample 2는 모두 32이므로 매칭 안됨
# → 이 iteration은 out에 0을 더함 (기여 없음)

# Iteration 1: feat_size=16
weight = self.weight[1] * self.mask[1]  # (768, 128), 첫 16개만 유효
torch.eq(in_feat_size, 16).shape = (2, 160)
# [[True, True, ..., True],    # Sample 1의 모든 위치 매칭!
#  [False, False, ..., False]] # Sample 2는 매칭 안됨
mask = torch.eq(in_feat_size, 16).unsqueeze(-1)  # (2, 160, 1)
projection = einsum(weight, x, "out inp, ... inp -> ... out")  # (2, 160, 768)
# Sample 1의 위치들만 projection 값이 더해지고, Sample 2는 0이 더해짐
out = out + mask * projection

# Iteration 2: feat_size=32
weight = self.weight[2] * self.mask[2]  # (768, 128), 첫 32개만 유효
torch.eq(in_feat_size, 32).shape = (2, 160)
# [[False, False, ..., False], # Sample 1은 매칭 안됨
#  [True, True, ..., True]]    # Sample 2의 모든 위치 매칭!
mask = torch.eq(in_feat_size, 32).unsqueeze(-1)  # (2, 160, 1)
projection = einsum(weight, x, "out inp, ... inp -> ... out")  # (2, 160, 768)
# Sample 2의 위치들만 projection 값이 더해지고, Sample 1은 0이 더해짐
out = out + mask * projection

# Iteration 3-4: feat_size=64, 128
# 둘 다 매칭 안되므로 0 더해짐

# 최종 결과
out.shape = (2, 160, 768)
# out[0, :, :]: Sample 1의 모든 위치에 weight[1] (patch_size=16용) 적용된 결과
# out[1, :, :]: Sample 2의 모든 위치에 weight[2] (patch_size=32용) 적용된 결과
```

### 3.4 핵심 메커니즘

1. **모든 patch size의 weight를 단일 tensor에 저장**: `(num_sizes, out_features, max_size)`
2. **Mask로 유효한 input dimension만 활성화**:
   - patch_size=16이면 첫 16개 dimension만 weight 적용
   - 나머지 112개는 mask로 0 곱해짐
3. **torch.eq로 현재 patch_size 매칭**:
   - 각 iteration에서 현재 patch_size와 일치하는 위치만 선택
   - 일치하는 위치만 projection 결과가 더해짐
4. **Batch 내 다른 patch size 동시 처리**:
   - Sample 1은 iteration 1에서 처리
   - Sample 2는 iteration 2에서 처리
   - 모두 같은 forward pass 내에서 처리됨

---

## 4. MultiOutSizeLinear (Output Projection)

비슷한 방식으로 output projection도 처리됩니다.

**src/uni2ts/module/ts_embed.py:175-243**:
```python
class MultiOutSizeLinear(nn.Module):
    def __init__(
        self,
        in_features: int,                    # 768
        out_features_ls: tuple[int, ...],    # (8, 16, 32, 64, 128)
        dim: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_ls = out_features_ls
        self.dim = dim

        # 모든 patch size의 weight 저장
        self.weight = nn.Parameter(
            torch.empty(
                (len(out_features_ls), max(out_features_ls), in_features)
            )
        )
        # Shape: (5, 128, 768)

        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(out_features_ls), max(out_features_ls)))
            )
        # Shape: (5, 128)

        # Mask
        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(out_features_ls), torch.as_tensor(out_features_ls)),
                "num_feats max_feat -> num_feats max_feat 1",
            ),
        )

    def forward(
        self,
        x: Float[torch.Tensor, "*batch in_feat"],       # (batch, seq, 768)
        out_feat_size: Int[torch.Tensor, "*batch"],     # (batch, seq)
    ) -> Float[torch.Tensor, "*batch max_feat"]:        # (batch, seq, 128)
        out = 0
        for idx, feat_size in enumerate(self.out_features_ls):
            weight = self.weight[idx] * self.mask[idx]  # (128, 768), 유효한 출력만
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(out_feat_size, feat_size // self.dim).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out
```

**동작 방식**:
- Input: `(batch, seq, 768)`
- Output: `(batch, seq, 128)` - 모든 샘플이 max_patch_size로 padding됨
- 각 위치의 patch_size에 맞는 weight matrix 적용
- 출력도 padding되어 128 dimension이지만, 실제 유효한 값은 patch_size만큼

---

## 5. 전체 흐름 요약

### 5.1 Training 시

```python
# 1. Patch size 랜덤 선택 (데이터마다 다를 수 있음)
GetPatchSize()
# Sample 1: patch_size=16
# Sample 2: patch_size=32

# 2. Patchify (non-overlapping)
Patchify(max_patch_size=128)
# Sample 1: (512,) → (32, 16) → pad → (32, 128)
# Sample 2: (512,) → (16, 32) → pad → (16, 128)

# 3. Flatten
# Sample 1: (5 vars, 32 patches, 128) → (160, 128)
# Sample 2: (5 vars, 16 patches, 128) → (80, 128)

# 4. Collate (batch)
# After padding: (2, 160, 128)
# patch_size tensor: (2, 160) with [16, 16, ..., 32, 32, ...]

# 5. Forward pass
model.in_proj(x, patch_size)
# MultiInSizeLinear automatically handles different patch sizes in the batch
# Output: (2, 160, 768)

# ... Transformer ...

# 6. Output projection
model.param_proj(reprs, patch_size)
# MultiOutSizeLinear handles different patch sizes
# Output: (2, 160, 128)
```

### 5.2 왜 이렇게 설계했나?

1. **Non-overlapping patches**:
   - 메모리 효율적: 각 timestep이 한 번만 처리됨
   - 계산 효율적: Overlapping이 없어 중복 계산 없음
   - 간단한 구현: einops rearrange로 쉽게 구현

2. **Multi patch size**:
   - **Flexibility**: 다양한 frequency와 길이의 시계열 처리
   - **Robustness**: 학습 시 다양한 resolution으로 학습
   - **Efficiency**: 하나의 모델로 모든 patch size 처리

3. **Single tensor + mask 방식**:
   - **Batch 처리**: 다른 patch size를 가진 샘플들을 같은 batch에서 처리
   - **메모리 효율**: 각 patch size마다 별도 layer 불필요
   - **학습 효율**: 모든 patch size의 weight가 동시에 학습됨

---

## 6. 실제 사용 예시

### Example 1: Daily 주가 데이터 (1년 = 252 trading days)

```python
# Input: (252,) - 252 trading days
freq = "D"
constraints = (16, 32)

# GetPatchSize
min_time_patches = 8
patch_size_ceil = 252 // 8 = 31
candidates = [16]  # 32는 patch_size_ceil 초과
patch_size = 16

# Patchify (non-overlapping)
# 252 // 16 = 15.75... → crop to 240 (15 * 16)
# (240,) → reshape → (15, 16) → pad → (15, 128)
# [d0, d1, ..., d15 | d16, d17, ..., d31 | ... | d224, ..., d239]
# └───── patch 0 ────┘ └───── patch 1 ────┘       └──── patch 14 ────┘

# 각 patch는 16일의 연속된 데이터를 담고 있음
# Patch 0: 첫 16일
# Patch 1: 다음 16일 (overlap 없음)
# ...
```

### Example 2: Hourly 센서 데이터 (1주 = 168 hours)

```python
# Input: (168,) - 1 week hourly
freq = "H"
constraints = (32, 64)

# GetPatchSize
min_time_patches = 4
patch_size_ceil = 168 // 4 = 42
candidates = [32]  # 64는 patch_size_ceil 초과
patch_size = 32

# Patchify (non-overlapping)
# 168 // 32 = 5.25 → crop to 160 (5 * 32)
# (160,) → reshape → (5, 32) → pad → (5, 128)
# 각 patch는 32시간 = 1.33일의 데이터

# MultiInSizeLinear
# weight[2] (patch_size=32용) 사용
# mask[2]로 첫 32개 dimension만 활성화
# Projection: (5, 32) → (5, 768)
```

---

## 참고: 코드 위치

- **Patchify**: `src/uni2ts/transform/patch.py:124-159`
- **GetPatchSize**: `src/uni2ts/transform/patch.py:78-120`
- **MultiInSizeLinear**: `src/uni2ts/module/ts_embed.py:37-111`
- **MultiOutSizeLinear**: `src/uni2ts/module/ts_embed.py:175-251`
- **MoiraiModule**: `src/uni2ts/model/moirai/module.py:82-199`
- **DefaultPatchSizeConstraints**: `src/uni2ts/transform/patch.py:57-74`
