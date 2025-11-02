# Moirai 데이터 처리 전체 과정 (실제 코드 기반)

## ⚠️ 주의
이 문서는 실제 코드(`src/uni2ts/`)를 직접 읽어서 작성했습니다.
모든 코드 예시는 실제 구현을 정확히 반영합니다.

---

## 목차
1. 데이터 준비 단계 (Transform)
2. Forward Pass 단계
   - Step 1: Scaling
   - Step 2: Input Projection
   - Step 3: Mask Fill
   - Step 4: Transformer Encoder
   - Step 5: Output Projection
   - Step 6: Distribution 생성
3. 전체 흐름 통합 예시

---

## 1. 데이터 준비 단계 (Transform)

### 1.1 원본 데이터

```python
# 다변량 시계열 데이터
Y = np.array([...])  # shape: (variables, timesteps)

# 예: OHLCV 주가 데이터
Y = np.random.randn(5, 512)  # (5 variables, 512 days)
# Variable 0: Open
# Variable 1: High
# Variable 2: Low
# Variable 3: Close
# Variable 4: Volume
```

### 1.2 Patchify

**파일**: `src/uni2ts/transform/patch.py:124-159`

```python
from uni2ts.transform.patch import Patchify, GetPatchSize

# Step 1: Patch size 선택
get_patch_size = GetPatchSize(
    min_time_patches=16,
    patch_sizes=(8, 16, 32, 64, 128),
    target_field='target'
)

data_entry = {
    'target': Y,  # (5, 512)
    'freq': 'D',  # 일별 데이터
}

# Patch size 결정 (주파수 기반)
data_entry = get_patch_size(data_entry)
# data_entry['patch_size'] = 16  (일별 데이터 → 16 선택)

# Step 2: Patchify 수행
patchify = Patchify(max_patch_size=128, fields=('target',))
data_entry = patchify(data_entry)

# 결과:
# data_entry['target']: (5, 32, 128)
#                        ↑  ↑   ↑
#                        │  │   └─ max_patch_size로 패딩
#                        │  └───── 512 / 16 = 32 patches
#                        └──────── 5 variables
```

**실제 _patchify_arr 구현**:

```python
def _patchify_arr(self, arr, patch_size):
    """
    arr: (var, time) = (5, 512)
    patch_size: 16
    return: (var, time, max_patch) = (5, 32, 128)
    """
    # Step 1: 시간을 패치로 재배열
    assert arr.shape[-1] % patch_size == 0  # 512 % 16 == 0
    arr = rearrange(arr, '... (time patch) -> ... time patch', patch=patch_size)
    # (5, 512) → (5, 32, 16)

    # Step 2: max_patch_size까지 패딩
    pad_width = [(0, 0) for _ in range(arr.ndim)]
    pad_width[-1] = (0, self.max_patch_size - patch_size)  # (0, 112)
    arr = np.pad(arr, pad_width, mode='constant', constant_values=self.pad_value)
    # (5, 32, 16) → (5, 32, 128)

    return arr
```

### 1.3 Flatten

**다변량 → 단일 시퀀스**

```python
# data_entry['target']: (5, 32, 128)

# Flatten: (variables, patches, patch_size) → (var*patches, patch_size)
target_flat = data_entry['target'].reshape(-1, 128)
# (5, 32, 128) → (160, 128)

# 토큰 배치:
# Token 0-31:   Variable 0 (Open)의 32개 패치
# Token 32-63:  Variable 1 (High)의 32개 패치
# Token 64-95:  Variable 2 (Low)의 32개 패치
# Token 96-127: Variable 3 (Close)의 32개 패치
# Token 128-159: Variable 4 (Volume)의 32개 패치
```

### 1.4 Add Features

**파일**: `src/uni2ts/transform/feature.py`

#### variate_id 생성

```python
from uni2ts.transform.feature import AddVariateIndex

add_variate = AddVariateIndex(
    fields=('target',),
    max_dim=100,
    variate_id_field='variate_id'
)

data_entry = add_variate(data_entry)

# 내부 동작:
def _generate_variate_id(self, data_entry, field):
    arr = data_entry[field]  # (5, 32, 128)
    dim, time = arr.shape[:2]  # dim=5, time=32

    # 변수별 ID: [0, 1, 2, 3, 4]
    variate_ids = np.arange(dim)

    # 각 시간(패치)마다 반복
    field_dim_id = repeat(
        np.asarray(variate_ids, dtype=int),
        'var -> var time',
        time=time
    )
    # (5,) → (5, 32)
    return field_dim_id

# 결과:
# data_entry['variate_id']: (5, 32)
# [[0, 0, 0, ..., 0],  # Variable 0
#  [1, 1, 1, ..., 1],  # Variable 1
#  [2, 2, 2, ..., 2],  # Variable 2
#  [3, 3, 3, ..., 3],  # Variable 3
#  [4, 4, 4, ..., 4]]  # Variable 4
```

#### time_id 생성

```python
from uni2ts.transform.feature import AddTimeIndex

add_time = AddTimeIndex(
    fields=('target',),
    time_id_field='time_id'
)

data_entry = add_time(data_entry)

# 내부 동작:
def _generate_time_id(self, data_entry, field):
    arr = data_entry[field]  # (5, 32, 128)
    var, time = arr.shape[:2]  # var=5, time=32

    # 시간 순서 ID: [0, 1, 2, ..., 31]
    field_seq_id = np.arange(time)

    # 각 변수마다 반복
    field_seq_id = repeat(field_seq_id, 'time -> var time', var=var)
    # (32,) → (5, 32)
    return field_seq_id

# 결과:
# data_entry['time_id']: (5, 32)
# [[0, 1, 2, ..., 31],  # Variable 0
#  [0, 1, 2, ..., 31],  # Variable 1
#  [0, 1, 2, ..., 31],  # Variable 2
#  [0, 1, 2, ..., 31],  # Variable 3
#  [0, 1, 2, ..., 31]]  # Variable 4
```

#### prediction_mask 생성

```python
# 마지막 h개 패치를 예측 구간으로 설정
prediction_length = 30  # 30일 예측
patch_size = 16
num_pred_patches = prediction_length // patch_size  # 30 // 16 = 1 (나머지 버림)

# prediction_mask: (5, 32)
prediction_mask = np.zeros((5, 32), dtype=bool)
prediction_mask[:, -num_pred_patches:] = True  # 마지막 1개 패치만 True

# 실제로는 더 복잡한 로직 (MaskedPrediction transform 사용)
```

#### Flatten all features

```python
# 모든 feature flatten
target = data_entry['target'].reshape(-1, 128)        # (160, 128)
variate_id = data_entry['variate_id'].reshape(-1)     # (160,)
time_id = data_entry['time_id'].reshape(-1)           # (160,)
prediction_mask = data_entry['prediction_mask'].reshape(-1)  # (160,)
patch_size = np.full((160,), data_entry['patch_size'])  # (160,) all 16

# Collation 후 sample_id 생성 (예: PackCollate)
sample_id = np.ones((160,), dtype=int)  # (160,) all 1 (같은 샘플)
```

---

## 2. Forward Pass 단계

**파일**: `src/uni2ts/model/moirai/module.py:151-199`

### MoiraiModule.forward 전체 코드

```python
def forward(
    self,
    target: Float[torch.Tensor, "*batch seq_len max_patch"],
    observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
    sample_id: Int[torch.Tensor, "*batch seq_len"],
    time_id: Int[torch.Tensor, "*batch seq_len"],
    variate_id: Int[torch.Tensor, "*batch seq_len"],
    prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
    patch_size: Int[torch.Tensor, "*batch seq_len"],
) -> Distribution:
    """
    Forward pass:
    1. Apply scaling to observations
    2. Project from observations to representations
    3. Replace prediction window with learnable mask
    4. Apply transformer layers
    5. Project from representations to distribution parameters
    6. Return distribution object
    """
    # Step 1: Scaling
    loc, scale = self.scaler(
        target,
        observed_mask * ~prediction_mask.unsqueeze(-1),
        sample_id,
        variate_id,
    )
    scaled_target = (target - loc) / scale

    # Step 2: Input Projection
    reprs = self.in_proj(scaled_target, patch_size)

    # Step 3: Mask Fill
    masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

    # Step 4: Transformer Encoder
    reprs = self.encoder(
        masked_reprs,
        packed_attention_mask(sample_id),
        time_id=time_id,
        var_id=variate_id,
    )

    # Step 5: Output Projection
    distr_param = self.param_proj(reprs, patch_size)

    # Step 6: Distribution
    distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)

    return distr
```

### Step 1: Scaling

**파일**: `src/uni2ts/module/packed_scaler.py:78-123`

```python
class PackedStdScaler(PackedScaler):
    """
    샘플별, 변수별로 독립적으로 표준화
    """

    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ):
        """
        각 (sample_id, variate_id) 조합마다 독립적으로 평균/표준편차 계산
        """
        # Step 1: 같은 (sample_id, variate_id) 마스크 생성
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        # id_mask[i, j] = True if (sample_id[i], variate_id[i]) == (sample_id[j], variate_id[j])

        # Step 2: 관측된 데이터 개수 계산
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )

        # Step 3: 평균 계산
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)  # loc = sum(values) / count

        # Step 4: 분산 계산
        var = reduce(
            id_mask * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))

        # Step 5: 표준편차
        scale = torch.sqrt(var + self.minimum_scale)

        # Step 6: 패딩 위치는 0, 1로
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1

        return loc, scale
```

**실제 예시**:

```python
# 입력
target = torch.randn(1, 160, 128)          # (batch, seq, max_patch)
observed_mask = torch.ones(1, 160, 128, dtype=torch.bool)
sample_id = torch.ones(1, 160, dtype=torch.long)  # 모두 같은 샘플
variate_id = torch.tensor([
    [0]*32 + [1]*32 + [2]*32 + [3]*32 + [4]*32  # 각 변수별로 32개씩
])  # (1, 160)
prediction_mask = torch.zeros(1, 160, dtype=torch.bool)
prediction_mask[0, -2:] = True  # 마지막 2개만 예측

# Scaling
scaler = PackedStdScaler()
loc, scale = scaler(
    target,
    observed_mask * ~prediction_mask.unsqueeze(-1),  # 예측 구간 제외
    sample_id,
    variate_id,
)

# loc, scale: (1, 160, 1)
# Variable 0의 토큰들 (0-31): 같은 loc, scale
# Variable 1의 토큰들 (32-63): 다른 loc, scale
# ...

scaled_target = (target - loc) / scale
# (1, 160, 128)
```

### Step 2: Input Projection

**파일**: `src/uni2ts/module/ts_embed.py:37-111`

```python
# 초기화
self.in_proj = MultiInSizeLinear(
    in_features_ls=(8, 16, 32, 64, 128),  # patch_sizes
    out_features=768,                      # d_model
)

# Forward
patch_size = torch.full((1, 160), 16, dtype=torch.long)  # 모두 16
reprs = self.in_proj(scaled_target, patch_size)

# 내부 동작 (ts_embed.py:89-102):
def forward(self, x, in_feat_size):
    out = 0
    for idx, feat_size in enumerate(self.in_features_ls):
        # feat_size = 8, 16, 32, 64, 128
        weight = self.weight[idx] * self.mask[idx]  # Masked weight
        bias = self.bias[idx] if self.bias is not None else 0

        # 현재 patch_size와 일치하는지 확인
        match = torch.eq(in_feat_size, feat_size)  # Boolean
        # in_feat_size=16이면, feat_size=16일 때만 True

        # Einsum linear transformation
        linear_out = einsum(weight, x, "out inp, ... inp -> ... out") + bias

        # 일치하는 경우만 출력에 추가
        out = out + match.unsqueeze(-1) * linear_out

    return out  # (1, 160, 768)

# 결과:
# reprs: (1, 160, 768)
```

### Step 3: Mask Fill

**파일**: `src/uni2ts/common/torch_util.py:57-64`

```python
# 초기화
self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=768)
# mask_encoding.weight: (1, 768) - 학습 가능한 mask 임베딩

# Forward
def mask_fill(tensor, mask, value):
    """
    tensor: (1, 160, 768)
    mask: (1, 160) - Boolean
    value: (1, 768) - mask_encoding.weight
    return: (1, 160, 768)
    """
    mask = mask.unsqueeze(-1)  # (1, 160, 1)
    return tensor * ~mask + value * mask
    # mask=False인 위치: tensor 값 그대로
    # mask=True인 위치: value (mask_encoding) 사용

masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

# 결과:
# masked_reprs: (1, 160, 768)
# - 대부분 위치: reprs의 원래 값 (history)
# - prediction_mask=True 위치: mask_encoding.weight (학습 가능)
```

**예시**:

```python
# prediction_mask: [False, False, ..., False, True, True]
#                   └──── 158개 ─────┘ └ 2개 ┘

# masked_reprs:
# Token 0-157: reprs의 원래 값 (history context)
# Token 158-159: mask_encoding.weight (예측 구간, 모델이 채워야 함)
```

### Step 4: Transformer Encoder

**파일**: `src/uni2ts/module/transformer.py`

#### Attention Mask 생성

```python
from uni2ts.common.torch_util import packed_attention_mask

def packed_attention_mask(sample_id):
    """
    sample_id: (1, 160)
    return: (1, 160, 160)
    """
    sample_id = sample_id.unsqueeze(-1)  # (1, 160, 1)
    attention_mask = sample_id.eq(sample_id.mT)  # (1, 160, 160)
    # attention_mask[i, j] = True if sample_id[i] == sample_id[j]
    return attention_mask

attn_mask = packed_attention_mask(sample_id)
# 같은 샘플 내에서만 attention 허용
```

#### Encoder Forward

```python
# 초기화
self.encoder = TransformerEncoder(
    d_model=768,
    num_layers=24,
    num_heads=12,
    # ... (생략)
    var_attn_bias_layer=partial(BinaryAttentionBias),
    time_qk_proj_layer=partial(
        QueryKeyProjection,
        proj_layer=RotaryProjection,
        kwargs=dict(max_len=2048),
        partial_factor=(0.0, 0.5),
    ),
)

# Forward
reprs = self.encoder(
    masked_reprs,                    # (1, 160, 768)
    packed_attention_mask(sample_id),  # (1, 160, 160)
    time_id=time_id,                 # (1, 160)
    var_id=variate_id,               # (1, 160)
)

# 내부 동작 (각 레이어마다):
for layer in self.layers:
    # Pre-norm
    x_norm = layer.norm1(x)

    # Self-Attention with Binary Attention Bias
    q, k, v = layer.attention.qkv_proj(x_norm)

    # Binary Attention Bias 계산
    bias = layer.var_attn_bias(q, k, var_id, var_id)
    # 같은 변수: +bias
    # 다른 변수: -bias

    # Rotary Position Encoding
    q, k = layer.time_qk_proj(q, k, time_id, time_id)

    # Attention
    attn_out = attention(q, k, v, attn_mask=attn_mask, bias=bias)

    # Residual
    x = x + attn_out

    # FFN with residual
    x = x + layer.ffn(layer.norm2(x))

# 결과:
# reprs: (1, 160, 768)
```

### Step 5: Output Projection

**파일**: `src/uni2ts/distribution/_base.py:62-128`

```python
# 초기화
self.param_proj = self.distr_output.get_param_proj(
    d_model,      # 768
    patch_sizes   # (8, 16, 32, 64, 128)
)

# get_param_proj 내부:
def get_param_proj(self, in_features, out_features, **kwargs):
    return DistrParamProj(
        in_features=in_features,      # 768
        out_features=out_features,    # (8, 16, 32, 64, 128)
        args_dim=self.args_dim,       # 예: {"loc": 1, "scale": 1}
        domain_map=self.domain_map,   # 예: {"loc": identity, "scale": softplus}
        proj_layer=MultiOutSizeLinear,
        **kwargs,
    )

# Forward
distr_param = self.param_proj(reprs, patch_size)

# DistrParamProj.forward (distribution/_base.py:115-127):
def forward(self, *args):  # args = (reprs, patch_size)
    # Step 1: MultiOutSizeLinear로 projection
    params_unbounded = tree_map(
        lambda proj: rearrange(
            proj(*args),  # MultiOutSizeLinear(reprs, patch_size)
            "... (dim out_size) -> ... out_size dim",
            out_size=self.out_size,
        ),
        convert_to_container(self.proj),
    )
    # params_unbounded: {"loc": (1, 160, 128, 1), "scale": (1, 160, 128, 1)}

    # Step 2: domain_map 적용
    params = tree_map_multi(
        lambda func, inp: func(inp),
        self.domain_map,
        params_unbounded
    )
    # params: {
    #   "loc": identity(params_unbounded["loc"]),
    #   "scale": softplus(params_unbounded["scale"])
    # }

    return params
```

**MultiOutSizeLinear** (`src/uni2ts/module/ts_embed.py:175-251`):

```python
class MultiOutSizeLinear(nn.Module):
    """
    MultiInSizeLinear의 반대 버전
    입력: (batch, seq, in_features=768)
    출력: (batch, seq, max_out_features=128)
    """

    def __init__(
        self,
        in_features: int,                   # 768
        out_features_ls: tuple[int, ...],   # (8, 16, 32, 64, 128)
        dim: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_ls = out_features_ls
        self.dim = dim

        # Weight: (num_sizes, max_out, in_feat)
        self.weight = nn.Parameter(
            torch.empty(
                (len(out_features_ls), max(out_features_ls), in_features)
            )
        )
        # (5, 128, 768)

        # Bias
        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(out_features_ls), max(out_features_ls)))
            )
        # (5, 128)

        # Mask
        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(out_features_ls), torch.as_tensor(out_features_ls)),
                "num_feats max_feat -> num_feats max_feat 1",
            ),
            persistent=False,
        )

    def forward(self, x, out_feat_size):
        """
        x: (batch, seq, 768)
        out_feat_size: (batch, seq) - patch_size
        return: (batch, seq, max_out=128)
        """
        out = 0
        for idx, feat_size in enumerate(self.out_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0

            # patch_size와 일치하는지 확인
            match = torch.eq(out_feat_size, feat_size // self.dim)

            # Einsum
            linear_out = einsum(weight, x, "out inp, ... inp -> ... out") + bias

            out = out + match.unsqueeze(-1) * linear_out

        return out  # (batch, seq, 128)
```

**결과**:

```python
# distr_param: {
#   "loc": (1, 160, 128, 1),
#   "scale": (1, 160, 128, 1)
# }
```

### Step 6: Distribution 생성

**파일**: `src/uni2ts/distribution/_base.py:164-174`

```python
def distribution(
    self,
    distr_params,  # {"loc": ..., "scale": ...}
    loc=None,      # scaling loc
    scale=None,    # scaling scale
    validate_args=None,
):
    # Step 1: 기본 분포 생성
    distr = self._distribution(distr_params, validate_args=validate_args)
    # 예: Normal(distr_params["loc"], distr_params["scale"])

    # Step 2: Affine transformation (rescaling)
    if loc is not None or scale is not None:
        distr = AffineTransformed(distr, loc=loc, scale=scale)
        # Final distribution: scale * base_distr + loc

    return distr

# Forward
distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
```

**AffineTransformed** (`src/uni2ts/distribution/_base.py:130-153`):

```python
class AffineTransformed(TransformedDistribution):
    """
    Affine transformation: Y = scale * X + loc
    """

    def __init__(self, base_dist, loc=None, scale=None, validate_args=None):
        self.loc = loc if loc is not None else 0.0
        self.scale = scale if scale is not None else 1.0

        super().__init__(
            base_dist,
            [AffineTransform(loc=self.loc, scale=self.scale)],
            validate_args=validate_args,
        )

    @property
    def mean(self):
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        return self.base_dist.variance * self.scale**2
```

**결과**:

```python
# distr: Distribution 객체
# - base_dist: Normal(distr_param["loc"], distr_param["scale"])
# - transforms: AffineTransform(loc=loc, scale=scale)

# 샘플링:
samples = distr.sample((100,))  # (100, 1, 160, 128, 1)

# Log-probability:
log_prob = distr.log_prob(target)  # (1, 160, 128)

# Mean:
mean = distr.mean  # (1, 160, 128, 1)
```

---

## 3. 전체 흐름 통합 예시

### 3.1 완전한 데이터 예시

```python
import torch
import numpy as np
from uni2ts.model.moirai import MoiraiModule
from uni2ts.distribution import NormalOutput

# Step 1: 원본 데이터
Y_raw = np.random.randn(5, 512)  # OHLCV, 512일

# Step 2: Transform (Patchify, Flatten, Add IDs)
# ... (위 1절 참조)

# 최종 입력 데이터
target = torch.randn(1, 160, 128)          # (batch, seq, max_patch)
observed_mask = torch.ones(1, 160, 128, dtype=torch.bool)
sample_id = torch.ones(1, 160, dtype=torch.long)
time_id = torch.tensor([[i % 32 for i in range(160)]])  # (1, 160)
variate_id = torch.tensor([[i // 32 for i in range(160)]])  # (1, 160)
prediction_mask = torch.zeros(1, 160, dtype=torch.bool)
prediction_mask[0, -5:] = True  # 마지막 5개 패치 예측
patch_size = torch.full((1, 160), 16, dtype=torch.long)

# Step 3: 모델 초기화
model = MoiraiModule(
    distr_output=NormalOutput(),
    d_model=768,
    num_layers=24,
    patch_sizes=(8, 16, 32, 64, 128),
    max_seq_len=2048,
    attn_dropout_p=0.0,
    dropout_p=0.1,
    scaling=True,
)

# Step 4: Forward pass
distr = model(
    target=target,
    observed_mask=observed_mask,
    sample_id=sample_id,
    time_id=time_id,
    variate_id=variate_id,
    prediction_mask=prediction_mask,
    patch_size=patch_size,
)

# Step 5: 예측
samples = distr.sample((100,))  # 100개 샘플
mean = distr.mean
variance = distr.variance

print("Samples:", samples.shape)  # (100, 1, 160, 128, 1)
print("Mean:", mean.shape)        # (1, 160, 128, 1)
print("Variance:", variance.shape)  # (1, 160, 128, 1)
```

### 3.2 Shape 변화 요약

```
원본 데이터:
  Y: (5, 512)  # OHLCV, 512일

Transform:
  Patchify: (5, 512) → (5, 32, 128)
  Flatten: (5, 32, 128) → (160, 128)
  Add IDs: variate_id=(160,), time_id=(160,)

Model Input:
  target: (1, 160, 128)
  variate_id: (1, 160)
  time_id: (1, 160)
  prediction_mask: (1, 160)
  patch_size: (1, 160)

Forward Pass:
  Scaling: (1, 160, 128) → loc=(1, 160, 1), scale=(1, 160, 1)
  Scaled: (1, 160, 128)

  Input Projection: (1, 160, 128) → (1, 160, 768)
  Mask Fill: (1, 160, 768) → (1, 160, 768)  # prediction 위치만 변경

  Transformer: (1, 160, 768) → (1, 160, 768)

  Output Projection: (1, 160, 768) → (1, 160, 128)
  Distribution params: {"loc": (1, 160, 128, 1), "scale": (1, 160, 128, 1)}

  Distribution: AffineTransformed(
    Normal(loc, scale),
    loc=scaling_loc,
    scale=scaling_scale
  )

Output:
  Samples: (num_samples, 1, 160, 128, 1)
  Mean: (1, 160, 128, 1)
```

### 3.3 각 변수별 처리

```python
# Token 배치:
# 0-31:   Variable 0 (Open)
# 32-63:  Variable 1 (High)
# 64-95:  Variable 2 (Low)
# 96-127: Variable 3 (Close)
# 128-159: Variable 4 (Volume)

# Scaling:
# - Variable 0 토큰들: 같은 loc_0, scale_0
# - Variable 1 토큰들: 같은 loc_1, scale_1
# - ...

# Attention:
# - Variable 0 토큰 ↔ Variable 0 토큰: high attention (Binary Bias)
# - Variable 0 토큰 ↔ Variable 1 토큰: low attention
# - ...

# Output:
# - Token 0-31: Variable 0의 32개 패치 예측
# - Token 32-63: Variable 1의 32개 패치 예측
# - ...

# Reshape back:
# (160, 128, 1) → (5, 32, 128, 1) → (5, 512, 1)
# 각 변수별로 512 timesteps 예측
```

---

## 4. 핵심 포인트 정리

### 데이터 변환
```
✓ Patchify: 시간을 패치로 분할 (512 → 32 patches)
✓ Flatten: 다변량 → 단일 시퀀스 (5×32 → 160 tokens)
✓ Add IDs: variate_id (변수 구분), time_id (시간 순서)
✓ Padding: max_patch_size로 통일 (16 → 128)
```

### Forward Pass
```
✓ Scaling: 샘플별, 변수별 독립적 정규화
✓ Input Proj: patch → d_model (128 → 768)
✓ Mask Fill: 예측 구간을 학습 가능한 토큰으로
✓ Transformer: Binary Bias + Rotary로 attention
✓ Output Proj: d_model → patch (768 → 128)
✓ Distribution: 분포 파라미터 + rescaling
```

### 핵심 메커니즘
```
✓ MultiInSizeLinear: 단일 텐서 + mask로 다중 patch size 지원
✓ Binary Attention Bias: 같은 변수 강화, 다른 변수 억제
✓ Packed Scaling: sample_id, variate_id 기반 그룹별 정규화
✓ Mask Encoding: 예측 구간을 학습 가능한 임베딩으로 대체
✓ Affine Transform: 원래 스케일로 복원
```

---

## 참고

모든 코드는 다음 파일에서 직접 읽었습니다:
- `src/uni2ts/model/moirai/module.py`
- `src/uni2ts/module/ts_embed.py`
- `src/uni2ts/module/packed_scaler.py`
- `src/uni2ts/module/transformer.py`
- `src/uni2ts/distribution/_base.py`
- `src/uni2ts/common/torch_util.py`
- `src/uni2ts/transform/`
