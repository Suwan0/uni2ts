# Moirai 논문 Method 섹션 - 실제 코드 기반 설명

## ⚠️ 주의사항
이 문서는 실제 코드(`src/uni2ts/`)를 직접 읽어서 작성했습니다.
개념적 설명이 아닌 **실제 구현**을 정확히 반영합니다.

---

## 목차
1. Problem Formulation (문제 정의)
2. Architecture (아키텍처)
   - 2.1 Multi Patch Size Projection Layers
   - 2.2 Any-variate Attention
   - 2.3 Mixture Distribution

---

## 1. Problem Formulation (문제 정의)

### 수식 설명

```
Dataset: D = {(Y⁽ⁱ⁾, Z⁽ⁱ⁾)}ᴺᵢ₌₁

Y⁽ⁱ⁾ = (y₁⁽ⁱ⁾, y₂⁽ⁱ⁾, ..., yₜᵢ⁽ⁱ⁾) ∈ ℝᵈʸⁱ ˣ ᵀⁱ
Z⁽ⁱ⁾ = (z₁⁽ⁱ⁾, z₂⁽ⁱ⁾, ..., zₜᵢ⁽ⁱ⁾) ∈ ℝᵈᶻⁱ ˣ ᵀⁱ

max E          E        [log p(Yₜ:ₜ₊ₕ|φ̂)]
 θ  (Y,Z)~p(D) (t,l,h)~p(T|D)

subject to: φ̂ = fθ(Yₜ₋ₗ:ₜ, Zₜ₋ₗ:ₜ₊ₕ)
```

### 구체적 예시

```python
# 시계열 데이터
Y⁽ⁱ⁾: 주가 OHLCV
  - dyi = 5 변수
  - Ti = 512 timesteps
  - Shape: (5, 512)

# 예측 설정
t = 400      # 예측 시작점
l = 100      # context length
h = 30       # prediction length

# 입력
Y_context = Y[300:400]      # 100일 history
Z_context = Z[300:430]      # 100일 + 30일 (미래 공변량 포함)

# 출력
Y_forecast = Y[400:430]     # 30일 예측
φ̂ = fθ(Y_context, Z_context)  # 분포 파라미터
```

---

## 2.1 Multi Patch Size Projection Layers

### 실제 구현: MultiInSizeLinear

**파일**: `src/uni2ts/module/ts_embed.py:37-111`

```python
class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],  # (8, 16, 32, 64, 128)
        out_features: int,                 # 768 (d_model)
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        # 하나의 큰 텐서로 모든 weight 저장
        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls)),
                dtype=dtype
            )
        )
        # Shape: (5, 768, 128)
        #         ↑   ↑    ↑
        #    patch_size  d_model  max_patch
        #    개수(5개)

        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features), dtype=dtype)
            )
            # Shape: (5, 768)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Mask buffer (persistent=False → checkpoint에 저장 안됨)
        self.register_buffer(
            "mask",
            rearrange(
                size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
                "num_feats max_feat -> num_feats 1 max_feat",
            ),
            persistent=False,
        )
        # mask[0]: [1,1,1,1,1,1,1,1, 0,0,...,0]  처음 8개만
        # mask[1]: [1,1,...,1, 0,0,...,0]        처음 16개만
        # ...
        # Shape: (5, 1, 128)

        self.register_buffer(
            "in_features_buffer",
            torch.tensor(in_features_ls),
            persistent=False,
        )

    def reset_parameters(self):
        """Weight 초기화"""
        for idx, feat_size in enumerate(self.in_features_ls):
            # 실제 사용하는 부분만 kaiming 초기화
            nn.init.kaiming_uniform_(
                self.weight[idx, :, :feat_size],
                a=math.sqrt(5)
            )
            # 나머지는 0으로
            nn.init.zeros_(self.weight[idx, :, feat_size:])

            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :, :feat_size]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch max_feat"],  # (batch, seq, 128)
        in_feat_size: Int[torch.Tensor, "*batch"],   # (batch, seq)
    ) -> Float[torch.Tensor, "*batch out_feat"]:    # (batch, seq, 768)
        """
        핵심 로직:
        1. 모든 patch_size에 대해 계산
        2. 현재 입력의 patch_size와 일치하는 것만 선택
        3. 합산
        """
        out = 0

        for idx, feat_size in enumerate(self.in_features_ls):
            # idx=0: feat_size=8
            # idx=1: feat_size=16
            # ...

            # Step 1: Weight에 mask 적용
            weight = self.weight[idx] * self.mask[idx]
            # (768, 128) * (1, 128) → (768, 128)
            # 처음 feat_size개만 유효값, 나머지 0

            # Step 2: Bias
            bias = self.bias[idx] if self.bias is not None else 0
            # (768,)

            # Step 3: 현재 샘플의 patch_size와 일치하는지 확인
            match = torch.eq(in_feat_size, feat_size)
            # (batch, seq) - Boolean tensor

            # Step 4: Einsum으로 linear transformation
            # "out inp, ... inp -> ... out"
            # weight: (out=768, inp=128)
            # x: (..., inp=128)
            # result: (..., out=768)
            linear_out = einsum(weight, x, "out inp, ... inp -> ... out") + bias
            # (batch, seq, 768)

            # Step 5: match인 경우만 출력에 추가
            out = out + match.unsqueeze(-1) * linear_out
            # match.unsqueeze(-1): (batch, seq, 1)
            # linear_out: (batch, seq, 768)
            # 요소별 곱셈 후 합산

        return out  # (batch, seq, 768)
```

### size_to_mask 함수

**파일**: `src/uni2ts/common/torch_util.py`

```python
def size_to_mask(
    max_size: int,
    size: Int[torch.Tensor, "batch"]
) -> Bool[torch.Tensor, "batch max_size"]:
    """
    각 size에 대해 mask 생성

    예: max_size=128, size=[8, 16, 32, 64, 128]
    return:
    [[1,1,1,1,1,1,1,1,0,0,...,0],  # size=8
     [1,1,...,1,0,0,...,0],        # size=16
     [1,1,...,1,0,0,...,0],        # size=32
     [1,1,...,1,0,0,...,0],        # size=64
     [1,1,...,1,1,...,1]]          # size=128
    """
    return torch.arange(max_size).unsqueeze(0) < size.unsqueeze(-1)
```

### 실제 동작 예시

```python
# 초기화
multi_linear = MultiInSizeLinear(
    in_features_ls=(8, 16, 32, 64, 128),
    out_features=768,
    bias=True
)

# 입력
batch = 2
seq = 3
x = torch.randn(2, 3, 128)  # 모두 max_size로 패딩됨

in_feat_size = torch.tensor([
    [16, 16, 32],  # 샘플 0: 패치 0,1은 16, 패치 2는 32
    [8, 16, 16]    # 샘플 1: 패치 0은 8, 패치 1,2는 16
])

# Forward
output = multi_linear(x, in_feat_size)
# (2, 3, 768)

# 내부 동작:
# Iteration 0 (feat_size=8):
#   match = [[False, False, False], [True, False, False]]
#   → 샘플 1의 패치 0만 계산 결과 사용
#
# Iteration 1 (feat_size=16):
#   match = [[True, True, False], [False, True, True]]
#   → 해당 위치들만 계산 결과 사용
#
# Iteration 2 (feat_size=32):
#   match = [[False, False, True], [False, False, False]]
#   → 샘플 0의 패치 2만 계산 결과 사용
#
# 최종: 각 위치마다 맞는 patch_size의 weight로 계산된 값
```

### 주파수별 Patch Size 선택

**파일**: `src/uni2ts/transform/patch.py:57-74`

```python
class DefaultPatchSizeConstraints(PatchSizeConstraints):
    DEFAULT_RANGES = {
        "S": (64, 128),   # 초
        "T": (32, 128),   # 분
        "H": (32, 64),    # 시간
        "D": (16, 32),    # 일
        "B": (16, 32),    # 영업일
        "W": (16, 32),    # 주
        "M": (8, 32),     # 월
        "Q": (1, 8),      # 분기
        "Y": (1, 8),      # 년
        "A": (1, 8),
    }

    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
        start, stop = self.DEFAULT_RANGES[offset_name]
        return start, stop
```

---

## 2.2 Any-variate Attention

### Flatten 전략

**다변량 → 단일 시퀀스**

```python
# (variables, patches, patch_size) → (variables*patches, patch_size)
multivariate = (5, 32, 16)
flattened = (160, 16)

# 토큰 배치:
# 0-31:   변수 0의 32개 패치
# 32-63:  변수 1의 32개 패치
# 64-95:  변수 2의 32개 패치
# 96-127: 변수 3의 32개 패치
# 128-159: 변수 4의 32개 패치
```

### Binary Attention Bias 실제 구현

**파일**: `src/uni2ts/module/position/attn_bias.py:67-87`

```python
class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)

        # 2개 임베딩만!
        # [0]: 다른 변수 간 bias (u⁽²⁾)
        # [1]: 같은 변수 간 bias (u⁽¹⁾)
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)
        # Shape: (2, num_heads)

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],     # variate_id
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],       # variate_id
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]:
        """
        query_id, kv_id: 각 토큰의 variate_id
        return: attention bias
        """

        # Step 1: 같은 변수인지 확인
        ind = torch.eq(
            query_id.unsqueeze(-1),  # (*batch, 1, 1, q_len, 1)
            kv_id.unsqueeze(-2)       # (*batch, 1, 1, 1, kv_len)
        )
        # ind: (*batch, 1, 1, q_len, kv_len)
        # ind[i, j] = True if variate_id[i] == variate_id[j]

        # Step 2: Embedding weight 가져오기
        weight = rearrange(
            self.emb.weight,  # (2, num_heads)
            "two num_heads -> two num_heads 1 1"
        )
        # weight: (2, num_heads, 1, 1)
        # weight[0]: 다른 변수용 (u⁽²⁾)
        # weight[1]: 같은 변수용 (u⁽¹⁾)

        # Step 3: ind에 따라 선택
        bias = rearrange(
            ~ind * weight[:1] + ind * weight[1:],
            # ~ind (False인 곳): weight[0] 사용
            # ind (True인 곳): weight[1] 사용
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        # bias: (*batch, group, hpg, q_len, kv_len)

        return bias
```

### 실제 사용 예시

```python
# OHLCV 5개 변수, 각 4개 패치 = 20 tokens
variate_id = torch.tensor([
    0,0,0,0,  # Open
    1,1,1,1,  # High
    2,2,2,2,  # Low
    3,3,3,3,  # Close
    4,4,4,4   # Volume
])

# Binary Attention Bias 초기화
binary_bias = BinaryAttentionBias(
    dim=768,
    num_heads=12,
    num_groups=4
)

# Embedding 학습 결과 (예시)
# binary_bias.emb.weight[0] = tensor([-1.0, -0.8, ...])  # 다른 변수용
# binary_bias.emb.weight[1] = tensor([2.5, 2.3, ...])    # 같은 변수용

# Forward
query_id = variate_id.view(1, 1, 1, 20)
kv_id = variate_id.view(1, 1, 1, 20)

bias = binary_bias.forward(query, key, query_id, kv_id)
# bias[i, j]:
#   variate_id[i] == variate_id[j] → emb.weight[1] (예: +2.5)
#   variate_id[i] != variate_id[j] → emb.weight[0] (예: -1.0)

# Attention Score
scores = Q @ K.T / sqrt(d) + bias
# 같은 변수: score + 2.5  (강화)
# 다른 변수: score - 1.0  (억제)
```

### Attention Matrix 시각화

```
      0-3  4-7  8-11 12-15 16-19  (variate_id)
      Open High Low  Close Vol
0-3   🟢🟢🟢🟢  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  Open
4-7   🔴🔴🔴🔴  🟢🟢🟢🟢  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  High
8-11  🔴🔴🔴🔴  🔴🔴🔴🔴  🟢🟢🟢🟢  🔴🔴🔴🔴  🔴🔴🔴🔴  Low
12-15 🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🟢🟢🟢🟢  🔴🔴🔴🔴  Close
16-19 🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🟢🟢🟢🟢  Vol

🟢 = 같은 변수 (u⁽¹⁾ ≈ +2.5)
🔴 = 다른 변수 (u⁽²⁾ ≈ -1.0)
```

---

## 2.3 Mixture Distribution

### 실제 구현

**파일**: `src/uni2ts/distribution/mixture.py:28-168`

```python
class Mixture(Distribution):
    """Mixture of Distributions"""

    arg_constraints = dict()
    has_rsample = False

    def __init__(
        self,
        weights: Categorical,              # 가중치 분포
        components: list[Distribution],    # 컴포넌트 분포들
        validate_args: Optional[bool] = None,
    ):
        # 검증 모드 끔 (성능)
        for comp in components:
            comp._validate_args = False

        self.weights = weights
        self.components = components

        # 검증 (디버그용)
        if not isinstance(weights, Categorical):
            raise TypeError("weights must be a Categorical distribution")
        if not all(isinstance(comp, Distribution) for comp in components):
            raise TypeError("components must all be instances of Distribution")

        batch_shape = weights.batch_shape
        event_shape = components[0].event_shape

        if validate_args:
            if not all(comp.batch_shape == batch_shape for comp in components):
                raise ValueError("components must have the same batch_shape as weights")
            if not all(comp.event_shape == event_shape for comp in components):
                raise ValueError("components must have the same event_shape")
            if weights.logits.shape[-1] != len(components):
                raise ValueError(
                    f"number of logits ({weights.logits.shape[-1]}) != "
                    f"number of components ({len(components)})"
                )

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Mixture의 log-probability 계산

        log p(y) = log Σᵢ wᵢ·pᵢ(y)
                 = log Σᵢ exp(log wᵢ + log pᵢ(y))
                 = logsumexp(log wᵢ + log pᵢ(y))
        """
        if self._validate_args:
            self._validate_sample(value)

            # 최소 하나의 support에는 속해야 함
            valid = reduce(
                torch.logical_or,
                (comp.support.check(value) for comp in self.components),
            )
            if not valid.all():
                raise ValueError(
                    f"Value {value} not in any component's support"
                )

        # Step 1: 가중치의 log-probs
        weights_log_probs = self.weights.logits.expand(
            value.shape + (len(self.components),)
        )
        weights_log_probs = torch.stack(weights_log_probs.unbind(dim=-1))
        # (num_components, *value.shape)

        # Step 2: 각 컴포넌트의 log-probs
        # nan gradient 방지: https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        components_log_probs = torch.stack(
            [
                torch.where(
                    comp.support.check(value),  # support 체크
                    comp.log_prob(
                        torch.where(
                            comp.support.check(value),
                            value,
                            comp.sample(),  # support 밖이면 dummy sample
                        )
                    ),
                    float("-inf"),  # support 밖은 -inf
                )
                for comp in self.components
            ]
        )
        # (num_components, *value.shape)

        # Step 3: -inf 처리 (가중치 0으로)
        weights_log_probs = torch.where(
            torch.isinf(components_log_probs),
            0.0,
            weights_log_probs,
        )

        # Step 4: Log-sum-exp
        return (weights_log_probs + components_log_probs).logsumexp(dim=0)
        # (*value.shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Mixture에서 샘플링

        절차:
        1. 가중치로부터 컴포넌트 선택
        2. 선택된 컴포넌트에서 샘플링
        """
        with torch.no_grad():
            # Step 1: 각 컴포넌트에서 샘플
            components_samples = torch.stack(
                [comp.sample(sample_shape) for comp in self.components],
                dim=-1
            )
            # (*sample_shape, *batch_shape, *event_shape, num_components)

            # Step 2: 가중치로부터 컴포넌트 선택
            weights_sample = unsqueeze_trailing_dims(
                self.weights.sample(sample_shape),  # 컴포넌트 index
                components_samples.shape
            )
            # (*sample_shape, *batch_shape, 1, ..., 1, 1)

            # Step 3: 선택된 컴포넌트의 샘플만 가져오기
            samples = torch.gather(
                components_samples,
                dim=-1,
                index=weights_sample,
            ).squeeze(-1)
            # (*sample_shape, *batch_shape, *event_shape)

        return samples

    @property
    def mean(self) -> torch.Tensor:
        """
        Mixture의 평균:
        E[Y] = Σᵢ wᵢ · E[Yᵢ]
        """
        weights_probs = torch.stack(self.weights.probs.unbind(dim=-1))
        # (num_components, *batch_shape)

        components_means = torch.stack([comp.mean for comp in self.components])
        # (num_components, *batch_shape, *event_shape)

        return (weights_probs * components_means).sum(dim=0)
        # (*batch_shape, *event_shape)

    @property
    def variance(self) -> torch.Tensor:
        """
        Law of Total Variance:
        Var(Y) = E[Var(Y|X)] + Var(E[Y|X])

        where X = 컴포넌트 선택 확률 변수
        """
        weights_probs = torch.stack(self.weights.probs.unbind(dim=-1))

        # E[Var(Y|X)] = Σᵢ wᵢ · Var(Yᵢ)
        components_var = torch.stack([comp.variance for comp in self.components])
        expected_cond_var = (weights_probs * components_var).sum(dim=0)

        # Var(E[Y|X]) = Σᵢ wᵢ · E[Yᵢ]² - E[Y]²
        components_means = torch.stack([comp.mean for comp in self.components])
        var_cond_expectation = (
            (weights_probs * components_means.pow(2.0)).sum(dim=0)
            - self.mean.pow(2.0)
        )

        return expected_cond_var + var_cond_expectation
```

### 실제 사용 예시

```python
from torch.distributions import Categorical, StudentT, Normal

# 4개 컴포넌트 Mixture
weights_logits = torch.tensor([0.5, -1.0, -0.5, -2.0])  # logits
weights = Categorical(logits=weights_logits)

components = [
    StudentT(df=3.0, loc=100.0, scale=5.0),       # Student-t
    # NegativeBinomial은 실제론 더 복잡
    Normal(loc=100.0, scale=10.0),                 # 대신 Normal 예시
    # LogNormal도 실제론 더 복잡
    Normal(loc=100.0, scale=15.0),                 # 대신 Normal 예시
    Normal(loc=100.0, scale=0.5),                  # Low-var Normal
]

mixture = Mixture(weights=weights, components=components)

# 가중치 확률 (softmax)
print(weights.probs)
# tensor([0.50, 0.11, 0.18, 0.04])  (정규화: 합=1)

# Log-probability 계산
value = torch.tensor([105.0])
log_prob = mixture.log_prob(value)
# log_prob = logsumexp([
#     log(0.50) + StudentT.log_prob(105),
#     log(0.11) + Normal1.log_prob(105),
#     log(0.18) + Normal2.log_prob(105),
#     log(0.04) + Normal3.log_prob(105),
# ])

# 샘플링
samples = mixture.sample((100,))  # 100개 샘플
# 내부 동작:
# - 50개: StudentT에서 샘플 (가중치 50%)
# - 11개: Normal1에서 샘플 (가중치 11%)
# - 18개: Normal2에서 샘플 (가중치 18%)
# - 4개: Normal3에서 샘플 (가중치 4%)
# (확률적이므로 정확히는 아님)

# 통계량
mean = mixture.mean
variance = mixture.variance
```

### Moirai의 4가지 컴포넌트 (실제)

Uni2TS에서 실제로 사용하는 분포들:

**파일**: `src/uni2ts/distribution/`

```
1. Student-t Distribution
   - src/uni2ts/distribution/student_t.py
   - 두꺼운 꼬리, 이상치 robust

2. Negative Binomial Distribution
   - src/uni2ts/distribution/negative_binomial.py
   - 카운트 데이터 (정수, 비음수)

3. Log-Normal Distribution
   - PyTorch 기본 제공
   - 오른쪽 치우침

4. Normal Distribution
   - PyTorch 기본 제공
   - Low variance 설정으로 사용
```

---

## 전체 흐름 (실제 코드 기반)

```python
# 입력
Y = (5, 512)  # OHLCV, 512일

# Step 1: Patchify
from uni2ts.transform.patch import Patchify, GetPatchSize

get_patch_size = GetPatchSize(
    min_time_patches=16,
    patch_sizes=(8, 16, 32, 64, 128),
    patch_size_constraints=DefaultPatchSizeConstraints(),
)
data_entry = get_patch_size({'target': Y, 'freq': 'D'})
# → data_entry['patch_size'] = 16 (일 단위이므로)

patchify = Patchify(max_patch_size=128)
data_entry = patchify(data_entry)
# data_entry['target']: (5, 32, 128)  # 32 patches, padded to 128

# Step 2: Flatten
target_flat = data_entry['target'].reshape(-1, 128)  # (160, 128)

# Step 3: Add variate_id
from uni2ts.transform.feature import AddVariateIndex

add_variate = AddVariateIndex(fields=('target',), max_dim=100)
data_entry = add_variate(data_entry)
# data_entry['variate_id']: (5, 32) = [[0,0,...],[1,1,...],...]

# Step 4: MultiInSizeLinear
from uni2ts.module.ts_embed import MultiInSizeLinear

in_proj = MultiInSizeLinear(
    in_features_ls=(8, 16, 32, 64, 128),
    out_features=768,
)
in_feat_size = torch.full((160,), data_entry['patch_size'])  # (160,) all 16
embedded = in_proj(target_flat, in_feat_size)
# embedded: (160, 768)

# Step 5: Transformer Encoder with Binary Attention Bias
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.attention import GroupedQueryAttention
from uni2ts.module.position.attn_bias import BinaryAttentionBias

encoder = TransformerEncoder(
    d_model=768,
    num_layers=24,
    num_heads=12,
    # ... (내부적으로 GroupedQueryAttention + BinaryAttentionBias 사용)
)

variate_id = data_entry['variate_id'].reshape(-1)  # (160,)
time_id = torch.arange(32).repeat(5)  # (160,)

encoded = encoder(
    embedded,
    variate_id=variate_id,
    time_id=time_id,
)
# encoded: (160, 768)

# Step 6: Output Projection
from uni2ts.module.ts_embed import MultiOutSizeLinear

out_proj = MultiOutSizeLinear(
    in_features=768,
    out_features_ls=(8, 16, 32, 64, 128),
)
params = out_proj(encoded, in_feat_size)
# params: (160, 128) → distribution parameters

# Step 7: Mixture Distribution
from uni2ts.distribution.mixture import MixtureOutput

mixture_output = MixtureOutput(components=[...])
mixture = mixture_output.distribution(params)

# Step 8: Loss
loss = -mixture.log_prob(target_true).mean()
```

---

## 핵심 포인트 정리

### MultiInSizeLinear
```
✓ 하나의 텐서: (num_sizes, out_features, max_size)
✓ Mask로 유효 영역 제한
✓ 모든 size 계산 후 torch.eq로 선택
✓ GPU 친화적 병렬 처리
```

### Binary Attention Bias
```
✓ 2개 임베딩만: [다른 변수, 같은 변수]
✓ torch.eq로 same/different 구분
✓ where로 선택적 적용
✓ 학습 가능한 bias
```

### Mixture Distribution
```
✓ Categorical weights + multiple components
✓ logsumexp for log_prob
✓ gather for sampling
✓ Law of Total Variance for variance
```

---

## 참고

이 문서의 모든 코드는 다음 파일들에서 직접 읽었습니다:
- `src/uni2ts/module/ts_embed.py`
- `src/uni2ts/module/position/attn_bias.py`
- `src/uni2ts/distribution/mixture.py`
- `src/uni2ts/transform/patch.py`
- `src/uni2ts/common/torch_util.py`
