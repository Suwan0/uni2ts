# Moirai 논문 Method 섹션 상세 설명

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
  ↑                                    ↑      ↑
  목표 시계열                      변수 수   시간 길이

Z⁽ⁱ⁾ = (z₁⁽ⁱ⁾, z₂⁽ⁱ⁾, ..., zₜᵢ⁽ⁱ⁾) ∈ ℝᵈᶻⁱ ˣ ᵀⁱ
  ↑
  공변량(covariates) - 보조 특징들
```

### 구체적 예시

```python
# 시계열 i
Y⁽ⁱ⁾: 주가 데이터
  - dyi = 5 변수 (Open, High, Low, Close, Volume)
  - Ti = 512 timesteps (512일)
  - Shape: (5, 512)

Z⁽ⁱ⁾: 공변량 (예: 거시경제 지표)
  - dzi = 2 변수 (금리, 환율)
  - Ti = 512 timesteps
  - Shape: (2, 512)

목표: t 시점에서 h 길이의 미래 예측
  - 입력: Yₜ₋ₗ:ₜ (과거 l timesteps, context length)
         Zₜ₋ₗ:ₜ₊ₕ (과거 + 미래의 공변량)
  - 출력: Yₜ:ₜ₊ₕ의 확률 분포 p(Yₜ:ₜ₊ₕ|φ̂)
```

### 최적화 목표 (Equation 1)

```
max E          E        [log p(Yₜ:ₜ₊ₕ|φ̂)]
 θ  (Y,Z)~p(D) (t,l,h)~p(T|D)

subject to: φ̂ = fθ(Yₜ₋ₗ:ₜ, Zₜ₋ₗ:ₜ₊ₕ)
```

**해석**:
```
1. 데이터셋 D에서 시계열 (Y, Z) 샘플링
2. 각 시계열에서 task (t, l, h) 샘플링
   - t: 예측 시작 시점
   - l: context length (lookback window)
   - h: prediction length (forecast horizon)
3. 모델 fθ가 분포 파라미터 φ̂를 예측
4. 실제 값 Yₜ:ₜ₊ₕ의 log-likelihood 최대화
```

### 구체적 예시

```python
# 예시 task
t = 400      # 400일 시점
l = 100      # 100일 history 사용
h = 30       # 30일 예측

입력:
  Y_context = Y[300:400]      # timestep 300~399 (100일)
  Z_context = Z[300:430]      # timestep 300~429 (100일 + 30일)
                              # 미래 공변량도 포함!

출력:
  Y_forecast = Y[400:430]     # timestep 400~429 (30일)
  φ̂ = fθ(Y_context, Z_context)

  # φ̂는 분포 파라미터 (예: mean, std)
  p(Y_forecast | φ̂) = Normal(μ=φ̂[0], σ=φ̂[1])

Loss:
  -log p(Y_forecast | φ̂)
```

---

## 2. Architecture (아키텍처)

### 전체 구조

```
입력 시계열 (dyi, Ti)
    ↓
[Flatten] 다변량 → 단일 시퀀스
    ↓
[Patchify] 시간을 패치로 분할
    ↓
[Multi Patch Size Input Projection] 패치 → 벡터
    ↓
[Mask Fill] 예측 구간을 [MASK] 토큰으로 대체
    ↓
[Transformer Encoder]
  - Pre-normalization (RMSNorm)
  - Any-variate Attention (Binary attention bias)
  - SwiGLU FFN
  - No biases
    ↓
[Multi Patch Size Output Projection] 벡터 → 분포 파라미터
    ↓
[Mixture Distribution] φ̂로 확률 분포 생성
    ↓
예측 분포 p(Yₜ:ₜ₊ₕ|φ̂)
```

### 아키텍처 개선사항

논문에서 언급한 최신 LLM 기법들:

```
1. Pre-normalization (Xiong et al., 2020)
   기존: x → Attention → Norm → FFN → Norm
   개선: x → Norm → Attention → x + ... → Norm → FFN → x + ...
   효과: 학습 안정성 향상

2. RMSNorm (Zhang & Sennrich, 2019)
   기존: LayerNorm = (x - mean) / std
   개선: RMSNorm = x / RMS(x), where RMS = sqrt(mean(x²))
   효과: 계산량 감소, mean 제거

3. Query-Key Normalization (Henry et al., 2020)
   Q와 K를 각각 정규화
   효과: attention score 안정화

4. SwiGLU (Shazeer, 2020)
   기존: FFN(x) = W₂ · ReLU(W₁x)
   개선: SwiGLU(x) = (W₁x ⊙ σ(W₂x)) · W₃
   효과: 표현력 향상 (gating mechanism)

5. No biases
   모든 linear layer에서 bias 제거
   효과: 파라미터 수 감소, 일반화 성능 향상
```

---

## 2.1 Multi Patch Size Projection Layers

### 문제: 단일 patch_size의 한계

```
기존 방식 (고정 patch_size):
  patch_size = 16

고주파 데이터 (분 단위):
  1440 timesteps (1일) → 90 patches
  문제: 너무 많은 패치 → 계산 비용 과다 ❌

저주파 데이터 (월 단위):
  120 timesteps (10년) → 7 patches
  문제: 너무 적은 패치 → 정보 손실 ❌
```

### 해결책: 다중 Patch Size

```python
# 주파수별 patch_size 자동 선택
PATCH_SIZE_MAP = {
    'S': (64, 128),   # 초 단위: 큰 patch
    'T': (32, 128),   # 분 단위: 큰 patch
    'H': (32, 64),    # 시간 단위: 중간 patch
    'D': (16, 32),    # 일 단위: 작은 patch
    'W': (16, 32),    # 주 단위: 작은 patch
    'M': (8, 32),     # 월 단위: 작은 patch
}

# 동적 선택
if freq == 'T':  # 분 단위
    patch_size = 128  # 큰 패치로 계산량 감소
elif freq == 'D':  # 일 단위
    patch_size = 16   # 작은 패치로 세밀한 패턴 포착
```

### 구현: MultiInSizeLinear

**⚠️ 중요: 아래는 개념적 설명입니다. 실제 구현은 `multi_in_size_linear_actual_implementation.md` 참조!**

**개념적 아이디어** (이해를 위한 단순화):
- 각 patch_size마다 별도의 linear layer
- 입력의 patch_size에 따라 적절한 layer 선택

**실제 구현** (`src/uni2ts/module/ts_embed.py`):
```python
class MultiInSizeLinear(nn.Module):
    def __init__(self, in_features_ls, out_features):
        # 하나의 큰 텐서로 모든 weight 저장!
        self.weight = nn.Parameter(
            torch.empty((len(in_features_ls), out_features, max(in_features_ls)))
        )
        # Shape: (5, 768, 128) for in_features_ls=(8,16,32,64,128)

        # Mask로 유효 영역만 활성화
        self.mask = size_to_mask(...)
        # mask[0]: [1,1,1,1,1,1,1,1, 0,0,...,0]  # 처음 8개만
        # mask[1]: [1,1,...,1, 0,0,...,0]        # 처음 16개만

    def forward(self, x, in_feat_size):
        out = 0
        # 모든 patch_size에 대해 계산 후 선택
        for idx, feat_size in enumerate(self.in_features_ls):
            weight_masked = self.weight[idx] * self.mask[idx]
            match = torch.eq(in_feat_size, feat_size)
            linear_out = einsum(weight_masked, x, "out inp, ... inp -> ... out")
            out = out + match.unsqueeze(-1) * linear_out
        return out
```

**핵심 차이**:
- ✗ nn.ParameterDict 사용 (제가 개념 설명용으로 단순화)
- ✓ 단일 텐서 + mask 방식 (실제 구현, GPU 효율적)

자세한 내용은 `docs/multi_in_size_linear_actual_implementation.md` 참조!

### 장점

```
1. 계산 효율성:
   고주파 데이터 → 큰 patch_size → 패치 수 감소 → Attention O(n²) 감소

2. 정보 보존:
   저주파 데이터 → 작은 patch_size → 패치 수 증가 → 세밀한 패턴 포착

3. 유연성:
   하나의 모델로 다양한 주파수 처리

4. Weight 공유:
   같은 patch_size는 weight 공유 → 파라미터 효율
```

### 실제 예시

```python
# 분 단위 데이터 (1일 = 1440분)
data_min = np.random.randn(1, 1440)
patch_size = 128
num_patches = 1440 // 128 = 11 patches

# 일 단위 데이터 (3년 = 1095일)
data_day = np.random.randn(1, 1095)
patch_size = 16
num_patches = 1095 // 16 = 68 patches

# 같은 모델, 다른 patch_size로 처리!
model(data_min, patch_size=128)
model(data_day, patch_size=16)
```

---

## 2.2 Any-variate Attention

### 문제: 임의 개수 변수 처리

```
기존 Transformer:
  Embedding: ℝᵈʸ → ℝᵈʰ

문제:
  1. dy(변수 수)가 고정되어야 함
  2. dy=5인 데이터로 학습 → dy=10인 데이터 처리 불가 ❌

목표:
  임의의 변수 수를 가진 다변량 시계열 처리
```

### 해결책 1: Flatten

**다변량을 단일 시퀀스로**

```python
# 기존: 다변량을 별도로 처리
multivariate_data = (5, 32, 16)  # (variables, patches, patch_size)
# → 각 변수마다 독립적인 시퀀스

# Moirai: Flatten
multivariate_data = (5, 32, 16)
  ↓
flattened = (5*32, 16) = (160, 16)  # 단일 시퀀스!

각 토큰 = 하나의 패치
토큰 0-31:   변수 0의 32개 패치
토큰 32-63:  변수 1의 32개 패치
토큰 64-95:  변수 2의 32개 패치
토큰 96-127: 변수 3의 32개 패치
토큰 128-159: 변수 4의 32개 패치
```

### 해결책 2: Binary Attention Bias

**문제**: Flatten 후 어떤 토큰이 어느 변수인지 구분 필요

**요구사항**:
```
1. Permutation Equivariance (변수 순서에 대한 동변성)
   [Var0, Var1, Var2] ≠ [Var2, Var0, Var1]
   순서가 바뀌면 출력도 일관되게 변해야 함

2. Permutation Invariance (변수 인덱스에 대한 불변성)
   Var0의 ID=0이든 ID=5든 같은 결과
   절대적 ID 값은 중요하지 않음

3. Arbitrary Number of Variates (임의 개수)
   변수가 5개든 100개든 처리 가능
```

**기존 방법의 한계**:
```
Sinusoidal Encoding:
  pos_emb[i] = sin(i / 10000^(2k/d))
  문제: ID 값 자체에 의존 (불변성 X) ❌

Learned Embedding:
  var_emb = nn.Embedding(max_vars, dim)
  문제: max_vars가 고정 (임의 개수 X) ❌
```

### Binary Attention Bias 수식 (Equation 2-3)

```
Attention Score:

Eᵢⱼ,ₘₙ = (Wᵠxᵢ,ₘ)ᵀ Rᵢ₋ⱼ (Wᴷxⱼ,ₙ) + u⁽¹⁾·𝟙{m=n} + u⁽²⁾·𝟙{m≠n}
         └─── 기본 attention ──┘   └───── Binary Bias ─────┘

Aᵢⱼ,ₘₙ = exp{Eᵢⱼ,ₘₙ} / Σₖ,ₒ exp{Eᵢₖ,ₘₒ}
```

**수식 구성 요소**:
```
i, j: 시간 인덱스 (패치 번호)
m, n: 변수 인덱스 (variate ID)

Wᵠxᵢ,ₘ: Query 벡터 (i번째 패치, m번째 변수)
Wᴷxⱼ,ₙ: Key 벡터 (j번째 패치, n번째 변수)
Rᵢ₋ⱼ: Rotary matrix (시간 위치 인코딩)

u⁽¹⁾: 같은 변수 간 bias (학습 가능)
u⁽²⁾: 다른 변수 간 bias (학습 가능)

𝟙{m=n}: 지시 함수
  = 1 if m == n (같은 변수)
  = 0 if m ≠ n  (다른 변수)
```

### 구현 코드

```python
# src/uni2ts/module/position/attn_bias.py:67-87

class BinaryAttentionBias(nn.Module):
    def __init__(self, dim, num_heads, num_groups):
        super().__init__()
        # 2개 임베딩: [다른 변수, 같은 변수]
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=num_heads)

    def forward(self, query, key, query_id, kv_id):
        """
        query_id: (batch, 1, 1, q_len) - Query의 variate_id
        kv_id: (batch, 1, 1, kv_len)   - Key/Value의 variate_id

        return: (batch, group, hpg, q_len, kv_len)
        """
        # Step 1: 같은 변수인지 확인
        # (batch, 1, 1, q_len, 1) == (batch, 1, 1, 1, kv_len)
        # → (batch, 1, 1, q_len, kv_len)
        ind = torch.eq(
            query_id.unsqueeze(-1),  # (batch, 1, 1, q_len, 1)
            kv_id.unsqueeze(-2)       # (batch, 1, 1, 1, kv_len)
        )
        # ind[i,j] = True if variate_id[i] == variate_id[j]

        # Step 2: Embedding weight 가져오기
        weight = self.emb.weight  # (2, num_heads)
        # weight[0]: 다른 변수 간 bias (u⁽²⁾)
        # weight[1]: 같은 변수 간 bias (u⁽¹⁾)

        # Step 3: ind에 따라 선택
        # ~ind * weight[0] + ind * weight[1]
        bias = torch.where(
            ind.unsqueeze(1),     # (batch, 1, 1, q_len, kv_len)
            weight[1].view(1, num_heads, 1, 1),  # 같은 변수
            weight[0].view(1, num_heads, 1, 1)   # 다른 변수
        )

        return bias  # (batch, num_heads, q_len, kv_len)
```

### 구체적 예시

```python
# OHLCV 5개 변수, 각 4개 패치
variate_id = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]
             └ Open ┘ └High ┘ └ Low ┘ └Close┘ └Volume┘

# Binary Bias 학습 결과 (예시)
u⁽¹⁾ = 2.5   # 같은 변수 간 boost
u⁽²⁾ = -1.0  # 다른 변수 간 suppress

# Token 5 (High의 두 번째 패치) → Token 9 (High의 네 번째 패치)
variate_id[5] = 1, variate_id[9] = 1
same_var = True
bias = u⁽¹⁾ = 2.5  ← Attention 강화!

# Token 5 (High) → Token 13 (Close)
variate_id[5] = 1, variate_id[13] = 3
same_var = False
bias = u⁽²⁾ = -1.0  ← Attention 억제!

# Attention Score 계산
score = Q[5] @ K[9] / sqrt(d) + 2.5   # 같은 변수: 높은 score
score = Q[5] @ K[13] / sqrt(d) - 1.0  # 다른 변수: 낮은 score
```

### 효과

```
1. 변수 구분:
   같은 변수 내에서 강한 attention
   다른 변수 간에는 약한 attention

2. 유연성:
   변수 개수에 무관 (2개든 100개든 동작)

3. 학습 가능:
   u⁽¹⁾, u⁽²⁾가 데이터에서 학습됨

4. 효율성:
   단 2개의 스칼라만 학습 (파라미터 최소)
```

### 시각화

```
Attention Matrix (5개 변수, 각 4개 패치 = 20 tokens)

          Token 0-3  4-7  8-11 12-15 16-19
                Open High Low  Close Vol
Token 0-3  Open    🟢🟢🟢🟢  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴
Token 4-7  High    🔴🔴🔴🔴  🟢🟢🟢🟢  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴
Token 8-11 Low     🔴🔴🔴🔴  🔴🔴🔴🔴  🟢🟢🟢🟢  🔴🔴🔴🔴  🔴🔴🔴🔴
Token 12-15 Close  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🟢🟢🟢🟢  🔴🔴🔴🔴
Token 16-19 Vol    🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🔴🔴🔴🔴  🟢🟢🟢🟢

🟢 = u⁽¹⁾ (같은 변수, high attention)
🔴 = u⁽²⁾ (다른 변수, low attention)
```

---

## 2.3 Mixture Distribution

### 문제: 단일 분포의 한계

```
Normal Distribution만 사용:
  p(y) = N(μ, σ²)

한계:
  1. Skewed data (치우친 데이터) 표현 불가 ❌
     예: 주가 수익률 (오른쪽 꼬리가 긴 분포)

  2. Count data (카운트 데이터) 표현 불가 ❌
     예: 일일 방문자 수 (정수, 비음수)

  3. Outliers (이상치) 처리 어려움 ❌
     예: 금융 위기 시 극단적 변동
```

### 해결책: Mixture Distribution

```
여러 분포의 가중 평균:

p(Y|φ̂) = Σᵢ₌₁ᶜ wᵢ · pᵢ(Y|φ̂ᵢ)

c: 컴포넌트 개수 (예: 4개)
wᵢ: i번째 컴포넌트의 가중치 (Σwᵢ = 1)
pᵢ: i번째 컴포넌트의 분포
φ̂ᵢ: i번째 컴포넌트의 파라미터
```

### Moirai의 4가지 컴포넌트

```python
1. Student's t-distribution (일반적 시계열)
   p(y) = StudentT(μ, σ, ν)
   특징:
   - 두꺼운 꼬리 (heavy tail)
   - 이상치에 robust
   - ν → ∞일 때 Normal에 수렴

2. Negative Binomial (카운트 데이터)
   p(y) = NegativeBinomial(μ, α)
   특징:
   - y ∈ {0, 1, 2, ...} (정수)
   - y ≥ 0 (비음수)
   - Overdispersion (분산 > 평균)

3. Log-Normal (오른쪽 치우침)
   p(y) = LogNormal(μ, σ)
   특징:
   - y > 0 (양수)
   - 오른쪽 꼬리가 긴 분포
   - 경제, 자연 현상에 흔함

4. Low-variance Normal (높은 확신)
   p(y) = Normal(μ, small σ)
   특징:
   - 낮은 분산 (예: σ < 0.1)
   - 확실한 예측에 사용
   - 패턴이 명확한 시계열
```

### 구현 코드

```python
# src/uni2ts/distribution/mixture.py:28-168

class Mixture(Distribution):
    def __init__(
        self,
        weights: Categorical,        # 가중치 분포
        components: list[Distribution],  # 컴포넌트들
    ):
        self.weights = weights
        self.components = components

    def log_prob(self, value):
        """
        value: 실제 관측값
        return: log p(value | mixture)
        """
        # Step 1: 가중치의 log-probability
        weights_log_probs = self.weights.logits  # log(wᵢ)

        # Step 2: 각 컴포넌트의 log-probability
        components_log_probs = [
            comp.log_prob(value)  # log pᵢ(value)
            for comp in self.components
        ]
        # components_log_probs[i] = log pᵢ(value)

        # Step 3: Log-sum-exp
        # log Σᵢ wᵢ·pᵢ(value)
        # = log Σᵢ exp(log wᵢ + log pᵢ(value))
        # = logsumexp(log wᵢ + log pᵢ(value))

        return (weights_log_probs + components_log_probs).logsumexp(dim=0)

    def sample(self, sample_shape):
        """
        샘플링 절차:
        1. 가중치로부터 컴포넌트 선택
        2. 선택된 컴포넌트에서 샘플링
        """
        # Step 1: 어느 컴포넌트를 사용할지 선택
        component_idx = self.weights.sample(sample_shape)
        # component_idx ∈ {0, 1, 2, 3}

        # Step 2: 각 컴포넌트에서 샘플
        components_samples = [
            comp.sample(sample_shape)
            for comp in self.components
        ]

        # Step 3: 선택된 컴포넌트의 샘플 사용
        samples = torch.gather(
            torch.stack(components_samples, dim=-1),
            dim=-1,
            index=component_idx
        )

        return samples

    @property
    def mean(self):
        """
        Mixture의 평균:
        E[Y] = Σᵢ wᵢ · E[Yᵢ]
        """
        weights_probs = self.weights.probs  # [w₁, w₂, w₃, w₄]
        components_means = [comp.mean for comp in self.components]

        return (weights_probs * components_means).sum(dim=0)

    @property
    def variance(self):
        """
        Law of Total Variance:
        Var(Y) = E[Var(Y|X)] + Var(E[Y|X])

        where X = 컴포넌트 선택
        """
        # E[Var(Y|X)] = Σᵢ wᵢ · Var(Yᵢ)
        expected_cond_var = (
            self.weights.probs *
            torch.stack([comp.variance for comp in self.components])
        ).sum(dim=0)

        # Var(E[Y|X]) = Σᵢ wᵢ · E[Yᵢ]² - (E[Y])²
        var_cond_expectation = (
            self.weights.probs *
            torch.stack([comp.mean.pow(2) for comp in self.components])
        ).sum(dim=0) - self.mean.pow(2)

        return expected_cond_var + var_cond_expectation
```

### 구체적 예시

```python
# 4개 컴포넌트 Mixture 예시

# 예측 파라미터
φ̂ = {
    'weights_logits': [0.5, -1.0, -0.5, -2.0],
    'components': [
        {'loc': 100.0, 'scale': 5.0, 'df': 3.0},      # Student-t
        {'mean': 95.0, 'total_count': 100},            # NegativeBinomial
        {'loc': 4.6, 'scale': 0.1},                    # LogNormal (μ=100, σ=10)
        {'loc': 100.0, 'scale': 0.5},                  # Low-var Normal
    ]
}

# 가중치 계산 (softmax)
weights = softmax([0.5, -1.0, -0.5, -2.0])
        = [0.50, 0.11, 0.18, 0.04]  (정규화: 합=1)

# Mixture 분포
p(y) = 0.50 × StudentT(y|100, 5, 3)      # 50% 가중
     + 0.11 × NegBinom(y|95, 100)        # 11% 가중
     + 0.18 × LogNormal(y|4.6, 0.1)      # 18% 가중
     + 0.04 × Normal(y|100, 0.5)         # 4% 가중

# 샘플링
1. 가중치로부터 컴포넌트 선택 (확률적)
   - 50% 확률로 Student-t 선택
   - 11% 확률로 NegativeBinomial 선택
   - ...

2. 선택된 컴포넌트에서 샘플
   예: Student-t 선택됨 → y ~ StudentT(100, 5, 3)
```

### 학습 과정

```python
# Forward pass
predictions = model(X)  # 모델 출력

# Mixture Distribution 생성
mixture = MixtureOutput(components=[
    StudentTOutput(),
    NegativeBinomialOutput(),
    LogNormalOutput(),
    LowVarianceNormalOutput(),
]).distribution(predictions)

# Log-likelihood 계산
log_prob = mixture.log_prob(Y_true)

# Loss (negative log-likelihood)
loss = -log_prob.mean()

# Backward pass
loss.backward()
```

### 왜 Mixture인가?

```
1. 유연성:
   다양한 데이터 특성 표현 가능
   - Skewed: LogNormal
   - Count: NegativeBinomial
   - Robust: Student-t
   - Confident: Low-var Normal

2. 자동 선택:
   가중치가 학습되어 적절한 컴포넌트 자동 선택

3. 불확실성:
   여러 컴포넌트의 조합으로 복잡한 불확실성 표현

4. 실용성:
   모든 컴포넌트가 효율적 샘플링/loss 계산 가능
```

### 실제 예측 예시

```python
# 정상 상황 (패턴 명확)
weights = [0.05, 0.05, 0.05, 0.85]
        #  ↑ Student-t (5%)
        #       ↑ NegBinom (5%)
        #            ↑ LogNormal (5%)
        #                 ↑ Low-var Normal (85%) ← 주로 사용!

예측: y ≈ 100.0 ± 0.5 (높은 확신)

# 이상 상황 (큰 변동성)
weights = [0.70, 0.10, 0.15, 0.05]
        #  ↑ Student-t (70%) ← 주로 사용!
        #       ↑ NegBinom (10%)
        #            ↑ LogNormal (15%)
        #                 ↑ Low-var Normal (5%)

예측: y ≈ 100.0 ± 15.0 (낮은 확신, 이상치 가능)

# Count 데이터
weights = [0.10, 0.75, 0.10, 0.05]
        #  ↑ Student-t (10%)
        #       ↑ NegBinom (75%) ← 주로 사용!
        #            ↑ LogNormal (10%)
        #                 ↑ Low-var Normal (5%)

예측: y ∈ {0, 1, 2, ...} (정수)
```

---

## 전체 흐름 요약

### End-to-End 예시

```python
# 입력: OHLCV 5개 변수, 512일
Y = (5, 512)  # 목표 시계열
Z = (2, 512)  # 공변량

# Task 설정
context_length = 100
prediction_length = 30

# Step 1: Patchify (patch_size 자동 선택)
patch_size = 16  # 일 단위 → 16 선택
Y_patched = (5, 32, 16)  # 512/16 = 32 patches

# Step 2: Flatten
Y_flat = (160, 16)  # 5*32 = 160 tokens

# Step 3: Add variate_id
variate_id = [0,0,...,0, 1,1,...,1, ..., 4,4,...,4]
              └─ 32 ─┘  └─ 32 ─┘       └─ 32 ─┘

# Step 4: Multi-size Input Projection
embedded = MultiInSizeLinear(Y_flat, patch_size=16)
         # (160, 16) → (160, 768)

# Step 5: Mask Fill
# prediction_mask에 해당하는 토큰 → [MASK]
embedded[prediction_mask] = mask_embedding

# Step 6: Transformer Encoder
for layer in encoder:
    # Any-variate Attention with Binary Bias
    q, k, v = layer.attention.qkv_proj(embedded)

    # Binary Attention Bias
    bias = binary_attention_bias(variate_id, variate_id)
    # Same var: +2.5, Different var: -1.0

    # Attention
    attn_out = attention(q, k, v, bias=bias)

    embedded = embedded + attn_out
    embedded = embedded + ffn(embedded)

# Step 7: Multi-size Output Projection
params = MultiOutSizeLinear(embedded)
       # (160, 768) → (160, num_params)

# Step 8: Mixture Distribution
mixture = MixtureOutput(
    components=[StudentT, NegBinom, LogNormal, LowVarNormal]
).distribution(params)

# Step 9: Loss
loss = -mixture.log_prob(Y_true).mean()

# Step 10: Prediction (Inference)
samples = mixture.sample((100,))  # 100 샘플
mean = mixture.mean
std = mixture.variance.sqrt()
```

---

## 핵심 포인트 정리

### 1. Multi Patch Size
```
✓ 주파수별로 적절한 patch_size 자동 선택
✓ 하나의 모델이 모든 주파수 처리
✓ 계산 효율성과 정보 보존 동시 달성
```

### 2. Any-variate Attention
```
✓ Flatten으로 임의 개수 변수 처리
✓ Binary Attention Bias로 변수 구분
✓ 같은 변수 내 강한 attention
✓ 다른 변수 간 약한 attention
```

### 3. Mixture Distribution
```
✓ 4개 컴포넌트로 다양한 데이터 표현
✓ Student-t: 일반적, 이상치 robust
✓ NegativeBinomial: 카운트 데이터
✓ LogNormal: 오른쪽 치우침
✓ Low-var Normal: 높은 확신
✓ 가중치 자동 학습
```

### Universal Forecasting의 핵심
```
하나의 모델로:
  ✓ 모든 주파수 (초 ~ 연)
  ✓ 모든 변수 개수 (1 ~ 수백)
  ✓ 모든 분포 특성 (정규, 치우침, 카운트)
  ✓ 모든 시계열 길이
```

이제 Moirai의 방법론을 완전히 이해하셨나요? 🎯
