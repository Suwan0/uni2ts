# Moirai 논문 Contribution 전체 요약

논문에서 제시한 4가지 핵심 contribution과 각 해결책의 구현 위치를 정리합니다.

## 논문 원문

> Starting from a masked encoder architecture which has been shown to be a strong candidate architecture for scaling up pre-trained time series forecasting models (Woo et al., 2023), we alleviate the above issues by introducing novel modifications which allows the architecture to handle the heterogeneity of arbitrary time series data.

---

## Contribution 체크리스트

| # | 문제점 | 해결책 | 설명 문서 | 구현 위치 |
|---|--------|--------|-----------|-----------|
| 1 | **Varying frequencies** | Multiple patch size projection layers | [`multi_patch_size_explained.md`](./multi_patch_size_explained.md) | `src/uni2ts/module/ts_embed.py` |
| 2 | **Varying dimensionality** | Any-variate Attention (Flatten + RoPE + Binary Attention Bias) | [`any_variate_attention.md`](./any_variate_attention.md) | `src/uni2ts/module/attention.py`, `src/uni2ts/module/position/` |
| 3 | **Flexible predictive distributions** | Mixture of parametric distributions | [`moirai_method_explained_accurate.md`](./moirai_method_explained_accurate.md) | `src/uni2ts/distribution/mixture.py` |
| 4 | **Flexible context/prediction lengths** | Random task distribution sampling | [`pretraining_task_and_data_distribution.md`](./pretraining_task_and_data_distribution.md) | `src/uni2ts/transform/crop.py`, `src/uni2ts/transform/task.py` |

---

## 1. Varying Frequencies (다양한 주기)

### 문제점
```
시계열 데이터는 다양한 주기를 가짐:
- Hourly: 센서 데이터, 트래픽
- Daily: 주가, 판매량
- Monthly: 경제 지표
- 등등...

기존 모델: 각 주기마다 별도 모델 필요
```

### 해결책: Multiple Patch Size Projection Layers

**핵심 아이디어**:
- 5가지 patch size 지원: `(8, 16, 32, 64, 128)`
- 각 주기마다 적절한 patch size 자동 선택
- 단일 tensor + mask로 모든 patch size 동시 처리

**구현**:
```python
# src/uni2ts/module/ts_embed.py
class MultiInSizeLinear(nn.Module):
    def __init__(self, in_features_ls=(8,16,32,64,128), out_features=768):
        # 모든 patch size의 weight를 단일 tensor에 저장
        self.weight = nn.Parameter(
            torch.empty((5, 768, 128))  # (num_sizes, out_feat, max_size)
        )
        self.mask = size_to_mask(...)  # 각 size별 유효 dimension 표시

    def forward(self, x, in_feat_size):
        # torch.eq로 현재 patch_size 매칭
        # 배치 내 다른 patch size 동시 처리!
        ...
```

**효과**:
- Hourly → patch_size=32
- Daily → patch_size=16
- Monthly → patch_size=8
- 모두 하나의 모델로 처리!

**자세한 설명**: [`multi_patch_size_explained.md`](./multi_patch_size_explained.md)

---

## 2. Varying Dimensionality (다양한 변수 개수)

### 문제점
```
시계열 데이터는 다양한 변수 개수를 가짐:
- Univariate: 1개 (온도, 주가)
- Multivariate: 5개 (주요 주가 지표)
- High-dimensional: 100개 (센서 네트워크)

기존 Transformer: 고정된 variate 수만 처리 가능
```

### 해결책: Any-variate Attention

**핵심 아이디어**:
1. **Flatten**: (variates, time) → (variates × time) 단일 sequence
2. **RoPE**: Time axis 정보 보존 (시간적 위치)
3. **Binary Attention Bias**: Variate axis 정보 보존 (같은/다른 변수)

**구현**:

#### 2.1 Flatten
```python
# src/uni2ts/transform/reshape.py
# (5 variates, 32 patches) → (160 tokens)
target = rearrange(target, "var time patch -> (var time) patch")
```

#### 2.2 RoPE (Rotary Position Encoding)
```python
# src/uni2ts/module/position/attn_projection.py
class RotaryProjection(Projection):
    def forward(self, x, seq_id):  # seq_id = time_id
        # 회전 변환으로 시간적 위치 인코딩
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)

# 효과: 상대적 시간 거리 보존
# Token(time=5) ↔ Token(time=6): High attention
# Token(time=5) ↔ Token(time=25): Low attention
```

#### 2.3 Binary Attention Bias
```python
# src/uni2ts/module/position/attn_bias.py
class BinaryAttentionBias(AttentionBias):
    def __init__(self, ...):
        # 2개의 학습 가능한 bias
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=num_heads)
        # emb.weight[0]: different variate bias
        # emb.weight[1]: same variate bias

    def forward(self, query, key, query_id, kv_id):  # query_id, kv_id = variate_id
        # Same variate인지 확인
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))

        # 해당하는 bias 선택
        bias = ~ind * weight[:1] + ind * weight[1:]
        return bias

# 효과:
# Same variate: weight[1] (예: +1.2) → attention 증가
# Different variate: weight[0] (예: -0.5) → attention 감소
```

#### 2.4 통합
```python
# src/uni2ts/module/attention.py
class GroupedQueryAttention(nn.Module):
    def forward(self, ..., var_id, time_id):
        # 1. Binary Attention Bias (variate axis)
        attn_bias = self.var_attn_bias(Q, K, query_id=var_id, kv_id=var_id)

        # 2. RoPE (time axis)
        Q_rot, K_rot = self.time_qk_proj(Q, K, query_id=time_id, kv_id=time_id)

        # 3. Attention with bias
        attn = softmax((Q_rot @ K_rot.T) / sqrt(d) + attn_bias)
        return attn @ V
```

**효과**:
- 1 variate도 OK
- 50 variates도 OK
- 128 variates도 OK
- 모두 하나의 모델로 처리!
- Zero-shot generalization to unseen variate counts

**자세한 설명**: [`any_variate_attention.md`](./any_variate_attention.md)

---

## 3. Flexible Predictive Distributions (유연한 예측 분포)

### 문제점
```
시계열 데이터는 다양한 분포를 가짐:
- Gaussian: 정규 분포 (온도, 센서)
- Count data: Negative Binomial (판매량, 방문자 수)
- Skewed: Log-normal (가격, 수익)
- Heavy-tailed: Student-t (금융 데이터)

단일 분포로는 모든 경우를 커버 불가
```

### 해결책: Mixture of Parametric Distributions

**핵심 아이디어**:
- 여러 분포를 혼합하여 복잡한 분포 표현
- 각 component의 가중치 학습
- Log-likelihood 최적화

**구현**:
```python
# src/uni2ts/distribution/mixture.py
class Mixture(Distribution):
    def __init__(
        self,
        component_distribution: Distribution,  # (num_components, *batch, *event)
        mixture_logits: torch.Tensor,          # (*batch, num_components)
    ):
        self.component_distribution = component_distribution
        self.mixture_distribution = Categorical(logits=mixture_logits)

    def log_prob(self, value):
        # log p(y) = log Σ_k π_k * p_k(y)
        #          = logsumexp(log π_k + log p_k(y))
        log_prob_k = self.component_distribution.log_prob(value)
        log_mix_prob = self.mixture_distribution.logits
        return torch.logsumexp(log_prob_k + log_mix_prob, dim=0)

    def sample(self, sample_shape):
        # 1. Sample component index k ~ Categorical(π)
        indices = self.mixture_distribution.sample(sample_shape)

        # 2. Sample from component p_k(y)
        samples = self.component_distribution.sample(sample_shape)

        # 3. Select samples according to indices
        return torch.gather(samples, dim=0, index=indices)
```

**실제 사용**:
```python
# Student-t mixture with 10 components
distr_output = StudentTMixtureOutput(num_components=10)

# Model predicts distribution parameters
distr_params = model(x)  # (batch, seq, num_params)

# Create mixture distribution
distr = distr_output.distribution(distr_params)

# Training: maximize log-likelihood
loss = -distr.log_prob(y_true).mean()

# Inference: sample or get mean
y_pred = distr.sample((100,)).median(dim=0).values
```

**효과**:
- 복잡한 분포 형태 모델링 가능
- Multimodal, skewed, heavy-tailed 모두 커버
- Uncertainty quantification (신뢰 구간)

**자세한 설명**: [`moirai_method_explained_accurate.md`](./moirai_method_explained_accurate.md)

---

## 4. Flexible Context/Prediction Lengths (유연한 lookback/horizon)

### 문제점
```
기존 forecasting 모델:
- 고정된 context_length (예: 512)
- 고정된 prediction_length (예: 96)
- 다른 설정에는 재학습 필요

실제 사용:
- 짧은 예측: 1일 후 (24 steps)
- 중간 예측: 1주일 후 (168 steps)
- 긴 예측: 1달 후 (720 steps)
```

### 해결책: Random Task Distribution Sampling

**핵심 아이디어**:
- 학습 시마다 다른 (context, prediction) 조합
- 다양한 길이에서 학습 → 유연한 모델

**구현**:

#### 4.1 Window Crop (Context + Prediction 전체)
```python
# src/uni2ts/transform/crop.py
class PatchCrop(Transformation):
    def _get_boundaries(self, data_entry):
        # Random window length
        num_patches = np.random.randint(
            self.min_time_patches,  # 2
            self.max_patches + 1     # 512
        )
        # 예: 20 patches 선택

        # Random start position
        first = np.random.randint(total_patches - num_patches + 1)

        start = offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop
```

#### 4.2 Prediction Mask (Lookback vs Horizon 구분)
```python
# src/uni2ts/transform/task.py
class MaskedPrediction(Transformation):
    def _generate_prediction_mask(self, target):
        var, time = target.shape[:2]

        # Random prediction ratio (0.15 ~ 0.5)
        mask_ratio = np.random.uniform(
            self.min_mask_ratio,  # 0.15
            self.max_mask_ratio   # 0.5
        )
        mask_length = max(1, round(time * mask_ratio))

        # Mask last mask_length patches
        prediction_mask = np.zeros((var, time), dtype=bool)
        prediction_mask[:, -mask_length:] = True
        return prediction_mask
```

**시각화**:
```
Sample 1:
├─────── Context (70%) ──────┤├─ Horizon (30%) ─┤
[observed data...............][ MASKED.........]

Sample 2:
├─ Context (60%) ─┤├──── Horizon (40%) ────┤
[observed data....][  MASKED...............]

Sample 3:
├────────── Context (85%) ───────────┤├ Horizon (15%)┤
[observed data......................][ MASKED...]

→ 모든 샘플이 다른 비율!
```

**효과**:
- 학습 시: 다양한 조합 경험
- 추론 시: 임의의 길이 처리 가능
- Zero-shot transfer to different horizons

**자세한 설명**: [`pretraining_task_and_data_distribution.md`](./pretraining_task_and_data_distribution.md)

---

## 추가 기여: Data Distribution

### 문제점
```
LOTSA 데이터셋:
- 27B observations
- 9 domains
- 다양한 frequencies
- 극심한 불균형 (finance >> weather)

단순 sampling: 큰 도메인만 학습됨
```

### 해결책: Sub-dataset Capping

**구현**:
```python
# From paper
p(D_k) = ω_k / Σ_i ω_i

where:
ω_k = min(|D_k| / Σ_i |D_i|, ε=0.001)
```

**예시**:
```
Before capping:
Finance: 80% → After: 0.1%
Weather: 15% → After: 0.1%
Energy: 5%  → After: 0.1%

→ 모든 sub-dataset이 균등하게 학습됨
```

**효과**:
- Domain balance
- Frequency balance
- Robust pre-trained model

**자세한 설명**: [`pretraining_task_and_data_distribution.md`](./pretraining_task_and_data_distribution.md)

---

## 전체 Architecture 흐름

```
Raw Data (varying everything)
  ↓
[Data Distribution: Sub-dataset sampling with capping]
  ↓
[Task Distribution: Random window & prediction length]
  ↓
[Patch Size Selection: Frequency-based constraints]
  ↓
Input: (batch, variates × time, patch_size)
  ↓
[MultiInSizeLinear: Multiple patch size projection] ← Contribution #1
  ↓
(batch, variates × time, d_model)
  ↓
[Transformer Encoder]
  ├─ [RoPE: Time axis encoding] ← Contribution #2
  ├─ [Binary Attention Bias: Variate axis encoding] ← Contribution #2
  └─ [Grouped Query Attention: Memory efficiency]
  ↓
(batch, variates × time, d_model)
  ↓
[MultiOutSizeLinear: Multiple patch size projection] ← Contribution #1
  ↓
[Distribution Parameters]
  ↓
[Mixture Distribution: Flexible prediction] ← Contribution #3
  ↓
Output: Distribution over (batch, variates × time, patch_size)
```

---

## 핵심 장점 요약

### Versatility (다재다능)
- **Any frequency**: Hourly to yearly
- **Any variates**: 1 to 128+
- **Any length**: Short to long
- **Any distribution**: Gaussian to heavy-tailed

### Robustness (강건성)
- **Domain balance**: All domains learned equally
- **Frequency balance**: All frequencies represented
- **Data augmentation**: Variate subsampling, multivariate construction

### Flexibility (유연성)
- **Zero-shot**: New frequencies, variate counts, lengths
- **Fine-tuning**: Easy adaptation to specific domains
- **Inference**: User-specified context and prediction lengths

---

## 문서 네비게이션

1. **Multi Patch Size**: [`multi_patch_size_explained.md`](./multi_patch_size_explained.md)
   - Non-overlapping patches
   - MultiInSizeLinear / MultiOutSizeLinear
   - Frequency-based constraints

2. **Any-variate Attention**: [`any_variate_attention.md`](./any_variate_attention.md)
   - Flatten strategy
   - RoPE (Rotary Position Encoding)
   - Binary Attention Bias

3. **Mixture Distribution**: [`moirai_method_explained_accurate.md`](./moirai_method_explained_accurate.md)
   - Mixture class implementation
   - Component distributions (Student-t, etc.)
   - Log-likelihood optimization

4. **Pre-training Pipeline**: [`pretraining_task_and_data_distribution.md`](./pretraining_task_and_data_distribution.md)
   - Data distribution with capping
   - Task distribution with random sampling
   - Data augmentation strategies

5. **Complete Data Flow**: [`complete_data_processing_flow.md`](./complete_data_processing_flow.md)
   - End-to-end pipeline
   - Step-by-step transformations
   - Integration example

6. **GQA Details**: [`gqa_detailed_explanation.md`](./gqa_detailed_explanation.md)
   - Grouped Query Attention mechanism
   - K/V sharing for memory efficiency
   - Shape transformations
