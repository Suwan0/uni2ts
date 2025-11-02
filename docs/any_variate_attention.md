# Any-variate Attention: RoPE와 Binary Attention Bias 상세 설명

실제 코드를 기반으로 Moirai의 Any-variate Attention 메커니즘을 설명합니다.

## 논문의 Contribution 체크리스트

| 문제점 | 해결책 | 설명 문서 | 상태 |
|--------|--------|-----------|------|
| 1. Varying frequencies (다양한 주기) | Multiple patch size projection layers | `multi_patch_size_explained.md` | ✓ 완료 |
| 2. Varying dimensionality (다양한 variate 수) | **Any-variate Attention** (Flatten + RoPE + Binary Attention Bias) | `any_variate_attention.md` (이 문서) | ✓ 완료 |
| 3. Flexible predictive distributions | Mixture of distributions | `moirai_method_explained_accurate.md` | ✓ 완료 |
| 4. Flexible context/prediction lengths | Task distribution with random sampling | `pretraining_task_and_data_distribution.md` | ✓ 완료 |

이 문서는 **#2 Any-variate Attention**을 상세히 설명합니다.

---

## 1. 문제점: Varying Dimensionality

### 1.1 기존 Transformer의 한계

**전통적인 Multivariate Time Series Transformer**:
```python
# Input: (batch, time, variates)
x.shape = (32, 512, 7)  # 32 samples, 512 timesteps, 7 variates

# Transformer encoder는 time dimension에만 attention 적용
for layer in transformer_layers:
    # Self-attention along time axis
    x = layer(x)  # (batch, time, variates) → (batch, time, variates)

# 문제점:
# 1. Variate 수가 고정되어야 함 (7 variates)
# 2. 다른 variate 수를 가진 데이터는 별도 모델 필요
# 3. 학습 시 본 variate 수만 처리 가능
```

**Moirai가 해결해야 하는 문제**:
- 1개 variate (univariate): (batch, time, 1)
- 5개 variates: (batch, time, 5)
- 100개 variates: (batch, time, 100)
- → **하나의 모델이 모든 경우를 처리해야 함**

### 1.2 해결책 개요: Any-variate Attention

**핵심 아이디어**: Time과 Variate를 **하나의 sequence**로 flatten

```python
# Input: (batch, variates, time)
x.shape = (1, 5, 32)  # 1 sample, 5 variates, 32 patches

# Flatten: (batch, variates × time)
x_flat = rearrange(x, "batch var time -> batch (var time)")
x_flat.shape = (1, 160)  # 5 × 32 = 160

# Now transformer can handle ANY number of variates!
# - 1 variate × 100 patches = 100 tokens
# - 5 variates × 32 patches = 160 tokens
# - 100 variates × 5 patches = 500 tokens
# All are just "sequence length" to the transformer
```

**하지만 문제**: Flatten하면 time과 variate 정보가 섞임
- Token 0: (variate=0, time=0)
- Token 1: (variate=0, time=1)
- Token 32: (variate=1, time=0)
- Token 33: (variate=1, time=1)

**해결**:
1. **RoPE (Rotary Position Encoding)**: time axis 정보 보존
2. **Binary Attention Bias**: variate axis 정보 보존

---

## 2. Rotary Position Encoding (RoPE) - Time Axis

### 2.1 개념

**목적**: Token의 **시간적 위치(temporal position)** 정보를 인코딩

**특징**:
- Query와 Key에만 적용 (Value에는 적용 안 함)
- Rotation matrix로 embedding을 회전
- 상대적 위치 정보를 dot product에 자연스럽게 인코딩

### 2.2 실제 구현 (src/uni2ts/module/position/attn_projection.py:55-106)

```python
class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,          # projection dimension (head_dim의 일부)
        num_heads: int,           # 12
        num_groups: int,          # 4
        max_len: int = 512,       # maximum sequence length
        base: int = 10000,        # base for frequency computation
    ):
        super().__init__(proj_width, num_heads, num_groups)

        # Step 1: Compute theta (frequency for each dimension)
        # theta_i = 1 / (base^(2i / proj_width))
        self.register_buffer(
            "theta",
            1.0 / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float) / self.proj_width,
            ),
        )
        # theta.shape = (proj_width // 2,)
        # Example: proj_width=32 → theta.shape = (16,)

        # Step 2: Pre-compute cos/sin for all positions
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        # position: [0, 1, 2, ..., max_len-1]
        position = torch.arange(max_len, device=self.theta.device, dtype=self.theta.dtype)

        # m_theta: outer product of position and theta
        # m_theta[pos, i] = position[pos] * theta[i]
        m_theta = einsum(position, self.theta, "length, width -> length width")
        # m_theta.shape = (max_len, proj_width // 2)

        # Repeat each frequency twice for rotation
        m_theta = repeat(m_theta, "length width -> length (width 2)")
        # m_theta.shape = (max_len, proj_width)

        # Pre-compute cos and sin
        self.register_buffer("cos", torch.cos(m_theta))
        self.register_buffer("sin", torch.sin(m_theta))
        # cos.shape = sin.shape = (max_len, proj_width)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate x by 90 degrees in 2D plane.
        [x1, x2] → [-x2, x1]
        """
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)

    def forward(
        self,
        x: torch.Tensor,           # query or key
        seq_id: torch.Tensor,      # time_id for each token
    ) -> torch.Tensor:
        # seq_id: time index for each position in sequence
        # Example: seq_id = [0,1,2,...,31, 0,1,2,...,31, ...]
        #                    └─ var 0 ─┘ └─ var 1 ─┘

        self._init_freq(max_len=seq_id.max() + 1)

        # Select cos/sin based on time_id
        rot_cos = self.cos[seq_id]  # (..., seq, proj_width)
        rot_sin = self.sin[seq_id]  # (..., seq, proj_width)

        # Apply rotation: x_rotated = cos * x + sin * rotate(x)
        return rot_cos * x + rot_sin * self._rotate(x)
```

### 2.3 RoPE의 작동 원리

**수식**:
```
RoPE(x, pos) = [
    x[0] * cos(pos * θ₀) - x[1] * sin(pos * θ₀),
    x[0] * sin(pos * θ₀) + x[1] * cos(pos * θ₀),
    x[2] * cos(pos * θ₁) - x[3] * sin(pos * θ₁),
    x[2] * sin(pos * θ₁) + x[3] * cos(pos * θ₁),
    ...
]

where θᵢ = 1 / (10000^(2i / d))
```

**핵심 속성**:
```python
# 상대적 위치 정보가 dot product에 자연스럽게 인코딩됨
q_pos_i = RoPE(q, pos=i)
k_pos_j = RoPE(k, pos=j)

# Attention score는 상대적 거리 (j - i)에만 의존
score = q_pos_i · k_pos_j = f(j - i)
```

### 2.4 구체적 예시

```python
# Flattened sequence
seq_len = 160  # 5 variates × 32 patches
time_id = [0, 1, 2, ..., 31, 0, 1, 2, ..., 31, ...]  # (160,)
#          └──── var 0 ────┘ └──── var 1 ────┘

# Query and Key after linear projection
q.shape = (1, 4, 3, 160, 64)  # (batch, groups, heads_per_group, seq, head_dim)
k.shape = (1, 4, 3, 160, 64)

# Apply RoPE to first 50% of head_dim (partial_factor=(0.0, 0.5))
proj_width = int(64 * (0.5 - 0.0)) = 32

# Split head_dim: [0:0, 0:32, 32:64]
#                 └─┘  └──┘  └──┘
#                 skip proj  skip
q_parts = q.split([0, 32, 32], dim=-1)

# Apply RoPE to middle part
q_parts[1] = RotaryProjection.forward(q_parts[1], seq_id=time_id)
# Input: (1, 4, 3, 160, 32)
# rot_cos = cos[time_id] → shape (160, 32)
# rot_sin = sin[time_id] → shape (160, 32)
# Output: rot_cos * q_parts[1] + rot_sin * rotate(q_parts[1])

# Concatenate back
q_rotated = torch.cat(q_parts, dim=-1)  # (1, 4, 3, 160, 64)

# Same for key
k_rotated = ...

# Now attention scores encode temporal relationships
# Token at time=5 has high attention with time=4,6 (nearby)
# Token at time=5 has low attention with time=25 (far away)
```

**시각화**:
```
Sequence (flattened):
Token   0   1   2  ...  31  32  33  ... 159
        ├───────────────┤   ├──────────┤
Var     0   0   0  ...  0   1   1   ... 4
Time    0   1   2  ...  31  0   1   ... 31
        ↓   ↓   ↓       ↓   ↓   ↓       ↓
RoPE   θ₀  θ₁  θ₂  ... θ₃₁ θ₀  θ₁  ... θ₃₁

Attention pattern (time=5 in var 0 → all tokens):
Time 0-10:   High attention (nearby in time)
Time 15-25:  Medium attention
Time 30:     Low attention (far in time)

→ Time proximity is preserved after flattening!
```

---

## 3. Binary Attention Bias - Variate Axis

### 3.1 개념

**목적**: **같은 variate vs 다른 variate**를 구분하여 attention 조절

**핵심 아이디어**:
- 같은 variate의 token끼리는 더 강한 attention
- 다른 variate의 token끼리는 상대적으로 약한 attention
- 2개의 학습 가능한 bias 값으로 구현

### 3.2 실제 구현 (src/uni2ts/module/position/attn_bias.py:67-87)

```python
class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int):
        super().__init__(dim, num_heads, num_groups)

        # 2개의 학습 가능한 embedding
        # emb.weight[0]: different variate bias
        # emb.weight[1]: same variate bias
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=num_heads)
        # emb.weight.shape = (2, num_heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_id: torch.Tensor,      # variate_id for query tokens
        kv_id: torch.Tensor,          # variate_id for key tokens
    ) -> torch.Tensor:
        # query_id.shape = (batch, 1, 1, q_len)
        # kv_id.shape = (batch, 1, 1, kv_len)

        # Step 1: Check if query and key have same variate_id
        # ind[i, j] = True if query[i] and key[j] are from same variate
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        # ind.shape = (batch, 1, 1, q_len, kv_len)

        # Step 2: Get learned bias weights
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        # weight.shape = (2, num_heads, 1, 1)
        # weight[0]: bias for different variates
        # weight[1]: bias for same variates

        # Step 3: Select appropriate bias
        # If ind[i,j] == False (different variate): use weight[0]
        # If ind[i,j] == True (same variate): use weight[1]
        bias = rearrange(
            ~ind * weight[:1] + ind * weight[1:],
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )
        # bias.shape = (batch, group, hpg, q_len, kv_len)

        return bias
```

### 3.3 Binary Attention Bias의 작동 원리

**Attention 계산에서의 역할**:
```python
# Standard attention
attn_weight = (Q @ K.T) / sqrt(d_k)
attn_weight = softmax(attn_weight)

# With Binary Attention Bias
attn_weight = (Q @ K.T) / sqrt(d_k) + binary_attention_bias
attn_weight = softmax(attn_weight)

# binary_attention_bias[i, j] =
#   - weight[1] if variate_id[i] == variate_id[j]  (same variate)
#   - weight[0] if variate_id[i] != variate_id[j]  (different variate)
```

**학습 과정**:
- 초기: weight[0]과 weight[1]이 랜덤 초기화
- 학습 중:
  - weight[1]이 양수로 증가 → 같은 variate 간 attention 증가
  - weight[0]이 음수 또는 작은 값 → 다른 variate 간 attention 감소
- 결과: 각 variate의 temporal pattern을 독립적으로 학습하면서도 필요시 cross-variate 정보 교환

### 3.4 구체적 예시

```python
# Flattened sequence
seq_len = 160  # 5 variates × 32 patches
variate_id = [0, 0, 0, ..., 0, 1, 1, 1, ..., 1, ..., 4, 4, ..., 4]  # (160,)
#             └──────────┘ └──────────┘       └──────────┘
#             32 tokens    32 tokens          32 tokens
#             (var 0)      (var 1)            (var 4)

# Binary Attention Bias weights (learned)
weight = self.emb.weight
# weight[0] = [-0.5, -0.3, 0.1, ...]  # 12 values for 12 heads (different variate)
# weight[1] = [1.2, 0.8, 1.5, ...]    # 12 values for 12 heads (same variate)

# Compute bias matrix
query_var_id = variate_id.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 160)
kv_var_id = variate_id.unsqueeze(0).unsqueeze(0).unsqueeze(0)     # (1, 1, 1, 160)

# ind[i, j] = True if variate_id[i] == variate_id[j]
ind = torch.eq(query_var_id.unsqueeze(-1), kv_var_id.unsqueeze(-2))
# ind.shape = (1, 1, 1, 160, 160)

# Example: ind matrix (simplified for 10 tokens: var 0,0,0,0,0, var 1,1,1,1,1)
#          Key:  0  0  0  0  0  1  1  1  1  1
# Query 0  var0  T  T  T  T  T  F  F  F  F  F
# Query 1  var0  T  T  T  T  T  F  F  F  F  F
# Query 2  var0  T  T  T  T  T  F  F  F  F  F
# Query 3  var0  T  T  T  T  T  F  F  F  F  F
# Query 4  var0  T  T  T  T  T  F  F  F  F  F
# Query 5  var1  F  F  F  F  F  T  T  T  T  T
# Query 6  var1  F  F  F  F  F  T  T  T  T  T
# Query 7  var1  F  F  F  F  F  T  T  T  T  T
# Query 8  var1  F  F  F  F  F  T  T  T  T  T
# Query 9  var1  F  F  F  F  F  T  T  T  T  T

# Apply bias (for head 0 as example)
bias = ~ind * weight[0, 0] + ind * weight[1, 0]
#    = ~ind * (-0.5) + ind * 1.2

# Result: bias matrix (head 0)
#          Key:   0     0     0     0     0     1     1     1     1     1
# Query 0  var0  1.2   1.2   1.2   1.2   1.2  -0.5  -0.5  -0.5  -0.5  -0.5
# Query 1  var0  1.2   1.2   1.2   1.2   1.2  -0.5  -0.5  -0.5  -0.5  -0.5
# ...
# Query 5  var1 -0.5  -0.5  -0.5  -0.5  -0.5   1.2   1.2   1.2   1.2   1.2
# Query 6  var1 -0.5  -0.5  -0.5  -0.5  -0.5   1.2   1.2   1.2   1.2   1.2
# ...

# Add to attention scores
attn_scores = Q @ K.T / sqrt(d_k)  # (-inf, +inf)
attn_scores = attn_scores + bias    # same variate: +1.2, diff variate: -0.5
attn_probs = softmax(attn_scores)   # (0, 1)

# Result:
# - Tokens from same variate get HIGHER attention (boosted by +1.2)
# - Tokens from different variates get LOWER attention (reduced by -0.5)
```

**시각화**:
```
Attention pattern for Token 35 (Var 1, Time 3):

Before Binary Attention Bias:
Var 0: ████████░░░░░░░░ (uniform attention)
Var 1: ████████████████ (uniform attention)
Var 2: ████████░░░░░░░░ (uniform attention)
Var 3: ████████░░░░░░░░ (uniform attention)
Var 4: ████████░░░░░░░░ (uniform attention)

After Binary Attention Bias:
Var 0: ██░░░░░░░░░░░░░░ (reduced by bias)
Var 1: ████████████████████ (boosted by bias)
Var 2: ██░░░░░░░░░░░░░░ (reduced by bias)
Var 3: ██░░░░░░░░░░░░░░ (reduced by bias)
Var 4: ██░░░░░░░░░░░░░░ (reduced by bias)

→ Focus more on same variate (Var 1)
```

---

## 4. Any-variate Attention 전체 메커니즘

### 4.1 GroupedQueryAttention Forward Pass

**src/uni2ts/module/attention.py:231-305**:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,                      # 768
        num_heads: int,                # 12
        num_groups: int,               # 4 (for GQA)
        var_attn_bias: Optional[...] = None,    # BinaryAttentionBias
        time_qk_proj: Optional[...] = None,     # RotaryProjection
    ):
        ...
        self.var_attn_bias = var_attn_bias() if var_attn_bias else None
        self.time_qk_proj = time_qk_proj() if time_qk_proj else None

    def forward(
        self,
        query: torch.Tensor,                  # (batch, seq_len, dim)
        key: torch.Tensor,                    # (batch, seq_len, dim)
        value: torch.Tensor,                  # (batch, seq_len, dim)
        attn_mask: Optional[torch.Tensor],    # (batch, seq_len, seq_len)
        query_var_id: Optional[torch.Tensor], # (batch, seq_len) - variate_id
        kv_var_id: Optional[torch.Tensor],    # (batch, seq_len) - variate_id
        query_time_id: Optional[torch.Tensor],# (batch, seq_len) - time_id
        kv_time_id: Optional[torch.Tensor],   # (batch, seq_len) - time_id
    ) -> torch.Tensor:
        # Step 1: Linear projections
        query = self.q_proj(query)  # (batch, seq_len, dim)
        key = self.k_proj(key)      # (batch, seq_len, dim)
        value = self.v_proj(value)  # (batch, seq_len, dim)

        # Step 2: Reshape for multi-head attention
        query = rearrange(
            query,
            "... q_len (group hpg dim) -> ... group hpg q_len dim",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )  # (batch, 4, 3, seq_len, 64)

        key = repeat(
            key,
            "... kv_len (group dim) -> ... group hpg kv_len dim",
            group=self.num_groups,
            hpg=self.heads_per_group,
        )  # (batch, 4, 3, seq_len, 64) - repeated for GQA

        value = repeat(...)  # Same as key

        # Step 3: Prepare IDs
        query_var_id = rearrange(query_var_id, "... q_len -> ... 1 1 q_len")
        kv_var_id = rearrange(kv_var_id, "... kv_len -> ... 1 1 kv_len")
        query_time_id = rearrange(query_time_id, "... q_len -> ... 1 1 q_len")
        kv_time_id = rearrange(kv_time_id, "... kv_len -> ... 1 1 kv_len")

        # Step 4: Apply Binary Attention Bias (variate axis)
        attn_bias = 0
        if self.var_attn_bias is not None:
            attn_bias = attn_bias + self.var_attn_bias(
                query, key,
                query_id=query_var_id,
                kv_id=kv_var_id,
            )
        # attn_bias.shape = (batch, group, hpg, seq_len, seq_len)

        # Step 5: Apply RoPE (time axis)
        if self.time_qk_proj is not None:
            query, key = self.time_qk_proj(
                query, key,
                query_id=query_time_id,
                kv_id=kv_time_id,
            )
        # Rotates first 50% of head_dim based on time_id

        # Step 6: Scaled Dot-Product Attention
        out = F.scaled_dot_product_attention(
            query,         # (batch, 4, 3, seq_len, 64)
            key,           # (batch, 4, 3, seq_len, 64)
            value,         # (batch, 4, 3, seq_len, 64)
            attn_mask=attn_bias,  # (batch, 4, 3, seq_len, seq_len)
            dropout_p=self.attn_dropout_p,
            scale=self.softmax_scale,
        )
        # out.shape = (batch, 4, 3, seq_len, 64)

        # Step 7: Reshape back
        out = rearrange(out, "... group hpg q_len dim -> ... q_len (group hpg dim)")
        # out.shape = (batch, seq_len, 768)

        return self.out_proj(out)
```

### 4.2 MoiraiModule에서의 사용

**src/uni2ts/model/moirai/module.py:126-147**:

```python
self.encoder = TransformerEncoder(
    d_model,
    num_layers,
    num_heads=None,  # Will default to appropriate value
    ...
    var_attn_bias_layer=partial(BinaryAttentionBias),
    time_qk_proj_layer=partial(
        QueryKeyProjection,
        proj_layer=RotaryProjection,
        kwargs=dict(max_len=max_seq_len),
        partial_factor=(0.0, 0.5),  # Apply RoPE to first 50% of head_dim
    ),
    shared_var_attn_bias=False,    # Each layer has own binary bias
    shared_time_qk_proj=True,      # All layers share RoPE
)
```

### 4.3 완전한 예시: End-to-End

```python
# ========== Input ==========
# After flattening: 5 variates × 32 patches = 160 tokens
target.shape = (1, 160, 128)  # (batch, seq_len, max_patch_size)
variate_id = [0,0,0,...,0, 1,1,1,...,1, ..., 4,4,...,4]  # (160,)
             └── 32 ──┘ └── 32 ──┘       └─ 32 ─┘
time_id = [0,1,2,...,31, 0,1,2,...,31, ..., 0,1,...,31]  # (160,)

# ========== Input Projection ==========
reprs = self.in_proj(target, patch_size)  # (1, 160, 768)

# ========== Transformer Layer ==========
# Each layer has GroupedQueryAttention

# 1. Linear projections (Q, K, V)
Q = self.q_proj(reprs)  # (1, 160, 768)
K = self.k_proj(reprs)  # (1, 160, 768)
V = self.v_proj(reprs)  # (1, 160, 768)

# 2. Reshape for multi-head
Q = rearrange(Q, "b s (g h d) -> b g h s d", g=4, h=3)  # (1, 4, 3, 160, 64)
K = repeat(K, "b s (g d) -> b g h s d", g=4, h=3)       # (1, 4, 3, 160, 64)
V = repeat(V, "b s (g d) -> b g h s d", g=4, h=3)       # (1, 4, 3, 160, 64)

# 3. Binary Attention Bias
var_bias = BinaryAttentionBias()(
    Q, K,
    query_id=variate_id,  # (160,) → (1, 1, 1, 160)
    kv_id=variate_id,     # (160,) → (1, 1, 1, 160)
)
# var_bias.shape = (1, 4, 3, 160, 160)
# var_bias[0, 0, 0, i, j] = weight[1] if variate_id[i] == variate_id[j]
#                         = weight[0] if variate_id[i] != variate_id[j]

# 4. RoPE
Q_rot, K_rot = RotaryProjection()(
    Q, K,
    query_id=time_id,  # (160,)
    kv_id=time_id,     # (160,)
)
# Rotates first 32 dimensions (50% of 64) based on time_id

# 5. Attention
attn_scores = Q_rot @ K_rot.transpose(-2, -1) / sqrt(64)
# attn_scores.shape = (1, 4, 3, 160, 160)

attn_scores = attn_scores + var_bias
# Add binary attention bias

attn_probs = softmax(attn_scores, dim=-1)
# attn_probs.shape = (1, 4, 3, 160, 160)

out = attn_probs @ V
# out.shape = (1, 4, 3, 160, 64)

# 6. Reshape back
out = rearrange(out, "b g h s d -> b s (g h d)")  # (1, 160, 768)

# ========== Result ==========
# Each token's representation now contains:
# - Temporal information from nearby timesteps (via RoPE)
# - Information primarily from same variate (via Binary Attention Bias)
# - Some cross-variate information (when beneficial)
```

**Attention Pattern 시각화** (Token 35: Var 1, Time 3):

```
                Time 0-31 (Var 0)   Time 0-31 (Var 1)   Time 0-31 (Var 2-4)
Attention       ███░░░░░░░░░░░░░    ████████████████    ███░░░░░░░░░░░░░
Strength        Low (diff var)      High (same var)     Low (diff var)

Within Var 1:   Time 0  1  2  3  4  5  ... 31
                ███ ███████████████ ███    ██
                    └─ High for nearby time (RoPE)

Combined effect:
- Highest attention: Same variate (Var 1) + nearby time (2, 3, 4, 5)
- Medium attention: Same variate (Var 1) + far time (0, 1, 30, 31)
- Low attention: Different variate (Var 0, 2, 3, 4) regardless of time
```

---

## 5. 왜 "Any-variate"인가?

### 5.1 유연성 (Flexibility)

**훈련 시**:
```python
# Sample 1: 3 variates × 40 patches = 120 tokens
# Sample 2: 15 variates × 10 patches = 150 tokens
# Sample 3: 1 variate × 200 patches = 200 tokens

# All can be processed in the same batch!
# Transformer only sees "sequence length", not "variates"
```

**추론 시**:
```python
# Model trained on 1-128 variates
# Can handle ANY number of variates in [1, 128]

# Example: 73 variates × 25 patches = 1825 tokens
# No problem! Just flatten and process
```

### 5.2 Generalization

**RoPE의 기여**:
- Temporal patterns는 variate 수와 무관
- Time=5와 Time=6의 관계는 1개 variate든 100개 variate든 동일
- → Temporal modeling은 일반화됨

**Binary Attention Bias의 기여**:
- Same vs Different variate 구분만 학습
- 구체적인 variate 개수는 무관
- → Variate modeling도 일반화됨

### 5.3 비교: 기존 방식 vs Any-variate Attention

| 측면 | 기존 방식 | Any-variate Attention |
|------|-----------|----------------------|
| **Input shape** | (batch, time, variates) 고정 | (batch, variates × time) 가변 |
| **Variate 수** | 고정 (예: 7) | 임의 (1~128) |
| **새 variate 수** | 재학습 필요 | 추가 학습 불필요 |
| **Position encoding** | Absolute position | RoPE (상대적 시간) |
| **Variate 관계** | 명시적 모델링 필요 | Binary Attention Bias |
| **Scalability** | 제한적 | 우수 |

---

## 6. 실제 효과

### 6.1 Pre-training

```python
# Batch with diverse variate counts
Sample 1: 5 variates (stock prices) → 160 tokens
Sample 2: 1 variate (temperature) → 32 tokens
Sample 3: 37 variates (sensors) → 1184 tokens

# All processed together!
# Model learns:
# - Temporal patterns (via RoPE)
# - Within-variate relationships (via Binary Attention Bias)
# - Cross-variate relationships (when weight[0] is not too negative)
```

### 6.2 Zero-shot Inference

```python
# Trained on 1-128 variates
# Test on new dataset with 50 variates (never seen 50 before)

# RoPE: Temporal patterns generalize
# - Time relationships are same regardless of variate count

# Binary Attention Bias: Variate separation generalizes
# - Same/different variate logic applies to any count

# Result: Good performance without fine-tuning!
```

---

## 7. 핵심 정리

### 7.1 Any-variate Attention의 3가지 핵심

1. **Flatten**: Time과 Variate를 하나의 sequence로 변환
   - 임의의 variate 수 처리 가능
   - Transformer의 유연성 활용

2. **RoPE** (Time Axis):
   - 시간적 위치 정보 인코딩
   - 상대적 거리 보존
   - Generalization to unseen sequence lengths

3. **Binary Attention Bias** (Variate Axis):
   - Same vs different variate 구분
   - Within-variate attention 강화
   - Cross-variate information 제어

### 7.2 문제 해결 확인

| 문제 | 해결책 | 메커니즘 |
|------|--------|----------|
| **Varying variate counts** | Flatten | (var, time) → (var × time) |
| **Temporal relationships** | RoPE | Rotation based on time_id |
| **Variate relationships** | Binary Attention Bias | Learned same/diff bias |
| **Generalization** | Relative encoding | 상대적 위치 & binary 구분 |

### 7.3 코드 위치

- **RotaryProjection**: `src/uni2ts/module/position/attn_projection.py:55-106`
- **BinaryAttentionBias**: `src/uni2ts/module/position/attn_bias.py:67-87`
- **GroupedQueryAttention**: `src/uni2ts/module/attention.py:58-306`
- **MoiraiModule setup**: `src/uni2ts/model/moirai/module.py:126-147`
- **Flatten operation**: `src/uni2ts/transform/reshape.py` (FlatPackCollection, FlatPackFields)
