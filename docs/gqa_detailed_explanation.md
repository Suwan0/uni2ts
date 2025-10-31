# Grouped Query Attention (GQA) 완전 정복 🎯

## 핵심 질문: K/V가 4개인데 어떻게 Q 12개와 매칭되나?

**답: "공유(Sharing)"와 "반복(Repeat)"을 통해!**

---

## 1. 기본 개념: Head Sharing

### 일반 Multi-Head Attention (MHA)

```
Q: 12개 heads, 각각 독립적
K: 12개 heads, 각각 독립적  → Q와 1:1 매칭
V: 12개 heads, 각각 독립적

┌────┐    ┌────┐
│ Q₀ │ ←→ │ K₀ │ ←→ │ V₀ │  Head 0
└────┘    └────┘    └────┘

┌────┐    ┌────┐    ┌────┐
│ Q₁ │ ←→ │ K₁ │ ←→ │ V₁ │  Head 1
└────┘    └────┘    └────┘

...

┌────┐    ┌────┐    ┌────┐
│Q₁₁ │ ←→ │K₁₁ │ ←→ │V₁₁ │  Head 11
└────┘    └────┘    └────┘

총 12개의 독립적인 attention 계산
```

### Grouped Query Attention (GQA)

```
Q: 12개 heads, 각각 독립적
K: 4개 groups (4개만!)      → 여러 Q가 하나의 K/V를 공유!
V: 4개 groups (4개만!)

Group 0:                Group 1:
┌────┐    ┌────┐       ┌────┐    ┌────┐
│ Q₀ │ ─→ │    │       │ Q₃ │ ─→ │    │
└────┘    │ K₀ │       └────┘    │ K₁ │
┌────┐    │    │       ┌────┐    │    │
│ Q₁ │ ─→ │    │  ←→  │ Q₄ │ ─→ │    │  ←→  V₁
└────┘    │    │  V₀   └────┘    │    │
┌────┐    │    │       ┌────┐    │    │
│ Q₂ │ ─→ │    │       │ Q₅ │ ─→ │    │
└────┘    └────┘       └────┘    └────┘

3개 Q가 1개 K/V 공유!    3개 Q가 1개 K/V 공유!

Group 2:                Group 3:
┌────┐    ┌────┐       ┌────┐    ┌────┐
│ Q₆ │ ─→ │    │       │ Q₉ │ ─→ │    │
└────┘    │ K₂ │       └────┘    │ K₃ │
┌────┐    │    │       ┌────┐    │    │
│ Q₇ │ ─→ │    │  ←→  │Q₁₀ │ ─→ │    │  ←→  V₃
└────┘    │    │  V₂   └────┘    │    │
┌────┐    │    │       ┌────┐    │    │
│ Q₈ │ ─→ │    │       │Q₁₁ │ ─→ │    │
└────┘    └────┘       └────┘    └────┘

결과: 12개 Q → 4개 K/V (3:1 비율)
```

---

## 2. 구체적인 Shape 변화로 이해하기

### 설정
```python
d_model = 768       # 모델 차원
num_heads = 12      # Query heads
num_groups = 4      # Key/Value groups
head_dim = 64       # 768 / 12 = 64
heads_per_group = 3 # 12 / 4 = 3
batch_size = 2
seq_len = 100
```

### Step-by-Step 변환

```python
# Input
x = [batch=2, seq_len=100, d_model=768]

# ============================================
# Step 1: Linear Projection
# ============================================

# Query: 모든 head에 대해 projection
q = Q_proj(x)  # [2, 100, 768]

# Key: group 수만큼만 projection (핵심!)
k = K_proj(x)  # [2, 100, 256]  ← 768이 아니라 256!
               #           ↑ head_dim * num_groups = 64 * 4

# Value: group 수만큼만 projection
v = V_proj(x)  # [2, 100, 256]

# ============================================
# Step 2: Reshape to Multi-Head Format
# ============================================

# Query: [batch, seq, d_model] → [batch, num_groups, heads_per_group, seq, head_dim]
q = q.view(2, 100, 12, 64)              # [2, 100, 12, 64]
q = q.permute(0, 2, 1, 3)                # [2, 12, 100, 64]
q = q.view(2, 4, 3, 100, 64)             # [2, 4, 3, 100, 64]
#             ↑  ↑
#       groups  heads_per_group

# Key: [batch, seq, head_dim*groups] → [batch, num_groups, 1, seq, head_dim]
k = k.view(2, 100, 4, 64)                # [2, 100, 4, 64]
k = k.permute(0, 2, 1, 3)                # [2, 4, 100, 64]
k = k.unsqueeze(2)                       # [2, 4, 1, 100, 64]
#                                                  ↑ 하나의 K만!

# Value: 동일
v = v.view(2, 100, 4, 64)                # [2, 100, 4, 64]
v = v.permute(0, 2, 1, 3)                # [2, 4, 100, 64]
v = v.unsqueeze(2)                       # [2, 4, 1, 100, 64]

# ============================================
# Step 3: Repeat/Expand K and V (핵심!)
# ============================================

# K를 heads_per_group만큼 복제
k = k.expand(2, 4, 3, 100, 64)  # [2, 4, 1, 100, 64] → [2, 4, 3, 100, 64]
#                   ↑ 1을 3으로 확장 (복사)

# V도 동일하게 복제
v = v.expand(2, 4, 3, 100, 64)  # [2, 4, 1, 100, 64] → [2, 4, 3, 100, 64]

# 이제 q, k, v 모두 shape 동일: [2, 4, 3, 100, 64]
```

### 시각화: Expand 과정

```
K의 Group 0 (shape: [2, 4, 1, 100, 64]):

┌─────────────────┐
│  K₀  (64-dim)   │  ← 하나의 K 벡터
└─────────────────┘

↓ expand to heads_per_group=3

┌─────────────────┐
│  K₀  (64-dim)   │  ← Q₀와 매칭
├─────────────────┤
│  K₀  (64-dim)   │  ← Q₁와 매칭 (복사본!)
├─────────────────┤
│  K₀  (64-dim)   │  ← Q₂와 매칭 (복사본!)
└─────────────────┘

동일한 K₀를 3개의 Query head가 공유!
```

---

## 3. Attention 계산 과정

### Group 0 예시

```python
# Group 0: Q₀, Q₁, Q₂가 K₀, V₀를 공유

Q_group0 = [
    Q₀: [q₀₀, q₀₁, q₀₂, ..., q₀₆₃],  # 64-dim
    Q₁: [q₁₀, q₁₁, q₁₂, ..., q₁₆₃],  # 64-dim
    Q₂: [q₂₀, q₂₁, q₂₂, ..., q₂₆₃],  # 64-dim
]

K_group0 = [k₀₀, k₀₁, k₀₂, ..., k₀₆₃]  # 64-dim (하나만!)
V_group0 = [v₀₀, v₀₁, v₀₂, ..., v₀₆₃]  # 64-dim (하나만!)

# Expand 후
K_group0_expanded = [
    K₀: [k₀₀, k₀₁, ..., k₀₆₃],  # Q₀용 (복사본)
    K₀: [k₀₀, k₀₁, ..., k₀₆₃],  # Q₁용 (복사본)
    K₀: [k₀₀, k₀₁, ..., k₀₆₃],  # Q₂용 (복사본)
]

# Attention Score 계산
score₀ = Q₀ @ K₀ᵀ / √64  # Q₀와 K₀
score₁ = Q₁ @ K₀ᵀ / √64  # Q₁와 동일한 K₀
score₂ = Q₂ @ K₀ᵀ / √64  # Q₂와 동일한 K₀

# Attention Weight
weight₀ = softmax(score₀)
weight₁ = softmax(score₁)
weight₂ = softmax(score₂)

# Attention Output
out₀ = weight₀ @ V₀  # Q₀의 출력
out₁ = weight₁ @ V₀  # Q₁의 출력 (동일한 V₀ 사용!)
out₂ = weight₂ @ V₀  # Q₂의 출력 (동일한 V₀ 사용!)
```

### 핵심: Q는 다르지만 K/V는 같다!

```
Q₀, Q₁, Q₂는 서로 다른 벡터
→ 각각 다른 "관점"에서 정보를 query

하지만 K₀, V₀는 동일
→ 동일한 "정보 저장소"를 참조

결과:
- Q₀는 K₀와 매칭해서 자신만의 attention weight 계산
- Q₁는 동일한 K₀와 매칭하지만 다른 attention weight 계산
- Q₂도 동일한 K₀와 매칭하지만 또 다른 attention weight 계산

같은 정보를 다른 관점에서 보는 것!
```

---

## 4. 메모리 비교

### Parameter 수 계산

```python
d_model = 768
head_dim = 64
num_heads = 12
num_groups = 4

# ============================================
# Multi-Head Attention (MHA)
# ============================================
Q_proj: 768 × 768 = 589,824 params
K_proj: 768 × 768 = 589,824 params
V_proj: 768 × 768 = 589,824 params
O_proj: 768 × 768 = 589,824 params
─────────────────────────────────
Total:  2,359,296 params

# ============================================
# Grouped Query Attention (GQA, groups=4)
# ============================================
Q_proj: 768 × 768     = 589,824 params
K_proj: 768 × 256     = 196,608 params  ← 1/3 감소!
V_proj: 768 × 256     = 196,608 params  ← 1/3 감소!
O_proj: 768 × 768     = 589,824 params
─────────────────────────────────
Total:  1,572,864 params

절약: 786,432 params (약 33% 감소)

# ============================================
# Multi-Query Attention (MQA, groups=1)
# ============================================
Q_proj: 768 × 768     = 589,824 params
K_proj: 768 × 64      = 49,152 params   ← 1/12 감소!
V_proj: 768 × 64      = 49,152 params   ← 1/12 감소!
O_proj: 768 × 768     = 589,824 params
─────────────────────────────────
Total:  1,277,952 params

절약: 1,081,344 params (약 46% 감소)
```

### KV Cache 크기 (Inference 시)

```python
# 설정
batch_size = 1
seq_len = 1024
num_layers = 24

# ============================================
# MHA
# ============================================
K_cache: [batch, num_heads, seq_len, head_dim]
       = [1, 12, 1024, 64] = 786,432 values per layer
V_cache: [1, 12, 1024, 64] = 786,432 values per layer

Total per layer: 1,572,864 values
Total for 24 layers: 37,748,736 values

# float16 기준: 37.7M × 2 bytes = 75.5 MB

# ============================================
# GQA (groups=4)
# ============================================
K_cache: [batch, num_groups, seq_len, head_dim]
       = [1, 4, 1024, 64] = 262,144 values per layer
V_cache: [1, 4, 1024, 64] = 262,144 values per layer

Total per layer: 524,288 values  ← 1/3로 감소!
Total for 24 layers: 12,582,912 values

# float16 기준: 12.6M × 2 bytes = 25.2 MB

메모리 절약: 50.3 MB (약 67% 감소!) 🚀
```

---

## 5. 실제 코드 구현 (Uni2TS)

```python
# src/uni2ts/module/attention.py

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        num_groups: int = 4,
    ):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups  # 12 // 4 = 3
        self.head_dim = dim // num_heads                 # 768 // 12 = 64

        # Query: 모든 head
        self.q_proj = nn.Linear(dim, dim)  # 768 → 768

        # Key: group만큼만 (핵심!)
        self.k_proj = nn.Linear(
            dim,
            self.head_dim * num_groups  # 768 → 64*4 = 256
        )

        # Value: group만큼만
        self.v_proj = nn.Linear(
            dim,
            self.head_dim * num_groups  # 768 → 64*4 = 256
        )

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_dims = x.shape[:-2]
        seq_len = x.shape[-2]

        # ============================================
        # Step 1: Projection
        # ============================================
        q = self.q_proj(x)  # [..., seq, 768]
        k = self.k_proj(x)  # [..., seq, 256]
        v = self.v_proj(x)  # [..., seq, 256]

        # ============================================
        # Step 2: Reshape
        # ============================================
        # Query: [..., seq, 768] → [..., group, hpg, seq, head_dim]
        q = rearrange(
            q,
            '... seq (group hpg head_dim) -> ... group hpg seq head_dim',
            group=self.num_groups,
            hpg=self.heads_per_group,
            head_dim=self.head_dim,
        )
        # [..., 4, 3, seq, 64]

        # Key: [..., seq, 256] → [..., group, 1, seq, head_dim]
        k = rearrange(
            k,
            '... seq (group head_dim) -> ... group seq head_dim',
            group=self.num_groups,
            head_dim=self.head_dim,
        )
        # [..., 4, seq, 64]
        k = k.unsqueeze(-3)  # [..., 4, 1, seq, 64]

        # Value: 동일
        v = rearrange(
            v,
            '... seq (group head_dim) -> ... group seq head_dim',
            group=self.num_groups,
            head_dim=self.head_dim,
        )
        v = v.unsqueeze(-3)  # [..., 4, 1, seq, 64]

        # ============================================
        # Step 3: Expand (핵심!)
        # ============================================
        k = k.expand(*batch_dims, self.num_groups, self.heads_per_group, seq_len, self.head_dim)
        # [..., 4, 1, seq, 64] → [..., 4, 3, seq, 64]

        v = v.expand(*batch_dims, self.num_groups, self.heads_per_group, seq_len, self.head_dim)
        # [..., 4, 1, seq, 64] → [..., 4, 3, seq, 64]

        # 이제 q, k, v 모두 [..., 4, 3, seq, 64]

        # ============================================
        # Step 4: Scaled Dot-Product Attention
        # ============================================
        # scores: [..., group, hpg, seq_q, seq_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn_weights = F.softmax(scores, dim=-1)

        # out: [..., group, hpg, seq, head_dim]
        out = torch.matmul(attn_weights, v)

        # ============================================
        # Step 5: Reshape and Project
        # ============================================
        out = rearrange(
            out,
            '... group hpg seq head_dim -> ... seq (group hpg head_dim)',
        )
        # [..., seq, 768]

        out = self.out_proj(out)

        return out
```

---

## 6. 왜 이렇게 설계했나?

### 동기: Autoregressive Inference 최적화

```
Training 시:
- 모든 토큰을 동시에 처리
- KV cache 필요 없음
- MHA도 충분히 빠름

Inference 시 (자동 회귀 생성):
- 토큰을 하나씩 생성
- 매번 이전 토큰들의 K, V를 재계산하지 않기 위해 cache
- KV cache가 병목!

┌────────────────────────────────────────┐
│ Step 1: "The"                          │
│   K_cache: [The]                       │
│   V_cache: [The]                       │
├────────────────────────────────────────┤
│ Step 2: "cat"                          │
│   K_cache: [The, cat]                  │
│   V_cache: [The, cat]                  │
├────────────────────────────────────────┤
│ Step 3: "sat"                          │
│   K_cache: [The, cat, sat]            │
│   V_cache: [The, cat, sat]            │
└────────────────────────────────────────┘

시퀀스가 길어질수록 KV cache가 급격히 증가!
```

### GQA의 해결책

```
MHA:
  각 head마다 독립적인 K, V
  → KV cache: num_heads × seq_len × head_dim

GQA:
  여러 head가 K, V 공유
  → KV cache: num_groups × seq_len × head_dim

효과:
  num_heads=12, num_groups=4
  → KV cache 크기: 1/3로 감소
  → 메모리 절약 → 더 긴 시퀀스 처리 가능
  → Batch size 증가 가능
```

### 성능 Trade-off

```
┌─────────┬──────────┬──────────┬────────────┐
│ Method  │ KV Size  │ Speed    │ Quality    │
├─────────┼──────────┼──────────┼────────────┤
│ MHA     │ Large    │ Slow     │ Best ⭐⭐⭐⭐⭐ │
│ GQA     │ Medium   │ Fast     │ Good ⭐⭐⭐⭐  │
│ MQA     │ Small    │ Very Fast│ OK ⭐⭐⭐    │
└─────────┴──────────┴──────────┴────────────┘

GQA = 성능과 효율성의 최적 균형!
```

---

## 7. 구체적 예시로 완전 이해

### 시나리오: 문장 번역

```
Input: "The cat sat on the mat"
Tokens: [The, cat, sat, on, the, mat]
seq_len = 6

설정:
- d_model = 768
- num_heads = 12
- num_groups = 4
- heads_per_group = 3
```

### Token "sat"의 Attention (Group 0)

```python
# Token "sat"의 Query (3개 head)
Q₀_sat = [0.1, 0.2, ..., 0.5]  # 64-dim, Head 0의 관점
Q₁_sat = [0.3, 0.1, ..., 0.7]  # 64-dim, Head 1의 관점
Q₂_sat = [0.2, 0.4, ..., 0.3]  # 64-dim, Head 2의 관점

# 모든 토큰의 Key (1개만!)
K_The = [0.8, 0.3, ..., 0.2]  # 64-dim
K_cat = [0.5, 0.6, ..., 0.4]
K_sat = [0.7, 0.2, ..., 0.9]
K_on  = [0.4, 0.5, ..., 0.1]
K_the = [0.9, 0.1, ..., 0.6]
K_mat = [0.3, 0.8, ..., 0.5]

# ============================================
# Head 0: Q₀_sat로 attention 계산
# ============================================
score₀_The = Q₀_sat · K_The / √64 = 0.12
score₀_cat = Q₀_sat · K_cat / √64 = 0.45  ← 높음!
score₀_sat = Q₀_sat · K_sat / √64 = 0.31
score₀_on  = Q₀_sat · K_on  / √64 = 0.08
score₀_the = Q₀_sat · K_the / √64 = 0.02
score₀_mat = Q₀_sat · K_mat / √64 = 0.02

weights₀ = softmax([0.12, 0.45, 0.31, 0.08, 0.02, 0.02])
         = [0.15, 0.45, 0.28, 0.08, 0.02, 0.02]

out₀ = 0.15*V_The + 0.45*V_cat + 0.28*V_sat + ...
     → Head 0: "cat"에 집중!

# ============================================
# Head 1: Q₁_sat로 attention 계산
# ============================================
score₁_The = Q₁_sat · K_The / √64 = 0.08
score₁_cat = Q₁_sat · K_cat / √64 = 0.15
score₁_sat = Q₁_sat · K_sat / √64 = 0.22
score₁_on  = Q₁_sat · K_on  / √64 = 0.38  ← 높음!
score₁_the = Q₁_sat · K_the / √64 = 0.12
score₁_mat = Q₁_sat · K_mat / √64 = 0.05

weights₁ = softmax([0.08, 0.15, 0.22, 0.38, 0.12, 0.05])
         = [0.10, 0.14, 0.18, 0.36, 0.13, 0.09]

out₁ = 0.10*V_The + 0.14*V_cat + 0.18*V_sat + 0.36*V_on + ...
     → Head 1: "on"에 집중!

# ============================================
# Head 2: Q₂_sat로 attention 계산
# ============================================
score₂_The = Q₂_sat · K_The / √64 = 0.25  ← 높음!
score₂_cat = Q₂_sat · K_cat / √64 = 0.18
score₂_sat = Q₂_sat · K_sat / √64 = 0.20
score₂_on  = Q₂_sat · K_on  / √64 = 0.15
score₂_the = Q₂_sat · K_the / √64 = 0.12
score₂_mat = Q₂_sat · K_mat / √64 = 0.10

weights₂ = softmax([0.25, 0.18, 0.20, 0.15, 0.12, 0.10])
         = [0.28, 0.21, 0.22, 0.16, 0.08, 0.05]

out₂ = 0.28*V_The + 0.21*V_cat + 0.22*V_sat + ...
     → Head 2: "The"에 집중!
```

### 핵심 포인트

```
동일한 K, V를 사용하지만:
- Q₀, Q₁, Q₂가 다르므로
- 각 head가 다른 attention pattern을 학습!

Head 0: 동사-주어 관계 ("sat" ← "cat")
Head 1: 동사-전치사 관계 ("sat" → "on")
Head 2: 문장 구조 ("sat" ← "The")

다양한 관점을 유지하면서도 메모리 절약!
```

---

## 8. MHA vs GQA vs MQA 비교표

```
┌───────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│           │ MHA         │ GQA         │ MQA         │ 비고        │
├───────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ num_heads │ 12          │ 12          │ 12          │ Q heads     │
│ num_groups│ 12          │ 4           │ 1           │ KV groups   │
│ heads_per │ 1           │ 3           │ 12          │ Q per group │
│   _group  │             │             │             │             │
├───────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Q_proj    │ 768→768     │ 768→768     │ 768→768     │             │
│ K_proj    │ 768→768     │ 768→256     │ 768→64      │ 메모리 차이 │
│ V_proj    │ 768→768     │ 768→256     │ 768→64      │             │
├───────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ KV params │ 1,179,648   │ 393,216     │ 98,304      │ 33% / 8%    │
│ KV cache  │ 1.5M        │ 0.5M        │ 0.125M      │ 33% / 8%    │
├───────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Speed     │ 1.0x        │ 1.5x        │ 2.0x        │ Inference   │
│ Quality   │ ⭐⭐⭐⭐⭐     │ ⭐⭐⭐⭐       │ ⭐⭐⭐         │ Downstream  │
├───────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ 사용 사례 │ 품질 최우선 │ 균형잡힌    │ 속도 최우선 │             │
│           │ 학습 단계   │ 대부분 상황 │ 엣지 디바이스│             │
└───────────┴─────────────┴─────────────┴─────────────┴─────────────┘

GQA = Sweet Spot! 🎯
```

---

## 9. 핵심 요약

### GQA의 본질

```
1. Query는 많이 (num_heads=12)
   → 다양한 관점에서 정보 수집

2. Key/Value는 적게 (num_groups=4)
   → 메모리 절약

3. 공유 메커니즘
   → 3개의 Query head가 1개의 Key/Value group을 공유
   → expand/repeat로 shape 맞춤

4. 결과
   → Q가 다르므로 다양한 attention pattern 유지
   → KV가 적으므로 메모리 절약
   → 품질과 효율성의 균형!
```

### 왜 동작하는가?

```
핵심 통찰:
"Attention의 다양성은 주로 Query에서 나온다!"

- Query가 다르면 → attention weight가 달라짐
- Key/Value가 같아도 → 각 head는 다른 정보 추출

예:
Q₀: "주어와의 관계"에 집중하는 query
Q₁: "목적어와의 관계"에 집중하는 query
Q₂: "수식어와의 관계"에 집중하는 query

동일한 K, V라도:
→ Q₀는 주어에 높은 attention
→ Q₁는 목적어에 높은 attention
→ Q₂는 수식어에 높은 attention

다양성 유지하면서 메모리 절약! 💡
```

---

## 참고 자료

- **GQA 논문**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)
- **MQA 논문**: "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)
- **Uni2TS 구현**: `src/uni2ts/module/attention.py:58-306`
- **Llama 2**: GQA를 사용하는 대표적인 LLM (num_groups=8)
