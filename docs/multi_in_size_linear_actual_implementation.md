# MultiInSizeLinear 실제 구현 상세 분석

## 문서의 설명 vs 실제 구현

### ❌ 제가 문서에 잘못 쓴 내용

```python
# 이건 제가 개념적으로 단순화한 것 (실제 코드 아님!)
self.weights = nn.ParameterDict({
    str(size): nn.Parameter(torch.randn(size, out_features))
    for size in in_features
})
```

### ✅ 실제 구현 (src/uni2ts/module/ts_embed.py)

```python
class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],  # 예: (8, 16, 32, 64, 128)
        out_features: int,                 # 예: 768
        bias: bool = True,
    ):
        super().__init__()

        # 핵심: 하나의 큰 텐서로 모든 weight 저장!
        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls))
            )
        )
        # Shape: (5, 768, 128)
        #         ↑   ↑    ↑
        #         │   │    └─ max patch_size (128)
        #         │   └────── output features (d_model)
        #         └────────── patch_size 개수 (5개)
```

## 왜 이렇게 구현했나?

### 구조 분석

```python
in_features_ls = (8, 16, 32, 64, 128)
out_features = 768

# Weight tensor shape
self.weight: (5, 768, 128)

# 각 인덱스별 의미:
weight[0]:  (768, 128)  # patch_size=8용 weight
weight[1]:  (768, 128)  # patch_size=16용 weight
weight[2]:  (768, 128)  # patch_size=32용 weight
weight[3]:  (768, 128)  # patch_size=64용 weight
weight[4]:  (768, 128)  # patch_size=128용 weight
```

### Masking 전략

**문제**: patch_size=8인데 weight shape는 (768, 128)

**해결**: Mask를 사용해서 실제 사용하는 부분만 활성화!

```python
# src/uni2ts/module/ts_embed.py (초기화 부분)

self.register_buffer(
    "mask",
    rearrange(
        size_to_mask(max(in_features_ls), torch.as_tensor(in_features_ls)),
        "num_feats max_feat -> num_feats 1 max_feat",
    ),
    persistent=False,
)
```

### size_to_mask 함수

```python
# src/uni2ts/module/ts_embed.py

def size_to_mask(
    max_size: int,
    size: Int[torch.Tensor, "batch"]
) -> Bool[torch.Tensor, "batch max_size"]:
    """
    각 size에 대해 마스크 생성

    예: max_size=128, size=[8, 16, 32, 64, 128]

    return:
    [[1,1,1,1,1,1,1,1,0,0,0,...,0],  # size=8:  처음 8개만 True
     [1,1,1,...,1,0,0,...,0],        # size=16: 처음 16개만 True
     [1,1,1,...,1,0,0,...,0],        # size=32: 처음 32개만 True
     [1,1,1,...,1,0,0,...,0],        # size=64: 처음 64개만 True
     [1,1,1,...,1,1,1,...,1]]        # size=128: 전부 True
    """
    return torch.arange(max_size).unsqueeze(0) < size.unsqueeze(-1)
```

### Mask 시각화

```python
in_features_ls = (8, 16, 32, 64, 128)

# mask shape: (5, 1, 128)

mask[0]: [1,1,1,1,1,1,1,1, 0,0,0,...,0]  # 처음 8개만 활성화
         └────── 8개 ─────┘ └── 120개 ──┘

mask[1]: [1,1,1,...,1, 0,0,0,...,0]      # 처음 16개만 활성화
         └─── 16개 ──┘ └── 112개 ──┘

mask[2]: [1,1,1,...,1, 0,0,0,...,0]      # 처음 32개만 활성화
         └─── 32개 ──┘ └── 96개 ───┘

mask[3]: [1,1,1,...,1, 0,0,0,...,0]      # 처음 64개만 활성화
         └─── 64개 ──┘ └── 64개 ───┘

mask[4]: [1,1,1,...,1,1,1,1,1,1,1,1]     # 전부 활성화 (128개)
         └──────── 128개 ─────────┘
```

## Forward Pass 상세 분석

```python
def forward(
    self,
    x: Float[torch.Tensor, "*batch max_feat"],     # (batch, seq, 128)
    in_feat_size: Int[torch.Tensor, "*batch"],      # (batch, seq)
) -> Float[torch.Tensor, "*batch out_feat"]:       # (batch, seq, 768)

    out = 0  # 초기화

    # 모든 patch_size에 대해 순회
    for idx, feat_size in enumerate(self.in_features_ls):
        # idx=0, feat_size=8
        # idx=1, feat_size=16
        # ...
        # idx=4, feat_size=128

        # Step 1: Weight에 mask 적용
        weight = self.weight[idx] * self.mask[idx]
        # weight: (768, 128)
        # mask[idx]: (1, 128)
        # 결과: (768, 128)에서 처음 feat_size개만 유효, 나머지는 0

        # Step 2: Bias 가져오기
        bias = self.bias[idx] if self.bias is not None else 0
        # bias: (768,)

        # Step 3: 현재 샘플의 patch_size와 일치하는지 확인
        # in_feat_size: (batch, seq) = [16, 16, 16, ..., 16]
        # feat_size: 8 (첫 번째 iteration)
        match = torch.eq(in_feat_size, feat_size)
        # match: (batch, seq) = [False, False, ..., False] (모두 16이므로)

        # feat_size: 16 (두 번째 iteration)
        match = torch.eq(in_feat_size, feat_size)
        # match: (batch, seq) = [True, True, ..., True] (모두 16!)

        # Step 4: Linear transformation
        # einsum: "out inp, ... inp -> ... out"
        # weight: (768, 128)
        # x: (batch, seq, 128)
        # result: (batch, seq, 768)
        linear_out = einsum(weight, x, "out inp, ... inp -> ... out") + bias

        # Step 5: match인 경우만 출력에 추가
        out = out + match.unsqueeze(-1) * linear_out
        # match.unsqueeze(-1): (batch, seq, 1)
        # linear_out: (batch, seq, 768)
        # 결과: match=True인 위치만 linear_out 값 사용

    return out
```

## 구체적 예시

### 입력

```python
batch_size = 2
seq_len = 3
in_features_ls = (8, 16, 32)
out_features = 4  # 간단히 4로 설정

x = torch.randn(2, 3, 32)  # (batch, seq, max_patch_size)
in_feat_size = torch.tensor([
    [16, 16, 32],  # 샘플 0: 패치 0,1은 size=16, 패치 2는 size=32
    [8, 16, 16]    # 샘플 1: 패치 0은 size=8, 패치 1,2는 size=16
])  # (2, 3)
```

### Weight & Mask

```python
# weight shape: (3, 4, 32)
weight[0]: (4, 32)  # patch_size=8용
weight[1]: (4, 32)  # patch_size=16용
weight[2]: (4, 32)  # patch_size=32용

# mask shape: (3, 1, 32)
mask[0]: [1,1,1,1,1,1,1,1, 0,0,...,0]  # 처음 8개만
mask[1]: [1,1,1,...,1, 0,0,...,0]      # 처음 16개만
mask[2]: [1,1,1,...,1,1,1,...,1]       # 전부 (32개)
```

### Forward Pass - Iteration 0 (feat_size=8)

```python
idx = 0, feat_size = 8

# Step 1: Masked weight
weight_masked = weight[0] * mask[0]
# (4, 32), 처음 8개만 유효값, 나머지는 0

# Step 2: Match 확인
match = torch.eq(in_feat_size, 8)
# [[False, False, False],  # 샘플 0: 모두 16 또는 32
#  [True,  False, False]]  # 샘플 1: 첫 패치만 8!

# Step 3: Linear transformation
linear_out = einsum(weight_masked, x, "out inp, ... inp -> ... out")
# (2, 3, 4)

# Step 4: 선택적 추가
out = 0 + match.unsqueeze(-1) * linear_out
# match.unsqueeze(-1): (2, 3, 1)
# 결과: 샘플 1의 패치 0만 값이 들어감, 나머지는 0
```

### Forward Pass - Iteration 1 (feat_size=16)

```python
idx = 1, feat_size = 16

# Match 확인
match = torch.eq(in_feat_size, 16)
# [[True,  True,  False],  # 샘플 0: 패치 0,1이 16
#  [False, True,  True]]   # 샘플 1: 패치 1,2가 16

# Masked weight
weight_masked = weight[1] * mask[1]
# (4, 32), 처음 16개만 유효값

# Linear transformation + 추가
linear_out = einsum(weight_masked, x, "out inp, ... inp -> ... out")
out = out + match.unsqueeze(-1) * linear_out

# 현재 out 상태:
# - 샘플 0, 패치 0,1: feat_size=16의 linear_out
# - 샘플 1, 패치 0: feat_size=8의 linear_out
# - 샘플 1, 패치 1,2: feat_size=16의 linear_out
```

### Forward Pass - Iteration 2 (feat_size=32)

```python
idx = 2, feat_size = 32

# Match 확인
match = torch.eq(in_feat_size, 32)
# [[False, False, True],   # 샘플 0: 패치 2만 32
#  [False, False, False]]  # 샘플 1: 없음

# Masked weight
weight_masked = weight[2] * mask[2]
# (4, 32), 전부 유효값 (마스크가 전부 1)

# Linear transformation + 추가
linear_out = einsum(weight_masked, x, "out inp, ... inp -> ... out")
out = out + match.unsqueeze(-1) * linear_out

# 최종 out:
# - 샘플 0, 패치 0,1: feat_size=16의 출력
# - 샘플 0, 패치 2:   feat_size=32의 출력
# - 샘플 1, 패치 0:   feat_size=8의 출력
# - 샘플 1, 패치 1,2: feat_size=16의 출력
```

## 왜 이렇게 구현했나?

### 장점

```
1. 메모리 효율성:
   - 하나의 연속된 텐서 (5, 768, 128)
   - nn.ParameterDict보다 메모리 locality 좋음
   - GPU에서 효율적

2. 병렬 처리:
   - 모든 patch_size에 대해 한 번에 계산
   - 조건문 없이 선택 (torch.eq + broadcasting)

3. 배치 처리:
   - 배치 내 서로 다른 patch_size 동시 처리 가능
   - 샘플마다 다른 patch_size 사용 가능

4. 구현 단순성:
   - 모든 weight가 같은 shape
   - 초기화 루프만 필요
```

### 단점

```
1. 메모리 낭비:
   - weight[0]은 (768, 128)인데 8개만 사용, 120개는 0
   - 전체 weight의 상당 부분이 0으로 채워짐

2. 계산 낭비:
   - 모든 patch_size에 대해 einsum 계산
   - 대부분은 match=False여서 버려짐
   - 하지만 GPU 병렬 처리로 상쇄
```

## 실제 사용 예시

```python
# 모델 초기화
multi_linear = MultiInSizeLinear(
    in_features_ls=(8, 16, 32, 64, 128),
    out_features=768,
    bias=True
)

# 파라미터 크기
print(multi_linear.weight.shape)  # (5, 768, 128)
print(multi_linear.bias.shape)    # (5, 768)
print(multi_linear.mask.shape)    # (5, 1, 128)

# Forward pass
batch = 2
seq = 100
x = torch.randn(batch, seq, 128)  # 모든 패치가 128로 패딩됨

# 각 패치의 실제 크기
in_feat_size = torch.randint(0, 5, (batch, seq))  # 0~4 인덱스
# 0 → patch_size=8
# 1 → patch_size=16
# 2 → patch_size=32
# 3 → patch_size=64
# 4 → patch_size=128

# 실제로는 patch_size 값을 직접 전달
in_feat_size = torch.full((batch, seq), 16)  # 모두 16

output = multi_linear(x, in_feat_size)
print(output.shape)  # (batch, seq, 768)
```

## 핵심 포인트 정리

### 구조

```
1개의 큰 텐서:
  weight: (num_sizes, out_features, max_size)
         = (5, 768, 128)

Mask로 유효 영역 제한:
  mask[i]: 처음 in_features_ls[i]개만 1, 나머지 0

Forward에서 선택:
  - 모든 size에 대해 계산
  - torch.eq로 현재 샘플의 size와 매칭
  - 매칭되는 것만 출력에 추가
```

### 효율성

```
✓ GPU 친화적: 병렬 계산
✓ 배치 유연성: 샘플마다 다른 patch_size 가능
✓ 구현 간결성: 조건문 없는 선택

✗ 메모리 낭비: 0으로 채워진 부분 존재
✗ 계산 낭비: 매칭 안 되는 계산 버려짐
  (하지만 GPU에서는 병렬이므로 문제 없음)
```

---

## 정정 사항

제가 이전 문서(`moirai_method_explained.md`)에서 설명한 내용은 **개념적으로 맞지만 구현이 다릅니다**.

**개념**: 각 patch_size마다 별도 weight ✅
**구현**: nn.ParameterDict ❌ → 하나의 큰 텐서 + mask ✅

앞으로 이 문서를 참고하시면 정확한 구현을 이해하실 수 있습니다!
