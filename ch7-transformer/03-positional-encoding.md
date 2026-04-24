# 3. Positional Encoding의 필요성과 설계

## 🎯 핵심 질문

자기-어텐션은 순서 정보가 없는데, 어떻게 시퀀스의 순서를 이해할 수 있는가? 다양한 위치 인코딩 방법들(sinusoidal, learned, RoPE, ALiBi)의 장단점은 무엇인가?

## 🔍 필요성

자기-어텐션 메커니즘은 순열에 대해 동변(permutation-equivariant)입니다. 즉, 입력 토큰의 순서가 바뀌면 출력의 순서도 같은 방식으로 바뀝니다. 이는 강력한 대칭성이지만, 언어처리에서는 순서가 매우 중요합니다. "개가 고양이를 쫓았다"와 "고양이가 개를 쫓았다"는 완전히 다른 의미입니다. 따라서 순서 정보를 어텐션에 명시적으로 추가해야 합니다.

## 📐 선행 지식

- 자기-어텐션 메커니즘 (Ch7-01)
- 삼각함수와 푸리에 분석
- 회전 행렬과 복소수
- 선형대수 기본

## 📖 직관

위치 인코딩을 생각하는 세 가지 방식:

1. **Absolute**: 각 위치 $p$에 고정된 벡터를 더하거나 연결 → "1번 위치는 항상 이 벡터"

2. **Relative**: 두 토큰의 거리 $(p - p')$에만 의존 → "거리 3은 항상 같은 관계"

3. **Rotary**: 쿼리와 키에 위치 의존 회전을 적용 → "내적 계산에 상대 위치가 자동으로 포함됨"

이 세 가지는 모두 시퀀스의 순서를 표현하지만, 계산 효율, 외삽(extrapolation) 능력, 해석 용이성 측면에서 다릅니다.

## ✏️ 정의

### 1. Permutation Equivariance 증명

**정리**: 자기-어텐션 함수는 순열 동변입니다.

입력 시퀀스 $X \in \mathbb{R}^{n \times d}$와 순열행렬 $P \in \mathbb{R}^{n \times n}$에 대해:

$$\text{Attention}(PX) = P \cdot \text{Attention}(X)$$

**증명**:

$$\text{Attention}(PX) = \text{softmax}\left(\frac{(PX)W_Q (PX)W_K^T}{\sqrt{d_k}}\right)(PX)W_V$$

$$= \text{softmax}\left(\frac{PXW_Q W_K^T X^T P^T}{\sqrt{d_k}}\right)PXW_V$$

$$P^T P = I$ 이고 행렬 곱의 성질에 의해:

$$= P \cdot \text{softmax}\left(\frac{XW_Q W_K^T X^T}{\sqrt{d_k}}\right)XW_V$$

$$= P \cdot \text{Attention}(X)$$

따라서 자기-어텐션은 입력 순열에 대해 불변 구조를 가집니다. $\square$

### 2. Sinusoidal Positional Encoding

Vaswani et al. (2017) "Attention Is All You Need" 에서 제안:

$$\text{PE}(p, 2i) = \sin\left(\frac{p}{10000^{2i/d}}\right)$$

$$\text{PE}(p, 2i+1) = \cos\left(\frac{p}{10000^{2i/d}}\right)$$

여기서:
- $p$: 토큰의 위치 ($0 \leq p < n$)
- $i$: 차원 인덱스 ($0 \leq i < d/2$)
- 입력은 $X + \text{PE}$ 형태로 더해짐

**성질**:

$$\text{PE}(p+k) = M(k) \cdot \text{PE}(p) + \text{correction}$$

여기서 $M(k)$는 거리 $k$에 대한 선형 변환 행렬입니다. 이는 상대 위치가 선형 변환으로 표현 가능함을 시사합니다.

### 3. Learned Positional Encoding

간단한 방식: 위치별 임베딩을 직접 학습

$$\text{PE}[p] \in \mathbb{R}^d, \quad p = 0, 1, \ldots, n_{\max}-1$$

각 위치마다 학습 가능한 벡터가 있으며, 훈련 중 그래디언트로 업데이트됩니다.

**장점**: 유연성, 데이터 기반 학습
**단점**: 훈련 길이 이상 외삽 불가

### 4. RoPE (Rotary Position Embedding)

Su et al. (2021) 제안:

쿼리 $q_m$과 키 $k_n$에 위치 의존 회전을 적용:

$$\tilde{q}_m = e^{imθ} q_m \quad \text{(복소수 표현)}$$

$$\tilde{k}_n = e^{inθ} k_n$$

여기서 $θ$는 주파수 기반 각도입니다.

내적 계산에서:

$$\langle \tilde{q}_m, \tilde{k}_n \rangle = \text{Re}(e^{i(m-n)θ} \langle q_m, k_n \rangle)$$

따라서 상대 위치 $m - n$이 자동으로 내적에 인코딩됩니다.

**장점**: 외삽 능력, 상대 위치 직접 인코딩
**단점**: 구현 복잡도

### 5. ALiBi (Attention with Linear Biases)

Press et al. (2022) 제안:

어텐션 점수에 직접 위치 의존 바이어스를 더합니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{bias}(i, j)\right)V$$

$$\text{bias}(i, j) = -\alpha \cdot |i - j|$$

여기서 $\alpha$는 학습 가능한 스칼라 (또는 헤드별로 다름).

**장점**: 극단적으로 간단, 외삽 능력 우수
**단점**: 거리만 고려, 방향성 정보 부재

## 🔬 증명

### 정리 1: Sinusoidal PE의 상대 위치 선형성

**명제**: Sinusoidal PE에서 $\text{PE}(p+k) - \text{PE}(p)$를 $\text{PE}(p)$로 표현 가능합니다 (근사).

**증명 스케치**:

삼각함수 덧셈 공식:

$$\sin(p + k) = \sin(p)\cos(k) + \cos(p)\sin(k)$$

$$\cos(p + k) = \cos(p)\cos(k) - \sin(p)\sin(k)$$

각 주파수별로 이는 선형 변환:

$$\begin{pmatrix} \sin(p+k) \\ \cos(p+k) \end{pmatrix} = \begin{pmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{pmatrix} \begin{pmatrix} \sin(p) \\ \cos(p) \end{pmatrix}$$

따라서 거리 $k$는 고정된 회전 행렬로 표현되며, 이는 학습 가능함을 의미합니다. $\square$

### 정리 2: RoPE와 상대 거리의 동변성

**명제**: RoPE는 상대 거리 의존성을 만족합니다.

**증명**:

$$\text{sim}(\tilde{q}_m, \tilde{k}_n) = \text{Re}(e^{i(m-n)θ} \langle q_m, k_n \rangle)$$

$m' = m + \delta, n' = n + \delta$ 로 변환하면:

$$\text{sim}(\tilde{q}_{m'}, \tilde{k}_{n'}) = \text{Re}(e^{i(m' - n')θ} \langle q_{m'}, k_{n'} \rangle) = \text{Re}(e^{i(m-n)θ} \lots)$$

상대 거리 $m - n$이 불변입니다. $\square$

### 정리 3: ALiBi의 외삽 성능

**명제**: ALiBi는 훈련 길이보다 긴 시퀀스에 우수한 성능을 보입니다.

**직관**: ALiBi는 거리만 인코딩하므로, 훈련 중 보지 못한 거리도 자연스럽게 처리 가능합니다.

예: 훈련에서 최대 길이 512이고 ALiBi로 학습했다면, 길이 1024 추론에서도 대부분 작동합니다.

## 💻 NumPy 구현

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Sinusoidal PE
def sinusoidal_pe(seq_len, d_model, base=10000):
    """
    Sinusoidal Positional Encoding
    
    Args:
        seq_len: 시퀀스 길이
        d_model: 임베딩 차원
        base: 주파수 기본값 (기본 10000)
    
    Returns:
        pe: (seq_len, d_model) 위치 인코딩
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)  # 짝수 인덱스
    if d_model % 2 == 1:
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)  # 홀수 인덱스
    
    return pe

# 2. Learned PE (간단한 버전)
def learned_pe(seq_len, d_model):
    """학습 가능한 위치 인코딩"""
    return np.random.randn(seq_len, d_model) * 0.02

# 3. ALiBi (attention에 더할 바이어스)
def alibi_bias(seq_len, heads=8):
    """
    ALiBi 바이어스 계산
    
    Returns:
        bias: (seq_len, seq_len) 각 헤드에 더할 바이어스
    """
    # 거리 행렬
    i = np.arange(seq_len)[:, np.newaxis]
    j = np.arange(seq_len)[np.newaxis, :]
    distance = np.abs(i - j)
    
    # 헤드별로 다른 기울기
    # 보통 1/2, 1/4, 1/8, ... 처럼 설정
    slopes = 2.0 ** np.arange(heads) / (2.0 ** heads)
    
    return -slopes[:, np.newaxis, np.newaxis] * distance

# 실험 1: Sinusoidal PE 시각화
seq_len, d_model = 100, 64
pe = sinusoidal_pe(seq_len, d_model)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Heatmap
im = axes[0, 0].imshow(pe.T, cmap='coolwarm', aspect='auto')
axes[0, 0].set_title('Sinusoidal PE Heatmap')
axes[0, 0].set_xlabel('Position')
axes[0, 0].set_ylabel('Dimension')
plt.colorbar(im, ax=axes[0, 0])

# 특정 위치의 PE
for p in [0, 25, 50, 75]:
    axes[0, 1].plot(pe[p], label=f'pos={p}', alpha=0.7)
axes[0, 1].set_title('PE Values by Position')
axes[0, 1].set_xlabel('Dimension')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 상대 위치 분석: PE(p+k) 와 PE(p)의 거리
def relative_position_distance(pe, k):
    """거리 k의 상대 위치 간 거리 계산"""
    distances = []
    for p in range(len(pe) - k):
        dist = np.linalg.norm(pe[p+k] - pe[p])
        distances.append(dist)
    return np.mean(distances)

k_values = np.arange(1, 50)
distances = [relative_position_distance(pe, k) for k in k_values]
axes[1, 0].plot(k_values, distances, 'o-')
axes[1, 0].set_title('Average Distance for Relative Position k')
axes[1, 0].set_xlabel('Distance k')
axes[1, 0].set_ylabel('L2 Distance')
axes[1, 0].grid(True, alpha=0.3)

# Learned PE와 비교
learned_pe_vals = learned_pe(seq_len, d_model)
im2 = axes[1, 1].imshow(learned_pe_vals.T, cmap='coolwarm', aspect='auto')
axes[1, 1].set_title('Learned PE Heatmap (초기)')
axes[1, 1].set_xlabel('Position')
axes[1, 1].set_ylabel('Dimension')
plt.colorbar(im2, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('/tmp/positional_encoding.png', dpi=100, bbox_inches='tight')
print("그래프 저장: /tmp/positional_encoding.png")

# 실험 2: ALiBi 바이어스 시각화
seq_len = 32
heads = 8
alibi = alibi_bias(seq_len, heads)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, head_idx in enumerate([0, 3, 5, 7]):
    ax = axes[idx // 2, idx % 2]
    im = ax.imshow(alibi[head_idx], cmap='RdBu', vmin=-5, vmax=0)
    ax.set_title(f'ALiBi Head {head_idx}')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('/tmp/alibi_bias.png', dpi=100, bbox_inches='tight')
print("그래프 저장: /tmp/alibi_bias.png")

# 실험 3: 외삽 능력 비교
print("\n" + "="*60)
print("외삽 능력 비교")
print("="*60)

def extrapolation_test(pe_func, train_len, test_len):
    """
    외삽 능력 테스트
    - 훈련: 길이 train_len
    - 테스트: 길이 test_len (> train_len)
    """
    pe_train = pe_func(train_len, 64)
    pe_test = pe_func(test_len, 64)
    
    # 훈련 범위 내에서의 거리
    intra_dist = []
    for i in range(train_len - 1):
        dist = np.linalg.norm(pe_train[i+1] - pe_train[i])
        intra_dist.append(dist)
    
    # 테스트에서 새로운 거리 (훈련 범위 초과)
    extra_dist = []
    for i in range(train_len, test_len - 1):
        dist = np.linalg.norm(pe_test[i+1] - pe_test[i])
        extra_dist.append(dist)
    
    intra_mean = np.mean(intra_dist) if intra_dist else 0
    extra_mean = np.mean(extra_dist) if extra_dist else 0
    
    return intra_mean, extra_mean

train_len, test_len = 512, 1024

intra_sine, extra_sine = extrapolation_test(sinusoidal_pe, train_len, test_len)
intra_learned, extra_learned = extrapolation_test(learned_pe, train_len, test_len)

print(f"\nSinusoidal PE:")
print(f"  훈련 범위 평균 거리: {intra_sine:.4f}")
print(f"  외삽 범위 평균 거리: {extra_sine:.4f}")
print(f"  비율: {extra_sine / (intra_sine + 1e-8):.4f}")

print(f"\nLearned PE:")
print(f"  훈련 범위 평균 거리: {intra_learned:.4f}")
print(f"  외삽 범위 평균 거리: {extra_learned:.4f}")
print(f"  비율: {extra_learned / (intra_learned + 1e-8):.4f}")

print("\n(주: Learned PE는 훈련 길이 초과 외삽이 일반적으로 나쁨)")

```

## 🔗 실전

위치 인코딩 선택 기준:

1. **Sinusoidal** (기본): 대부분의 경우 안정적
2. **Learned**: 고정된 최대 길이가 있을 때 (예: 문서 분류)
3. **RoPE**: 매우 긴 시퀀스, 외삽 필요 (LLaMA, PaLM)
4. **ALiBi**: 극단적으로 효율적, 매우 긴 시퀀스 (BLOOM)

최근 추세: RoPE와 ALiBi가 대규모 모델에서 주도적

## ⚖️ 한계

1. **Sinusoidal의 선택**: 왜 10000인가? → 경험적 선택
2. **Learned의 길이 제약**: 훈련 길이 초과 사용 불가
3. **RoPE의 구현**: 복소수 연산으로 인한 복잡도
4. **ALiBi의 단순성**: 거리만 고려하면 실제 의미론적 관계 제한 가능

## 📌 핵심 정리

| 방법 | 공식 | 장점 | 단점 |
|------|------|------|------|
| Sinusoidal | $\sin(p/10000^{2i/d})$ | 안정적, 상대 위치 학습 가능 | 고정 주파수 |
| Learned | 훈련 가능 임베딩 | 유연함 | 길이 고정 |
| RoPE | 쿼리/키 회전 | 상대 위치 직접, 외삽 우수 | 구현 복잡 |
| ALiBi | 어텐션에 바이어스 추가 | 극단적 단순, 외삽 우수 | 거리만 고려 |

## 🤔 문제

1. **문제 3.1**: Sinusoidal PE에서 기본값 10000 대신 100 또는 100000을 사용하면 어떻게 될까요? 상대 위치 특성에 미치는 영향을 분석하세요.

2. **문제 3.2**: RoPE를 NumPy로 구현하고 (복소수 표현), Sinusoidal PE와 비교해보세요.

3. **문제 3.3**: ALiBi를 자기-어텐션에 통합하고, Sinusoidal PE와의 수렴 속도를 비교하세요.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Multi-Head Attention](./02-multi-head-attention.md) | [📚 README로 돌아가기](../README.md) | [04. Transformer의 범용성 (Yun 2020) ▶](./04-transformer-uat.md) |

</div>
