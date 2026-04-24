# 1. Self-Attention의 수학과 √d_k 스케일링

## 🎯 핵심 질문

자기-어텐션(Self-Attention)이 무엇이고, 왜 쿼리-키 내적을 $\sqrt{d_k}$로 정규화해야 하는가? 이 정규화가 그래디언트 흐름에 어떤 영향을 미치는가?

## 🔍 필요성

Transformer 아키텍처의 핵심 메커니즘인 자기-어텐션은 시퀀스 내 모든 토큰 쌍 사이의 관계를 학습할 수 있게 해줍니다. 하지만 $d_k$가 커질수록 내적 값이 폭발적으로 커져서 softmax가 포화되고 그래디언트가 사라지는 문제가 발생합니다. 이를 해결하는 원리를 이해하는 것은 대규모 모델 훈련에서 필수적입니다.

## 📐 선행 지식

- 행렬 연산 및 확률론 기초
- Softmax 함수와 그 미분
- 중심극한정리(Central Limit Theorem)
- 그래디언트 역전파

## 📖 직관

자기-어텐션을 "소프트 룩업(soft lookup)" 메커니즘으로 생각할 수 있습니다:
- **쿼리(Query)**: "무엇을 찾을 것인가"
- **키(Key)**: "이것이 무엇인가"
- **값(Value)**: "이것의 특징은 무엇인가"

쿼리와 키 사이의 유사도(내적)를 계산해서, 유사도가 높은 토큰들의 값을 가중 평균합니다. 

하지만 $d_k$가 크면, 모든 내적 값이 크게 나와서 softmax가 "원-핫" 같은 형태가 되어 거의 모든 주의가 하나의 토큰에 집중됩니다. 이는 정보 손실과 그래디언트 부족으로 이어집니다. $\sqrt{d_k}$로 나누면 내적 값의 분산을 정상화해서 softmax가 "부드러운" 분포를 유지하게 됩니다.

## ✏️ 정의

**자기-어텐션(Self-Attention)**

입력 시퀀스 $X \in \mathbb{R}^{n \times d}$ (시퀀스 길이 $n$, 임베딩 차원 $d$)에 대해:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

여기서 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ (또는 $d \times d_v$)는 학습 가능한 가중치 행렬입니다.

따라서:
- $Q \in \mathbb{R}^{n \times d_k}$
- $K \in \mathbb{R}^{n \times d_k}$
- $V \in \mathbb{R}^{n \times d_v}$

**어텐션 출력(Attention Output)**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

자세히 풀어쓰면:

$$\text{Attention}_{ij} = \sum_{k=1}^n \text{softmax}\left(\frac{Q_i K_k^T}{\sqrt{d_k}}\right)_k V_k$$

어텐션 가중치 행렬을 $A$라 하면:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}$$

$A_{ij}$는 토큰 $i$가 토큰 $j$에 어떤 정도로 주의를 기울이는지를 나타냅니다.

**행렬 모양 명시(Shape)**

- 입력: $X \in \mathbb{R}^{n \times d}$
- 쿼리: $Q \in \mathbb{R}^{n \times d_k}$
- 키: $K \in \mathbb{R}^{n \times d_k}$
- 값: $V \in \mathbb{R}^{n \times d_v}$
- 내적: $QK^T \in \mathbb{R}^{n \times n}$
- 어텐션 가중치: $A \in \mathbb{R}^{n \times n}$
- 출력: $\text{Attention}(Q,K,V) \in \mathbb{R}^{n \times d_v}$

## 🔬 증명

### 정리 1: 내적의 분산 증가

**가정**: $Q, K$는 독립이고, 각 원소 $Q_{ij}, K_{ij} \sim \mathcal{N}(0, 1)$ (평균 0, 분산 1)이라 가정합시다.

**명제**: $(QK^T)_{ij}$의 분산은 $d_k$입니다.

**증명**:

$$QK^T \text{의 }(i,j) \text{ 원소} = \sum_{l=1}^{d_k} Q_{il} K_{jl}$$

각 $Q_{il}, K_{jl}$이 독립이고 표준정규분포이므로, $Q_{il}K_{jl}$의 분산은:

$$\text{Var}(Q_{il}K_{jl}) = \mathbb{E}[Q_{il}^2 K_{jl}^2] = \mathbb{E}[Q_{il}^2]\mathbb{E}[K_{jl}^2] = 1 \cdot 1 = 1$$

독립성에 의해:

$$\text{Var}\left(\sum_{l=1}^{d_k} Q_{il}K_{jl}\right) = \sum_{l=1}^{d_k} \text{Var}(Q_{il}K_{jl}) = d_k$$

중심극한정리에 의해 합은 근사적으로 정규분포를 따릅니다:

$$(QK^T)_{ij} \approx \mathcal{N}(0, d_k)$$

따라서 $d_k$가 크면 내적 값들이 매우 커집니다. $\square$

### 정리 2: 정규화의 효과

$$\frac{QK^T}{\sqrt{d_k}}$$

에 대해, 각 원소의 분산은:

$$\text{Var}\left(\frac{(QK^T)_{ij}}{\sqrt{d_k}}\right) = \frac{1}{d_k}\text{Var}((QK^T)_{ij}) = \frac{d_k}{d_k} = 1$$

따라서 정규화 후 값들은 $\mathcal{N}(0, 1)$을 따릅니다. 이제 softmax 입력이 $O(1)$ 스케일로 유지되므로, softmax는 너무 포화되지 않습니다.

### 정리 3: Softmax 포화와 그래디언트

Softmax 함수: $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

그래디언트:

$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i(\delta_{ij} - \text{softmax}(z)_j)$$

$z$의 한 성분이 매우 크면 (예: $z_1 \gg z_2, z_3, \ldots$), softmax는 $(1, 0, 0, \ldots)$에 가까워지고:

$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} \approx 0 \quad (i \neq 1, j \neq 1)$$

그래디언트가 거의 0이 되어 vanishing gradient 문제가 발생합니다.

정규화를 통해 입력을 $O(1)$ 범위로 유지하면, softmax는 엔트로피가 높은 분포를 유지해서 그래디언트가 전달되지 않는 것을 방지합니다.

## 💻 NumPy 구현

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, scale=None):
    """Scaled dot-product attention"""
    if scale is None:
        scale = np.sqrt(Q.shape[1])
    
    scores = np.dot(Q, K.T) / scale
    
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
    
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# 실험: 정규화 효과 비교
np.random.seed(42)
n, d_k, d_v = 8, 64, 64
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
V = np.random.randn(n, d_v)

def entropy(p):
    return -np.sum(p[p > 1e-10] * np.log(p[p > 1e-10] + 1e-10))

output_no_scale, attn_no = scaled_dot_product_attention(Q, K, V, scale=1.0)
output_with_scale, attn_with = scaled_dot_product_attention(Q, K, V, scale=np.sqrt(d_k))

ent_no = np.mean([entropy(attn_no[i]) for i in range(n)])
ent_yes = np.mean([entropy(attn_with[i]) for i in range(n)])

print(f"정규화 없음: 엔트로피 {ent_no:.4f}")
print(f"정규화 있음: 엔트로피 {ent_yes:.4f}")
print(f"개선: {(ent_yes - ent_no):.4f}")
```

## 🔗 실전

자기-어텐션은 다음과 같이 활용됩니다:

1. **Transformer의 핵심**: 각 문장 내에서 단어 간 관계를 학습
2. **Vision Transformer**: 이미지 패치 간 관계 학습
3. **BERT**: 양방향 문맥 인코딩
4. **GPT**: 자동회귀적 언어모델링

실전에서 주의할 점:
- Causal masking: 미래 토큰 참조 방지 (생성형 모델)
- Dropout: 어텐션 가중치에 정규화
- Flash Attention: 메모리 효율적 구현

## ⚖️ 한계

1. **계산 복잡도**: $O(n^2)$ 메모리와 시간 → 긴 시퀀스 불가
2. **상대적 위치 정보 부재**: 자기-어텐션 자체는 순서를 모름 (포지셔널 인코딩 필요)
3. **엔트로피와 소프트니스 트레이드오프**: 정규화가 모든 가중치를 약간 평탄하게 만들 수 있음
4. **고정 스케일**: $\sqrt{d_k}$는 경험적 선택이며, 최적값은 아닐 수 있음

## 📌 핵심 정리

| 항목 | 내용 |
|------|------|
| **정의** | $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ |
| **행렬 형상** | $Q, K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$, 출력 $\in \mathbb{R}^{n \times d_v}$ |
| **내적 분산** | $(QK^T)_{ij} \sim \mathcal{N}(0, d_k)$ → 정규화 필수 |
| **정규화 목적** | Softmax 포화 방지, 그래디언트 전달 보장 |
| **포화 위험** | 정규화 없으면 $d_k$ 커질수록 원-핫 같은 분포 → 그래디언트 소실 |

## 🤔 문제

1. **문제 1.1**: $d_k = 128$일 때 정규화 전후 $(QK^T)_{ij}$의 분산을 계산하세요.

2. **문제 1.2**: 정규화 없이 엔트로피가 감소하는 현상을 설명하세요.

3. **문제 1.3**: Causal masking을 추가하여 자기-어텐션을 수정하세요.

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch6-04. Echo State Network](../ch6-rnn/04-echo-state-network.md) | [📚 README로 돌아가기](../README.md) | [02. Multi-Head Attention ▶](./02-multi-head-attention.md) |

</div>
