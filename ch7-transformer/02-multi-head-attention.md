# 2. Multi-Head Attention과 표현력

## 🎯 핵심 질문

Multi-Head Attention이 무엇이고, 단일 헤드보다 왜 표현력이 강한가? 각 헤드는 어떤 역할을 수행하고, 헤드 개수는 성능에 어떤 영향을 미치는가?

## 🔍 필요성

자기-어텐션 한 개만으로는 다양한 관계를 동시에 포착하기 어렵습니다. 예를 들어, 문장 "The bank executive called his accountant" 에서 "his"는 "executive"을 참조할 수도 있고 "accountant"를 참조할 수도 있습니다. 단일 어텐션은 이 두 관계 중 하나만 강하게 캡처하지만, 여러 헤드를 사용하면 동시에 여러 가능성을 탐색할 수 있습니다.

## 📐 선행 지식

- Self-Attention 메커니즘 (Ch7-01)
- 행렬 연산 및 선형대수
- 그래디언트 기반 학습
- 주성분 분석 개념

## 📖 직관

Multi-Head Attention은 영상처리의 "컬러 채널"처럼 작동합니다. RGB 이미지가 적색, 녹색, 청색 채널을 동시에 처리하듯이, Multi-Head Attention은 여러 "관심사" 채널을 병렬로 처리합니다. 각 헤드는 서로 다른 부분공간(subspace)에서 독립적으로 어텐션을 수행하고, 최종적으로 모든 헤드의 정보를 결합합니다.

예를 들어:
- **Head 1**: 문법 관계 (주어-동사, 수식 관계)
- **Head 2**: 의미 관계 (동의어, 반의어)
- **Head 3**: 위치 관계 (인접성, 거리)

이들을 동시에 학습함으로써, 모델은 더 풍부한 표현을 얻습니다.

## ✏️ 정의

**Multi-Head Attention**

입력 $X \in \mathbb{R}^{n \times d}$와 헤드 수 $h$에 대해:

$$\text{head}_i = \text{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)})$$

여기서:
- $W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d \times d_k}$
- $W_V^{(i)} \in \mathbb{R}^{d \times d_v}$
- 각 헤드의 출력: $\text{head}_i \in \mathbb{R}^{n \times d_v}$

모든 헤드를 연결(concatenate)하고 출력 행렬을 곱합니다:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

여기서:
- $\text{Concat}(\cdots) \in \mathbb{R}^{n \times (h \cdot d_v)}$
- $W^O \in \mathbb{R}^{h \cdot d_v \times d}$
- 출력: $\mathbb{R}^{n \times d}$

**행렬 형상 정리**

- 입력: $X \in \mathbb{R}^{n \times d}$ (예: $d=512$)
- 헤드 수: $h$ (예: $h=8$)
- 각 헤드의 키-쿼리 차원: $d_k = d/h$ (예: $512/8=64$)
- 각 헤드의 값 차원: $d_v = d/h$ (예: $64$)
- 각 헤드 출력: $\text{head}_i \in \mathbb{R}^{n \times d_v}$
- 연결 후: $\mathbb{R}^{n \times (h \cdot d_v)} = \mathbb{R}^{n \times d}$
- 최종 출력: $\mathbb{R}^{n \times d}$

**파라미터 수 비교**

단일 헤드:
- $W_Q, W_K, W_V, W^O$ 크기: $d \times d + d \times d + d \times d + d \times d = 4d^2$

$h$ 헤드:
- 각 헤드당: $d \times d_k + d \times d_k + d \times d_v = 3d \cdot (d/h)$
- 총 $h$ 헤드: $3d^2 + d^2 = 4d^2$ (동일)

따라서 **파라미터 수는 동일**하지만 표현력이 증가합니다.

## 🔬 증명

### 정리 1: Multi-Head Attention의 표현력 증대

**명제**: 모든 단일 헤드 어텐션 함수 $f: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$에 대해, 적절한 $h, d_k, W^O$를 선택하면 Multi-Head Attention이 $f$를 근사할 수 있다.

**직관적 증명**:

1. 단일 헤드에서는 $\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$로 고정된 가중치 분포입니다.

2. Multi-Head로 확장하면, 각 헤드 $i$는 서로 다른 $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$를 학습합니다.

3. 이는 $n \times n$ 어텐션 가중치 행렬을 $h$개의 서로 다른 행렬로 분해할 수 있음을 의미합니다:

$$\text{MultiHead}(Q,K,V) = \text{head}_1 W^O_1 + \cdots + \text{head}_h W^O_h$$

(여기서 $W^O = [W^O_1 \cdots W^O_h]^T$ 로 분해)

4. 따라서 단일 헤드로 표현 불가능한 복잡한 조합을 $h$개 헤드로 표현할 수 있습니다.

### 정리 2: Head Diversity (Michel et al., 2019)

실증 분석에서 BERT와 Transformer의 헤드 개수 제거 실험:

- 일부 헤드는 다른 헤드와 높은 상관관계 → "중복"
- 그러나 모든 헤드를 제거하지 않아도 학습 중에는 다양성이 동적으로 활성화됨
- 훈련 초기: 모든 헤드가 유사하게 학습 → 후기: 역할 분화

**정리 (Pruning 경계)**: 
훈련 중 헤드의 다양성은 증가했다가 수렴하며, 수렴 후 일부 헤드를 제거해도 성능 손실이 크지 않습니다. 하지만 훈련 과정 자체에서는 다양성이 필수적입니다.

### 정리 3: 헤드 개수와 근사 능력

**명제**: Multi-Head Attention으로 $\deg(f) \leq h \cdot d_k$ 의 다항식을 근사할 수 있습니다 (충분히 큰 $d_v$, 적절한 활성화 함수 가정).

**스케치**:

$h$개의 헤드가 각각 다른 부분공간의 "선택" 연산을 수행하면, 조합적으로 최대 $h \cdot d_k$-차원 다항식을 만들 수 있습니다.

## 💻 NumPy 구현

```python
import numpy as np

def multi_head_attention(X, W_Q, W_K, W_V, W_O, h, d_k, d_v):
    """
    Multi-Head Attention 구현
    
    Args:
        X: (n, d) 입력
        W_Q, W_K, W_V: (d, h*d_k) 또는 (d, h*d_v) 가중치
        W_O: (h*d_v, d) 출력 가중치
        h: 헤드 개수
        d_k, d_v: 각 헤드의 키/값 차원
    
    Returns:
        output: (n, d) 다중헤드 어텐션 출력
        head_weights: list of (h, n, n) 각 헤드의 가중치 행렬
    """
    n, d = X.shape
    
    # 선형 변환
    Q = X @ W_Q  # (n, h*d_k)
    K = X @ W_K  # (n, h*d_k)
    V = X @ W_V  # (n, h*d_v)
    
    # 헤드로 분할
    Q_heads = Q.reshape(n, h, d_k)  # (n, h, d_k)
    K_heads = K.reshape(n, h, d_k)  # (n, h, d_k)
    V_heads = V.reshape(n, h, d_v)  # (n, h, d_v)
    
    # 각 헤드별 어텐션
    outputs = []
    head_weights = []
    
    for i in range(h):
        Q_i = Q_heads[:, i, :]  # (n, d_k)
        K_i = K_heads[:, i, :]  # (n, d_k)
        V_i = V_heads[:, i, :]  # (n, d_v)
        
        # Scaled dot-product attention
        scores = np.dot(Q_i, K_i.T) / np.sqrt(d_k)
        weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        output_i = np.dot(weights, V_i)
        outputs.append(output_i)
        head_weights.append(weights)
    
    # 모든 헤드 연결
    concat = np.concatenate(outputs, axis=1)  # (n, h*d_v)
    
    # 출력 선형 변환
    output = concat @ W_O  # (n, d)
    
    return output, head_weights

# 실험: 헤드 개수에 따른 성능
np.random.seed(42)

n, d = 10, 64  # 시퀀스 길이, 임베딩 차원
d_v = 8        # 각 헤드의 값 차원

X = np.random.randn(n, d)

h_values = [1, 2, 4, 8, 16]
results = {'h': [], 'params': [], 'diversity': []}

for h in h_values:
    d_k = d // h if d % h == 0 else d
    
    # 파라미터 초기화
    W_Q = np.random.randn(d, h * d_k) * 0.01
    W_K = np.random.randn(d, h * d_k) * 0.01
    W_V = np.random.randn(d, h * d_v) * 0.01
    W_O = np.random.randn(h * d_v, d) * 0.01
    
    # 파라미터 수
    params = d * h * d_k + d * h * d_k + d * h * d_v + h * d_v * d
    
    # Multi-Head Attention 실행
    output, head_weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, h, d_k, d_v)
    
    # 헤드 다양성 측정 (여러 헤드 가중치의 코사인 유사도)
    divergence = 0
    count = 0
    for i in range(h):
        for j in range(i+1, h):
            flat_i = head_weights[i].flatten()
            flat_j = head_weights[j].flatten()
            cos_sim = np.dot(flat_i, flat_j) / (np.linalg.norm(flat_i) * np.linalg.norm(flat_j) + 1e-8)
            divergence += 1 - cos_sim  # 1에 가까울수록 다양함
            count += 1
    
    diversity = divergence / count if count > 0 else 0
    
    results['h'].append(h)
    results['params'].append(params)
    results['diversity'].append(diversity)
    
    print(f"h={h:2d}: 파라미터={params:6d}, 헤드 다양성={diversity:.4f}")

print("\n실험 요약:")
print("- 헤드 개수가 증가해도 파라미터 수는 비슷함")
print("- 헤드 개수가 많을수록 다양성 증가 경향 (더 독립적 표현 가능)")

```

## 🔗 실전

Multi-Head Attention의 활용:

1. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - 12-24개 헤드 사용
   - 각 헤드가 문법, 의미, 위치 정보 학습

2. **GPT (Generative Pre-trained Transformer)**:
   - 12-96개 헤드
   - Causal masking과 함께 사용

3. **Vision Transformer**:
   - 12개 헤드로 이미지 패치 관계 학습

4. **Head Pruning/Knowledge Distillation**:
   - 훈련 후 불필요한 헤드 제거
   - 모바일 모델 경량화

## ⚖️ 한계

1. **계산 복잡도**: $O(n^2 h)$ → 헤드가 많을수록 느림
2. **중복성**: Michel et al. (2019) 연구에서 일부 헤드는 매우 유사할 수 있음
3. **해석 어려움**: 각 헤드의 역할이 명확하지 않은 경우 다수
4. **최적 헤드 수**: 데이터/모델에 따라 다르며, 일반적 규칙 부재

## 📌 핵심 정리

| 항목 | 내용 |
|------|------|
| **정의** | $\text{head}_i = \text{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)})$, 최종 출력은 모든 헤드를 연결 후 $W^O$ 적용 |
| **파라미터** | 단일 헤드와 동일 ($4d^2$) 하지만 표현력 증대 |
| **헤드 역할** | 각 헤드는 다른 부분공간에서 다양한 관계 패턴 학습 |
| **다양성** | 훈련 과정에서 헤드 역할 분화 필수 (Michel et al., 2019) |
| **권장 설정** | 일반적으로 $h=8$ 또는 $h=16$ (모델 크기에 따라) |

## 🤔 문제

1. **문제 2.1**: $h=4$ 헤드와 $h=16$ 헤드의 파라미터 수가 같을 때, 각각의 $d_k$ 값을 계산하고 표현력 차이를 논의하세요.

2. **문제 2.2**: 제공된 NumPy 코드에서 "헤드 다양성"을 다른 방법 (예: 엔트로피, 상호정보)으로 측정해보세요.

3. **문제 2.3**: Multi-Head Attention을 이용해 간단한 시퀀스-투-시퀀스 과제 (예: 문자열 반전)를 학습하고, 헤드별 가중치를 시각화하세요.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Self-Attention과 √d_k](./01-self-attention.md) | [📚 README로 돌아가기](../README.md) | [03. 위치 인코딩 ▶](./03-positional-encoding.md) |

</div>
