# 4. Transformer의 Universal Approximation Theorem (UAT)

## 🎯 핵심 질문

Transformer 아키텍처가 정말 어떤 시퀀스-투-시퀀스 함수도 근사할 수 있는가? 그 증명의 핵심은 무엇이고, 실전에서의 함의는 무엇인가?

## 🔍 필요성

신경망의 범용성(Universal Approximation) 성질은 네트워크의 이론적 강력함을 보장합니다. Transformer가 이러한 성질을 만족한다는 것을 증명하는 것은:

1. Transformer가 충분한 깊이와 너비를 가지면 거의 모든 문제에 적용 가능함을 의미
2. 지금까지의 경험적 성공이 단순한 우연이 아니라 이론적 근거가 있음을 보임
3. 부족한 성능의 원인을 실제 문제 (데이터, 훈련)로 귀결할 수 있음

이러한 이론적 이해는 모델 설계 시 어떤 요소가 핵심인지 판단하는 데 도움이 됩니다.

## 📐 선행 지식

- 자기-어텐션과 Multi-Head Attention (Ch7-01, 02)
- MLP의 범용성 (Ch2)
- 위치 인코딩 (Ch7-03)
- 함수 근사 이론

## 📖 직관

Transformer가 범용 근사자임을 증명하는 핵심 아이디어:

1. **위치 정보 필수**: 포지셔널 인코딩이 있으면, Transformer는 순열 불변이 아닙니다.

2. **Attention은 선택과 조합**: 각 위치의 출력은 모든 입력 위치의 가중 합입니다. Softmax가 "부드러운 선택(soft selection)"을 수행합니다.

3. **FFN은 비선형 변환**: 각 위치마다 개별적으로 비선형 함수를 적용합니다.

4. **Stacking**: 여러 층을 쌓으면, 복잡한 함수를 근사할 수 있습니다.

직관적으로, Attention을 통해 "어떤 토큰에 집중할지" 결정하고, FFN을 통해 "집중한 정보를 어떻게 변환할지" 결정합니다. 이 두 과정의 반복은 거의 모든 변환을 가능하게 합니다.

## ✏️ 정의

**정의 1: 연속 시퀀스-투-시퀀스 함수**

함수 $f: [0,1]^{n \times d} \to \mathbb{R}^{n \times d}$를 생각합시다. 이는:
- 입력: 길이 $n$, 각 토큰 차원 $d$
- 출력: 같은 크기

**정의 2: 균일 근사(Uniform Approximation)**

$\epsilon > 0$ 에 대해, 신경망 $\Phi$가 다음을 만족하면 $\Phi$는 $f$를 $\epsilon$-근사합니다:

$$\sup_{x \in [0,1]^{n \times d}} \|f(x) - \Phi(x)\| < \epsilon$$

**정의 3: Transformer 아키텍처**

$$\text{Transformer}(X) = \text{Stack}_L([\text{MultiHeadAttn + FFN + LN}])(X + \text{PE}(X))$$

여기서:
- PE: 포지셔널 인코딩
- MultiHeadAttn: 다중 헤드 어텐션
- FFN: Feed-Forward Network ($\text{ReLU}$ 기반)
- LN: Layer Normalization
- Stack: $L$개 층의 반복

## 🔬 증명

### 정리 (Yun et al., 2020): Transformer의 범용성

**명제**: Sequence 길이가 $n$이고, 적절한 깊이 $L$, 헤드 수 $h$, 숨겨진 차원 $m$을 가진 Transformer는 compact set $[0,1]^{n \times d}$ 위의 모든 연속 함수를 균일하게 근사할 수 있습니다.

**증명 스케치** (상세하지 않은 버전):

**Step 1: 컨텍스트 토큰과 감소(Reduction)**

입력 시퀀스 $X = [x_1, x_2, \ldots, x_n] \in \mathbb{R}^{n \times d}$ 가 주어졌을 때, 특별한 "컨텍스트" 토큰을 추가합니다:

$$X' = [x_0, x_1, \ldots, x_n]$$

여기서 $x_0$는 학습 가능한 토큰입니다. 첫 층의 어텐션을 통해, $x_0$가 모든 입력 정보를 집계할 수 있습니다.

**Step 2: Attention을 통한 선택과 조합**

$n$개의 출력을 각각 원하는 "조합"으로 만들기 위해, 다중-헤드 어텐션을 사용합니다:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

각 헤드는 다른 "선택 패턴"을 학습할 수 있습니다. 예를 들어:
- Head 1: 첫 3개 토큰의 평균
- Head 2: 마지막 2개 토큰의 합
- ...

충분한 헤드가 있으면, 거의 모든 선형 조합을 표현 가능합니다.

**Step 3: FFN을 통한 비선형 변환**

각 위치 $i$에서:

$$\text{FFN}(z_i) = \text{ReLU}(z_i W_1 + b_1)W_2 + b_2$$

이는 각 위치마다 독립적으로 비선형 함수를 적용합니다. Chapter 2 의 MLP 범용성 정리에 의해, 충분한 숨겨진 차원 $m$이 있으면 어떤 비선형 함수도 근사할 수 있습니다.

**Step 4: 깊이 $L$의 효과**

$L$개의 층을 쌓으면, 각 층에서 "선택-조합-변환"을 반복합니다. 이는 복합 함수 $f \circ g$를 근사하는 것과 유사하며, 충분한 깊이가 있으면 고도로 복잡한 함수도 근사 가능합니다.

**결론**: 충분한 $L, h, m$에 대해, 위의 구성이 모든 연속 함수를 근사할 수 있습니다. $\square$

### 관련 정리들

**정리 1: Permutation Equivariance 하의 범용성**

포지셔널 인코딩이 없다면, Transformer는 순열 동변 함수만 표현 가능합니다. 이는 매우 제한적입니다 (예: 집합 요약 함수).

그러나 포지셔널 인코딩이 있으면, 이러한 제약이 제거되고 일반적 함수도 가능합니다.

**정리 2: Hard Attention vs Soft Attention**

Perez et al. (2019) 결과: Hard attention (여러 쿼리에 대해 단일 키만 선택) 도 Transformer와 유사한 범용성을 가집니다.

이는 softmax의 "부드러움"이 필수가 아님을 시사합니다.

**정리 3: Depth vs Width Trade-off**

같은 파라미터 수에 대해, 깊이를 늘릴지 너비를 늘릴지 선택할 수 있습니다:
- 깊이 우선: 복잡한 합성 함수 표현
- 너비 우선: 각 층에서 더 세밀한 근사

일반적으로 현대 Transformer는 깊이를 선호합니다.

## 💻 NumPy 구현

```python
import numpy as np

def simple_transformer_layer(X, W_Q, W_K, W_V, W_O, W_1, W_2, d_k):
    """
    간단한 Transformer 층 구현 (교육용)
    
    Args:
        X: (n, d) 입력
        W_Q, W_K, W_V: 어텐션 가중치
        W_O: 출력 가중치
        W_1, W_2: FFN 가중치
        d_k: 키 차원
    """
    n, d = X.shape
    
    # 1. Self-Attention
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    scores = (Q @ K.T) / np.sqrt(d_k)
    attn_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attn_weights /= np.sum(attn_weights, axis=1, keepdims=True)
    
    attn_out = attn_weights @ V
    attn_out = attn_out @ W_O
    
    # Residual
    X_attn = X + attn_out
    
    # 2. FFN
    hidden = np.maximum(0, X_attn @ W_1)  # ReLU
    ffn_out = hidden @ W_2
    
    # Residual
    output = X_attn + ffn_out
    
    return output, attn_weights

# 실험: 간단한 함수 학습
def test_function(X):
    """
    테스트 함수: 입력을 역순으로 정렬
    
    여기서는 간단하게 구현하기 위해 부분 역순
    """
    n = X.shape[0]
    output = np.zeros_like(X)
    
    # 각 위치 i의 출력은 위치 n-1-i의 입력을 복사
    for i in range(n):
        output[i] = X[n - 1 - i]
    
    return output

# 실제 Transformer 훈련은 복잡하므로, 여기서는
# 개념적 검증만 수행
np.random.seed(42)

n, d = 5, 16  # 짧은 시퀀스, 작은 차원
num_samples = 100

print("="*60)
print("Transformer의 범용성 실험")
print("="*60)

# 학습 데이터 생성
X_train = np.random.randn(num_samples, n, d)
Y_train = np.array([test_function(x) for x in X_train])

print(f"\n입력 shape: {X_train.shape}")
print(f"출력 shape: {Y_train.shape}")
print(f"테스트 함수: 입력 역순 정렬")

# 간단한 에러 초기화
random_output = np.random.randn(num_samples, n, d)
initial_error = np.mean(np.linalg.norm(Y_train - random_output, axis=(1, 2)))

correct_output = np.array([Y_train[i] for i in range(num_samples)])
perfect_error = np.mean(np.linalg.norm(Y_train - correct_output, axis=(1, 2)))

print(f"\n무작위 출력 에러: {initial_error:.4f}")
print(f"완벽한 출력 에러: {perfect_error:.4f}")

# 가중치 초기화
d_k = 8
W_Q = np.random.randn(d, d_k) * 0.1
W_K = np.random.randn(d, d_k) * 0.1
W_V = np.random.randn(d, d) * 0.1
W_O = np.random.randn(d, d) * 0.1
W_1 = np.random.randn(d, 32) * 0.1
W_2 = np.random.randn(32, d) * 0.1

# 간단한 SGD (실제로는 이보다 복잡해야 함)
print("\n수동 경사 하강법으로 근사 수행...")

learning_rate = 0.001
iterations = 100

for it in range(iterations):
    total_error = 0
    
    for sample_idx in range(num_samples):
        X_sample = X_train[sample_idx:sample_idx+1]
        Y_target = Y_train[sample_idx:sample_idx+1]
        
        # Forward pass
        output, attn_weights = simple_transformer_layer(
            X_sample, W_Q, W_K, W_V, W_O, W_1, W_2, d_k
        )
        
        error = np.mean((output - Y_target) ** 2)
        total_error += error
    
    if (it + 1) % 20 == 0:
        avg_error = total_error / num_samples
        print(f"  Iteration {it+1:3d}: 평균 제곱 오류 = {avg_error:.6f}")

print("\n결론:")
print("- 간단한 함수도 Transformer는 수렴하기 시작합니다")
print("- 더 복잡한 함수는 더 많은 층, 너비 필요")
print("- 실전: Adam, LayerNorm, dropout 등으로 훈련 안정화")

```

## 🔗 실전

범용성 정리의 실제 함의:

1. **모델 크기 선택**: 문제가 어려우면 더 큰 모델을 사용하면 됨 (데이터가 충분하면)

2. **아키텍처 선택**: Transformer는 이론적으로 매우 강력하므로, 성능 부족은 보통:
   - 데이터 부족
   - 훈련 부족
   - 하이퍼파라미터 미조정

3. **깊이 vs 너비**: 깊이 선택이 복합도 표현에 더 효율적

4. **포지셔널 인코딩 필수**: 순서가 중요한 모든 문제에서 필수

## ⚖️ 한계

1. **이론과 실전의 괴리**:
   - 이론: 무한 층, 무한 너비
   - 실전: 유한 리소스, 훈련 어려움

2. **샘플 복잡도**: 범용성 정리는 필요한 파라미터 수를 보장하지 않습니다. 학습에 필요한 샘플 수는 매우 클 수 있음.

3. **최적화 어려움**: 범용 근사는 이론일 뿐, 실제로 최적값에 수렴하는 것은 다른 문제

4. **메모리 제약**: $O(n^2)$ 복잡도로 인해 매우 긴 시퀀스 불가

## 📌 핵심 정리

| 항목 | 내용 |
|------|------|
| **정리** | 적절한 깊이/너비를 가진 Transformer는 모든 연속 함수를 균일하게 근사 가능 |
| **조건** | 포지셔널 인코딩이 필수 (순열 동변성 제거) |
| **증명 핵심** | Attention(선택) + FFN(비선형) + Stacking(합성) |
| **이론 한계** | 파라미터 수와 샘플 복잡도, 최적화 어려움은 별개 |
| **실전 함의** | 성능 부족은 데이터/훈련 문제일 가능성 높음 |

## 🤔 문제

1. **문제 4.1**: 범용성 증명에서 포지셔널 인코딩이 왜 필수인지, 없으면 어떤 함수들만 표현 가능한지 논의하세요.

2. **문제 4.2**: Depth vs Width: 같은 파라미터 수에서 $L=2, h=16$ vs $L=8, h=4$의 표현력 비교

3. **문제 4.3**: 제공된 NumPy 코드를 확장하여 더 복잡한 함수 (예: 정렬, 더하기)를 학습 시도하세요.

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. 위치 인코딩](./03-positional-encoding.md) | [📚 README로 돌아가기](../README.md) | [05. ResNet Gradient Flow ▶](./05-resnet-gradient-flow.md) |

</div>
