# 03. Reverse-Mode AD = Backpropagation

## 🎯 핵심 질문

- **Reverse-mode AD의 복잡도 이점은 정확히 무엇인가?**
  - 왜 신경망에서 $10^6$배 더 빠를까?
  - Memory-computation trade-off는 무엇인가?

- **JVP와 VJP의 차이는?**
  - 어느 상황에서 어느 것을 사용할까?

- **Checkpointing 기법은 어떻게 역전파를 최적화할까?**
  - 메모리와 시간의 trade-off를 어떻게 조절할까?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

역전파(Backpropagation)는 **reverse-mode AD의 신경망 구현**입니다. 현대 딥러닝의 스케일을 가능하게 합니다:

- **1980년대 이전**: 신경망 학습 불가능
  - 가중치가 많으면 각 가중치마다 forward pass 필요 → exponential 복잡도
  
- **역전파 발견 (1986, Rumelhart et al.)**
  - 모든 가중치에 대한 gradient를 **한 번**의 backward pass로 계산
  - **가능해진 것**: 수백만 개 파라미터의 깊은 네트워크 학습

- **현대 대규모 모델**
  - GPT-3: 1,750억 개 파라미터 (역전파 없이는 불가능)
  - 각 배치마다 backward pass 1회로 모든 gradient 계산

따라서 **역전파의 정확한 수학 = 딥러닝 이론의 핵심**입니다.

## 📐 수학적 선행 조건

- Jacobian과 연쇄법칙 (01번 문서)
- Computational graph (02번 문서)
- VJP 개념: $\bar{x} = \bar{y}^T J_f(x)$ (02번 문서)
- 행렬 전치의 성질: $(AB)^T = B^T A^T$

## 📖 직관적 이해

### 왜 Reverse-Mode가 신경망에서 필수인가?

신경망:
- **입력 차원** $n$: 입력값 개수 (이미지: $32 \times 32 \times 3 = 3072$)
- **파라미터 개수** (= 가중치): $p \approx 10^6$ (작은 네트워크)
- **손실** $L$: 스칼라 (1개 값)

Forward-mode AD:
- 각 입력/파라미터 방향에 대해 JVP 계산
- 필요한 forward pass 수: $p \approx 10^6$
- **비용**: $O(10^6 \times T)$

Reverse-mode AD:
- 손실에서 역방향으로 모든 파라미터로 VJP 계산
- 필요한 backward pass 수: 1 (손실은 스칼라)
- **비용**: $O(1 \times T) = O(T)$

**속도 비교**: $\frac{O(10^6 \times T)}{O(T)} = 10^6$배 더 빠름!

---

### Memory Trade-off

**Reverse-mode의 메모리 요구:**
- Forward pass 중 모든 중간 활성값(activation) 저장
- 깊이 100, 활성값 크기 1MB → 100MB 메모리
- 큰 배치(예: 10000) → 1GB 이상 필요 가능

**Checkpointing (Gradient Checkpointing):**
- Forward pass: 선택된 계층의 활성값만 저장
- Backward pass: 필요할 때 해당 구간을 다시 계산
- **Trade-off**: 메모리 $O(\sqrt{L})$ vs 계산 시간 2-3배 증가

---

## ✏️ 엄밀한 정의

### 1. JVP (Jacobian-Vector Product)

**정의:** 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$와 벡터 $\mathbf{v} \in \mathbb{R}^n$에 대해,
$$\text{JVP}(f, \mathbf{v}) := J_f(x) \mathbf{v} \in \mathbb{R}^m$$

**의미**: 입력의 $\mathbf{v}$ 방향 변화가 출력에 미치는 영향

**계산**: Forward-mode AD로 계산
$$\mathbf{v} \xrightarrow{f} J_f(x) \mathbf{v}$$

**복잡도**: 각 $\mathbf{v}$마다 $O(T)$ (T = primitive 연산 수)

### 2. VJP (Vector-Jacobian Product)

**정의:** 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$와 벡터 $\mathbf{u} \in \mathbb{R}^m$에 대해,
$$\text{VJP}(f, \mathbf{u}) := \mathbf{u}^T J_f(x) = J_f(x)^T \mathbf{u} \in \mathbb{R}^n$$

**의미**: 출력의 $\mathbf{u}$ 방향 변화가 입력에 미친 영향 (역으로)

**계산**: Reverse-mode AD (backpropagation)로 계산
$$\mathbf{u} \xleftarrow{f} J_f(x)^T \mathbf{u}$$

**복잡도**: 각 $\mathbf{u}$마다 $O(T)$

### 3. Reverse-Mode AD의 형식적 알고리즘

**입력**: 계산 그래프, 손실 함수 $L: y \in \mathbb{R}^m \to \mathbb{R}$ (스칼라)

**단계:**

1. **초기화**: $\bar{y}_L := 1$ (손실의 미분)

2. **Backward traversal**: 그래프를 역순으로 순회
   ```
   for each node (l = L-1, ..., 0):
       for each incoming edge from z to x:
           ∂f/∂x = Jacobian of f at current point
           x̄ += (∂f/∂x)^T · ȳ   // VJP 누적
   ```

3. **결과**: 각 입력 $x_i$에 대한 $\bar{x}_i = \frac{\partial L}{\partial x_i}$

### 4. 신경망 역전파의 구체적 형식

**가정**: $L$ 계층 신경망, 각 계층 $l$:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

**Forward pass:**
```
a^(0) = x  (입력)
for l = 1, ..., L:
    z^(l) = W^(l) @ a^(l-1) + b^(l)
    a^(l) = σ(z^(l))
```

**Backward pass (역전파):**
```
δ^(L) = ∇_{a^(L)} L ⊙ σ'(z^(L))  // 출력층 error
for l = L-1, ..., 1:
    δ^(l) = (W^(l+1))^T @ δ^(l+1) ⊙ σ'(z^(l))  // 역전파 공식
    ∂L/∂W^(l) = δ^(l) @ (a^(l-1))^T
    ∂L/∂b^(l) = δ^(l)
```

**여기서:**
- $\delta^{(l)} := \frac{\partial L}{\partial z^{(l)}}$ (error signal)
- $\odot$: element-wise 곱
- $\sigma'(z) = \frac{d\sigma}{dz}(z)$ (활성함수 미분)

## 🔬 정리와 증명

**정리 3.1 (역전파의 정확성)**

위의 backward pass 알고리즘으로 계산된 $\delta^{(l)}$은 정확히:
$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

**증명:**
귀납법. $L = $ 손실 = 스칼라이므로,

**Base case** (출력층 $l = L$):
$$\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} = \nabla_{a^{(L)}} L \odot \sigma'(z^{(L)})$$
✓

**귀납 단계** ($l < L$, $l+1$에서 성립한다고 가정):

연쇄법칙:
$$\frac{\partial L}{\partial z^{(l)}} = \frac{\partial L}{\partial z^{(l+1)}} \cdot \frac{\partial z^{(l+1)}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}}$$

여기서:
- $\frac{\partial L}{\partial z^{(l+1)}} = \delta^{(l+1)}$ (귀납 가정)
- $\frac{\partial z^{(l+1)}}{\partial a^{(l)}} = (W^{(l+1)})^T$ (선형 변환)
- $\frac{\partial a^{(l)}}{\partial z^{(l)}} = \sigma'(z^{(l)})$ (벡터 함수, 대각 Jacobian)

따라서:
$$\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})$$
✓ $\square$

**정리 3.2 (가중치 미분 공식)**

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

**증명:**
$$L \to z^{(l)} \to a^{(l)} \to \cdots \to L$$

$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$이므로,
$$\frac{\partial z^{(l)}}{\partial W^{(l)}} = a^{(l-1)} \text{ (각 성분별로)}$$

VJP를 사용하면:
$$\frac{\partial L}{\partial W^{(l)}} = J_{z^{(l)} \text{ wrt } W^{(l)}}^T \delta^{(l)} = \delta^{(l)} (a^{(l-1)})^T$$

**차원 확인**: $\delta^{(l)} \in \mathbb{R}^{m_l} \times (a^{(l-1)})^T \in \mathbb{R}^{m_{l-1}} \times [1 \times n_l] \to [m_l \times n_l]$ ✓ $\square$

**정리 3.3 (복잡도 정리)**

신경망 $f: \mathbb{R}^n \to \mathbb{R}$ (손실은 스칼라), $T$ 개의 primitive 연산:

| 방법 | Forward pass | Backward pass | 총 복잡도 | 비용 (n = 10^6) |
|------|-------------|---------------|---------|--------------|
| Forward-mode AD | $O(n \times T)$ | - | $O(n \times T)$ | 매우 비쌈 |
| Reverse-mode AD | $O(T)$ | $O(T)$ | $O(2T)$ | 매우 저렴 |
| 속도 비 | - | - | $\frac{n}{2} \approx n$ | **500,000배 빠름** |

## 💻 NumPy로 바닥부터 구현

### 간단한 2-계층 MLP

```python
import numpy as np

class SimpleNN:
    """
    2-계층 신경망: x -> σ(W₁x + b₁) -> σ(W₂ + b₂) -> output
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((output_dim, 1))
        
        # Backward 저장용
        self.cache = {}
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # 수치 안정성
    
    def sigmoid_prime(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, x):
        """Forward pass, cache 저장"""
        # 1번째 계층
        z1 = self.W1 @ x + self.b1
        a1 = self.sigmoid(z1)
        
        # 2번째 계층
        z2 = self.W2 @ a1 + self.b2
        a2 = self.sigmoid(z2)
        
        # Cache (backward용)
        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        
        return a2
    
    def backward(self, dL_da2):
        """
        Reverse-mode AD (역전파)
        dL_da2: 손실에 대한 출력의 미분
        """
        cache = self.cache
        
        # 출력층 역전파
        dL_dz2 = dL_da2 * self.sigmoid_prime(cache['z2'])
        dL_dW2 = dL_dz2 @ cache['a1'].T
        dL_db2 = dL_dz2
        
        # 은닉층으로 역전파
        dL_da1 = self.W2.T @ dL_dz2
        dL_dz1 = dL_da1 * self.sigmoid_prime(cache['z1'])
        dL_dW1 = dL_dz1 @ cache['x'].T
        dL_db1 = dL_dz1
        
        return {
            'dL_dW1': dL_dW1, 'dL_db1': dL_db1,
            'dL_dW2': dL_dW2, 'dL_db2': dL_db2
        }

# 테스트: XOR 문제
np.random.seed(42)
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])  # [2, 4]
Y = np.array([[0, 1, 1, 0]])  # [1, 4]

nn = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)

print("=== Forward Pass ===")
for i in range(4):
    x_i = X[:, i:i+1]
    y_i = nn.forward(x_i)
    print(f"Input {X[:, i]} -> Output {y_i[0, 0]:.4f}")

print("\n=== Backward Pass (역전파) ===")
x_sample = X[:, 0:1]
y_pred = nn.forward(x_sample)
y_true = Y[:, 0:1]
loss = 0.5 * (y_pred - y_true)**2

dL_dy = y_pred - y_true  # BCE/MSE loss gradient
grads = nn.backward(dL_dy)

print(f"Loss: {loss[0, 0]:.6f}")
print(f"dL_dW1 shape: {grads['dL_dW1'].shape}")
print(f"dL_dW1:\n{grads['dL_dW1']}")
print(f"dL_dW2 shape: {grads['dL_dW2'].shape}")
print(f"dL_db2: {grads['dL_db2']}")

# 수치 미분으로 검증
def numerical_gradient_W1(nn, x, y_true, eps=1e-5):
    """W1의 수치 gradient"""
    grad = np.zeros_like(nn.W1)
    for i in range(nn.W1.shape[0]):
        for j in range(nn.W1.shape[1]):
            nn.W1[i, j] += eps
            y_plus = nn.forward(x)
            loss_plus = 0.5 * (y_plus - y_true)**2
            
            nn.W1[i, j] -= 2*eps
            y_minus = nn.forward(x)
            loss_minus = 0.5 * (y_minus - y_true)**2
            
            nn.W1[i, j] += eps  # 원래대로 복원
            
            grad[i, j] = (loss_plus - loss_minus)[0, 0] / (2 * eps)
    return grad

grad_numerical = numerical_gradient_W1(nn, x_sample, y_true)
print(f"\n수치 미분 오차 (W1):")
print(f"  Max abs diff: {np.max(np.abs(grads['dL_dW1'] - grad_numerical)):.2e}")
```

### Batched Backpropagation

```python
def batch_forward_backward(nn, X_batch, Y_batch):
    """
    배치 처리 역전파
    X_batch: [n_features, batch_size]
    Y_batch: [output_dim, batch_size]
    """
    batch_size = X_batch.shape[1]
    
    # Accumulate gradients
    dL_dW1_accum = np.zeros_like(nn.W1)
    dL_db1_accum = np.zeros_like(nn.b1)
    dL_dW2_accum = np.zeros_like(nn.W2)
    dL_db2_accum = np.zeros_like(nn.b2)
    
    total_loss = 0.0
    
    for i in range(batch_size):
        x_i = X_batch[:, i:i+1]
        y_i = Y_batch[:, i:i+1]
        
        # Forward
        y_pred = nn.forward(x_i)
        loss = 0.5 * np.sum((y_pred - y_i)**2)
        total_loss += loss
        
        # Backward
        dL_dy = y_pred - y_i
        grads = nn.backward(dL_dy)
        
        # Accumulate
        dL_dW1_accum += grads['dL_dW1']
        dL_db1_accum += grads['dL_db1']
        dL_dW2_accum += grads['dL_dW2']
        dL_db2_accum += grads['dL_db2']
    
    # Average
    dL_dW1_accum /= batch_size
    dL_db1_accum /= batch_size
    dL_dW2_accum /= batch_size
    dL_db2_accum /= batch_size
    
    return {
        'dL_dW1': dL_dW1_accum,
        'dL_db1': dL_db1_accum,
        'dL_dW2': dL_dW2_accum,
        'dL_db2': dL_db2_accum
    }, total_loss / batch_size

# Batched backprop 테스트
print("\n=== Batched Backpropagation ===")
grads_batch, loss_batch = batch_forward_backward(nn, X, Y)
print(f"Batch loss: {loss_batch:.6f}")
print(f"Batch gradient dL_dW1 shape: {grads_batch['dL_dW1'].shape}")

# 학습 루프
print("\n=== Training Loop (10 iterations) ===")
learning_rate = 0.5
nn = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)
---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Computational Graph와 AD](./02-computational-graph-ad.md) | [📚 README로 돌아가기](../README.md) | [04. MLP 역전파 공식 유도 ▶](./04-mlp-backprop-formula.md) |

</div>
