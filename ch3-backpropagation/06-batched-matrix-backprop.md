# 06. Batched Computation과 Matrix 미분

## 🎯 핵심 질문

- **신경망을 배치(batch) 단위로 처리하면 역전파는 어떻게 변할까?**
  - 개별 샘플 대신 행렬로 동시 처리: 수학적으로는?

- **배치 역전파의 행렬 형태는?**
  - $\partial L / \partial W$를 배치 차원을 포함해 계산하려면?

- **Trace trick과 행렬 미분은 무엇인가?**
  - 복잡한 행렬 미분을 어떻게 간단히 계산할 것인가?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

**배치 처리는 현대 신경망 학습의 핵심입니다:**

- **계산 효율성**
  - 100개 샘플을 개별 처리: 100번 forward + 100번 backward
  - 배치 처리: 1번 forward + 1번 backward (행렬 연산)
  - **GPU/TPU 활용**: BLAS 최적화된 행렬 곱 → 100배 빠름

- **메모리 효율성**
  - Vectorized operations는 캐시 친화적
  - Sequential processing보다 메모리 대역폭 활용도 높음

- **통계적 이점**
  - 배치 정규화(Batch Normalization)
  - 더 안정적인 gradient 추정

따라서 **배치 수준에서의 수학적 이해 = 신경망 프레임워크 설계의 기초**입니다.

## 📐 수학적 선행 조건

- 행렬 곱셈: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$
- MLP 역전파 공식 (04번 문서)
- Trace 연산: $\text{tr}(A) = \sum_i A_{ii}$, $\text{tr}(ABC) = \text{tr}(CAB)$
- Frobenius norm: $\|A\|_F^2 = \text{tr}(A^T A)$

## 📖 직관적 이해

### 배치 처리 구조

**개별 샘플 처리:**
```
for i in range(batch_size):
    x_i ∈ ℝ^{n}
    forward: y_i = model(x_i)  # 스칼라 또는 벡터
    backward: ∂L/∂W
```

**배치 처리:**
```
X ∈ ℝ^{B × n}  (B = batch_size, n = input_dim)
forward: Y = X @ W^T  ∈ ℝ^{B × m}  (m = output_dim)
backward: ∂L/∂W ∈ ℝ^{m × n}
```

### 행렬 관점의 변화

| 요소 | 개별 샘플 | 배치 |
|------|---------|------|
| 입력 | $x \in \mathbb{R}^n$ | $X \in \mathbb{R}^{B \times n}$ |
| 가중치 | $W \in \mathbb{R}^{m \times n}$ | $W \in \mathbb{R}^{m \times n}$ (공유) |
| 출력 | $y = Wx \in \mathbb{R}^m$ | $Y = XW^T \in \mathbb{R}^{B \times m}$ |
| Error | $\delta \in \mathbb{R}^m$ | $\Delta \in \mathbb{R}^{B \times m}$ |

---

## ✏️ 엄밀한 정의

### 1. 배치 Forward Pass

**입력**: $X \in \mathbb{R}^{B \times n_{\text{in}}}$ (B개 샘플, n차원 입력)
**가중치**: $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$ (공유 파라미터)
**편향**: $b \in \mathbb{R}^{n_{\text{out}}}$ (또는 broadcasting)

```
Z = XW^T + b^T  ∈ ℝ^{B × n_out}
A = σ(Z)        ∈ ℝ^{B × n_out}
```

**행렬 구조:**
$$Z_{bi} = \sum_{j=1}^{n_{\text{in}}} X_{bj} W_{ij} + b_i$$

여기서 $b = (b_1, \ldots, b_{n_{\text{out}}})$는 모든 샘플에 broadcast.

### 2. 배치 Backward Pass

**출력 손실**: $L = \sum_{b=1}^B L_b$ (각 샘플의 손실 합)

**Error signal 행렬**:
$$\Delta_{bi} := \frac{\partial L}{\partial Z_{bi}} \in \mathbb{R}^{B \times n_{\text{out}}}$$

### 3. 배치 가중치 미분

**정리**: 배치 처리에서,
$$\frac{\partial L}{\partial W} = \Delta^T X \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$$

**증명:**

연쇄법칙:
$$\frac{\partial L}{\partial W_{ij}} = \sum_{b=1}^B \frac{\partial L}{\partial Z_{bi}} \frac{\partial Z_{bi}}{\partial W_{ij}}$$

$Z_{bi} = \sum_k X_{bk} W_{ik} + b_i$이므로:
$$\frac{\partial Z_{bi}}{\partial W_{ij}} = X_{bj}$$

따라서:
$$\frac{\partial L}{\partial W_{ij}} = \sum_{b=1}^B \Delta_{bi} X_{bj}$$

행렬 표기:
$$\boxed{\frac{\partial L}{\partial W} = \Delta^T X}$$

**차원 확인**: $[n_{\text{out}} \times B] \times [B \times n_{\text{in}}] = [n_{\text{out}} \times n_{\text{in}}]$ ✓

### 4. 배치 입력 미분

역전파를 위해 입력에 대한 gradient:
$$\frac{\partial L}{\partial X} = \Delta W \in \mathbb{R}^{B \times n_{\text{in}}}$$

**증명:**
$$\frac{\partial L}{\partial X_{bj}} = \sum_i \frac{\partial L}{\partial Z_{bi}} \frac{\partial Z_{bi}}{\partial X_{bj}} = \sum_i \Delta_{bi} W_{ij}$$

행렬 표기:
$$\boxed{\frac{\partial L}{\partial X} = \Delta W}$$

**차원**: $[B \times n_{\text{out}}] \times [n_{\text{out}} \times n_{\text{in}}] = [B \times n_{\text{in}}]$ ✓

### 5. 배치 편향 미분

$$\frac{\partial L}{\partial b_i} = \sum_{b=1}^B \Delta_{bi}$$

행렬 표기:
$$\boxed{\frac{\partial L}{\partial b} = \sum_{b=1}^B \Delta[b, :] = \mathbf{1}^T \Delta}$$

여기서 $\mathbf{1} \in \mathbb{R}^B$는 모두 1인 벡터.

### 6. Trace Trick (행렬 미분의 핵심)

**정의**: $A, B$가 행렬일 때,
$$d(\text{tr}(AB)) = \text{tr}(dA \cdot B + A \cdot dB) = \text{tr}(B \cdot dA) + \text{tr}(A \cdot dB)$$

**핵심 성질**:
$$\text{tr}(A^T B) = \sum_{ij} A_{ij} B_{ij} = \langle A, B \rangle_F$$

(Frobenius inner product)

**예시: 손실이 행렬 곱의 Frobenius norm일 때**

$$L = \frac{1}{2} \|Y - Z\|_F^2 = \frac{1}{2} \text{tr}((Y - Z)^T(Y - Z))$$

$Y = XW^T$이면:
$$dL = \text{tr}((Y - Z)^T d(XW^T))$$
$$= \text{tr}((Y - Z)^T (dX \cdot W^T + X \cdot dW^T))$$
$$= \text{tr}(W(Y - Z)^T dX) + \text{tr}((Y - Z)^T dX W^T)$$

미분을 추출:
$$\frac{\partial L}{\partial X} = (Y - Z) W$$
$$\frac{\partial L}{\partial W} = (Y - Z)^T X$$

---

## 🔬 정리와 증명

**정리 6.1 (배치 선형층 역전파)**

배치 입력 $X \in \mathbb{R}^{B \times n_{\text{in}}}$, $Z = XW^T + b$, 손실 $L$에 대해:

$$\frac{\partial L}{\partial W} = \Delta^T X, \quad \frac{\partial L}{\partial b} = \mathbf{1}^T \Delta, \quad \frac{\partial L}{\partial X} = \Delta W$$

여기서 $\Delta_{bi} = \frac{\partial L}{\partial Z_{bi}}$.

**증명:**
위의 연쇄법칙 계산 참조. $\square$

**정리 6.2 (배치 비선형층 역전파)**

배치 활성화 $A = \sigma(Z)$일 때:

$$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot \sigma'(Z)$$

여기서 $\odot$는 element-wise 곱.

**증명:**
$$\frac{\partial L}{\partial Z_{bi}} = \frac{\partial L}{\partial A_{bi}} \sigma'(Z_{bi})$$

행렬 표기로 element-wise operation. $\square$

**정리 6.3 (계산 복잡도)**

| 방식 | Forward | Backward | 총계 |
|------|---------|----------|------|
| Sequential (for loop) | $B \times O(n_{\text{in}} n_{\text{out}})$ | $B \times O(n_{\text{in}} n_{\text{out}})$ | $O(2B \cdot n_{\text{in}} n_{\text{out}})$ |
| Batched (GEMM) | $O(B \cdot n_{\text{in}} n_{\text{out}})$ | $O(2B \cdot n_{\text{in}} n_{\text{out}})$ | **동일** |
| GPU 활용 | Sequential: 낮은 utilization | Batched: 높은 utilization | **100배 이상 차이** |

---

## 💻 NumPy로 바닥부터 구현

### 배치 처리 선형층

```python
import numpy as np
import time

class LinearLayer:
    """배치 처리가 가능한 선형층"""
    
    def __init__(self, input_dim, output_dim):
        # Xavier initialization
        self.W = np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
        self.b = np.zeros((output_dim,))
        
        self.cache = {}
    
    def forward(self, X):
        """
        X: [batch_size, input_dim]
        Output: Z ∈ [batch_size, output_dim]
        """
        # Z = X @ W^T + b (broadcasting)
        Z = X @ self.W.T + self.b
        
        self.cache = {'X': X}
        return Z
    
    def backward(self, dL_dZ):
        """
        역전파
        dL_dZ: [batch_size, output_dim] (upstream gradient)
        반환: dL_dW, dL_db, dL_dX
        """
        X = self.cache['X']
        batch_size = X.shape[0]
        
        # ∂L/∂W = dL_dZ^T @ X
        dL_dW = dL_dZ.T @ X / batch_size
        
        # ∂L/∂b = sum_batch dL_dZ
        dL_db = np.sum(dL_dZ, axis=0) / batch_size
        
        # ∂L/∂X = dL_dZ @ W (for next layer)
        dL_dX = dL_dZ @ self.W
        
        return dL_dW, dL_db, dL_dX
    
    def update(self, dL_dW, dL_db, learning_rate):
        """경사 하강"""
        self.W -= learning_rate * dL_dW
        self.b -= learning_rate * dL_db


class BatchedNN:
    """Batched 처리가 가능한 MLP"""
    
    def __init__(self, layer_dims):
        """
        layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(LinearLayer(layer_dims[i], layer_dims[i+1]))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        """Forward pass with activations"""
        A = [X]
        Z = []
        
        for layer in self.layers[:-1]:
            z = layer.forward(A[-1])
            a = self.sigmoid(z)
            Z.append(z)
            A.append(a)
        
        # Output layer (no activation)
        z_out = self.layers[-1].forward(A[-1])
        Z.append(z_out)
        A.append(z_out)
        
        return A, Z
    
    def backward(self, A, Z, dL_dA_out):
        """
        역전파
        A: activations [a0, a1, ..., aL]
        Z: pre-activations [z1, ..., zL]
        dL_dA_out: output layer의 gradient
        """
        grads = []
        
        # 출력층부터 역순
        dL_dZ = dL_dA_out  # Output layer (no activation)
        
        for i in range(len(self.layers) - 1, -1, -1):
            dL_dW, dL_db, dL_dA = self.layers[i].backward(dL_dZ)
            grads.insert(0, (dL_dW, dL_db))
            
            if i > 0:
                # 활성함수 미분 곱하기
                dL_dZ = dL_dA * self.sigmoid_derivative(A[i])
        
        return grads
    
    def compute_loss(self, X, Y):
        """MSE loss"""
        A, Z = self.forward(X)
        Y_pred = A[-1]
        loss = np.mean((Y_pred - Y) ** 2) / 2
        return loss, A, Z
    
    def update(self, grads, learning_rate):
        """모든 계층 업데이트"""
        for i, (dL_dW, dL_db) in enumerate(grads):
            self.layers[i].update(dL_dW, dL_db, learning_rate)


# 테스트 1: 간단한 배치 역전파
print("=== Test 1: Batched Backward ===")
np.random.seed(42)

layer = LinearLayer(input_dim=3, output_dim=2)
X = np.random.randn(5, 3)  # batch_size=5, input_dim=3

# Forward
Z = layer.forward(X)
print(f"Input shape: {X.shape}")
print(f"Output shape: {Z.shape}")
print(f"W shape: {layer.W.shape}")

# Simulated gradient
dL_dZ = np.random.randn(5, 2)  # batch_size=5, output_dim=2

# Backward
dL_dW, dL_db, dL_dX = layer.backward(dL_dZ)
print(f"∂L/∂W shape: {dL_dW.shape} (expected [2, 3])")
print(f"∂L/∂b shape: {dL_db.shape} (expected [2])")
print(f"∂L/∂X shape: {dL_dX.shape} (expected [5, 3])")

# 수치 미분으로 검증 (첫 원소만)
def numerical_grad_W(layer, X, dL_dZ, idx, eps=1e-5):
    layer.W[idx] += eps
    Z_plus = layer.forward(X)
    loss_plus = np.sum(dL_dZ * Z_plus)
    
    layer.W[idx] -= 2*eps
    Z_minus = layer.forward(X)
    loss_minus = np.sum(dL_dZ * Z_minus)
    
    layer.W[idx] += eps
    return (loss_plus - loss_minus) / (2 * eps)

grad_num_w00 = numerical_grad_W(layer, X, dL_dZ, (0, 0))
print(f"\nGradient checking (W[0,0]):")
print(f"  Analytical: {dL_dW[0, 0]:.6f}")
print(f"  Numerical:  {grad_num_w00:.6f}")
print(f"  Error: {abs(dL_dW[0, 0] - grad_num_w00):.2e}")

# 테스트 2: 배치 vs 순차 처리 정확도
print("\n=== Test 2: Batched vs Sequential (Accuracy) ===")

def sequential_backward(X, dL_dZ, layer):
    """순차 처리 역전파"""
    dL_dW_accum = np.zeros_like(layer.W)
    dL_db_accum = np.zeros_like(layer.b)
    
    for b in range(X.shape[0]):
        x_b = X[b:b+1]
        dL_dz_b = dL_dZ[b:b+1]
        
        z_b = x_b @ layer.W.T + layer.b
        dL_dW_b = dL_dz_b.T @ x_b
        dL_db_b = np.sum(dL_dz_b, axis=0)
        
        dL_dW_accum += dL_dW_b
        dL_db_accum += dL_db_b
    
    dL_dW_accum /= X.shape[0]
    dL_db_accum /= X.shape[0]
    
    return dL_dW_accum, dL_db_accum

dL_dW_seq, dL_db_seq = sequential_backward(X, dL_dZ, layer)
print(f"Batched   dL/dW: {dL_dW.flatten()[:3]}")
print(f"Sequential dL/dW: {dL_dW_seq.flatten()[:3]}")
print(f"Max difference: {np.max(np.abs(dL_dW - dL_dW_seq)):.2e}")

# 테스트 3: 속도 비교
print("\n=== Test 3: Speed Comparison (1000 iterations) ===")

batch_size = 1000
input_dim = 100
output_dim = 10
num_iterations = 1000

layer = LinearLayer(input_dim, output_dim)
X = np.random.randn(batch_size, input_dim)
dL_dZ = np.random.randn(batch_size, output_dim)

# Batched
start = time.time()
for _ in range(num_iterations):
    dL_dW, dL_db, dL_dX = layer.backward(dL_dZ)
---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Softmax + Cross-Entropy](./05-softmax-crossentropy-grad.md) | [📚 README로 돌아가기](../README.md) | [Ch4-01. 초기화와 Symmetry Breaking ▶](../ch4-initialization/01-symmetry-breaking.md) |

</div>
