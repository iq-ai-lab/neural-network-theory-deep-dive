# 04. MLP 역전파 공식 완전 유도

## 🎯 핵심 질문

- **완전 연결 계층(Dense layer)의 역전파를 한 줄씩 유도할 수 있는가?**
  - Forward: $z_l = W_l a_{l-1} + b_l$, $a_l = \sigma(z_l)$
  - Backward: 어떻게 거꾸로 계산할 것인가?

- **Error signal $\delta_l$의 정확한 정의는?**
  - 왜 $\delta_l = \nabla_{z_l} L$로 정의할까?

- **Vanishing/Exploding Gradient는 어떻게 수학으로 나타나나?**
  - 깊은 네트워크에서 gradient가 어떻게 변하는가?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

**역전파의 핵심은 수식입니다.** 신경망 학습의 모든 것이 여기에 있습니다:

- **1986년 Rumelhart et al.** "Learning Representations by Backpropagating Errors"
  - 이 정확한 공식을 발견함으로써 신경망 학습의 문을 열음
  - 이후 40년간 딥러닝의 기초

- **현대 프레임워크 (PyTorch, TensorFlow)**
  - 모두 이 공식을 자동으로 계산
  - 하지만 원리를 모르면 debugging 불가능

- **이상 증상 진단:**
  - Gradient가 NaN?이 공식으로 어디가 폭발했는지 찾을 수 있음
  - Learning 안 되는 이유? Vanishing gradient의 수학적 증명 필요

따라서 **역전파 공식의 완전한 이해 = 신경망 엔지니어링의 기초**입니다.

## 📐 수학적 선행 조건

- 편미분과 연쇄법칙 (01번 문서)
- Jacobian의 행렬 형식 (01번 문서)
- Reverse-mode AD의 VJP (02, 03번 문서)
- 행렬 미분: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$

## 📖 직각적 이해

### MLP의 구조

```
x (입력, n₀ 차원)
  ↓ W₁ [n₁ × n₀] + b₁ [n₁ × 1]
z₁ [n₁ × 1]
  ↓ σ (sigmoid, tanh, ReLU 등)
a₁ [n₁ × 1]
  ↓ W₂ [n₂ × n₁] + b₂
z₂ [n₂ × 1]
  ↓ σ
a₂ [n₂ × 1]
  ↓ W₃ [1 × n₂] + b₃ (출력층, 손실)
z₃ [1 × 1] = L
```

### Forward Pass의 직관

각 계층:
1. **선형 변환**: $z_l = W_l a_{l-1} + b_l$ (입력 $a_{l-1}$ → 가중합)
2. **비선형 활성**: $a_l = \sigma(z_l)$ (비선형성 추가)

### Backward Pass의 직관

역순으로, 각 계층:
1. **출력층에서 시작**: 손실의 gradient $\nabla L$
2. **가중합으로 역전파**: $\delta_l = \nabla_{z_l} L$
3. **입력으로 전파**: $\nabla_{a_{l-1}} L$
4. **가중치/편향 미분**: $\nabla_{W_l} L$, $\nabla_{b_l} L$

---

## ✏️ 엄밀한 정의

### 1. Forward Pass (한 계층)

**입력**: $a_{l-1} \in \mathbb{R}^{n_{l-1}}$ (이전 계층의 활성값)
**파라미터**: $W_l \in \mathbb{R}^{n_l \times n_{l-1}}$, $b_l \in \mathbb{R}^{n_l}$
**출력**: $a_l \in \mathbb{R}^{n_l}$ (이 계층의 활성값)

```
z_l = W_l @ a_{l-1} + b_l     # [n_l × 1] = [n_l × n_{l-1}] @ [n_{l-1} × 1] + [n_l × 1]
a_l = σ(z_l)                   # element-wise 활성함수 적용
```

### 2. Error Signal (핵심 정의)

$\delta_l$을 정의:
$$\delta_l := \frac{\partial L}{\partial z_l} \in \mathbb{R}^{n_l}$$

**의미**: 손실 $L$이 가중합 $z_l$에 얼마나 민감한가?

### 3. 역전파의 단계별 유도

#### **Step 1: 출력층 역전파**

출력층 (계층 $L$): $z_L$ → $a_L$ → $L$ (손실은 스칼라)

연쇄법칙:
$$\delta_L = \frac{\partial L}{\partial z_L} = \frac{\partial L}{\partial a_L} \cdot \frac{\partial a_L}{\partial z_L}$$

여기서:
- $\frac{\partial L}{\partial a_L} = \nabla_{a_L} L$ (손실의 gradient, 주어짐 또는 계산 가능)
- $\frac{\partial a_L}{\partial z_L} = \sigma'(z_L)$ (활성함수 미분, element-wise)

따라서:
$$\boxed{\delta_L = \nabla_{a_L} L \odot \sigma'(z_L)}$$

여기서 $\odot$는 element-wise 곱.

#### **Step 2: 중간 계층 역전파**

계층 $l < L$:

먼저 $\nabla_{a_l} L$을 구해야 함. $a_l$은 다음 계층 $l+1$의 입력이므로:
$$\frac{\partial L}{\partial a_l} = \frac{\partial L}{\partial z_{l+1}} \cdot \frac{\partial z_{l+1}}{\partial a_l}$$

연쇄법칙:
$$\nabla_{a_l} L = (W_{l+1})^T \delta_{l+1}$$

**이유**: $z_{l+1} = W_{l+1} a_l + b_{l+1}$이므로,
$$\frac{\partial z_{l+1}}{\partial a_l} = W_{l+1}$$

따라서:
$$\delta_l = \nabla_{a_l} L \odot \sigma'(z_l) = \left((W_{l+1})^T \delta_{l+1}\right) \odot \sigma'(z_l)$$

$$\boxed{\delta_l = \left((W_{l+1})^T \delta_{l+1}\right) \odot \sigma'(z_l)}$$

#### **Step 3: 가중치와 편향의 미분**

$z_l = W_l a_{l-1} + b_l$에서:

**가중치에 대한 미분:**
$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial z_l} \cdot \frac{\partial z_l}{\partial W_l}$$

$z_{l,i} = \sum_j W_{l,ij} a_{l-1,j} + b_{l,i}$이므로:
$$\frac{\partial z_{l,i}}{\partial W_{l,ij}} = a_{l-1,j}$$

따라서:
$$\frac{\partial L}{\partial W_{l,ij}} = \delta_{l,i} \cdot a_{l-1,j}$$

행렬 형태:
$$\boxed{\frac{\partial L}{\partial W_l} = \delta_l \cdot (a_{l-1})^T \in \mathbb{R}^{n_l \times n_{l-1}}}$$

**편향에 대한 미분:**
$$\frac{\partial L}{\partial b_{l,i}} = \delta_{l,i}$$

행렬 형태:
$$\boxed{\frac{\partial L}{\partial b_l} = \delta_l \in \mathbb{R}^{n_l}}$$

---

## 🔬 정리와 증명

**정리 4.1 (MLP 역전파의 정확성 — 귀납법 증명)**

$L$-계층 MLP에서, 위의 공식으로 계산된 $\delta_l$과 $\frac{\partial L}{\partial W_l}$은:

$$\delta_l = \frac{\partial L}{\partial z_l}, \quad \frac{\partial L}{\partial W_l} = \delta_l (a_{l-1})^T$$

**증명:**

귀납법. 계층 $L$에서 계층 1까지.

**Base case** ($l = L$):

출력층에서 손실 $L$까지의 경로:
$$L(\text{logits}, \text{labels})$$

따라서:
$$\delta_L = \frac{\partial L}{\partial z_L} = \frac{\partial L}{\partial a_L} \odot \sigma'(z_L) = \nabla_{a_L} L \odot \sigma'(z_L)$$

✓

**귀납 단계** ($l < L$, $\delta_{l+1}$이 올바르다고 가정):

$a_l$은 계층 $l+1$의 입력이므로:
$$\frac{\partial L}{\partial a_l} = \frac{\partial L}{\partial z_{l+1}} \cdot \frac{\partial z_{l+1}}{\partial a_l}$$

$z_{l+1} = W_{l+1} a_l + b_{l+1}$에서:
$$\frac{\partial z_{l+1}}{\partial a_l} = W_{l+1}$$

따라서:
$$\frac{\partial L}{\partial a_l} = (W_{l+1})^T \frac{\partial L}{\partial z_{l+1}} = (W_{l+1})^T \delta_{l+1}$$

그러므로:
$$\delta_l = \frac{\partial L}{\partial z_l} = \frac{\partial L}{\partial a_l} \odot \sigma'(z_l) = \left((W_{l+1})^T \delta_{l+1}\right) \odot \sigma'(z_l)$$

✓

**가중치 미분:**
$z_l = W_l a_{l-1} + b_l$에서, $\frac{\partial z_{l,i}}{\partial W_{l,ij}} = a_{l-1,j}$:

$$\frac{\partial L}{\partial W_{l,ij}} = \sum_i \frac{\partial L}{\partial z_{l,i}} \frac{\partial z_{l,i}}{\partial W_{l,ij}} = \delta_{l,i} a_{l-1,j}$$

행렬 표기: $\frac{\partial L}{\partial W_l} = \delta_l (a_{l-1})^T$ ✓

$\square$

**정리 4.2 (Gradient의 크기 관계 — Vanishing/Exploding Gradient)**

깊은 네트워크에서:
$$\delta_1 = (W_2)^T \sigma'(z_2) \cdots (W_L)^T \sigma'(z_L) \delta_L$$

만약 모든 고유값 $|\lambda_i(W_l)| < 1$ (또는 $\sigma'(z_l) < \tau < 1$)이면:
$$\|\delta_1\| \lesssim \prod_{l=2}^{L} |\lambda_{\max}(W_l)| \cdot \max_l \sigma'(z_l) \cdot \|\delta_L\|$$

깊이 $L$에서:
$$\|\delta_1\| \sim \rho^{L-1} \|\delta_L\|$$

여기서 $\rho = \lambda_{\max} \cdot \sigma'_{\max} < 1$이면:
$$\|\delta_1\| \sim \rho^L \|\delta_L\| \to 0 \text{ as } L \to \infty$$

**결론**: Sigmoid ($\sigma'(z) \leq 0.25$)와 작은 가중치 ($|\lambda| < 0.5$)로 깊은 네트워크를 학습하려면 gradient가 exponential하게 감소합니다.

## 💻 NumPy로 바닥부터 구현

### 3-계층 MLP 전체 구현

```python
import numpy as np

class MLP:
    """3-계층 MLP: input -> hidden1 -> hidden2 -> output"""
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=42):
        np.random.seed(seed)
        
        # Xavier initialization
        self.W1 = np.random.randn(hidden_dim1, input_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((hidden_dim1, 1))
        
        self.W2 = np.random.randn(hidden_dim2, hidden_dim1) / np.sqrt(hidden_dim1)
        self.b2 = np.zeros((hidden_dim2, 1))
        
        self.W3 = np.random.randn(output_dim, hidden_dim2) / np.sqrt(hidden_dim2)
        self.b3 = np.zeros((output_dim, 1))
        
        self.cache = {}
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        """a = σ(z), σ'(z) = σ(z)(1 - σ(z)) = a(1-a)"""
        return a * (1 - a)
    
    def forward(self, x):
        """Forward pass, cache 저장"""
        # Layer 1
        z1 = self.W1 @ x + self.b1
        a1 = self.sigmoid(z1)
        
        # Layer 2
        z2 = self.W2 @ a1 + self.b2
        a2 = self.sigmoid(z2)
        
        # Layer 3 (output)
        z3 = self.W3 @ a2 + self.b3
        a3 = self.sigmoid(z3)
        
        self.cache = {
            'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3
        }
        
        return a3
    
    def backward(self, y_true):
        """
        Backpropagation (역전파)
        y_true: 정답 (one-hot 또는 연속값)
        """
        cache = self.cache
        m = cache['x'].shape[1] if cache['x'].ndim > 1 else 1
        
        # 출력층 error signal
        # Loss = 0.5 * (a3 - y_true)^2
        # dL/da3 = a3 - y_true
        # dL/dz3 = dL/da3 * σ'(z3)
        dL_da3 = cache['a3'] - y_true
        delta3 = dL_da3 * self.sigmoid_derivative(cache['a3'])
        
        # Layer 3 파라미터 미분
        dL_dW3 = delta3 @ cache['a2'].T / m
        dL_db3 = np.sum(delta3, axis=1, keepdims=True) / m
        
        # 계층 2로 역전파
        dL_da2 = self.W3.T @ delta3
        delta2 = dL_da2 * self.sigmoid_derivative(cache['a2'])
        
        # Layer 2 파라미터 미분
        dL_dW2 = delta2 @ cache['a1'].T / m
        dL_db2 = np.sum(delta2, axis=1, keepdims=True) / m
        
        # 계층 1로 역전파
        dL_da1 = self.W2.T @ delta2
        delta1 = dL_da1 * self.sigmoid_derivative(cache['a1'])
        
        # Layer 1 파라미터 미분
        dL_dW1 = delta1 @ cache['x'].T / m
        dL_db1 = np.sum(delta1, axis=1, keepdims=True) / m
        
        return {
            'dW1': dL_dW1, 'db1': dL_db1,
            'dW2': dL_dW2, 'db2': dL_db2,
            'dW3': dL_dW3, 'db3': dL_db3,
            'delta1': delta1, 'delta2': delta2, 'delta3': delta3
        }
    
    def update(self, grads, learning_rate):
        """Gradient descent update"""
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']
        self.W3 -= learning_rate * grads['dW3']
        self.b3 -= learning_rate * grads['db3']
    
    def compute_loss(self, x, y):
        """MSE loss"""
        a3 = self.forward(x)
        loss = 0.5 * np.mean((a3 - y) ** 2)
        return loss


# 테스트: XOR 문제
print("=== XOR Problem ===")
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=np.float32)
Y = np.array([[0, 1, 1, 0]], dtype=np.float32)

mlp = MLP(input_dim=2, hidden_dim1=4, hidden_dim2=4, output_dim=1)

print(f"Initial loss: {mlp.compute_loss(X, Y):.6f}")

# Forward + Backward
print("\n=== Gradient Checking (수치 미분 vs 해석적 미분) ===")

x_single = X[:, 0:1]
y_single = Y[:, 0:1]

# 해석적 미분
_ = mlp.forward(x_single)
grads = mlp.backward(y_single)

# 수치 미분 (W1 일부만 확인)
def numerical_grad_W1(mlp, x, y, idx, eps=1e-5):
    mlp.W1[idx] += eps
    loss_plus = mlp.compute_loss(x, y)
    mlp.W1[idx] -= 2*eps
    loss_minus = mlp.compute_loss(x, y)
    mlp.W1[idx] += eps
    return (loss_plus - loss_minus) / (2 * eps)

# 첫 5개 원소만 확인
print(f"Analytical gradient (first 5): {grads['dW1'].flatten()[:5]}")
numerical_grads = [numerical_grad_W1(mlp, x_single, y_single, (i, j))
                   for i in range(min(2, mlp.W1.shape[0]))
                   for j in range(min(3, mlp.W1.shape[1]))]
print(f"Numerical gradient (first 5): {numerical_grads[:5]}")
print(f"Max error: {np.max(np.abs(np.array(numerical_grads[:5]) - grads['dW1'].flatten()[:5])):.2e}")

# 학습 루프
print("\n=== Training (20 iterations) ===")
mlp = MLP(input_dim=2, hidden_dim1=8, hidden_dim2=8, output_dim=1)
learning_rate = 1.0

for epoch in range(20):
    # Mini-batch gradient descent
    total_loss = 0
    for i in range(X.shape[1]):
        x_i = X[:, i:i+1]
        y_i = Y[:, i:i+1]
        
        mlp.forward(x_i)
        grads = mlp.backward(y_i)
        mlp.update(grads, learning_rate)
        
        loss_i = mlp.compute_loss(x_i, y_i)
        total_loss += loss_i
    
    avg_loss = total_loss / X.shape[1]
    
    if epoch % 4 == 0:
        print(f"Epoch {epoch:2d}: Loss = {avg_loss:.6f}")

print("\n=== Final Predictions ===")
for i in range(X.shape[1]):
---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Reverse-Mode AD = Backpropagation](./03-reverse-mode-backprop.md) | [📚 README로 돌아가기](../README.md) | [05. Softmax + Cross-Entropy의 Gradient ▶](./05-softmax-crossentropy-grad.md) |

</div>
