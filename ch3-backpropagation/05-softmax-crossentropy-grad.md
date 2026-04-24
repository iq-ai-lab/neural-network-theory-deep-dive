# 05. Softmax + Cross-Entropy의 Gradient

## 🎯 핵심 질문

- **Softmax의 Jacobian을 어떻게 구할까?**
  - 각 출력 $\hat{y}_i$가 모든 입력 $z_j$에 의존하는 경우의 미분

- **Cross-entropy loss의 gradient를 정확히 계산하려면?**
  - One-hot 라벨 $y$에 대해 $\frac{\partial L}{\partial z_i}$의 간단한 형태

- **왜 softmax + cross-entropy 결합이 특별한가?**
  - Gradient가 기적적으로 간단해지는 수학적 이유

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

**Multi-class 분류는 딥러닝의 기본 문제입니다:**

- 이미지 분류 (ImageNet 1000 클래스)
- 텍스트 분류 (감정 분석, 주제 분류)
- 언어 모델 (GPT: 50,000 토큰 클래스)

**표준 접근:** Softmax (확률) + Cross-Entropy (손실)

**중요한 성질:**
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

이 **기적적으로 간단한 형태** 때문에:
- 수치적으로 안정적 (underflow/overflow 회피)
- 계산이 빠름 (단순한 뺄셈)
- 교과서 등에서 가장 먼저 배우는 역전파

따라서 **이 공식의 완전한 이해 = 분류 문제의 기초**입니다.

## 📐 수학적 선행 조건

- Softmax 함수의 정의: $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
- Cross-entropy loss: $L = -\sum_i y_i \log \hat{y}_i$
- 연쇄법칙과 편미분 (01번 문서)
- 행렬 미분 (01번 문서)

## 📖 직관적 이해

### Softmax의 역할

**입력**: 출력층의 로짓(logits) $z = (z_1, \ldots, z_K) \in \mathbb{R}^K$ (K = 클래스 수)

**출력**: 확률 분포 $\hat{y} = (\hat{y}_1, \ldots, \hat{y}_K)$
$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \in (0, 1), \quad \sum_i \hat{y}_i = 1$$

**의미:**
- 가장 큰 $z_i$에 해당하는 $\hat{y}_i$가 가장 큼 (하지만 정확한 확률로 표현)
- 모든 클래스에 어느 정도의 확률 부여 (soft assignment)

### Cross-Entropy Loss의 역할

**정답**: One-hot 벡터 $y = (0, \ldots, 1, \ldots, 0)$ (정답 클래스 위치만 1)

**손실**:
$$L = -\sum_{i=1}^K y_i \log \hat{y}_i = -\log \hat{y}_c$$

여기서 $c$ = 정답 클래스 인덱스

**의미:**
- 정답 클래스의 확률이 높으면 손실이 작음
- 정답 클래스의 확률이 낮으면 손실이 큼

### Gradient의 직관

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

**직관:**
- 정답 클래스 ($i = c$): $\hat{y}_c - 1 < 0$ (그 클래스를 더 높이라)
- 오답 클래스 ($i \neq c$): $\hat{y}_i > 0$ (그 클래스를 낮추라)
- Magnitude: 얼마나 잘못했는가 (0~1 사이)

---

## ✏️ 엄밀한 정의

### 1. Softmax 함수

**정의:**
$$\hat{y}_i = \text{softmax}(z)_i := \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

벡터 표기:
$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$$

### 2. Softmax의 Jacobian 계산

**핵심**: $\hat{y}_i$가 모든 $z_j$에 의존함!

$$\frac{\partial \hat{y}_i}{\partial z_j} = ?$$

분자 미분:
$$\frac{\partial}{\partial z_j} e^{z_i} = \delta_{ij} e^{z_i}$$

몫의 미분 (Quotient rule):
$$\frac{\partial}{\partial z_j} \left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right) = \frac{\delta_{ij} e^{z_i} \sum_k e^{z_k} - e^{z_i} e^{z_j}}{(\sum_k e^{z_k})^2}$$

$$= \frac{\delta_{ij} e^{z_i}}{\sum_k e^{z_k}} - \frac{e^{z_i} e^{z_j}}{(\sum_k e^{z_k})^2}$$

$$= \delta_{ij} \hat{y}_i - \hat{y}_i \hat{y}_j$$

**따라서:**
$$\boxed{\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)}$$

행렬 형태로, Jacobian $J \in \mathbb{R}^{K \times K}$:
$$J_{ij} = \hat{y}_i(\delta_{ij} - \hat{y}_j)$$

또는:
$$J = \text{diag}(\hat{\mathbf{y}}) - \hat{\mathbf{y}}\hat{\mathbf{y}}^T$$

### 3. Cross-Entropy Loss

**정의** (One-hot 라벨 가정):
$$L = -\sum_{i=1}^K y_i \log \hat{y}_i$$

One-hot: $y_c = 1$ (정답), $y_i = 0$ (i ≠ c)이므로:
$$L = -\log \hat{y}_c$$

### 4. 손실에 대한 Softmax 입력의 미분 (완전 유도)

연쇄법칙:
$$\frac{\partial L}{\partial z_j} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \cdot \frac{\partial \hat{\mathbf{y}}}{\partial z_j}$$

먼저 $\frac{\partial L}{\partial \hat{y}_i}$:
$$L = -\sum_{i=1}^K y_i \log \hat{y}_i$$

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$$

따라서:
$$\frac{\partial L}{\partial z_j} = \sum_{i=1}^K \frac{\partial L}{\partial \hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_j}$$

$$= \sum_{i=1}^K \left(-\frac{y_i}{\hat{y}_i}\right) \hat{y}_i(\delta_{ij} - \hat{y}_j)$$

$$= \sum_{i=1}^K (-y_i)(\delta_{ij} - \hat{y}_j)$$

$$= -y_j + \sum_{i=1}^K y_i \hat{y}_j$$

$$= -y_j + \hat{y}_j \sum_{i=1}^K y_i$$

One-hot 가정 ($\sum_i y_i = 1$):
$$= -y_j + \hat{y}_j$$

$$\boxed{\frac{\partial L}{\partial z_j} = \hat{y}_j - y_j}$$

**결론**: 이 기적적으로 간단한 형태가 softmax + cross-entropy의 표준 선택 이유!

---

## 🔬 정리와 증명

**정리 5.1 (Softmax Jacobian)**

Softmax 함수 $\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$의 Jacobian:

$$J = \text{diag}(\hat{\mathbf{y}}) - \hat{\mathbf{y}}\hat{\mathbf{y}}^T$$

또는 성분별로:
$$\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)$$

**증명:**
위의 quotient rule 계산 참조. $\square$

**정리 5.2 (Softmax + Cross-Entropy Gradient)**

One-hot 라벨 $y$에 대해, softmax + cross-entropy의 손실에 대한 logit의 미분:

$$\frac{\partial L}{\partial z_j} = \hat{y}_j - y_j$$

또는 벡터 형태:
$$\nabla_{\mathbf{z}} L = \hat{\mathbf{y}} - \mathbf{y}$$

**증명:**
위의 연쇄법칙 계산 참조. $\square$

**정리 5.3 (Log-Sum-Exp 수치 안정성)**

Softmax 계산 시 underflow/overflow 회피:

$$\log \sum_{j=1}^K e^{z_j} = m + \log \sum_{j=1}^K e^{z_j - m}$$

여기서 $m = \max_j z_j$.

**증명:**
$$\log \sum_{j=1}^K e^{z_j} = \log \left(e^m \sum_{j=1}^K e^{z_j - m}\right) = m + \log \sum_{j=1}^K e^{z_j - m}$$

**효과**: $z_j - m \leq 0$ → exponential이 underflow하지만, 합으로는 수렴. $\square$

---

## 💻 NumPy로 바닥부터 구현

### Stable Softmax + Cross-Entropy

```python
import numpy as np

def softmax_stable(z):
    """Numerically stable softmax"""
    z = np.asarray(z)
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cross_entropy_loss(logits, labels):
    """
    Cross-entropy loss (one-hot labels assumed)
    logits: [batch_size, num_classes] or [num_classes]
    labels: [batch_size, num_classes] or [num_classes] (one-hot)
    """
    # Handle both single sample and batch
    logits = np.atleast_2d(logits)
    labels = np.atleast_2d(labels)
    
    # Stable softmax
    probs = softmax_stable(logits)
    
    # Cross-entropy: -sum(y * log(p))
    # For numerical stability, use log-softmax
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    
    loss = -np.sum(labels * log_probs, axis=-1)
    
    return np.mean(loss), probs

def cross_entropy_gradient(probs, labels):
    """
    Gradient of cross-entropy w.r.t. logits: p - y
    probs: softmax outputs [batch_size, num_classes]
    labels: one-hot labels [batch_size, num_classes]
    """
    return probs - labels

# 테스트 1: 간단한 예제
print("=== Test 1: Simple Example ===")
z = np.array([1.0, 2.0, 0.5])
y = np.array([0, 1, 0])  # 정답: 클래스 1

probs = softmax_stable(z)
print(f"Logits: {z}")
print(f"Softmax: {probs}")
print(f"Sum (should be 1.0): {np.sum(probs)}")

loss, _ = cross_entropy_loss(z, y)
print(f"Loss: {loss:.6f}")

grad = cross_entropy_gradient(probs, y)
print(f"Gradient (∂L/∂z): {grad}")
print(f"Should be [ŷ₀ - y₀, ŷ₁ - y₁, ŷ₂ - y₂]: {probs - y}")

# 테스트 2: Softmax Jacobian 검증
print("\n=== Test 2: Softmax Jacobian ===")

def softmax_jacobian_numerical(z, eps=1e-5):
    """수치 Jacobian"""
    J = np.zeros((len(z), len(z)))
    for j in range(len(z)):
        z_plus = z.copy()
        z_plus[j] += eps
        z_minus = z.copy()
        z_minus[j] -= eps
        
        J[:, j] = (softmax_stable(z_plus) - softmax_stable(z_minus)) / (2 * eps)
    return J

def softmax_jacobian_analytical(probs):
    """해석적 Jacobian: diag(ŷ) - ŷŷᵀ"""
    return np.diag(probs) - np.outer(probs, probs)

z = np.array([1.0, 2.0, 0.5])
probs = softmax_stable(z)

J_num = softmax_jacobian_numerical(z)
J_ana = softmax_jacobian_analytical(probs)

print(f"Numerical Jacobian:\n{J_num}")
print(f"Analytical Jacobian:\n{J_ana}")
print(f"Max error: {np.max(np.abs(J_num - J_ana)):.2e}")

# 테스트 3: Batched 계산
print("\n=== Test 3: Batched Cross-Entropy ===")

batch_logits = np.array([
    [1.0, 2.0, 0.5],
    [2.0, 1.0, 3.0],
    [0.5, 0.5, 0.5]
])
batch_labels = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

loss, probs = cross_entropy_loss(batch_logits, batch_labels)
print(f"Batch loss shape: {loss.shape if loss.ndim > 0 else 'scalar'}")
print(f"Batch loss (mean): {loss:.6f}")

# 각 샘플별 손실
for i in range(batch_logits.shape[0]):
    loss_i, _ = cross_entropy_loss(batch_logits[i:i+1], batch_labels[i:i+1])
    print(f"  Sample {i}: {loss_i[0]:.6f}")

# 테스트 4: Gradient checking
print("\n=== Test 4: Gradient Checking ===")

def numerical_gradient_logits(logits, labels, idx, eps=1e-5):
    """Logit에 대한 손실의 수치 gradient"""
    logits_plus = logits.copy()
    logits_plus[idx] += eps
    loss_plus, _ = cross_entropy_loss(logits_plus, labels)
    
    logits_minus = logits.copy()
    logits_minus[idx] -= eps
    loss_minus, _ = cross_entropy_loss(logits_minus, labels)
    
    return (loss_plus - loss_minus)[0] / (2 * eps)

z = np.array([[1.0, 2.0, 0.5]])
y = np.array([[0, 1, 0]])

loss_val, probs = cross_entropy_loss(z, y)
grad_ana = cross_entropy_gradient(probs, y)

grad_num = np.array([numerical_gradient_logits(z, y, i) for i in range(z.shape[1])])

print(f"Analytical gradient: {grad_ana[0]}")
print(f"Numerical gradient: {grad_num}")
print(f"Max error: {np.max(np.abs(grad_ana[0] - grad_num)):.2e}")

# 테스트 5: 분류 학습 루프
print("\n=== Test 5: Simple Classification ===")

np.random.seed(42)
num_classes = 3
num_samples = 100

# 합성 데이터
X_train = np.random.randn(num_samples, 2)
y_train = np.random.randint(0, num_classes, num_samples)
y_train_onehot = np.eye(num_classes)[y_train]

# 모델: 선형 분류기
W = np.random.randn(num_classes, 2) * 0.01
b = np.zeros(num_classes)

learning_rate = 0.1

print(f"Initial loss: {cross_entropy_loss(X_train @ W.T + b, y_train_onehot)[0]:.6f}")

# SGD 학습
for epoch in range(50):
    # Mini-batch
    idx = np.random.choice(num_samples, 32, replace=False)
    X_batch = X_train[idx]
    y_batch = y_train_onehot[idx]
    
    # Forward
    logits = X_batch @ W.T + b
    loss, probs = cross_entropy_loss(logits, y_batch)
    
    # Backward
    grad_logits = cross_entropy_gradient(probs, y_batch)
    dW = grad_logits.T @ X_batch / len(idx_batch)
    db = np.mean(grad_logits, axis=0)
    
    # Update
    W -= learning_rate * dW
    b -= learning_rate * db
    
    if epoch % 10 == 0:
        loss_full, _ = cross_entropy_loss(X_train @ W.T + b, y_train_onehot)
        print(f"Epoch {epoch}: Loss = {loss_full:.6f}")

# 정확도
logits_test = X_train @ W.T + b
probs_test = softmax_stable(logits_test)
preds = np.argmax(probs_test, axis=1)
accuracy = np.mean(preds == y_train)
print(f"Final accuracy: {accuracy:.4f}")
```

### 다중 클래스 분류 모델

```python
class SoftmaxRegression:
    """Softmax regression with cross-entropy loss"""
    
    def __init__(self, input_dim, num_classes):
        self.W = np.random.randn(num_classes, input_dim) * 0.01
        self.b = np.zeros(num_classes)
        self.num_classes = num_classes
    
    def forward(self, X):
        """X: [batch_size, input_dim]"""
        logits = X @ self.W.T + self.b
        probs = softmax_stable(logits)
        return logits, probs
---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. MLP 역전파 공식](./04-mlp-backprop-formula.md) | [📚 README로 돌아가기](../README.md) | [06. Batched 행렬 미분 ▶](./06-batched-matrix-backprop.md) |

</div>
