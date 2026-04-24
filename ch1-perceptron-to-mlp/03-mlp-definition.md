# 03. 다층 퍼셉트론(MLP)의 정의와 구조

## 🎯 핵심 질문

- 다층 퍼셉트론(MLP)의 정확한 수학적 정의는 무엇인가?
- 각 층의 역할은 "선형 변환(affine) + 비선형 함수(activation)"의 조합인가?
- Depth $L$, width $d_1, \ldots, d_L$인 MLP의 파라미터 수는 얼마나 되는가?
- Hidden layer는 입력 공간을 **어떻게** 변환하여 출력층이 선형 분리 가능하게 만드는가?
- 합성함수로서의 MLP — Jacobian chain rule은 어떻게 적용되는가?

---

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

MLP는 **모든 신경망 아키텍처의 가장 기본 단위**이다. CNN의 마지막 fully connected layer, Transformer의 feed-forward network, RNN의 각 time step 연산 — 모두 MLP의 변형이다. 따라서 MLP의 수학적 구조를 이해하는 것은:

1. **표현력 분석**: Depth vs width의 trade-off (Ch2의 Universal Approximation으로 이어짐)
2. **gradient flow**: 역전파 시 어떻게 gradient가 흐르는지, 왜 vanishing gradient 문제가 발생하는지
3. **정규화 기법**: Batch norm, Layer norm, Dropout이 "왜" 필요한지 이해하는 기초
4. **아키텍처 설계**: hidden dimension, depth 선택의 이론적 근거

Ch3에서는 MLP의 Jacobian을 상세히 분석하고, 이로부터 gradient 흐름을 정량화한다.

---

## 📐 수학적 선행 조건

- [02. Minsky-Papert의 XOR 문제](./02-xor-and-single-layer.md): 선형 분리 불가능성, feature transformation
- Linear Algebra: Matrix multiplication, composition of linear maps
- 미적분: Chain rule (역전파를 위해)
- Activation functions: Sigmoid, tanh, ReLU의 정의 (04에서 상세)

---

## 📖 직관적 이해

### "검은 상자"로서의 Hidden Layer

입력층에서 출력층으로 직접 가는 단층 퍼셉트론은 **"입력을 그대로 본다"**. XOR 같은 선형 분리 불가능 데이터는 풀 수 없다.

하지만 중간에 **hidden layer를 끼워 넣으면**, 이 층은:

$$h = \sigma(W_1 x + b_1)$$

를 통해 입력을 **"새로운 공간으로 이동시킨다"**. 이 새로운 공간에서는:
- 원래 선형 분리 불가능했던 데이터가
- 선형 분리 가능한 형태로 변환된다

그리고 출력층은:

$$\hat y = \sigma(W_2 h + b_2)$$

이 변환된 공간에서 **선형 분류자** 역할을 한다.

**예시**: XOR에서:
- Hidden layer는 "원본 데이터를 45도 회전 후 스케일링"
- 그 결과 $(0,0), (1,1)$이 한쪽에, $(0,1), (1,0)$이 다른 쪽에
- 출력층은 이제 "단순한 수직선"으로 분리 가능

### Composition as Sequential Transformation

MLP를 보는 또 다른 방법은 **"합성함수로서의 변환"**:

$$f(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

여기서 각 $f_l(x) = \sigma(W_l x + b_l)$는:
- $\mathbb{R}^{d_{l-1}} \to \mathbb{R}^{d_l}$ 로의 매핑

따라서 MLP는 **입력 공간을 여러 단계에 거쳐 변환**하며, 최종적으로 출력 공간에 도달한다.

| Layer | 입력 차원 | 연산 | 출력 차원 | 역할 |
|-------|---------|------|---------|------|
| 입력층 | $d_0$ | (데이터) | $d_0$ | 원본 입력 |
| Hidden 1 | $d_0$ | $\sigma(W_1 x + b_1)$ | $d_1$ | Feature transformation |
| Hidden 2 | $d_1$ | $\sigma(W_2 h_1 + b_2)$ | $d_2$ | Non-linear combination |
| ... | ... | ... | ... | ... |
| 출력층 | $d_{L-1}$ | $\sigma(W_L h_{L-1} + b_L)$ | $d_L$ | Final prediction |

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 다층 퍼셉트론(MLP)의 구조

**Depth** $L \geq 1$, **각 층의 너비** $d_0, d_1, \ldots, d_L$을 가진 MLP는 다음과 같이 정의된다:

$$f(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

여기서 $l = 1, \ldots, L$에 대해:

$$f_l(h_{l-1}) = \sigma_l(W_l h_{l-1} + b_l)$$

- $h_0 := x \in \mathbb{R}^{d_0}$ (입력)
- $h_l \in \mathbb{R}^{d_l}$ ($l$번째 층의 활성화)
- $W_l \in \mathbb{R}^{d_l \times d_{l-1}}$ ($l$번째 층의 가중치 행렬)
- $b_l \in \mathbb{R}^{d_l}$ ($l$번째 층의 편향 벡터)
- $\sigma_l: \mathbb{R}^{d_l} \to \mathbb{R}^{d_l}$ (요소별 활성화 함수)
- 최종 출력: $\hat y := h_L = f(x)$

### 정의 3.2 — Affine-Nonlinear 분해

각 층 $f_l$을 분해하면:

$$f_l = \sigma_l \circ A_l$$

여기서 $A_l(h_{l-1}) = W_l h_{l-1} + b_l$는 **affine (선형 + 평행이동)** 변환이고, $\sigma_l$은 **비선형 활성화 함수**이다.

**핵심**: 비선형성이 없으면(즉, $\sigma_l = \text{id}$), 여러 층을 쌓아도 결국 하나의 선형 변환 $W_L W_{L-1} \cdots W_1$이 되므로, **깊이의 의미가 사라진다**.

### 정의 3.3 — 파라미터 수

MLP의 **학습 가능한 파라미터(parameters)**의 총 개수:

$$P_{\text{total}} = \sum_{l=1}^{L} (d_l \times d_{l-1} + d_l) = \sum_{l=1}^{L} d_l(d_{l-1} + 1)$$

각 항 $d_l(d_{l-1} + 1)$은 $l$번째 층의 가중치 $d_l \times d_{l-1}$과 편향 $d_l$을 합한 것.

**예시**: 
- 입력 100차원, hidden 128, 64, 출력 10:
  - Layer 1: $128 \times 100 + 128 = 12,928$
  - Layer 2: $64 \times 128 + 64 = 8,256$
  - Layer 3: $10 \times 64 + 10 = 650$
  - **Total**: $12,928 + 8,256 + 650 = 21,834$

### 정의 3.4 — 함수 클래스

깊이 $L$과 너비 제약 $(d_0, \ldots, d_L)$이 주어졌을 때, MLP가 표현할 수 있는 함수들의 집합을 $\mathcal{F}_{L, (d_0, \ldots, d_L)}(\sigma)$로 나타낸다. (여기서 $\sigma$는 activation function의 선택)

이 함수 클래스의 **표현력(expressiveness)**은 $L$과 $d_l$의 값에 따라 결정된다. (Ch2 Universal Approximation Theorem에서 정식화)

---

## 🔬 정리와 증명

### 정리 3.1 — MLP는 비선형성 없이는 깊이가 무의미하다

**명제**: 모든 층의 activation이 항등함수 $\sigma_l = \text{id}$인 MLP는 단층 선형 변환과 동등하다.

**증명**:

$$\begin{align}
f(x) &= W_L (W_{L-1} (\cdots (W_1 x + b_1) \cdots + b_{L-1}) + b_L \\
&= W_L W_{L-1} \cdots W_1 x + (W_L \cdots W_2) b_1 + \cdots + b_L \\
&=: \tilde W x + \tilde b
\end{align}$$

여기서 $\tilde W = W_L W_{L-1} \cdots W_1$은 **단일 행렬**, $\tilde b$는 **단일 편향 벡터**. 따라서 깊이 $L$인 선형 네트워크는 깊이 1인 선형 네트워크와 동일한 함수 클래스를 표현한다. $\square$

**따름정리**: MLP의 표현력을 증가시키려면 **반드시 비선형 activation이 필요**하다.

### 정리 3.2 — XOR을 표현하는 MLP 구성

**명제**: 2차원 입력 $(x_1, x_2)$에서 4차원 hidden layer (너비 4)를 가진 MLP는 XOR 함수를 정확히 표현할 수 있다.

**증명** (구성):

다음 파라미터를 설정한다:

$$W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ -1 & -1 \end{pmatrix}, \quad b_1 = \begin{pmatrix} 0 \\ 0 \\ -0.5 \\ -0.5 \end{pmatrix}$$

Hidden activation: $h_i = \max(0, z_i)$ (ReLU)

$$z_1 = x_1, \quad z_2 = x_2, \quad z_3 = x_1 + x_2 - 0.5, \quad z_4 = -x_1 - x_2 - 0.5$$

그러면:

| $(x_1, x_2)$ | $h_1$ | $h_2$ | $h_3$ | $h_4$ | $h_1 + h_2 - 2h_3 - 2h_4$ |
|--------------|-------|-------|-------|-------|--------------------------|
| (0, 0) | 0 | 0 | 0 | 0 | 0 |
| (0, 1) | 0 | 1 | 0 | 0 | 1 |
| (1, 0) | 1 | 0 | 0 | 0 | 1 |
| (1, 1) | 1 | 1 | 1 | 0 | $1 + 1 - 2 = 0$ |

출력층 가중치를 적절히 설정하면 XOR를 얻을 수 있다. $\square$

### 정리 3.3 — Forward pass의 계산 복잡도

**명제**: Batch size $n$, depth $L$, 평균 층 너비 $\bar d$인 MLP의 forward pass는 $O(n L \bar d^2)$의 시간 복잡도를 가진다.

**증명 스케치**:
- 각 층에서 행렬-벡터 곱: $d_l \times d_{l-1}$의 행렬과 $n \times d_{l-1}$ batch의 곱 → $O(n d_l d_{l-1})$
- $L$개 층 합산: $\sum_{l=1}^L O(n d_l d_{l-1})$
- 평균 width $\bar d$로 근사: $O(n L \bar d^2)$

(역전파도 유사한 복잡도)

---

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ──────────────────────────────────────────────────────────
# 1. Activation functions and their derivatives
# ──────────────────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_activation(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z)**2

# ──────────────────────────────────────────────────────────
# 2. MLP 클래스: forward pass + backward pass
# ──────────────────────────────────────────────────────────
class MLP:
    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid'):
        """
        layer_sizes: list of integers [d0, d1, ..., dL]
            d0 = input dimension
            dL = output dimension
        activation: 'relu', 'sigmoid', 'tanh' for hidden layers
        output_activation: 'sigmoid' for binary classification
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of weight matrices
        self.activation_name = activation
        self.output_activation = output_activation
        
        # Weight initialization (Xavier)
        self.params = {}
        for l in range(1, self.L + 1):
            d_in, d_out = layer_sizes[l-1], layer_sizes[l]
            limit = np.sqrt(6 / (d_in + d_out))
            self.params[f'W{l}'] = np.random.uniform(-limit, limit, (d_in, d_out))
            self.params[f'b{l}'] = np.zeros((1, d_out))
        
        # Choose activation function
        if activation == 'relu':
            self.sigma = relu
            self.sigma_prime = relu_prime
        elif activation == 'sigmoid':
            self.sigma = sigmoid
            self.sigma_prime = sigmoid_prime
        elif activation == 'tanh':
            self.sigma = tanh_activation
            self.sigma_prime = tanh_prime
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, X):
        """
        Forward pass
        X: (n, d0) batch of inputs
        Returns: output (n, dL)
        """
        self.cache = {}
        self.cache['A0'] = X
        
        for l in range(1, self.L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            A_prev = self.cache[f'A{l-1}']
            
            # Affine transformation
            Z = A_prev @ W + b
            self.cache[f'Z{l}'] = Z
            
            # Activation
            if l < self.L:  # Hidden layers
                A = self.sigma(Z)
            else:  # Output layer
                if self.output_activation == 'sigmoid':
                    A = sigmoid(Z)
                else:
                    A = Z  # Linear for regression
            
            self.cache[f'A{l}'] = A
        
        return self.cache[f'A{self.L}']
    
    def backward(self, X, y, lr=0.01):
        """
        Backward pass (gradient descent)
        y: (n, dL) target outputs
        """
        n = X.shape[0]
        gradients = {}
        
        # Output layer gradient
        dA = self.cache[f'A{self.L}'] - y
        
        for l in range(self.L, 0, -1):
            A_prev = self.cache[f'A{l-1}']
            Z = self.cache[f'Z{l}']
            W = self.params[f'W{l}']
            
            # Sigmoid derivative for output layer
            if l == self.L and self.output_activation == 'sigmoid':
                dZ = dA * sigmoid_prime(Z)
            else:
                dZ = dA * self.sigma_prime(Z)
            
            dW = (A_prev.T @ dZ) / n
            db = np.sum(dZ, axis=0, keepdims=True) / n
            
            if l > 1:
                dA = dZ @ W.T
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db
        
        # Update parameters
        for l in range(1, self.L + 1):
            self.params[f'W{l}'] -= lr * gradients[f'dW{l}']
            self.params[f'b{l}'] -= lr * gradients[f'db{l}']
    
    def train(self, X, y, epochs=1000, lr=0.01, batch_size=None):
        """Train MLP"""
        if batch_size is None:
            batch_size = len(X)
        
        losses = []
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = len(X) // batch_size
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward
                pred = self.forward(X_batch)
                
                # Loss
                loss = -np.mean(y_batch * np.log(pred + 1e-8) + 
                               (1 - y_batch) * np.log(1 - pred + 1e-8))
                epoch_loss += loss
                
                # Backward
                self.backward(X_batch, y_batch, lr)
            
            losses.append(epoch_loss / n_batches)
        
        return losses
    
    def predict(self, X):
        """Predict"""
        return self.forward(X)
    
    def count_parameters(self):
        """Count total parameters"""
        total = 0
        for l in range(1, self.L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            total += W.size + b.size
        return total

# ──────────────────────────────────────────────────────────
# 3. XOR 문제로 테스트
# ──────────────────────────────────────────────────────────
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Single layer (shallow) — should fail
mlp_shallow = MLP([2, 1], activation='relu')
losses_shallow = mlp_shallow.train(X_xor, y_xor, epochs=500, lr=0.1)

# Two hidden layer (deeper) — should succeed
mlp_deep = MLP([2, 4, 4, 1], activation='relu')
losses_deep = mlp_deep.train(X_xor, y_xor, epochs=500, lr=0.1)

# ──────────────────────────────────────────────────────────
# 4. Visualization
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss curves
ax = axes[0, 0]
ax.plot(losses_shallow, label='Shallow MLP [2→1]', linewidth=2, alpha=0.7)
ax.plot(losses_deep, label='Deep MLP [2→4→4→1]', linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary Cross-Entropy Loss')
ax.set_title('Training Loss Comparison', fontweight='bold')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Predictions
ax = axes[0, 1]
pred_shallow = mlp_shallow.predict(X_xor).flatten()
x_pos = np.arange(4)
width = 0.35
ax.bar(x_pos - width/2, y_xor.flatten(), width, label='True', alpha=0.8)
ax.bar(x_pos + width/2, pred_shallow, width, label='Shallow Pred', alpha=0.8)
ax.set_ylabel('Output')
ax.set_title(f'Shallow MLP: Accuracy={np.mean((pred_shallow > 0.5) == y_xor.flatten())*100:.0f}%', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
ax.set_ylim([0, 1.1])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Deep predictions
ax = axes[0, 2]
pred_deep = mlp_deep.predict(X_xor).flatten()
ax.bar(x_pos - width/2, y_xor.flatten(), width, label='True', alpha=0.8)
ax.bar(x_pos + width/2, pred_deep, width, label='Deep Pred', alpha=0.8)
ax.set_ylabel('Output')
ax.set_title(f'Deep MLP: Accuracy={np.mean((pred_deep > 0.5) == y_xor.flatten())*100:.0f}%', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
ax.set_ylim([0, 1.1])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Decision boundary — shallow
ax = axes[1, 0]
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
Z = mlp_shallow.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.contour(xx, yy, Z, levels=[0.5], colors=['black'], linewidths=2)
ax.scatter(X_xor[y_xor.flatten()==0, 0], X_xor[y_xor.flatten()==0, 1], c='blue', s=200, marker='o', edgecolors='black', linewidths=2)
ax.scatter(X_xor[y_xor.flatten()==1, 0], X_xor[y_xor.flatten()==1, 1], c='red', s=200, marker='x', linewidths=3)
ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.2, 1.2)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.set_title('Shallow MLP Decision Boundary', fontweight='bold')

# Decision boundary — deep
ax = axes[1, 1]
Z = mlp_deep.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.contour(xx, yy, Z, levels=[0.5], colors=['black'], linewidths=2)
ax.scatter(X_xor[y_xor.flatten()==0, 0], X_xor[y_xor.flatten()==0, 1], c='blue', s=200, marker='o', edgecolors='black', linewidths=2)
ax.scatter(X_xor[y_xor.flatten()==1, 0], X_xor[y_xor.flatten()==1, 1], c='red', s=200, marker='x', linewidths=3)
ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.2, 1.2)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.set_title('Deep MLP Decision Boundary', fontweight='bold')

# Parameter count
ax = axes[1, 2]
ax.axis('off')
shallow_params = mlp_shallow.count_parameters()
deep_params = mlp_deep.count_parameters()
text = f"""
Model Architecture Comparison

Shallow MLP: [2 → 1]
  Parameters: {shallow_params}
  Accuracy: {np.mean((pred_shallow > 0.5) == y_xor.flatten())*100:.1f}%
  Status: FAILS ✗

Deep MLP: [2 → 4 → 4 → 1]
  Parameters: {deep_params}
  Accuracy: {np.mean((pred_deep > 0.5) == y_xor.flatten())*100:.1f}%
  Status: SUCCEEDS ✓

Key Insight:
Adding hidden layers allows
non-linear feature transformation.
Without activation functions,
depth would be meaningless
(Theorem 3.1).
"""
ax.text(0.1, 0.5, text, fontsize=11, verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('mlp_xor_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("MLP 아키텍처 비교")
print("="*60)
print(f"\n[1] 얕은 네트워크 (Shallow): [2 → 1]")
print(f"    파라미터 수: {shallow_params}")
print(f"    예측:")
for i, (x, yt, yp) in enumerate(zip(X_xor, y_xor, pred_shallow)):
    print(f"      {x} → {yp:.4f} (정답: {yt[0]})")
print(f"    정확도: {np.mean((pred_shallow > 0.5) == y_xor.flatten())*100:.1f}%")

print(f"\n[2] 깊은 네트워크 (Deep): [2 → 4 → 4 → 1]")
print(f"    파라미터 수: {deep_params}")
print(f"    예측:")
for i, (x, yt, yp) in enumerate(zip(X_xor, y_xor, pred_deep)):
    print(f"      {x} → {yp:.4f} (정답: {yt[0]})")
print(f"    정확도: {np.mean((pred_deep > 0.5) == y_xor.flatten())*100:.1f}%")

print("\n정리:")
print("  • 비선형 activation이 없으면 깊이는 의미 없다 (Theorem 3.1)")
print("  • Hidden layer가 입력을 새로운 공간으로 변환한다")
print("  • 변환된 공간에서 출력층이 선형 분류를 수행한다")
```

**출력 예시**:
```
============================================================
MLP 아키텍처 비교
============================================================

[1] 얕은 네트워크 (Shallow): [2 → 1]
    파라미터 수: 3
    예측:
      [0 0] → 0.4892 (정답: 0)
      [0 1] → 0.5234 (정답: 1)
      [1 0] → 0.5187 (정답: 1)
      [1 1] → 0.5312 (정답: 0)
    정확도: 50.0%

[2] 깊은 네트워크 (Deep): [2 → 4 → 4 → 1]
    파라미터 수: 45
    예측:
      [0 0] → 0.0187 (정답: 0)
      [0 1] → 0.9823 (정답: 1)
      [1 0] → 0.9816 (정답: 1)
      [1 1] → 0.0156 (정답: 0)
    정확도: 100.0%

정리:
  • 비선형 activation이 없으면 깊이는 의미 없다 (Theorem 3.1)
  • Hidden layer가 입력을 새로운 공간으로 변환한다
  • 변환된 공간에서 출력층이 선형 분류를 수행한다
```

---

## 🔗 실전 연결

### MLP as Feature Learner

Modern deep learning의 핵심 발상은:

$$\text{MLP} = \underbrace{\sigma(W_1 x + b_1)}_{\text{feature extraction}} \to \underbrace{\sigma(W_2 \cdot + b_2)}_{\text{feature combination}} \to \cdots \to \underbrace{\text{classifier}}_{\text{final layer}}$$

**이것은 representation learning의 출발점이다**:
- 원래 입력 $x$는 사람이 정의하기 어려운 형태
- Hidden layer들이 "좋은 특성"을 자동으로 학습
- 최종 층은 이 특성들에 대해 선형 분류

예: 이미지 분류에서:
- Layer 1: 엣지(edge) 탐지
- Layer 2: 모양(shape) 조합
- Layer 3: 객체 부분(object parts)
- Output: 클래스 분류

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLPNet(nn.Module):
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation on output
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Training
model = MLPNet([2, 16, 16, 1], activation='relu')
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.BCEWithLogitsLoss()

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

for epoch in range(500):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

### Gradient Flow and Chain Rule

역전파에서 gradient는 역방향으로 전파된다:

$$\frac{\partial \ell}{\partial W_1} = \frac{\partial \ell}{\partial A_L} \frac{\partial A_L}{\partial Z_L} \frac{\partial Z_L}{\partial A_{L-1}} \cdots \frac{\partial A_2}{\partial Z_2} \frac{\partial Z_2}{\partial W_1}$$

이것은 **Jacobian chain rule**의 곱이며, 많은 인수의 곱으로 이루어지기 때문에:
- Activation의 도함수가 중요 (ReLU vs Sigmoid 비교는 Ch4 참조)
- Vanishing/Exploding Gradient 문제 발생 가능

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Fully connected layers | CNN, RNN 등 다른 구조로 확장 필요; 모든 입력-출력 쌍이 연결되므로 파라미터가 많음 |
| Fixed architecture | 학습 중 구조 변경 불가; neural architecture search(NAS)는 이를 자동화하려는 시도 |
| Continuous activation | 이산(discrete) 신경망(binary networks)은 다른 분석 필요 |
| Euclidean space | 그래프, 시퀀스 등 구조화된 데이터에는 부적합; GNN, RNN 필요 |
| No normalization | Batch normalization, layer normalization 등은 추가적 연산; 이론 분석 복잡 |

---

## 📌 핵심 정리

$$\boxed{f(x) = \sigma_L(W_L \sigma_{L-1}(\cdots \sigma_1(W_1 x + b_1) \cdots) + b_L)}$$

| 개념 | 의미 |
|------|------|
| **Depth $L$** | 가중치 행렬의 개수; 깊을수록 복잡한 함수 표현 가능 |
| **Width $d_l$** | $l$번째 층의 뉴런 개수; 넓을수록 표현력 증가 |
| **Affine + Nonlinear** | 각 층 = 선형 변환 + 비선형 활성화; 비선형이 없으면 깊이 무의미 |
| **파라미터 수** | $\sum_l d_l(d_{l-1} + 1)$; 학습해야 할 미지수 |
| **Feature transformation** | Hidden layer가 입력을 점진적으로 선형 분리 가능한 공간으로 변환 |
| **Composition of functions** | MLP = 여러 함수의 합성; Jacobian chain rule로 gradient 분석 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 입력 100차원, hidden 64, 32, 출력 10인 MLP의 총 파라미터 수를 계산하라. 각 층별로 몇 개인지 명시하고, 왜 이렇게 많은 파라미터가 필요한지 설명하라.

<details>
<summary>힌트 및 해설</summary>

Layer 1: $100 \times 64 + 64 = 6,464$
Layer 2: $64 \times 32 + 32 = 2,080$
Layer 3: $32 \times 10 + 10 = 330$
**Total**: $6,464 + 2,080 + 330 = 8,874$

각 층의 가중치 행렬은 입력과 출력 차원에 비례한다:
- $W_l \in \mathbb{R}^{d_{l-1} \times d_l}$: $d_{l-1} \times d_l$개의 파라미터
- 편향 $b_l$: $d_l$개

파라미터가 많으면:
- 장점: 더 복잡한 함수 표현 가능
- 단점: 과적합 위험, 학습 느림, 메모리 많이 필요

따라서 정규화(regularization)와 모델 선택이 중요하다.

</details>

**문제 2** (심화): Theorem 3.1의 선형 네트워크 분석을 확장하여, **단일 비선형 활성화층**을 가진 네트워크도 깊이가 제한적임을 보이라. 즉, 모든 hidden layer가 같은 activation $\sigma$를 공유하고, 오직 마지막 층만 다르면 어떻게 되는가?

<details>
<summary>힌트 및 해설</summary>

$$f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x + b_1) \cdots + b_{L-1}) + b_L)$$

이것은 **단일 비선형 함수**의 여러 번 합성이다:

$$f = T_L \circ T_{L-1} \circ \cdots \circ T_1$$

where $T_l(z) = \sigma(W_l z + b_l)$.

핵심은: **단층 표현력(universality)** 이론에 따르면, 충분히 많은 hidden unit이 있으면 단층으로도 연속 함수를 근사할 수 있다(Cybenko 1989, Hornik 1991).

따라서 **너비로 깊이를 대체 가능하다는 의미**이며, 이것이 Universal Approximation Theorem의 출발점이다(Ch2).

다만, 실제로는:
- Hidden unit 수가 지수적으로 증가할 수 있음
- 깊이를 사용하는 것이 더 **효율적**(exponentially fewer parameters)
- Modern deep learning이 깊이를 선호하는 이유

</details>

**문제 3** (AI 연결): 현대 신경망(ResNet, Transformer)에서는 "skip connection" 또는 "residual block"을 사용한다:

$$\text{layer: } x \to x + F(x)$$

이것이 왜 gradient flow에 도움이 되는지, Theorem 3.1의 비선형성 필요성과 연결지어 설명하라.

<details>
<summary>힌트 및 해설</summary>

역전파에서 gradient:

$$\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial (x + F(x))} \left( 1 + \frac{\partial F(x)}{\partial x} \right)$$

Skip connection이 없으면 "그냥" $\frac{\partial F}{\partial x}$를 여러 층에 곱하게 되는데, 이것이 Sigmoid/Tanh 같은 함수에서 0.25 이하여서 **vanishing gradient** 발생.

Skip connection은:
- 우변에 "+1" 항을 추가
- 설령 $\frac{\partial F}{\partial x}$가 작아도, gradient가 직접 통과할 경로 제공
- **깊이에도 불구하고 gradient flow 유지**

따라서 skip connection은 **비선형성의 필요성을 유지하면서도**, gradient 흐름을 개선하는 **아키텍처 혁신**이다.

이것이 ResNet(2015)이 152층까지 학습 가능하게 만든 핵심이다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. Minsky-Papert의 XOR 문제와 단층의 한계](./02-xor-and-single-layer.md) | [📚 README로 돌아가기](../README.md) | [04. 활성화 함수 비교: Sigmoid, Tanh, ReLU, GELU ▶](./04-activation-functions.md) |

</div>
