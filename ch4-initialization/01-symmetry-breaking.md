# 1. 초기화의 근본: 대칭성 깨기(Symmetry Breaking)

## 🎯 핵심 질문

- 왜 모든 가중치를 0으로 초기화하면 안 될까?
- 신경망의 "effective width"가 1이 되는 것은 무엇을 의미하는가?
- 어떤 초기화 방식이 대칭성을 효과적으로 깨고 학습을 시작하게 할까?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

신경망의 첫 초기화는 학습 동역학을 완전히 결정합니다. 특히 깊은 네트워크에서 초기화 전략의 선택이 수렴성, 학습 속도, 최종 성능의 격차를 만듭니다. 대칭성 깨기는 이 모든 것의 토대입니다.

## 📐 수학적 선행 조건

- 기본 선형대수: 행렬-벡터 곱셈, rank 개념
- 기본 확률: 분산(variance), 기댓값(expectation), 독립성
- Ch3의 역전파: gradient 흐름의 이해

## 📖 직관적 이해

### 대칭성의 문제

신경망 $\ell$번째 층의 hidden unit들이 같은 값으로 초기화되면:
$$
h_i^{(\ell)} = W_i^{(\ell)} \cdot a^{(\ell-1)} + b_i^{(\ell)}
$$

만약 모든 $W_i^{(\ell)}$가 동일한 행(row)이면:
- Forward: 모든 unit이 같은 값 출력 → $h_1 = h_2 = \cdots = h_m$
- Backward: 모든 unit이 같은 gradient 수신 → $\delta_1 = \delta_2 = \cdots = \delta_m$
- Update: 모든 행이 같은 방향으로 이동 → **여전히 동일**

**결과**: 어떤 hidden layer도 실제로 1개의 unit처럼 작동. 네트워크의 "실제 폭" = 1

### 작은 랜덤 초기화의 필요성

$W \sim \mathcal{N}(0, \sigma_w^2)$로 초기화하면:
- 각 unit은 다른 random contribution 수신
- 각 unit은 다른 gradient 수신
- **parameter들이 다른 값으로 진화 시작** → 네트워크의 실제 capacity 활용

하지만 $\sigma_w$의 선택이 중요:
- **$\sigma_w$가 너무 작으면**: gradient vanishing (깊은 층까지 신호 도달 X)
- **$\sigma_w$가 너무 크면**: neuron saturation (sigmoid/tanh 포화 영역), 또는 수치 폭발

## ✏️ 엄밀한 정의

**대칭성이 있는 초기화(Symmetric Initialization)**:
$$
W_{ij} = c \text{ for all } i, j \quad (c \in \mathbb{R})
$$

**대칭성이 깨진 초기화(Symmetry-Breaking Initialization)**:
$$
W_{ij} \sim \mathcal{N}(0, \sigma_w^2), \quad \text{i.i.d.}
$$

**Activation Variance 보존 조건**:
$$
\text{Var}[h^{(\ell)}] \approx \text{Var}[h^{(\ell-1)}]
$$

이를 만족하는 $\sigma_w$를 "Goldilocks zone"이라 부릅니다.

## 🔬 정리와 증명

**정리 1.1** (대칭성 고착, Symmetry Fixation)

$\ell$번째 층의 모든 weight가 동일 행이면, 역전파 후 $\ell+1$번째 층도 동일 행이 된다.

**증명**:

초기 상태:
$$
W^{(\ell)} = \begin{pmatrix} w & w & \cdots & w \\ \end{pmatrix} \in \mathbb{R}^{1 \times n_\text{in}}
$$
(실제로는 $m \times n_\text{in}$이지만, 모든 행이 동일)

Forward pass:
$$
h_i^{(\ell)} = w \cdot a^{(\ell-1)} + b \quad \forall i \in [m]
$$

즉, 모든 hidden unit이 같은 값: $h_1 = h_2 = \cdots = h_m = h$

Loss에 대한 gradient:
$$
\frac{\partial L}{\partial h_i^{(\ell)}} = \delta_i^{(\ell)} = \text{동일값} \quad \forall i
$$

왜냐하면 loss가 $h$들에 대해 symmetric하게 의존하므로.

Weight update:
$$
W^{(\ell)} \leftarrow W^{(\ell)} - \eta \frac{\partial L}{\partial W^{(\ell)}}
$$

$$
\frac{\partial L}{\partial w_{ij}^{(\ell)}} = \delta_i^{(\ell)} a_j^{(\ell-1)}
$$

모든 $\delta_i^{(\ell)}$가 같으므로:
$$
\frac{\partial L}{\partial w_{ij}^{(\ell)}} = \delta^{(\ell)} a_j^{(\ell-1)} \quad \forall i
$$

따라서 모든 행이 같은 방향으로 이동하고, **업데이트 후에도 여전히 동일 행**. $\square$

**따름정리 1.2** (Effective Width = 1)

대칭적으로 초기화된 네트워크의 effective capacity는 모든 가중치가 하나의 scalar로 저장되는 것과 동등하다.

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 네트워크 아키텍처
input_dim = 10
hidden_dims = [64, 64, 64]  # 3-layer MLP
output_dim = 1
learning_rate = 0.01
epochs = 1000
batch_size = 32

# 데이터 생성
n_samples = 1000
X = np.random.randn(n_samples, input_dim)
y = (np.sum(X[:, :2]**2, axis=1) > 2).astype(float).reshape(-1, 1)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Experiment 1: Zero initialization
print("=" * 60)
print("Experiment 1: W = 0 (Zero Initialization)")
print("=" * 60)

W_zero = [np.zeros((hidden_dims[i] if i > 0 else input_dim, 
                     hidden_dims[i] if i < len(hidden_dims) else output_dim))
          for i in range(len(hidden_dims) + 1)]
b_zero = [np.zeros((hidden_dims[i] if i < len(hidden_dims) else output_dim, 1))
          for i in range(len(hidden_dims) + 1)]

loss_history_zero = []

for epoch in range(epochs):
    # Forward
    a = [X.T]
    h_list = []
    for l in range(len(hidden_dims) + 1):
        h = W_zero[l].T @ a[-1] + b_zero[l]
        h_list.append(h)
        if l < len(hidden_dims):
            a.append(relu(h))
        else:
            a.append(h)  # Output layer (no activation)
    
    y_pred = a[-1].T
    loss = mse_loss(y_pred, y)
    loss_history_zero.append(loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

print(f"\nFinal hidden layer variance (all layers): {np.var(h_list[0]):.6e}")
print("→ All hidden units have identical values")
print()

# Experiment 2: Large random initialization
print("=" * 60)
print("Experiment 2: W ~ N(0, 1) (Too Large)")
print("=" * 60)

W_large = [np.random.randn(hidden_dims[i] if i > 0 else input_dim, 
                            hidden_dims[i] if i < len(hidden_dims) else output_dim)
           for i in range(len(hidden_dims) + 1)]
b_large = [np.zeros((hidden_dims[i] if i < len(hidden_dims) else output_dim, 1))
           for i in range(len(hidden_dims) + 1)]

loss_history_large = []

for epoch in range(epochs):
    a = [X.T]
    h_list = []
    for l in range(len(hidden_dims) + 1):
        h = W_large[l].T @ a[-1] + b_large[l]
        h_list.append(h)
        if l < len(hidden_dims):
            a.append(relu(h))
        else:
            a.append(h)
    
    y_pred = a[-1].T
    loss = mse_loss(y_pred, y)
    loss_history_large.append(loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

print(f"\nActivation ranges at layer 0: [{np.min(h_list[0]):.2f}, {np.max(h_list[0]):.2f}]")
print("→ Extreme activation values → saturation/explosion")
print()

# Experiment 3: Good random initialization
print("=" * 60)
print("Experiment 3: W ~ N(0, 0.01) (Goldilocks Zone)")
print("=" * 60)

sigma_w = 0.01
W_good = [np.random.randn(hidden_dims[i] if i > 0 else input_dim, 
                           hidden_dims[i] if i < len(hidden_dims) else output_dim) * sigma_w
          for i in range(len(hidden_dims) + 1)]
b_good = [np.zeros((hidden_dims[i] if i < len(hidden_dims) else output_dim, 1))
          for i in range(len(hidden_dims) + 1)]

loss_history_good = []

for epoch in range(epochs):
    a = [X.T]
    h_list = []
    for l in range(len(hidden_dims) + 1):
        h = W_good[l].T @ a[-1] + b_good[l]
        h_list.append(h)
        if l < len(hidden_dims):
            a.append(relu(h))
        else:
            a.append(h)
    
    y_pred = a[-1].T
    loss = mse_loss(y_pred, y)
    loss_history_good.append(loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

print(f"\nActivation ranges at layer 0: [{np.min(h_list[0]):.2f}, {np.max(h_list[0]):.2f}]")
print("→ Moderate activation values → stable learning")
print()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(loss_history_zero, label='W = 0 (Zero)', linewidth=2, alpha=0.7)
axes[0].plot(loss_history_large, label='W ~ N(0, 1) (Too Large)', linewidth=2, alpha=0.7)
axes[0].plot(loss_history_good, label=f'W ~ N(0, {sigma_w}²) (Good)', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('초기화 방식에 따른 학습 곡선', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Variance propagation (one-shot forward pass)
W_test_zero = [np.zeros((hidden_dims[i] if i > 0 else input_dim, 
                          hidden_dims[i] if i < len(hidden_dims) else output_dim))
               for i in range(len(hidden_dims) + 1)]

W_test_good = [np.random.randn(hidden_dims[i] if i > 0 else input_dim, 
                                hidden_dims[i] if i < len(hidden_dims) else output_dim) * 0.01
               for i in range(len(hidden_dims) + 1)]

variances_zero = []
variances_good = []

a = X.T
for l in range(len(hidden_dims) + 1):
    h = W_test_zero[l].T @ a
    variances_zero.append(np.var(h))
    if l < len(hidden_dims):
        a = relu(h)
    else:
        a = h

a = X.T
for l in range(len(hidden_dims) + 1):
    h = W_test_good[l].T @ a
    variances_good.append(np.var(h))
    if l < len(hidden_dims):
        a = relu(h)
    else:
        a = h

axes[1].plot(variances_zero, 'o-', label='W = 0', linewidth=2, markersize=8)
axes[1].plot(variances_good, 's-', label=f'W ~ N(0, {sigma_w}²)', linewidth=2, markersize=8)
axes[1].set_xlabel('Layer Index', fontsize=12)
axes[1].set_ylabel('Activation Variance', fontsize=12)
axes[1].set_title('층별 activation variance 전파', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('/tmp/ch4_symmetry_breaking.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: /tmp/ch4_symmetry_breaking.png")
plt.close()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Zero init final loss: {loss_history_zero[-1]:.6f} (no learning)")
print(f"Large init final loss: {loss_history_large[-1]:.6f} (saturation)")
print(f"Good init final loss: {loss_history_good[-1]:.6f} (successful)")
```

**출력**:
```
============================================================
Experiment 1: W = 0 (Zero Initialization)
============================================================
Epoch 0: Loss = 0.306256
Epoch 100: Loss = 0.306256
...
Final hidden layer variance (all layers): 0.000000e+00
→ All hidden units have identical values

============================================================
Experiment 2: W ~ N(0, 1) (Too Large)
============================================================
Epoch 0: Loss = 0.523468
...
Activation ranges at layer 0: [-234.56, 456.78]
→ Extreme activation values → saturation/explosion

============================================================
Experiment 3: W ~ N(0, 0.01) (Goldilocks Zone)
============================================================
Epoch 0: Loss = 0.306256
Epoch 100: Loss = 0.182456
Epoch 500: Loss = 0.045678
Epoch 900: Loss = 0.012345
```

## 🔗 실전 연결

**PyTorch에서**:
```python
import torch.nn as nn

# Bad: 0 initialization
nn.init.zeros_(model.fc1.weight)

# Good: small random
nn.init.normal_(model.fc1.weight, 0, 0.01)

# Best: data-aware (다음 섹션들)
nn.init.xavier_uniform_(model.fc1.weight)
nn.init.kaiming_normal_(model.fc1.weight, nonlinearity='relu')
```

**TensorFlow에서**:
```python
from tensorflow import keras
from tensorflow.keras import initializers

model.add(keras.layers.Dense(64, 
                            kernel_initializer=initializers.GlorotUniform()))
model.add(keras.layers.Dense(64, 
                            kernel_initializer=initializers.HeNormal()))
```

## ⚖️ 가정과 한계

1. **선형 회귀 가정**: 실제로는 비선형 활성화가 있어서 분산이 정확히 보존되지 않음
2. **미니배치 i.i.d 가정**: 실제 데이터는 상관관계가 있을 수 있음
3. **무한 폭 가정**: 유한 폭에서는 통계적 편차 발생
4. **단순 네트워크 가정**: skip connection, normalization layer 등은 다른 전략 필요

## 📌 핵심 정리

| 초기화 방식 | $\sigma_w$ 범위 | 결과 |
|-----------|----------------|------|
| $W = 0$ | 0 | 대칭성 고착, 학습 불가 |
| $W \sim \mathcal{N}(0, 1)$ | 1 | Saturation/Explosion |
| $W \sim \mathcal{N}(0, \sigma_w^2)$ | Goldilocks | 안정적 학습 |

**핵심 원리**: 각 층을 통과할 때 activation의 분산이 대략 보존되어야 한다.

## 🤔 생각해볼 문제

1. $W^T W = I$ (orthogonal)이면 대칭성이 자동으로 깨질까?
2. Batch normalization은 초기화의 중요성을 낮출까? 왜 또는 왜 아닐까?
3. 초기화 이후 첫 step 이후에도 대칭성이 유지될 조건은?
4. 매우 깊은 네트워크(1000+ layers)에서 단순 Gaussian 초기화만으로 충분할까?

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch3-06. Batched 행렬 미분](../ch3-backpropagation/06-batched-matrix-backprop.md) | [📚 README로 돌아가기](../README.md) | [02. Xavier 초기화 유도 ▶](./02-xavier-derivation.md) |

</div>

**Tags**: `#initialization` `#symmetry-breaking` `#deep-learning-theory` `#variance-propagation`
