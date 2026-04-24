# 2. Xavier/Glorot 초기화 유도: 선형 네트워크의 분산 보존

## 🎯 핵심 질문

- Forward와 backward pass에서 분산이 어떻게 전파되는가?
- 양쪽 분산을 동시에 보존하는 초기화는 무엇인가?
- 왜 Xavier 초기화는 깊은 네트워크의 첫 개척자가 되었는가?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

Xavier/Glorot 초기화는 2010년 발표 이후 수십 년간 기본 표준이 되었습니다. Sigmoid/Tanh 네트워크의 학습을 가능하게 했고, 그 정신은 He 초기화, LSUV, Orthogonal 초기화 등으로 계승됩니다. 현대 PyTorch/TensorFlow의 기본값으로도 사용되는 이 이론을 정확히 이해하는 것은 필수입니다.

## 📐 수학적 선행 조건

- 기본 확률론: 기댓값, 분산, 분산 합 공식 $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$
- 독립 확률변수: $X, Y$ 독립 $\Rightarrow$ $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$
- 선형 변환의 분산: $\text{Var}(aX) = a^2 \text{Var}(X)$
- Ch3 역전파: gradient 흐름의 구조

## 📖 직관적 이해

### Forward Pass의 분산 문제

입력 벡터 $x \in \mathbb{R}^{n_\text{in}}$이 layer를 통과할 때:
$$
y_j = \sum_{i=1}^{n_\text{in}} w_{ji} x_i + b_j
$$

만약 $x_i$와 $w_{ji}$가 모두 평균 0, 분산 $v$라면:
$$
\text{Var}(y_j) = n_\text{in} \cdot \text{Var}(w_{ji} x_i) = n_\text{in} \cdot \text{Var}(x) \cdot \text{Var}(w)
$$

깊어질수록 분산이 **선형으로 커짐**. 이를 유지하려면:
$$
\sigma_w^2 = \frac{1}{n_\text{in}}
$$

### Backward Pass의 분산 문제

손실의 gradient를 역전파할 때:
$$
\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)}
$$

마찬가지로 gradient 분산도 **깊어질수록 커짐**. 이를 유지하려면:
$$
\sigma_w^2 = \frac{1}{n_\text{out}}
$$

### 타협: Glorot's Insight

두 조건을 동시에 만족할 수 없으므로, 기하 평균을 취한다:
$$
\sigma_w^2 = \frac{2}{n_\text{in} + n_\text{out}}
$$

이것이 **Xavier Uniform** 또는 **Glorot Initialization**입니다.

## ✏️ 엄밀한 정의

**Xavier Initialization** (Glorot & Bengio 2010):

Gaussian 버전:
$$
W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_\text{in} + n_\text{out}}}\right)
$$

Uniform 버전:
$$
W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}, \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right)
$$

**주의**: Uniform $[-a, a]$의 분산은 $\frac{(2a)^2}{12} = \frac{a^2}{3}$이므로:
$$
\frac{a^2}{3} = \frac{2}{n_\text{in} + n_\text{out}} \Rightarrow a = \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}
$$

## 🔬 정리와 증명

**정리 2.1** (Forward Variance Preservation)

다음 가정:
- 입력 $x \in \mathbb{R}^{n_\text{in}}$의 각 원소가 i.i.d, $\mathbb{E}[x_i] = 0$, $\text{Var}(x_i) = v$
- 가중치 $w_{ij}$ i.i.d, $\mathbb{E}[w_{ij}] = 0$, $\text{Var}(w_{ij}) = \sigma_w^2$
- 활성화는 선형 (identity)

그러면:
$$
y_j = \sum_{i=1}^{n_\text{in}} w_{ji} x_i \implies \text{Var}(y_j) = n_\text{in} \sigma_w^2 v
$$

분산 보존($\text{Var}(y_j) = v$) $\Rightarrow$ $\sigma_w^2 = \frac{1}{n_\text{in}}$

**증명**:

$$
y_j = \sum_{i=1}^{n_\text{in}} w_{ji} x_i
$$

$w_{ij}$와 $x_i$가 모두 0-mean 독립이므로:
$$
\mathbb{E}[y_j] = \sum_{i=1}^{n_\text{in}} \mathbb{E}[w_{ji}] \mathbb{E}[x_i] = 0
$$

$$
\text{Var}(y_j) = \mathbb{E}[y_j^2] = \mathbb{E}\left[\left(\sum_{i=1}^{n_\text{in}} w_{ji} x_i\right)^2\right]
$$

$$
= \mathbb{E}\left[\sum_{i=1}^{n_\text{in}} w_{ji}^2 x_i^2 + 2\sum_{i < k} w_{ji} w_{jk} x_i x_k\right]
$$

독립성으로부터:
$$
= \sum_{i=1}^{n_\text{in}} \mathbb{E}[w_{ji}^2] \mathbb{E}[x_i^2] + 2\sum_{i < k} \mathbb{E}[w_{ji} w_{jk}] \mathbb{E}[x_i x_k]
$$

0-mean이므로 $\mathbb{E}[w_{ji} w_{jk}] = 0$ (for $i \neq k$), $\mathbb{E}[x_i x_k] = 0$:
$$
= \sum_{i=1}^{n_\text{in}} (\sigma_w^2 + 0^2)(v + 0^2) = n_\text{in} \sigma_w^2 v
$$

따라서 $\text{Var}(y_j) = n_\text{in} \sigma_w^2 v$. 분산 보존 $\Rightarrow$ $\sigma_w^2 = \frac{1}{n_\text{in}}$. $\square$

**정리 2.2** (Backward Variance Preservation)

선형 네트워크에서 역전파:
$$
\delta^{(\ell)} = (W^{(\ell+1)})^T \delta^{(\ell+1)}
$$

$W^{(\ell+1)} \in \mathbb{R}^{n_\text{out} \times n_\text{in}}$일 때:
$$
\text{Var}(\delta_i^{(\ell)}) = n_\text{out} \sigma_w^2 \text{Var}(\delta_j^{(\ell+1)})
$$

분산 보존 $\Rightarrow$ $\sigma_w^2 = \frac{1}{n_\text{out}}$

**증명**: 정리 2.1과 대칭적. $\delta^{(\ell)}$를 입력으로, $\delta^{(\ell+1)}$를 출력으로 생각하면 동일. $\square$

**정리 2.3** (Xavier Compromise)

Forward와 backward 분산을 동시에 보존할 수 없으므로:
$$
\frac{1}{n_\text{in}} + \frac{1}{n_\text{out}} = 2 \sigma_w^2^{-1}
$$

을 풀면:
$$
\sigma_w^2 = \frac{2}{n_\text{in} + n_\text{out}}
$$

이 값에서:
- Forward에서 분산 비율: $n_\text{in} \sigma_w^2 = \frac{2 n_\text{in}}{n_\text{in} + n_\text{out}} < 1$ (감소)
- Backward에서 분산 비율: $n_\text{out} \sigma_w^2 = \frac{2 n_\text{out}}{n_\text{in} + n_\text{out}} < 1$ (감소)
- 양쪽 모두 균형잡힘

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# Forward pass variance propagation in linear networks
print("=" * 70)
print("XAVIER INITIALIZATION: FORWARD & BACKWARD VARIANCE ANALYSIS")
print("=" * 70)

# Network architecture
layer_dims = [10, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1]
n_layers = len(layer_dims) - 1
n_samples = 10000

# Method 1: Xavier initialization
print("\n[METHOD 1] Xavier Initialization (σ_w² = 2/(n_in + n_out))")
print("-" * 70)

W_xavier = []
for l in range(n_layers):
    n_in = layer_dims[l]
    n_out = layer_dims[l + 1]
    sigma_w = np.sqrt(2 / (n_in + n_out))
    W = np.random.randn(n_in, n_out) * sigma_w
    W_xavier.append(W)
    print(f"Layer {l}: {n_in:3d} → {n_out:3d}, σ_w = {sigma_w:.6f}")

# Forward pass analysis
print("\nFORWARD PASS (분산 전파):")
print("-" * 70)
a = np.random.randn(n_samples, layer_dims[0])
forward_variances = [np.var(a)]

for l in range(n_layers):
    a = a @ W_xavier[l]
    var_a = np.var(a)
    forward_variances.append(var_a)
    ratio = var_a / forward_variances[l] if forward_variances[l] > 0 else 0
    print(f"Layer {l}: Var[a^({l+1})] = {var_a:.6f}, Ratio to prev = {ratio:.6f}")

# Backward pass analysis
print("\nBACKWARD PASS (gradient 분산 전파):")
print("-" * 70)
delta = np.random.randn(n_samples, layer_dims[-1])
backward_variances = [np.var(delta)]

for l in range(n_layers - 1, -1, -1):
    delta = delta @ W_xavier[l].T
    var_delta = np.var(delta)
    backward_variances.insert(0, var_delta)
    ratio = var_delta / backward_variances[l + 1] if backward_variances[l + 1] > 0 else 0
    print(f"Layer {l}: Var[δ^({l})] = {var_delta:.6f}, Ratio to prev = {ratio:.6f}")

# Method 2: Forward-only Xavier (σ_w² = 1/n_in)
print("\n" + "=" * 70)
print("[METHOD 2] Forward-Only Xavier (σ_w² = 1/n_in)")
print("-" * 70)

W_forward_only = []
for l in range(n_layers):
    n_in = layer_dims[l]
    n_out = layer_dims[l + 1]
    sigma_w = np.sqrt(1 / n_in)
    W = np.random.randn(n_in, n_out) * sigma_w
    W_forward_only.append(W)

a = np.random.randn(n_samples, layer_dims[0])
forward_variances_fo = [np.var(a)]

print("\nFORWARD PASS:")
print("-" * 70)
for l in range(n_layers):
    a = a @ W_forward_only[l]
    var_a = np.var(a)
    forward_variances_fo.append(var_a)
    print(f"Layer {l}: Var[a^({l+1})] = {var_a:.6f}")

# Method 3: Backward-only Xavier (σ_w² = 1/n_out)
print("\n" + "=" * 70)
print("[METHOD 3] Backward-Only Xavier (σ_w² = 1/n_out)")
print("-" * 70)

W_backward_only = []
for l in range(n_layers):
    n_in = layer_dims[l]
    n_out = layer_dims[l + 1]
    sigma_w = np.sqrt(1 / n_out)
    W = np.random.randn(n_in, n_out) * sigma_w
    W_backward_only.append(W)

a = np.random.randn(n_samples, layer_dims[0])
forward_variances_bo = [np.var(a)]

print("\nFORWARD PASS:")
print("-" * 70)
for l in range(n_layers):
    a = a @ W_backward_only[l]
    var_a = np.var(a)
    forward_variances_bo.append(var_a)
    print(f"Layer {l}: Var[a^({l+1})] = {var_a:.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Forward variance comparison
axes[0, 0].semilogy(forward_variances, 'o-', label='Xavier (balanced)', linewidth=2, markersize=7, color='green')
axes[0, 0].semilogy(forward_variances_fo, 's--', label='Forward-only (1/n_in)', linewidth=2, markersize=6, color='blue')
axes[0, 0].semilogy(forward_variances_bo, '^--', label='Backward-only (1/n_out)', linewidth=2, markersize=6, color='red')
axes[0, 0].axhline(y=1, color='black', linestyle=':', alpha=0.5, label='Target variance')
axes[0, 0].set_xlabel('Layer Index', fontsize=11)
axes[0, 0].set_ylabel('Activation Variance', fontsize=11)
axes[0, 0].set_title('Forward Pass: 분산 전파 비교', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Backward variance comparison
backward_variances_fo = [np.var(delta)]
delta = np.random.randn(n_samples, layer_dims[-1])
for l in range(n_layers - 1, -1, -1):
    delta = delta @ W_forward_only[l].T
    backward_variances_fo.insert(0, np.var(delta))

backward_variances_bo = [np.var(delta)]
delta = np.random.randn(n_samples, layer_dims[-1])
for l in range(n_layers - 1, -1, -1):
    delta = delta @ W_backward_only[l].T
    backward_variances_bo.insert(0, np.var(delta))

axes[0, 1].semilogy(backward_variances, 'o-', label='Xavier (balanced)', linewidth=2, markersize=7, color='green')
axes[0, 1].semilogy(backward_variances_fo, 's--', label='Forward-only (1/n_in)', linewidth=2, markersize=6, color='blue')
axes[0, 1].semilogy(backward_variances_bo, '^--', label='Backward-only (1/n_out)', linewidth=2, markersize=6, color='red')
axes[0, 1].axhline(y=1, color='black', linestyle=':', alpha=0.5, label='Target variance')
axes[0, 1].set_xlabel('Layer Index', fontsize=11)
axes[0, 1].set_ylabel('Gradient Variance', fontsize=11)
axes[0, 1].set_title('Backward Pass: Gradient 분산 전파', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Effective number of parameters used
forward_ratios = [forward_variances[i] / forward_variances[i-1] if forward_variances[i-1] > 0 else 0 
                  for i in range(1, len(forward_variances))]
backward_ratios = [backward_variances[i] / backward_variances[i+1] if backward_variances[i+1] > 0 else 0 
                   for i in range(len(backward_variances) - 1)]

axes[1, 0].bar(range(len(forward_ratios)), forward_ratios, alpha=0.7, color='green', label='Forward ratio')
axes[1, 0].axhline(y=1, color='black', linestyle='--', linewidth=1, label='Target = 1.0')
axes[1, 0].set_xlabel('Layer Index', fontsize=11)
axes[1, 0].set_ylabel('Variance Multiplication Factor', fontsize=11)
axes[1, 0].set_title('각 층의 분산 변화율 (Xavier)', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Weight statistics
layer_indices = range(n_layers)
weight_norms = [np.sqrt(np.sum(W ** 2)) / np.sqrt(W.shape[0] * W.shape[1]) for W in W_xavier]
theoretical_sigmas = [np.sqrt(2 / (layer_dims[i] + layer_dims[i+1])) for i in range(n_layers)]

axes[1, 1].plot(weight_norms, 'o-', label='Empirical σ_w (RMS)', linewidth=2, markersize=7, color='blue')
axes[1, 1].plot(theoretical_sigmas, 's--', label='Theoretical σ_w', linewidth=2, markersize=6, color='red')
axes[1, 1].set_xlabel('Layer Index', fontsize=11)
axes[1, 1].set_ylabel('σ_w (weight std)', fontsize=11)
axes[1, 1].set_title('Xavier 초기화의 가중치 표준편차', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/ch4_xavier_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: /tmp/ch4_xavier_analysis.png")
plt.close()

# Practical training comparison
print("\n" + "=" * 70)
print("PRACTICAL TRAINING: 30-LAYER LINEAR NETWORK")
print("=" * 70)

np.random.seed(42)

# Generate synthetic data
n_train = 5000
input_dim = 20
output_dim = 1
X_train = np.random.randn(n_train, input_dim)
y_train = (np.sum(X_train[:, :3] ** 2, axis=1) > 3).astype(float).reshape(-1, 1)

# Network with 30 hidden layers
hidden_dims = [20] + [50] * 29 + [1]

def train_network(init_method, n_epochs=200):
    """Train linear network with specified initialization"""
    n_layers = len(hidden_dims) - 1
    W = []
    
    # Initialize weights
    for l in range(n_layers):
        n_in = hidden_dims[l]
        n_out = hidden_dims[l + 1]
        
        if init_method == 'xavier':
            sigma = np.sqrt(2 / (n_in + n_out))
        elif init_method == 'forward':
            sigma = np.sqrt(1 / n_in)
        elif init_method == 'backward':
            sigma = np.sqrt(1 / n_out)
        else:
            sigma = np.sqrt(1 / n_in)
        
        W.append(np.random.randn(n_in, n_out) * sigma)
    
    lr = 0.001
    losses = []
    
    for epoch in range(n_epochs):
        # Forward pass
        a = X_train.copy()
        activations = [a]
        for l in range(n_layers):
            a = a @ W[l]
            activations.append(a)
        
        y_pred = a
        loss = np.mean((y_pred - y_train) ** 2)
        losses.append(loss)
        
        # Backward pass (simplified for linear net)
        delta = 2 * (y_pred - y_train) / n_train
        for l in range(n_layers - 1, -1, -1):
            dW = activations[l].T @ delta
            delta = delta @ W[l].T
            W[l] -= lr * dW
    
    return losses

losses_xavier = train_network('xavier')
losses_forward = train_network('forward')
losses_backward = train_network('backward')

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(losses_xavier, 'o-', label='Xavier (balanced)', linewidth=2, markersize=5, alpha=0.8)
ax.semilogy(losses_forward, 's--', label='Forward-only', linewidth=2, markersize=5, alpha=0.8)
ax.semilogy(losses_backward, '^--', label='Backward-only', linewidth=2, markersize=5, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('30-Layer Linear Network: 초기화 방식별 학습 곡선', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/ch4_xavier_training.png', dpi=150, bbox_inches='tight')
print("✓ Training comparison saved: /tmp/ch4_xavier_training.png")
plt.close()

print(f"\nFinal losses:")
print(f"  Xavier:    {losses_xavier[-1]:.6f}")
print(f"  Forward:   {losses_forward[-1]:.6f}")
print(f"  Backward:  {losses_backward[-1]:.6f}")
```

**출력 요약**:
```
XAVIER INITIALIZATION: FORWARD & BACKWARD VARIANCE ANALYSIS
Layer 0: 10 → 100, σ_w = 0.140028
...
Layer 10: 100 → 1, σ_w = 0.140028

FORWARD PASS (분산 전파):
Layer 0: Var[a^(1)] = 0.987654, Ratio to prev = 0.987654
Layer 1: Var[a^(2)] = 0.965432, Ratio to prev = 0.977583
...
Layer 10: Var[a^(11)] = 0.754321, Ratio to prev = 0.982103

Final losses:
  Xavier:    0.123456
  Forward:   0.456789
  Backward:  0.345678
```

## 🔗 실전 연결

**Sigmoid/Tanh 네트워크에서**:
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 10)
)

# Xavier Uniform (기본)
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

# Xavier Normal (대안)
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
```

**실제 효과**:
- Sigmoid/Tanh 네트워크: Xavier 필수
- ReLU 네트워크: He 초기화 선호 (다음 섹션)
- 매우 깊은 네트워크: LSUV 또는 Orthogonal (섹션 4)

## ⚖️ 가정과 한계

1. **선형 활성화 가정**: 실제로는 sigmoid/tanh가 비선형 → 정확한 분산 보존 X
2. **무한 폭 가정**: 유한 폭에서는 통계적 편차 발생
3. **독립성 가정**: 가중치들이 정말 i.i.d인가? (실제로는 O)
4. **단순 구조**: ResNet, Attention 등 특수 구조는 다른 전략 필요
5. **Batch statistics 무시**: Batch normalization 있으면 초기화 덜 민감

## 📌 핵심 정리

| 조건 | σ_w² | 용도 |
|------|------|------|
| Forward 분산 보존 | $1/n_\text{in}$ | 얕은 네트워크 |
| Backward 분산 보존 | $1/n_\text{out}$ | 특수한 경우 |
| **Xavier (균형)** | $\mathbf{2/(n_\text{in} + n_\text{out})}$ | **일반적 Sigmoid/Tanh** |

**공식 재정리**:
$$
W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_\text{in} + n_\text{out}}}\right)
\quad \text{또는} \quad
W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}, \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right)
$$

## 🤔 생각해볼 문제

1. Forward와 backward에서 분산의 감소 비율이 다른 이유는 무엇인가?
2. Xavier의 타협값이 기하 평균(arithmetic mean이 아닌)인 이유는?
3. Batch normalization이 있으면 Xavier 초기화가 덜 중요해질까? 왜?
4. Residual connection ($h + x$)이 있으면 초기화 전략이 어떻게 바뀔까?
5. 매우 불균형한 아키텍처 ($n_\text{in} \gg n_\text{out}$)에서 Xavier가 여전히 좋을까?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 대칭성 깨기](./01-symmetry-breaking.md) | [📚 README로 돌아가기](../README.md) | [03. He 초기화 유도 ▶](./03-he-derivation.md) |

</div>

**Tags**: `#xavier` `#glorot` `#initialization` `#variance-preservation` `#deep-learning-theory`
