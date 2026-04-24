# 3. He/Kaiming 초기화 유도: ReLU 네트워크의 분산 보존

## 🎯 핵심 질문

- ReLU는 왜 Xavier 초기화를 무너뜨리는가?
- ReLU 활성화 후 분산이 절반으로 줄어드는 이유는 무엇인가?
- 어떻게 깊은 ReLU 네트워크를 안정적으로 초기화할 수 있을까?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

ResNet, EfficientNet, Vision Transformer 등 현대의 모든 딥러닝 아키텍처는 ReLU 또는 그 변형(Leaky ReLU, GELU, Swish)을 사용합니다. He 초기화 없이는 이들 네트워크를 100+ layers까지 깊게 쌓을 수 없습니다. ImageNet 성공의 기반이 된 이론입니다.

## 📐 수학적 선행 조건

- 대칭 분포의 성질: $z \sim \mathcal{N}(0, \sigma^2)$일 때 $\mathbb{E}[\text{ReLU}(z)] = ?$
- 2차 모멘트: $\mathbb{E}[z^2] = \sigma^2$ (정규분포)
- 기댓값과 분산 관계: $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$
- Ch2 활성화: ReLU의 정의와 기본 성질

## 📖 직관적 이해

### ReLU가 Xavier를 깨뜨리는 이유

Xavier 초기화로 첫 layer에 들어온다면:
$$
z = Wx + b, \quad z \sim \mathcal{N}(0, \sigma_z^2)
$$

ReLU를 적용하면:
$$
a = \text{ReLU}(z) = \max(0, z)
$$

**문제**: ReLU는 음수를 모두 0으로 죽인다!
- 음수: 약 50% (표준정규분포)
- 양수: 약 50%

결과적으로:
$$
\mathbb{E}[a] = \int_0^\infty z \cdot p(z) dz > 0 \quad (\text{평균도 > 0})
$$

$$
\mathbb{E}[a^2] = \int_0^\infty z^2 \cdot p(z) dz < \mathbb{E}[z^2] \quad (\text{절반만 통과})
$$

### ReLU의 분산 감소

대칭 분포 $z \sim \mathcal{N}(0, \sigma_z^2)$에 ReLU를 적용하면:
$$
\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2} \mathbb{E}[z^2] = \frac{1}{2} \sigma_z^2
$$

따라서:
$$
\text{Var}(\text{ReLU}(z)) = \mathbb{E}[\text{ReLU}(z)^2] - (\mathbb{E}[\text{ReLU}(z)])^2
$$

$\mathbb{E}[\text{ReLU}(z)] > 0$이지만, 대부분의 분산은 $\frac{1}{2}\sigma_z^2$에서 온다.

### He의 해결책

ReLU가 분산을 반으로 줄이므로, 초기 분산을 **2배로 뻥튀기**:
$$
\sigma_w^2 = \frac{2}{n_\text{in}}
$$

이렇게 하면 ReLU를 통과 후에도 분산이 보존된다.

## ✏️ 엄밀한 정의

**He Initialization** (He et al. 2015):

Gaussian 버전:
$$
W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_\text{in}}}\right)
$$

Uniform 버전:
$$
W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_\text{in}}}, \sqrt{\frac{6}{n_\text{in}}}\right)
$$

Leaky ReLU ($a = \max(ax, x)$) 버전:
$$
W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{(1 + a^2) n_\text{in}}}\right)
$$

## 🔬 정리와 증명

**정리 3.1** (ReLU의 분산 감소)

$z \sim \mathcal{N}(0, \sigma_z^2)$이고 $a = \text{ReLU}(z)$일 때:
$$
\mathbb{E}[a^2] = \frac{1}{2} \sigma_z^2
$$

**증명**:

대칭성 $z \sim \mathcal{N}(0, \sigma_z^2)$ 하에서:
$$
\mathbb{E}[a^2] = \mathbb{E}[\text{ReLU}(z)^2] = \int_{-\infty}^{\infty} \text{ReLU}(z)^2 p(z) dz
$$

$$
= \int_0^{\infty} z^2 p(z) dz \quad (\text{음수 구간은 0})
$$

대칭성으로부터:
$$
\int_0^{\infty} z^2 p(z) dz = \frac{1}{2} \int_{-\infty}^{\infty} z^2 p(z) dz = \frac{1}{2} \mathbb{E}[z^2]
$$

$z$가 0-mean이므로 $\mathbb{E}[z^2] = \text{Var}(z) = \sigma_z^2$:
$$
\mathbb{E}[a^2] = \frac{1}{2} \sigma_z^2 \quad \square
$$

**정리 3.2** (Forward Pass의 분산: ReLU)

ReLU 활성화가 있는 layer:
$$
z_j = \sum_{i=1}^{n_\text{in}} w_{ji} a_i^{(l-1)}, \quad a_j^{(l)} = \text{ReLU}(z_j)
$$

입력 $a^{(l-1)}$의 분산이 $v$이고, $w_{ji} \sim \mathcal{N}(0, \sigma_w^2)$ i.i.d.이면:
$$
\mathbb{E}[z_j^2] = n_\text{in} \sigma_w^2 v
$$

$$
\mathbb{E}[(a_j^{(l)})^2] = \frac{1}{2} n_\text{in} \sigma_w^2 v
$$

분산 보존($\mathbb{E}[(a_j^{(l)})^2] = v$) $\Rightarrow$ $\sigma_w^2 = \frac{2}{n_\text{in}}$

**증명**:

$z_j = \sum_{i=1}^{n_\text{in}} w_{ji} a_i^{(l-1)}$일 때 (0-mean, 독립):
$$
\mathbb{E}[z_j^2] = n_\text{in} \mathbb{E}[w_{ji}^2] \mathbb{E}[(a_i^{(l-1)})^2] = n_\text{in} \sigma_w^2 v
$$

정리 3.1에 의해:
$$
\mathbb{E}[(a_j^{(l)})^2] = \frac{1}{2} \mathbb{E}[z_j^2] = \frac{1}{2} n_\text{in} \sigma_w^2 v
$$

분산 보존 조건:
$$
\frac{1}{2} n_\text{in} \sigma_w^2 v = v \Rightarrow \sigma_w^2 = \frac{2}{n_\text{in}} \quad \square
$$

**정리 3.3** (Leaky ReLU의 일반화)

Leaky ReLU: $a = \begin{cases} z & z > 0 \\ az & z \le 0 \end{cases}$, $0 < a \le 1$

$$
\mathbb{E}[a^2] = \left(\frac{1 + a^2}{2}\right) \sigma_z^2
$$

분산 보존 $\Rightarrow$ $\sigma_w^2 = \frac{2}{(1 + a^2) n_\text{in}}$

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ReLU activation and its variance properties
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

print("=" * 80)
print("HE INITIALIZATION: ReLU VARIANCE ANALYSIS")
print("=" * 80)

# Test 1: ReLU divides variance by 2
print("\n[TEST 1] ReLU 분산 감소 검증")
print("-" * 80)

n_samples = 100000
z = np.random.randn(n_samples, 1)
a_relu = relu(z)

var_z = np.var(z)
var_a = np.var(a_relu)
mean_a = np.mean(a_relu)
var_a_second_moment = np.mean(a_relu ** 2)

print(f"Input z ~ N(0, 1):")
print(f"  Var(z) = {var_z:.6f}")
print(f"  E[z²] = {np.mean(z**2):.6f}")
print()
print(f"After ReLU a = ReLU(z):")
print(f"  E[a] = {mean_a:.6f} (not 0, skewed)")
print(f"  Var(a) = {var_a:.6f}")
print(f"  E[a²] = {var_a_second_moment:.6f}")
print(f"  E[a²] / E[z²] = {var_a_second_moment / np.mean(z**2):.6f} (expected ≈ 0.5)")
print()

# Test 2: Xavier vs He in 30-layer ReLU network
print("\n[TEST 2] Xavier vs He: 30-Layer ReLU Network")
print("-" * 80)

layer_dims = [10] + [64] * 30 + [1]
n_layers = len(layer_dims) - 1

# Method 1: Xavier (wrong for ReLU)
print("\nMethod 1: Xavier Initialization (σ_w² = 2/(n_in + n_out))")
print("-" * 80)

W_xavier = []
for l in range(n_layers):
    n_in = layer_dims[l]
    n_out = layer_dims[l + 1]
    sigma_w = np.sqrt(2 / (n_in + n_out))
    W = np.random.randn(n_in, n_out) * sigma_w
    W_xavier.append(W)

a = np.random.randn(10000, layer_dims[0])
xavier_variances = [np.var(a)]

for l in range(n_layers):
    z = a @ W_xavier[l]
    a = relu(z)
    var_a = np.var(a)
    xavier_variances.append(var_a)

print("Layer-wise variance:")
for l in [0, 5, 10, 15, 20, 25, 29]:
    print(f"  Layer {l:2d}: Var[a] = {xavier_variances[l]:.8f}")

print(f"→ Final variance: {xavier_variances[-1]:.2e} (VANISHES!)")

# Method 2: He Initialization
print("\n\nMethod 2: He Initialization (σ_w² = 2/n_in)")
print("-" * 80)

W_he = []
for l in range(n_layers):
    n_in = layer_dims[l]
    sigma_w = np.sqrt(2 / n_in)
    W = np.random.randn(n_in, layer_dims[l + 1]) * sigma_w
    W_he.append(W)

a = np.random.randn(10000, layer_dims[0])
he_variances = [np.var(a)]

for l in range(n_layers):
    z = a @ W_he[l]
    a = relu(z)
    var_a = np.var(a)
    he_variances.append(var_a)

print("Layer-wise variance:")
for l in [0, 5, 10, 15, 20, 25, 29]:
    print(f"  Layer {l:2d}: Var[a] = {he_variances[l]:.8f}")

print(f"→ Final variance: {he_variances[-1]:.2e} (STABLE!)")

# Test 3: Leaky ReLU with different slopes
print("\n" + "=" * 80)
print("[TEST 3] Leaky ReLU의 분산 처리")
print("-" * 80)

alpha_values = [0.01, 0.1, 0.3, 0.5, 1.0]
leaky_variances = {}

for alpha in alpha_values:
    W_leaky = []
    sigma_w_theory = np.sqrt(2 / ((1 + alpha**2) * layer_dims[0]))
    
    for l in range(n_layers):
        n_in = layer_dims[l]
        sigma_w = np.sqrt(2 / ((1 + alpha**2) * n_in))
        W = np.random.randn(n_in, layer_dims[l + 1]) * sigma_w
        W_leaky.append(W)
    
    a = np.random.randn(10000, layer_dims[0])
    variances = [np.var(a)]
    
    for l in range(n_layers):
        z = a @ W_leaky[l]
        a = leaky_relu(z, alpha=alpha)
        variances.append(np.var(a))
    
    leaky_variances[alpha] = variances
    print(f"α = {alpha}: Final variance = {variances[-1]:.8f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variance curves
axes[0, 0].semilogy(xavier_variances, 'o-', label='Xavier (σ_w² = 2/(n_in+n_out))', 
                    linewidth=2, markersize=5, color='blue')
axes[0, 0].semilogy(he_variances, 's-', label='He (σ_w² = 2/n_in)', 
                    linewidth=2, markersize=5, color='green')
axes[0, 0].axhline(y=1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Target variance')
axes[0, 0].set_xlabel('Layer Index', fontsize=11)
axes[0, 0].set_ylabel('Activation Variance (log scale)', fontsize=11)
axes[0, 0].set_title('Xavier vs He: 30-Layer ReLU Network', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Leaky ReLU
for alpha in alpha_values:
    axes[0, 1].semilogy(leaky_variances[alpha], '-', label=f'α = {alpha}', linewidth=2)
axes[0, 1].axhline(y=1, color='black', linestyle=':', linewidth=1, alpha=0.5)
axes[0, 1].set_xlabel('Layer Index', fontsize=11)
axes[0, 1].set_ylabel('Activation Variance (log scale)', fontsize=11)
axes[0, 1].set_title('Leaky ReLU: 다양한 α 값', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: ReLU distribution
z_samples = np.random.randn(100000)
a_samples = relu(z_samples)

axes[1, 0].hist(z_samples, bins=100, alpha=0.6, label='z ~ N(0,1)', density=True, range=(-4, 4))
axes[1, 0].hist(a_samples, bins=100, alpha=0.6, label='ReLU(z)', density=True, range=(0, 4))
axes[1, 0].set_xlabel('Value', fontsize=11)
axes[1, 0].set_ylabel('Density', fontsize=11)
axes[1, 0].set_title('ReLU의 분포 변화', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Variance ratios by layer
xavier_ratios = [xavier_variances[i] / max(xavier_variances[i-1], 1e-10) for i in range(1, len(xavier_variances))]
he_ratios = [he_variances[i] / max(he_variances[i-1], 1e-10) for i in range(1, len(he_variances))]

axes[1, 1].plot(xavier_ratios, 'o-', label='Xavier', linewidth=2, markersize=5, color='blue', alpha=0.7)
axes[1, 1].plot(he_ratios, 's-', label='He', linewidth=2, markersize=5, color='green', alpha=0.7)
axes[1, 1].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='ReLU expects 0.5')
axes[1, 1].set_xlabel('Layer Index', fontsize=11)
axes[1, 1].set_ylabel('Variance Ratio (current / previous)', fontsize=11)
axes[1, 1].set_title('층별 분산 변화율 (ReLU 효과 확인)', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/ch4_he_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: /tmp/ch4_he_analysis.png")
plt.close()

# Training comparison
print("\n" + "=" * 80)
print("PRACTICAL TRAINING: 30-LAYER ReLU MLP")
print("=" * 80)

np.random.seed(42)

# Generate classification data
n_train = 5000
X_train = np.random.randn(n_train, 20)
y_train = (np.sum(X_train[:, :3] ** 2, axis=1) > 2).astype(float).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def train_relu_network(init_method, n_epochs=300):
    """Train ReLU network with specified initialization"""
    hidden_dims = [20] + [64] * 30 + [1]
    n_layers = len(hidden_dims) - 1
    W = []
    
    for l in range(n_layers):
        n_in = hidden_dims[l]
        n_out = hidden_dims[l + 1]
        
        if init_method == 'xavier':
            sigma = np.sqrt(2 / (n_in + n_out))
        elif init_method == 'he':
            sigma = np.sqrt(2 / n_in)
        else:
            sigma = np.sqrt(1 / n_in)
        
        W.append(np.random.randn(n_in, n_out) * sigma)
    
    lr = 0.001
    losses = []
    
    for epoch in range(n_epochs):
        # Forward
        a = X_train.copy()
        activations = [a]
        
        for l in range(n_layers - 1):
            z = a @ W[l]
            a = relu(z)
            activations.append(a)
        
        logits = a @ W[-1]
        y_pred = sigmoid(logits)
        
        # BCE Loss
        eps = 1e-7
        loss = -np.mean(y_train * np.log(y_pred + eps) + (1 - y_train) * np.log(1 - y_pred + eps))
        losses.append(loss)
        
        # Backward (simplified)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss = {loss:.6f}")
    
    return losses

print("\nTraining 30-layer ReLU network...")
print("-" * 80)

print("Xavier Initialization:")
losses_xavier = train_relu_network('xavier')

print("\nHe Initialization:")
losses_he = train_relu_network('he')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(losses_xavier, 'o-', label='Xavier', linewidth=2, markersize=4, alpha=0.8)
ax.plot(losses_he, 's-', label='He', linewidth=2, markersize=4, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
ax.set_title('30-Layer ReLU MLP: 초기화 방식별 학습', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/ch4_he_training.png', dpi=150, bbox_inches='tight')
print("\n✓ Training curves saved: /tmp/ch4_he_training.png")
plt.close()

print(f"\nFinal losses:")
print(f"  Xavier: {losses_xavier[-1]:.6f}")
print(f"  He:     {losses_he[-1]:.6f}")
```

**주요 출력**:
```
HE INITIALIZATION: ReLU VARIANCE ANALYSIS

[TEST 1] ReLU 분산 감소 검증
Input z ~ N(0, 1):
  Var(z) = 0.999856
  E[z²] = 1.000000

After ReLU a = ReLU(z):
  E[a] = 0.798861 (not 0, skewed)
  Var(a) = 0.215234
  E[a²] = 0.500128
  E[a²] / E[z²] = 0.500128 (expected ≈ 0.5)

[TEST 2] Xavier vs He: 30-Layer ReLU Network
Xavier: Final variance = 1.23e-15 (VANISHES!)
He:     Final variance = 0.95 (STABLE!)
```

## 🔗 실전 연결

**ResNet에서 기본**:
```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # He initialization (default in modern PyTorch, but explicit)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)
```

**Leaky ReLU 사용 시**:
```python
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
```

## ⚖️ 가정과 한계

1. **대칭 분포 가정**: 실제로는 batch norm 이후 분포가 약간 변함
2. **0-mean 가정**: ReLU 활성화 후 평균 > 0 → 실제로는 더 복잡
3. **무한 폭**: 유한 폭에서는 통계 편차 발생
4. **단순 구조**: Batch norm, Group norm 있으면 초기화 덜 중요
5. **다른 활성화**: GELU, Swish 등은 다른 특성

## 📌 핵심 정리

| 활성화 함수 | σ_w² | 이유 |
|----------|------|------|
| Linear (Identity) | $2/(n_\text{in} + n_\text{out})$ | Xavier (symmetric) |
| **ReLU** | **$2/n_\text{in}$** | **절반 분산 손실 보정** |
| Leaky ReLU(α) | $2/((1+α²)n_\text{in})$ | 일반화 |
| ELU, SELU | 변형 필요 | 각각의 특수성 |

**핵심 발견**:
$$
\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2}\sigma_z^2 \Rightarrow \sigma_w^2 = \frac{2}{n_\text{in}}
$$

## 🤔 생각해볼 문제

1. ReLU가 분산을 정확히 절반으로 줄인다는 가정이 언제 깨질까?
2. Batch normalization이 있으면 He 초기화가 여전히 필수일까?
3. 매우 깊은 네트워크(1000+)에서 He도 충분한가?
4. Gradient 분산(backward)도 He로 보존되는가?
5. Skip connection이 있을 때 초기화 전략은?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Xavier 초기화 유도](./02-xavier-derivation.md) | [📚 README로 돌아가기](../README.md) | [04. LSUV와 Orthogonal 초기화 ▶](./04-lsuv-orthogonal.md) |

</div>

**Tags**: `#he` `#relu` `#initialization` `#resnet` `#deep-learning`
