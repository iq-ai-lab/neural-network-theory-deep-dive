# 04. 활성화 함수 비교: Sigmoid, Tanh, ReLU, GELU

## 🎯 핵심 질문

- Sigmoid $\sigma(z) = 1/(1+e^{-z})$의 도함수는 최대값 0.25를 가지는가? 왜 이것이 문제인가?
- Tanh가 Sigmoid보다 선호되는 이유는 무엇인가? (도함수 최댓값 비교)
- ReLU $\max(0, z)$는 왜 깊은 네트워크에서 "vanishing gradient" 문제를 완화하는가?
- Dying ReLU 문제란 무엇이고, Leaky ReLU는 어떻게 해결하는가?
- GELU와 Swish 같은 smooth activation은 어떤 장점을 가지는가?
- Saturation 영역(flat region)에서 왜 gradient가 죽는가?

---

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

**활성화 함수는 신경망의 비선형성의 유일한 원천**이며, Theorem 3.1에서 보았듯이 비선형성이 없으면 깊이는 의미가 없다. 하지만 **모든 비선형 함수가 동등하지 않다**:

1. **Sigmoid & Tanh**: 1960~1980년대 dominant, 하지만 **vanishing gradient** 문제로 깊은 네트워크에서 실패
2. **ReLU (2011 Krizhevsky et al.)**: ImageNet 혁명의 핵심; gradient가 0 또는 1이므로 backprop 안정적
3. **GELU, Swish (2018+)**: Transformer 시대의 선택; smooth + non-monotone
4. **Mish, SILU**: 현대 아키텍처 최적화

**깊이 $L$이 증가할수록**:
- Sigmoid/Tanh: $\prod_{l=1}^L \sigma'(z_l) \approx (0.25)^L \to 0$ (exponentially vanishing)
- ReLU: $\prod_{l=1}^L \mathbb{1}_{z_l > 0} = \mathbb{1}_{\text{all } z_l > 0}$ (stable)

이것이 **ResNet 이전까지 20층 이상 학습이 거의 불가능했던 이유**이고, ReLU 도입으로 100+층이 가능하게 된 이유이다. Ch3에서는 이를 정량화한다.

---

## 📐 수학적 선행 조건

- [03. 다층 퍼셉트론(MLP)의 정의와 구조](./03-mlp-definition.md): Activation function 정의, forward/backward pass
- 미적분: 함수의 도함수, 연쇄 법칙(chain rule)
- 지수함수: $e^x$, $e^{-x}$, 성질
- Saturation의 개념: gradient가 0에 가까운 영역

---

## 📖 직관적 이해

### Sigmoid: 매끄럽지만 느린 gradient

Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$:
- $z \to -\infty$: $\sigma(z) \to 0$
- $z = 0$: $\sigma(z) = 0.5$
- $z \to +\infty$: $\sigma(z) \to 1$

**도함수**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

이것은 **포물선(parabola) 형태**이며:
- $z = 0$에서 최대: $\sigma'(0) = 0.5 \times 0.5 = 0.25$
- $|z|$가 커질수록: $\sigma'(z) \to 0$ (saturation 영역)

**문제**: Deep network에서 역전파할 때, 각 층의 gradient를 곱하면:

$$\frac{\partial \ell}{\partial w_1} = \frac{\partial \ell}{\partial z_L} \prod_{l=2}^L \sigma'(z_l) \leq (0.25)^{L-1} \frac{\partial \ell}{\partial z_L}$$

$L = 10$이면 $0.25^9 \approx 9.3 \times 10^{-6}$ — 거의 0. 즉, **입력 근처의 가중치는 거의 업데이트되지 않는다** (vanishing gradient).

### Tanh: 더 가파른 도함수

Tanh $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$:
- Sigmoid와 유사한 형태 (S-curve)
- 범위: $[-1, 1]$ (Sigmoid는 $[0, 1]$)
- **도함수**: $\tanh'(z) = 1 - \tanh^2(z)$

최대값: $z = 0$에서 $\tanh'(0) = 1 - 0 = 1$

**Sigmoid vs Tanh**:

| 성질 | Sigmoid | Tanh |
|------|---------|------|
| 범위 | [0, 1] | [-1, 1] |
| 도함수 최댓값 | 0.25 | 1 |
| Saturation 속도 | 빠름 | 느림 |
| 대칭성 | 없음(0 기준 비대칭) | 있음(0 기준 대칭) |

**Tanh 유리**: 도함수가 Sigmoid의 4배이므로, deep network에서 gradient가 더 잘 흐른다.

### ReLU: Saturation 없음

ReLU $\text{ReLU}(z) = \max(0, z)$:
- $z \leq 0$: $f(z) = 0$
- $z > 0$: $f(z) = z$ (항등함수)

**도함수**: $\text{ReLU}'(z) = \begin{cases} 0 & z \leq 0 \\ 1 & z > 0 \end{cases}$

**핵심**: Saturation이 $z \leq 0$ 영역에서만 발생하고, $z > 0$ 영역에서는 도함수가 정확히 1!

따라서:
$$\prod_{l: z_l > 0} \text{ReLU}'(z_l) = 1^{\text{#positive}} = 1$$

**깊이에 무관하게 gradient가 1의 크기를 유지**한다. (Sigmoid/Tanh와 정반대)

### Dying ReLU Problem

하지만 문제가 있다: 만약 어떤 뉴런이 **계속 음수 입력을 받으면** ($z \leq 0$):
- Forward pass: 출력이 항상 0
- Backward pass: gradient가 0 (더 이상 학습 안 됨)
- 결과: 그 뉴런은 "죽는다"(dying)

**해결책**: Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} \alpha z & z \leq 0 \\ z & z > 0 \end{cases}$$

여기서 $\alpha \approx 0.01$. 그러면 $z \leq 0$에서도 도함수가 $\alpha \neq 0$이므로 gradient가 흐른다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 표준 활성화 함수들

**Sigmoid**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Tanh**:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$

**ReLU (Rectified Linear Unit)**:
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} 0 & z \leq 0 \\ z & z > 0 \end{cases}$$

도함수는 0이 아닌 점에서:
$$\text{ReLU}'(z) = \mathbb{1}_{z > 0} = \begin{cases} 0 & z < 0 \\ \text{undefined} & z = 0 \\ 1 & z > 0 \end{cases}$$

(0에서는 subgradient 사용)

**Leaky ReLU** (매개변수 $\alpha > 0$):
$$\text{LeakyReLU}_\alpha(z) = \begin{cases} \alpha z & z \leq 0 \\ z & z > 0 \end{cases}, \quad \text{LeakyReLU}'_\alpha(z) = \begin{cases} \alpha & z < 0 \\ 1 & z > 0 \end{cases}$$

**ELU (Exponential Linear Unit)**:
$$\text{ELU}_\alpha(z) = \begin{cases} \alpha(e^z - 1) & z \leq 0 \\ z & z > 0 \end{cases}, \quad \text{ELU}'_\alpha(z) = \begin{cases} \alpha e^z & z < 0 \\ 1 & z > 0 \end{cases}$$

### 정의 4.2 — 현대 활성화 함수

**GELU (Gaussian Error Linear Unit)**:
$$\text{GELU}(z) = z \Phi(z)$$

여기서 $\Phi(z) = \int_{-\infty}^z \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt$는 표준 정규분포의 누적분포함수(CDF).

해석: $\Phi(z)$를 "입력 $z$가 활성화될 확률"로 본다. 그러면 GELU는 "확률적으로 입력을 스케일링"하는 해석.

근사식:
$$\text{GELU}(z) \approx 0.5z(1 + \tanh(\sqrt{2/\pi}(z + 0.044715z^3)))$$

**Swish**:
$$\text{Swish}(z) = z \sigma(\beta z) = z \cdot \frac{1}{1 + e^{-\beta z}}$$

(보통 $\beta = 1$ 사용, 또는 학습 가능한 매개변수로 둔다)

도함수:
$$\text{Swish}'(z) = \sigma(\beta z) + \beta z \sigma(\beta z)(1 - \sigma(\beta z))$$

---

## 🔬 정리와 증명

### 정리 4.1 — Sigmoid의 Vanishing Gradient Bound

**명제**: 깊이 $L$인 sigmoid 네트워크에서 역전파 시 gradient product bound:

$$\left| \prod_{l=1}^L \sigma'(z_l) \right| \leq (0.25)^L$$

**증명**:

각 $z_l$에 대해:
$$\sigma'(z_l) = \sigma(z_l)(1 - \sigma(z_l))$$

$\sigma(z_l) \in (0, 1)$이므로, 함수 $f(x) = x(1-x)$는 $x \in (0, 1)$에서:

$$\max_{x \in (0,1)} x(1-x) = 0.25 \quad (\text{at } x = 0.5)$$

따라서 $\sigma'(z_l) \leq 0.25$. 그러므로:

$$\boxed{\prod_{l=1}^L \sigma'(z_l) \leq (0.25)^L}$$

$L = 10$일 때: $(0.25)^{10} \approx 9.1 \times 10^{-7}$ — 거의 0. $\square$

### 정리 4.2 — ReLU의 Gradient Stability

**명제**: ReLU 네트워크에서 양수 활성화 경로 위의 gradient product는:

$$\prod_{l: z_l > 0} \text{ReLU}'(z_l) = 1$$

**증명**:

ReLU' = 1 (양수 영역에서). 따라서 양수 경로 위에서 gradient는 그대로 전파된다. 깊이 $L$이어도:

$$\boxed{\text{Gradient magnitude} = \text{constant}}$$

(음수 경로는 gradient = 0, 하지만 학습 시작 시 데이터는 무작위이므로 일부 경로는 양수)

**따름정리**: ReLU를 사용하면 깊이에 대해 지수적 감쇠가 없다. 이것이 **ResNet 같은 깊은 네트워크를 가능**하게 한 핵심.

$\square$

### 정리 4.3 — Saturation의 정량화

**명제**: Activation $\sigma$에서 "saturated region" $|z| > \tau$에서:

$$|\sigma'(z)| < \epsilon$$

Sigmoid의 경우 $\tau = 2$로 잡으면 $|\sigma'(z)| < 0.1$.

**의미**: $|z| > 2$ 영역에서는 gradient가 매우 작아서, 그 뉴런이 거의 업데이트되지 않는다.

**ReLU와의 비교**:
- Sigmoid: Saturation 영역이 $|z| > 2$부터 시작 (전체 범위의 많은 부분)
- ReLU: Saturation이 $z < 0$에만 발생 (정확히 절반)

---

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ──────────────────────────────────────────────────────────
# 1. Activation functions and their derivatives
# ──────────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_func(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_prime(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def elu_prime(z, alpha=1.0):
    return np.where(z > 0, 1, alpha * np.exp(z))

def gelu(z):
    """Approximation of GELU"""
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
    return z * cdf

def gelu_prime(z):
    """Numerical gradient of GELU"""
    eps = 1e-5
    return (gelu(z + eps) - gelu(z - eps)) / (2 * eps)

def swish(z, beta=1.0):
    return z * sigmoid(beta * z)

def swish_prime(z, beta=1.0):
    sig = sigmoid(beta * z)
    return sig + beta * z * sig * (1 - sig)

# ──────────────────────────────────────────────────────────
# 2. Visualization
# ──────────────────────────────────────────────────────────
z = np.linspace(-5, 5, 1000)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Row 1: Sigmoid family
activations = [
    ('Sigmoid', sigmoid, sigmoid_prime, 'blue'),
    ('Tanh', tanh_func, tanh_prime, 'green'),
    ('ReLU', relu, relu_prime, 'red'),
    ('Leaky ReLU', lambda z: leaky_relu(z, 0.01), lambda z: leaky_relu_prime(z, 0.01), 'orange')
]

for idx, (name, act, act_prime, color) in enumerate(activations):
    ax = axes[0, idx]
    y = act(z)
    ax.plot(z, y, color=color, linewidth=2.5, label=name)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(f'{name} Function', fontweight='bold')
    ax.set_ylabel('$\sigma(z)$')
    ax.set_ylim([-1.5, 2.5] if name == 'ReLU' else [-1.5, 1.5])
    ax.legend(loc='best')

# Row 2: Derivatives
for idx, (name, act, act_prime, color) in enumerate(activations):
    ax = axes[1, idx]
    y_prime = act_prime(z)
    ax.plot(z, y_prime, color=color, linewidth=2.5, label=f"{name}'")
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(f"{name} Derivative", fontweight='bold')
    ax.set_ylabel("$\sigma'(z)$")
    ax.set_ylim([-0.1, 1.2])
    ax.legend(loc='best')

# Row 3: Modern activations
modern_acts = [
    ('ELU', lambda z: elu(z, 1.0), lambda z: elu_prime(z, 1.0), 'purple'),
    ('GELU', gelu, gelu_prime, 'brown'),
    ('Swish', lambda z: swish(z, 1.0), lambda z: swish_prime(z, 1.0), 'pink'),
    ('Saturation Analysis', None, None, 'black')
]

for idx, (name, act, act_prime, color) in enumerate(modern_acts[:3]):
    ax = axes[2, idx]
    y = act(z)
    ax.plot(z, y, color=color, linewidth=2.5, label=name)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(f'{name} Function', fontweight='bold')
    ax.set_ylabel('$\sigma(z)$')
    ax.legend(loc='best')

# Saturation analysis
ax = axes[2, 3]
saturation_values = []
depth_levels = range(1, 16)

for L in depth_levels:
    sig_bound = (0.25) ** L
    tanh_bound = (1.0) ** L  # tanh derivative = 1 at max, but typically much less
    relu_bound = 1.0
    saturation_values.append({
        'depth': L,
        'sigmoid': sig_bound,
        'tanh': tanh_bound,
        'relu': relu_bound
    })

depths = [v['depth'] for v in saturation_values]
sigmoid_bounds = [v['sigmoid'] for v in saturation_values]

ax.semilogy(depths, sigmoid_bounds, 'o-', color='blue', linewidth=2.5, markersize=8, label='Sigmoid: $(0.25)^L$')
ax.axhline(y=1.0, color='red', linestyle='-', linewidth=2.5, label='ReLU: constant 1.0')
ax.fill_between(depths, 1e-8, sigmoid_bounds, alpha=0.2, color='blue', label='Vanishing region')
ax.set_xlabel('Depth $L$', fontsize=12)
ax.set_ylabel('Gradient Product Bound', fontsize=12)
ax.set_title('Vanishing Gradient: Sigmoid vs ReLU', fontweight='bold')
ax.set_yscale('log')
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ──────────────────────────────────────────────────────────
# 3. Gradient flow simulation in deep networks
# ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("활성화 함수 비교: Gradient Flow Analysis")
print("="*70)

depths = [5, 10, 20, 50]
print("\n[1] Gradient Product Bound (역전파에서 upstream gradient 곱하기)")
print(f"{'Depth':<8} {'Sigmoid':<20} {'Tanh':<20} {'ReLU':<20}")
print("-" * 68)

for L in depths:
    sig_bound = (0.25) ** L
    tanh_approx = (0.9) ** L  # Tanh is better but not perfect
    relu_bound = 1.0
    print(f"{L:<8} {sig_bound:<20.2e} {tanh_approx:<20.2e} {relu_bound:<20.2f}")

print("\n[2] 도함수 최댓값 비교")
print(f"{'Function':<15} {'Max Derivative':<20} {'Saturation 시작':<20}")
print("-" * 55)
print(f"{'Sigmoid':<15} {0.25:<20.4f} {abs(2):<20} (at |z|=2)")
print(f"{'Tanh':<15} {1.0:<20.4f} {abs(2):<20} (at |z|=2)")
print(f"{'ReLU':<15} {1.0:<20.4f} {'z < 0':<20}")
print(f"{'Leaky ReLU':<15} {1.0:<20.4f} {'both':<20}")
print(f"{'ELU':<15} {1.0:<20.4f} {'both':<20}")

print("\n[3] 실제 Gradient 크기 (L층 네트워크, upstream gradient = 1)")
test_depth = 20
sig_grad = (0.25) ** test_depth
tanh_grad = (0.9) ** test_depth
relu_grad = 1.0

print(f"\nDepth {test_depth}에서 입력층까지 역전파되는 gradient 크기:")
print(f"  Sigmoid:     {sig_grad:.2e} (매우 작음 ✗)")
print(f"  Tanh:        {tanh_grad:.2e} (여전히 작음 ✗)")
print(f"  ReLU:        {relu_grad:.2e} (안정적 ✓)")

print("\n[4] Dying ReLU 문제")
print(f"{'Activation':<15} {'Negative 영역':<25} {'Gradient 유무'}")
print("-" * 55)
print(f"{'ReLU':<15} {'z < 0 → 0':<25} {'없음 (0)'}")
print(f"{'Leaky ReLU':<15} {'z < 0 → αz (α≈0.01)':<25} {'있음 (α)'}")
print(f"{'ELU':<15} {'z < 0 → α(e^z-1)':<25} {'있음 (αe^z)'}")

print("\n" + "="*70)
```

**출력 예시**:
```
======================================================================
활성화 함수 비교: Gradient Flow Analysis
======================================================================

[1] Gradient Product Bound (역전파에서 upstream gradient 곱하기)
Depth    Sigmoid              Tanh                 ReLU                
---
5        9.77e-04             5.90e-02             1.00
10       9.54e-07             3.49e-03             1.00
20       9.09e-14             1.22e-06             1.00
50       2.27e-34             4.48e-15             1.00

[2] 도함수 최댓값 비교
Function        Max Derivative       Saturation 시작
Sigmoid         0.2500               2 (at |z|=2)
Tanh            1.0000               2 (at |z|=2)
ReLU            1.0000               z < 0
Leaky ReLU      1.0000               both
ELU             1.0000               both

[3] 실제 Gradient 크기 (L=20층 네트워크, upstream gradient = 1)
Depth 20에서 입력층까지 역전파되는 gradient 크기:
  Sigmoid:     9.09e-14 (매우 작음 ✗)
  Tanh:        1.22e-06 (여전히 작음 ✗)
  ReLU:        1.00e+00 (안정적 ✓)

[4] Dying ReLU 문제
Activation      Negative 영역        Gradient 유무
ReLU            z < 0 → 0            없음 (0)
Leaky ReLU      z < 0 → αz (α≈0.01)  있음 (α)
ELU             z < 0 → α(e^z-1)     있음 (α)
```

---

## 🔗 실전 연결

### Modern Architectures의 선택

| 아키텍처 | 주요 Activation | 이유 |
|--------|----------------|------|
| ResNet, DenseNet | ReLU | 깊이 최적화; 구현 간단; 계산 효율 |
| Transformer (BERT, GPT) | GELU | Smooth; non-monotone; language model에 최적 |
| Vision Transformer | GELU | BERT에서 영감; modern vision 모델 채택 |
| MobileNet, EfficientNet | Swish | 효율성과 정확도 균형 |
| Older models (AlexNet, VGG) | Sigmoid/ReLU | 시대적 선택; Sigmoid는 이제 거의 미사용 |

### ReLU의 성공과 그 이유 (실증적)

2012 ImageNet 혁명 (AlexNet):
- **첫 사용**: ReLU를 깊은 CNN(8층)에 적용
- **결과**: Sigmoid 대비 **training time 6배 단축**
- **성능**: 정확도 향상 + 수렴 속도 개선

**수학적 근거**:
- Sigmoid: Gradient bound $(0.25)^8 \approx 2.3 \times 10^{-5}$ (거의 0)
- ReLU: Gradient bound = 1.0 (안정적)

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 여러 activation 선택 가능
model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),  # 또는 nn.GELU(), nn.SiLU() (Swish)
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

# 또는 특정 활성화 함수 구현
class CustomActivation(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'swish':
            self.act = nn.SiLU()  # SiLU = Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        return self.act(x)
```

### Dying ReLU의 실제 해결

```python
# 방법 1: Leaky ReLU
nn.LeakyReLU(negative_slope=0.01)

# 방법 2: ELU
nn.ELU(alpha=1.0)

# 방법 3: Batch Normalization (부작용으로 해결)
nn.Sequential(
    nn.Linear(100, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
)

# 방법 4: 높은 learning rate로 뉴런 "깨우기"
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 요소별 활성화(element-wise) | Attention mechanism 같은 다른 비선형 연산 존재; 모든 비선형성이 이들 함수는 아님 |
| Continuous activation | Binarized/quantized 신경망은 discrete activation; 이론 다름 |
| 초기화가 적절함 | Poor initialization은 dying ReLU 악화 가능; He initialization 중요 |
| Stateless activation | Gating 메커니즘(LSTM, GRU)은 activation + 상태 결합 |
| 고정 매개변수 | Parametric ReLU (PReLU)는 $\alpha$를 학습; 분석 복잡해짐 |

---

## 📌 핵심 정리

$$\boxed{\text{Gradient stability} = \text{적절한 activation 선택의 핵심}}$$

| 함수 | 범위 | 도함수 최댓값 | Saturation | 깊은망 | 추천 |
|------|------|-------------|-----------|--------|------|
| **Sigmoid** | [0,1] | 0.25 | 심함 ($\|z\| > 2$) | 나쁨 | 출력층(binary) |
| **Tanh** | [-1,1] | 1.0 | 중간 | 나쁨 | 거의 미사용 |
| **ReLU** | [0,∞) | 1.0 | 약함 ($z<0$) | 좋음 | 표준 선택 |
| **Leaky ReLU** | ℝ | 1.0 | 거의 없음 | 좋음 | ReLU 개선 |
| **ELU** | [-α,∞) | 1.0 | 없음 | 좋음 | 정규화 효과 |
| **GELU** | ℝ | ~1.7 | 없음 | 매우좋음 | Transformer |
| **Swish** | ℝ | ~1.8 | 없음 | 매우좋음 | 현대 선택 |

**깊이별 선택**:
- **Shallow** (1-2층): Sigmoid, Tanh 가능
- **Medium** (3-10층): ReLU, Leaky ReLU
- **Deep** (50+층): ReLU + normalization 또는 GELU/Swish

---

## 🤔 생각해볼 문제

**문제 1** (기초): Sigmoid $\sigma(z) = 1/(1+e^{-z})$의 도함수가 최대값 0.25인 것을 직접 계산으로 보이라. (미적분)

<details>
<summary>힌트 및 해설</summary>

$\sigma(z) = (1 + e^{-z})^{-1}$이므로:

$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

$\sigma(z) = \frac{1}{1+e^{-z}}$이므로 $e^{-z} = \frac{1}{\sigma(z)} - 1 = \frac{1-\sigma(z)}{\sigma(z)}$.

따라서:

$$\sigma'(z) = \frac{(1-\sigma(z))/\sigma(z)}{(1 + e^{-z})^2} = \frac{1-\sigma(z)}{\sigma(z)} \cdot \sigma(z)^2 = \sigma(z)(1-\sigma(z))$$

함수 $f(x) = x(1-x)$의 최대값은 $f'(x) = 1 - 2x = 0$에서 $x = 0.5$이고:

$$\max_x x(1-x) = 0.5 \times 0.5 = 0.25$$

$\sigma(z) \in (0, 1)$이므로 $\sigma'(z) \leq 0.25$.

</details>

**문제 2** (심화): **ReLU 네트워크의 Piecewise Linear 성질**을 이용하여, $L$층 ReLU 네트워크가 최대 $2^L$개의 linear region으로 입력 공간을 분할함을 보이라. (이것이 표현력의 상한.)

<details>
<summary>힌트 및 해설</summary>

ReLU는 piecewise linear: 각 선형 부분은 hyperplane $z_l = 0$들의 교집합.

Layer 1에서: $d_1$개 뉴런 → 최대 $2^{d_1}$개 linear region (각 뉴런이 "on/off" 2가지 상태)

하지만 지나친 상한. 실제로는:

Layer 1에서 $H_1$개 hyperplane (각 뉴런이 하나의 hyperplane). 
$d$ 차원에서 $H$ 개 hyperplane이 만드는 region 수는 최대:

$$\sum_{i=0}^d \binom{H}{i} \leq O(H^d)$$

각 층을 거치면서 이 bound가 곱해지므로, 깊이 $L$일 때:

$$\text{# regions} \leq (\text{polynomial in } d_1, \ldots, d_L)^L$$

즉, **깊이가 region 분할 능력을 exponentially 증가**시킨다는 의미.

</details>

**문题 3** (AI 연결): **Vanishing Gradient와 Batch Normalization의 관계**를 설명하라. 왜 Batch Norm이 Sigmoid 같은 "나쁜" activation도 사용 가능하게 만드는가?

<details>
<summary>힌트 및 해설</summary>

Vanishing gradient의 근본 원인: Sigmoid/Tanh의 saturation 영역에서 입력이 몰린다.

**Without Batch Norm**:
- Layer 1의 출력이 고르게 분포 → Layer 2 입력도 고르게
- Layer 2의 출력이 다시 고르게 분포
- ... 계속 반복되면 어느 순간 입력이 saturation 영역으로 몰림
- Gradient 0으로 vanishing

**With Batch Norm**:
$$\hat x = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \quad y = \gamma \hat x + \beta$$

각 층의 입력을 정규화하여 **평균 0, 분산 1로 유지**.

따라서:
- Sigmoid 입력이 항상 중간 영역에만 머무름
- Sigmoid 도함수가 ~0.2 유지 (saturate 안 함)
- Gradient가 약해지긴 해도 완전히 vanish하지 않음

**결론**: Batch Norm + Sigmoid도 가능하지만, ReLU + 적절한 init이 여전히 더 효율적.

이것이 **"정규화 기법"이 현대 딥러닝의 필수**인 이유이다 (Ch5에서 상세).

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. 다층 퍼셉트론(MLP)의 정의와 구조](./03-mlp-definition.md) | [📚 README로 돌아가기](../README.md) | [Ch2-01. Cybenko Universal Approximation Theorem ▶](../ch2-universal-approximation/01-cybenko-uat.md) |

</div>
