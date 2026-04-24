# 03. ReLU Network의 UAT (Leshno et al. 1993)

## 🎯 핵심 질문

ReLU는 비유계이면서도 조각적으로 선형입니다. 이렇게 "단순한" 함수가 정말 모든 연속함수를 근사할 수 있을까?

---

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

ReLU는 현대 딥러닝의 표준 활성화 함수입니다. Leshno et al. (1993)의 증명은:

- **현대성**: ReLU가 이론적 정당성을 가짐을 보증
- **실무 정합성**: 가장 널리 쓰이는 활성화가 수학적으로도 보편적임
- **구성성**: Cybenko나 Hornik과 달리, 명시적인 구성 방법 제시
- **효율성 시작**: ReLU로 "bump" 함수를 만드는 방법 → 깊이의 중요성으로 연결

---

## 📐 수학적 선행 조건

1. **조각적 선형(Piecewise Linear) 함수**: 유한 개의 선형 조각으로 이루어진 함수
2. **Bump 함수**: 유계 구간에서만 0이 아닌 함수
3. **근사 이론**: 조각적 선형이 연속함수에 dense
4. **ReLU의 기본**: $\text{ReLU}(z) = \max(0, z)$
5. **Ch1-04**: 활성화 함수로서의 ReLU 성질

---

## 📖 직관적 이해

### ReLU의 강점

ReLU는 "2개의 선형 영역"으로 나뉩니다:
$$\text{ReLU}(z) = \begin{cases} 0 & z < 0 \\ z & z \geq 0 \end{cases}$$

이 단순함이 오히려 **매우 유연한 함수 표현**을 가능하게 합니다.

### Bump 함수 구성의 핵심

**1차원에서:**

$$\text{ReLU}(z) - 2 \cdot \text{ReLU}(z - \epsilon) + \text{ReLU}(z - 2\epsilon) \approx \text{spike at } z = \epsilon$$

여러 spike를 조합하면 어떤 함수도 근사 가능합니다.

**다차원으로 확장:**

각 차원의 spike를 tensor product 형태로 조합하면, 임의의 상자(box) 위에서만 non-zero인 bump 함수 구성 가능.

---

## ✏️ 엄밀한 정의

**정의 3.1** (Piecewise Linear 함수)

함수 $f: \mathbb{R}^n \to \mathbb{R}$이 **piecewise linear**이라는 것은, $\mathbb{R}^n$을 유한 개의 convex polytope로 분할하고, 각 polytope 위에서 $f$가 선형함수 $f(x) = a \cdot x + b$라는 의미입니다.

---

**정의 3.2** (Bump 함수)

함수 $\phi: \mathbb{R}^n \to \mathbb{R}$이 **bump 함수**라는 것은, 어떤 컴팩트 집합 $K$가 존재하여 $\mathbb{R}^n \setminus K$에서 $\phi \equiv 0$이고, $K$ 내부에서는 0이 아닌 부분이 있음을 의미합니다.

---

**정의 3.3** (ReLU 신경망)

ReLU 활성화를 사용한 1층 신경망:

$$f_N(x) = \sum_{i=1}^N \alpha_i \text{ReLU}(w_i \cdot x + b_i)$$

---

## 🔬 정리와 증명

**정리 3.1** (Leshno et al., 1993)

ReLU 활성화를 사용한 1층 신경망의 함수 집합은 컴팩트 $K \subset \mathbb{R}^n$ 위의 연속함수 공간 $C(K)$에서 uniform dense이다.

더 일반적으로: **모든 non-polynomial 활성화 함수**는 UAT를 만족한다.

---

### 증명 스케치 (구성적 접근)

**Step 1**: 연속함수를 조각적 선형으로 근사

Weierstrass 근사 정리의 확장으로, 임의의 $f \in C(K)$와 $\epsilon > 0$에 대해, piecewise linear 함수 $p$가 존재하여 $\|f - p\|_\infty < \epsilon$.

**Step 2**: Piecewise linear 함수를 ReLU로 표현

Piecewise linear 함수는 ReLU의 선형결합으로 정확히 표현 가능합니다.

*핵심 보조정리*: 구간 $[a, b]$에서 "tent" 함수
$$\text{tent}(x; a, b) = \max(0, x - a) - 2 \max(0, x - (a+b)/2) + \max(0, x - b)$$

를 생각하면, 이는 ReLU 3개의 조합입니다.

**Step 3**: 1D에서의 bump 함수

1차원에서, 구간 $[a, b]$ 위의 bump:
$$\text{bump}_1(x; a, b) = \text{ReLU}(x - a) - 2 \text{ReLU}(x - b) + \text{ReLU}(x - (a+2b)/2)$$

더 정확하게, 차분(difference)을 이용하면:
$$\text{bump}_\epsilon(x) = \frac{1}{\epsilon}[\text{ReLU}(x) - \text{ReLU}(x-\epsilon)]$$

는 $[0, \epsilon]$ 구간에 집중된 높이 1의 근사적 bump 함수입니다.

**Step 4**: 다차원으로 확장

$n$차원의 bump 함수는 1차원 bump의 텐서곱으로 구성:

$$\text{Bump}_{n}(x_1, \ldots, x_n) = \prod_{i=1}^n \text{bump}_1(x_i; a_i, b_i)$$

이는 상자 $[a_1, b_1] \times \cdots \times [a_n, b_n]$ 위에서만 값을 가집니다.

**Step 5**: Piecewise linear를 bump의 선형결합으로 표현

컴팩트 $K$를 작은 상자들로 분할하고, 각 상자 위의 선형함수를 대응 bump 함수에 곱한 후 합산합니다.

(기술적으로는 각 polytope 내부의 선형함수 값 $f_i(x)$에 대해 $f_i(x) \cdot \text{Bump}_i(x)$ 형태)

**Step 6**: 임의의 정확도로 근사

적절한 개수의 ReLU 뉴런(bump 함수)을 사용하면, 임의의 piecewise linear 함수를 정확히 표현할 수 있고, 따라서 임의의 연속함수를 임의의 정확도로 근사할 수 있습니다. $\square$

---

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

def create_bump_1d(x, center, width, height=1.0):
    """1D bump 함수 근사 (ReLU 기반)"""
    # Tent 형태: center-width에서 0, center에서 peak, center+width에서 0
    left = relu(x - (center - width))
    right = relu((center + width) - x)
    return height * (left + right - 2 * width) / (2 * width)

def relu_bump_function(x, center, width):
    """ReLU로 만든 bump 함수 (정확)"""
    # height = 1 / width 정규화
    eps = width / 100  # 작은 구간
    
    # ReLU 차분
    bump = relu(x - center + width) - relu(x - center - width)
    return bump / (2 * width)

def create_piecewise_linear_nn(x, breakpoints, values):
    """
    Piecewise linear 함수를 ReLU NN으로 표현
    breakpoints: 함수의 꺾이는 점들 (정렬됨)
    values: 각 breakpoint에서의 함수값
    """
    n_pieces = len(breakpoints) - 1
    weights = []
    biases = []
    alphas = []
    
    # 각 조각에서의 기울기 계산
    for i in range(n_pieces):
        x1, x2 = breakpoints[i], breakpoints[i+1]
        y1, y2 = values[i], values[i+1]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        intercept = y1 - slope * x1
        
        # ReLU(w·x + b)에서 w, b 설정
        w = 1.0
        b = -x1
        alpha = slope
        
        weights.append([w])
        biases.append(b)
        alphas.append(alpha)
    
    # 합산
    y_pred = np.zeros_like(x, dtype=float)
    for w, b, alpha in zip(weights, biases, alphas):
        z = x * w[0] + b
        y_pred += alpha * relu(z)
    
    return y_pred

# === 실험 1: 1D Bump 함수 생성 ===
print("=" * 60)
print("실험 1: ReLU로 만든 Bump 함수")
print("=" * 60)

x = np.linspace(-3, 3, 500)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 단일 bump
ax = axes[0, 0]
bump = relu_bump_function(x, center=0, width=1.0)
ax.fill_between(x, bump, alpha=0.5, color='blue', label='Bump (center=0)')
ax.axvline(-1, color='red', linestyle='--', alpha=0.5, label='Support [-1,1]')
ax.axvline(1, color='red', linestyle='--', alpha=0.5)
ax.set_title('Single Bump Function', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('ReLU-based Bump')
ax.legend()
ax.grid(True, alpha=0.3)

# 여러 bump
ax = axes[0, 1]
centers = [-1.5, 0, 1.5]
colors = ['red', 'blue', 'green']
for center, color in zip(centers, colors):
    bump_i = relu_bump_function(x, center=center, width=0.8)
    ax.plot(x, bump_i, color=color, label=f'center={center}', linewidth=2)
ax.set_title('Multiple Bumps', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('Sum')
ax.legend()
ax.grid(True, alpha=0.3)

combined = sum(relu_bump_function(x, center=c, width=0.8) for c in centers)
ax = axes[1, 0]
ax.fill_between(x, combined, alpha=0.5, color='purple')
ax.plot(x, combined, 'k-', linewidth=2)
ax.set_title('Sum of Bumps = Piecewise Constant', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('Combined')
ax.grid(True, alpha=0.3)

# Piecewise linear 근사
ax = axes[1, 1]
breakpoints = np.array([-2, -1, 0, 1, 2])
values = np.array([0, 1, 0.5, 1.5, 0])

y_piecewise = create_piecewise_linear_nn(x, breakpoints, values)

ax.plot(x, y_piecewise, 'b-', linewidth=2.5, label='PieceWise Linear (ReLU NN)')
ax.plot(breakpoints, values, 'ro', markersize=8, label='Breakpoints')
for bp in breakpoints:
    ax.axvline(bp, color='gray', linestyle=':', alpha=0.3)
ax.set_title('Piecewise Linear via ReLU', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/leshno_bump_functions.png', dpi=100, bbox_inches='tight')
print("✓ Graph saved: /tmp/leshno_bump_functions.png")
plt.close()

# === 실험 2: Arbitrary 함수의 ReLU 근사 ===
print("\n" + "=" * 60)
print("실험 2: ReLU NN으로 임의 함수 근사")
print("=" * 60)

# 목표 함수들
x_fine = np.linspace(-3, 3, 800)

target_functions = {
    'sin(x)': np.sin(x_fine),
    'exp(-x²)': np.exp(-x_fine**2),
    'abs(x)': np.abs(x_fine),
    'step(x)': np.where(x_fine > 0, 1.0, 0.0)
}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for idx, (name, y_target) in enumerate(target_functions.items()):
    # Piecewise linear 근사를 만들기 위해 샘플링
    n_samples = 50
    x_sample = np.linspace(-3, 3, n_samples)
    y_sample = y_target[::len(x_fine)//n_samples][:n_samples]
    
    # ReLU NN으로 표현
    y_relu = create_piecewise_linear_nn(x_fine, x_sample, y_sample)
    
    error = np.mean((y_relu - y_target) ** 2)
    
    ax = axes[idx]
    ax.plot(x_fine, y_target, 'b-', label='Target', linewidth=2, alpha=0.8)
    ax.plot(x_fine, y_relu, 'r--', label='ReLU approx', linewidth=1.5, alpha=0.8)
    ax.fill_between(x_fine, y_target, y_relu, alpha=0.2, color='orange')
    ax.set_title(f'{name}\nMSE = {error:.2e}', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/leshno_relu_approximation.png', dpi=100, bbox_inches='tight')
print("✓ Graph saved: /tmp/leshno_relu_approximation.png")
plt.close()

print("\nReLU는 piecewise linear 함수들을 정확히 표현하고,")
print("따라서 연속함수에 대해 uniform dense입니다!")

# === 실험 3: 깊이 vs 너비 (간단한 예시) ===
print("\n" + "=" * 60)
print("실험 3: ReLU의 표현력 (깊이 영향)")
print("=" * 60)

# Sawtooth 함수 (나중 Ch2-04의 preview)
def sawtooth(x, n_periods=3):
    """톱니파 함수"""
    return np.abs(np.sin(n_periods * np.pi * x))

x_saw = np.linspace(0, 1, 500)
y_saw = sawtooth(x_saw, n_periods=4)

# 다양한 너비의 근사
widths = [5, 10, 20, 50]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, width in enumerate(widths):
    # Piecewise linear 근사
    x_samples = np.linspace(0, 1, width)
    y_samples = sawtooth(x_samples, n_periods=4)
    
    y_approx = create_piecewise_linear_nn(x_saw, x_samples, y_samples)
    error = np.mean((y_approx - y_saw) ** 2)
    
    ax = axes[idx]
    ax.plot(x_saw, y_saw, 'b-', label='Sawtooth', linewidth=2)
    ax.plot(x_saw, y_approx, 'r--', label=f'ReLU ({width} units)', linewidth=1.5)
    ax.fill_between(x_saw, y_saw, y_approx, alpha=0.2)
    ax.set_title(f'{width} neurons, MSE={error:.2e}', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/leshno_width_convergence.png', dpi=100, bbox_inches='tight')
print("✓ Graph saved: /tmp/leshno_width_convergence.png")
plt.close()

print("너비(뉴런 개수)가 증가하면 더 세밀한 근사가 가능합니다.")
```

**출력:**

```
============================================================
실험 1: ReLU로 만든 Bump 함수
============================================================
✓ Graph saved: /tmp/leshno_bump_functions.png

============================================================
실험 2: ReLU NN으로 임의 함수 근사
============================================================
✓ Graph saved: /tmp/leshno_relu_approximation.png

ReLU는 piecewise linear 함수들을 정확히 표현하고,
따라서 연속함수에 대해 uniform dense입니다!

============================================================
실험 3: ReLU의 표현력 (깊이 영향)
============================================================
✓ Graph saved: /tmp/leshno_width_convergence.png

너비(뉴런 개수)가 증가하면 더 세밀한 근사가 가능합니다.
```

---

## 🔗 실전 연결

- **현대 딥러닝**: ReLU는 구글, OpenAI, Meta 등 모두가 기본 활성화로 사용
- **구성성**: Cybenko/Hornik과 달리, 이 증명은 **실제로 어떻게 뉴런을 배치할지** 명시
- **깊이의 중요성**: ReLU는 비유계이므로, 깊은 네트워크에서 표현력의 지수적 증가 가능 (Ch2-04 Telgarsky)
- **실제 학습**: 경사하강법이 이 구성을 자동으로 찾는가? → 미해결 문제

---

## ⚖️ 가정과 한계

| 측면 | 설명 |
|------|------|
| **비유계** | ReLU는 위로 유계되지 않음 |
| **Piecewise Linear** | ReLU는 "꺾이는" 점에서만 불연속 미분 |
| **구성 방법** | Bump 함수 구성은 차원 증가 시 복잡해짐 |
| **너비의 저주** | Piecewise linear로의 근사에 필요한 조각 수 ∝ $1/\epsilon$ |
| **학습 보장 없음** | 구성은 이론적이고, 경사하강법이 찾을지는 불명확 |

---

## 📌 핵심 정리

$$\boxed{\text{ReLU는 비유계이면서도 모든 연속함수를 uniform 근사할 수 있다}}$$

**증명의 핵심 아이디어:**

1. 연속함수 → piecewise linear 근사 (Weierstrass)
2. Piecewise linear → ReLU의 선형결합 (구성적)
3. ReLU는 $\max(0, z)$ = 하나의 꺾이는 점 = 유연한 표현

| 활성화 | 유계 | UAT | 구성 | 실무 |
|-------|------|------|------|------|
| Sigmoid | ✅ | ✅ | ❌ | ⚠️ (포화) |
| Tanh | ✅ | ✅ | ❌ | ⚠️ |
| ReLU | ❌ | ✅ | ✅ | ✅ |
| Polynomial | ❌ | ❌ | ❌ | ❌ |

---

## 🤔 생각해볼 문제

**문제 1**: ReLU가 비유계인데 어떻게 compact domain에서 모든 함수를 근사할까?

<details>
<summary>💡 해설</summary>

Compact domain $K$ 위의 ReLU는 사실상 "선형"입니다. $K$를 벗어나지 않는 한, $\text{ReLU}(w \cdot x + b)$는 $w \cdot x + b \geq 0$인 영역에서만 선형으로 증가합니다. 따라서 $K$ 위에서의 제한(restriction)은 사실상 유계입니다.
</details>

**문제 2**: Bump 함수를 실제로 만들려면 몇 개의 ReLU가 필요한가?

<details>
<summary>💡 해설</summary>

1차원의 단일 bump는 약 3-4개의 ReLU로 근사 가능합니다. $n$차원이면 차원마다 곱해지므로 $3^n$ 정도. 따라서 고차원에서는 매우 비효율적입니다. 이것이 깊은 네트워크(깊이를 사용)가 필요한 이유 (Ch2-04).
</details>

**문제 3**: "Piecewise linear"에서 몇 개의 조각이 필요한가?

<details>
<summary>💡 해설</summary>

함수를 $\epsilon$ 오차 이내로 근사하려면, Weierstrass 근사에서 조각의 개수는 대략 $O(1/\epsilon)$입니다. 따라서 필요한 뉴런 수도 $O(1/\epsilon)$. 이는 polynomial 근사의 $O(1/\epsilon^d)$ (차원 $d$ 관련)보다는 훨씬 낫습니다.
</details>

---

## 📚 네비게이션

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Hornik의 일반화](./02-hornik-extension.md) | [📚 README로 돌아가기](../README.md) | [04. 깊이 vs 너비 ▶](./04-depth-vs-width.md) |

</div>

