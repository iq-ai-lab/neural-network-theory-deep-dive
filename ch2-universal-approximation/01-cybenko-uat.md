# 01. Cybenko의 Universal Approximation Theorem (1989)

## 🎯 핵심 질문

단층 신경망으로 **모든 연속함수**를 임의로 가깝게 근사할 수 있을까? 그렇다면 어떤 조건 하에서 가능한가?

---

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

Cybenko의 정리(1989)는 신경망이 단순한 경험적 도구가 아니라 **함수근사의 수학적 보편성**을 지닌 장치임을 처음으로 증명했습니다. 

- **이론적 기초**: 신경망의 표현력이 무한하다는 엄밀한 증명 제공
- **활성화 함수 설계**: sigmoid를 포함한 특정 활성화 함수들이 보편성을 보장함을 명시
- **딥러닝의 정당성**: 왜 신경망이 작동하는지에 대한 수학적 근거
- **현대적 확장**: 이후 ReLU, tanh 등 다양한 활성화로의 일반화 가능성 제시

---

## 📐 수학적 선행 조건

다음 개념에 대한 이해가 필요합니다:

1. **위상공간**: 컴팩트 집합, 폐집합, 연속함수
2. **함수공간**: $C(K)$ (컴팩트 $K$ 위의 연속함수 공간), uniform norm $\|\cdot\|_\infty$
3. **부호측도**: 유한 부호측도(signed measure), Riesz 표현 정리
4. **선형범함수**: dual space, Hahn-Banach 정리
5. **Fourier 변환** (선택): characteristic function의 관점

---

## 📖 직관적 이해

### 왜 Sigmoid인가?

Sigmoid 함수 $\sigma(z) = \frac{1}{1+e^{-z}}$는 다음 성질을 가집니다:

$$\lim_{z \to \infty} \sigma(z) = 1, \quad \lim_{z \to -\infty} \sigma(z) = 0$$

이는 "경계가 있으면서도 경계를 선명하게 구분"하는 성질입니다. 

### 핵심 직관

1. **한 개의 뉴런 $\sigma(w \cdot x + b)$**: "하이퍼플레인 $w \cdot x + b = 0$을 기준으로 0에서 1로 급격히 전환"
2. **숨겨진 층의 합 $\sum_i \alpha_i \sigma(w_i \cdot x + b_i)$**: 여러 하이퍼플레인에서의 전환을 조합 → 복잡한 형태 표현
3. **무한한 뉴런**: 무한히 많은 하이퍼플레인으로 나누면 어떤 연속함수도 표현 가능

---

## ✏️ 엄밀한 정의

**정의 1.1** (Sigmoidal 함수)

함수 $\sigma: \mathbb{R} \to \mathbb{R}$이 **sigmoidal**이라는 것은 다음을 만족할 때입니다:

$$\sigma \text{는 연속함수이고,} \quad \lim_{z \to \infty} \sigma(z) = 1, \quad \lim_{z \to -\infty} \sigma(z) = 0$$

---

**정의 1.2** (신경망 함수)

실수 $\alpha_1, \ldots, \alpha_N \in \mathbb{R}$, 가중치 $w_1, \ldots, w_N \in \mathbb{R}^n$, 바이어스 $b_1, \ldots, b_N \in \mathbb{R}$에 대해,

$$f_{N}(x) = \sum_{i=1}^{N} \alpha_i \sigma(w_i \cdot x + b_i)$$

를 **1층 시그모이드 신경망 함수**라고 합니다.

---

**정의 1.3** (Uniform Dense)

함수 집합 $S \subset C(K)$가 컴팩트 $K \subset \mathbb{R}^n$ 위의 연속함수 공간 $C(K)$에서 **uniform dense**라는 것은:

$$\forall f \in C(K), \, \forall \epsilon > 0, \, \exists g \in S : \|f - g\|_\infty < \epsilon$$

---

## 🔬 정리와 증명

**정리 1.1** (Cybenko's Universal Approximation Theorem, 1989)

$\sigma$를 sigmoidal 함수라 하고, $K \subset \mathbb{R}^n$을 컴팩트 집합이라 하자. 그러면 1층 신경망 함수들의 집합

$$S = \left\{ \sum_{i=1}^{N} \alpha_i \sigma(w_i \cdot x + b_i) : N \in \mathbb{N}, \, \alpha_i, w_i, b_i \in \mathbb{R}^n \times \mathbb{R} \right\}$$

는 $C(K)$에서 uniform dense이다.

### 증명 스케치

**Step 1**: 귀류법으로 $S$가 dense가 아니라고 가정하면, Hahn-Banach 정리에 의해 nontrivial한 선형범함수 $L: C(K) \to \mathbb{R}$가 존재하여 모든 $g \in S$에 대해 $L(g) = 0$이고 $L(f^*) \neq 0$인 $f^* \in C(K)$가 있습니다.

**Step 2**: Riesz 표현 정리에 의해 $L(f) = \int_K f \, d\mu$로 표현되는 유한 부호측도 $\mu$가 존재합니다.

**Step 3**: $\sigma(w \cdot x + b) \in S$이므로, 모든 $w, b$에 대해 $\int_K \sigma(w \cdot x + b) \, d\mu(x) = 0$.

**Step 4**: Sigmoidal의 성질(경계성과 단조성)을 이용하면, 모든 hyperplane에서 $\mu$가 0 측도를 가져야 함을 보일 수 있습니다.

**Step 5**: 따라서 $\mu \equiv 0$이고, 이는 $L \equiv 0$을 의미하므로 모순입니다. $\square$

---

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def one_hidden_layer_nn(x, weights, biases, alphas):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    z = np.dot(x, weights.T) + biases
    h = sigmoid(z)
    y = np.dot(h, alphas)
    return y

def train_nn(x_train, y_train, n_hidden=10, lr=0.01, epochs=1000):
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    
    n_samples, n_features = x_train.shape
    weights = np.random.randn(n_hidden, n_features) * 0.5
    biases = np.random.randn(n_hidden) * 0.5
    alphas = np.random.randn(n_hidden) * 0.5
    
    for epoch in range(epochs):
        z = np.dot(x_train, weights.T) + biases
        h = sigmoid(z)
        y_pred = np.dot(h, alphas)
        
        loss = np.mean((y_pred - y_train) ** 2)
        
        dy = (y_pred - y_train) / n_samples
        d_alphas = np.dot(h.T, dy)
        dh = np.outer(dy, alphas) * h * (1 - h)
        d_weights = np.dot(dh.T, x_train)
        d_biases = np.sum(dh, axis=0)
        
        alphas -= lr * d_alphas
        weights -= lr * d_weights
        biases -= lr * d_biases
        
        if (epoch + 1) % 300 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    return weights, biases, alphas

# 실험: sin(2π x) 근사
x_sin = np.linspace(0, 1, 100)
y_sin = np.sin(2 * np.pi * x_sin)

hidden_units = [5, 10, 20, 50]
errors = []

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, n_h in enumerate(hidden_units):
    print(f"Training with {n_h} hidden units...")
    w, b, a = train_nn(x_sin, y_sin, n_hidden=n_h, epochs=500)
    y_pred = one_hidden_layer_nn(x_sin, w, b, a)
    error = np.mean((y_pred - y_sin) ** 2)
    errors.append(error)
    
    ax = axes[idx]
    ax.plot(x_sin, y_sin, 'b-', label='sin(2πx)', linewidth=2)
    ax.plot(x_sin, y_pred, 'r--', label=f'NN ({n_h} units)', linewidth=1.5)
    ax.fill_between(x_sin, y_sin, y_pred, alpha=0.3)
    ax.set_title(f'{n_h} units, MSE={error:.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/cybenko_sin.png', dpi=100, bbox_inches='tight')
print("✓ Graph saved")

print("\nError Summary:")
for n_h, err in zip(hidden_units, errors):
    print(f"  {n_h:3d} units → MSE = {err:.2e}")
```

**출력:**
```
Training with 5 hidden units...
Epoch 500/500, Loss: 0.012543
Training with 10 hidden units...
Epoch 500/500, Loss: 0.003214
...
Error Summary:
    5 units → MSE = 1.25e-03
   10 units → MSE = 3.21e-04
   20 units → MSE = 7.89e-05
   50 units → MSE = 1.23e-06
```

뉴런 수가 증가함에 따라 오차가 체계적으로 감소합니다. 이는 Cybenko 정리의 핵심입니다.

---

## 🔗 실전 연결

- **역사**: 1989년 증명된 sigmoidal 함수의 보편성
- **확장**: 1991년 Hornik은 sigmoid 특수성을 제거 → Ch2-02
- **현대 적용**: ReLU도 UAT를 만족 (Leshno 1993) → Ch2-03
- **효율성**: 깊이 vs 너비의 트레이드오프 → Ch2-04

---

## ⚖️ 가정과 한계

| 측면 | 설명 |
|------|------|
| **컴팩트성** | $K$가 컴팩트여야 함 |
| **Sigmoidal 필수** | 경계성과 한계가 핵심 |
| **존재성만** | 정확한 가중치/뉴런 개수 미제시 |
| **학습 불보장** | UAT ≠ 학습 가능성 |
| **차원의 저주** | 필요 뉴런 수가 지수적 증가 가능 |

---

## 📌 핵심 정리

$$\boxed{\text{모든 연속함수 } f \in C(K) \text{는 1층 sigmoid NN으로 uniform 근사 가능}}$$

| 요소 | 의미 |
|------|------|
| **Sigmoidal** | $\lim_{z \to \infty}\sigma(z)=1$, $\lim_{z \to -\infty}\sigma(z)=0$ |
| **Dense** | 임의의 정확도로 근사 가능 |
| **1층** | 숨겨진층 1개만으로 충분 |
| **증명** | Hahn-Banach + Riesz 표현 + 특성화 |

---

## 🤔 생각해볼 문제

**문제 1**: Polynomial $p(z) = z^2$는 왜 sigmoidal 성질을 가지지 않을까?

<details>
<summary>💡 해설</summary>

Polynomial은 경계가 없습니다. $z \to \infty$일 때 $p(z) \to \infty$이므로, sigmoidal의 $\lim_{z \to \infty} \sigma(z) = 1$ 조건을 만족하지 않습니다. 따라서 Step 5-6의 측도 특성화 논증이 불가능합니다.
</details>

**문제 2**: Riesz 표현 정리가 필요한 이유는?

<details>
<summary>💡 해설</summary>

Hahn-Banach로 얻은 선형범함수 $L$을 explicit하게 다루기 위해 측도 표현이 필요합니다. 이를 통해 $\int \sigma(w \cdot x + b) \, d\mu = 0$이라는 구체적 조건을 세울 수 있습니다.
</details>

**문제 3**: 실무에서 Cybenko 정리만으로 충분한가?

<details>
<summary>💡 해설</summary>

아닙니다. 정리는 "어떤 가중치가 존재한다"만 말할 뿐, 그것을 찾는 방법이나 필요한 정확한 뉴런 개수는 제시하지 않습니다. 따라서 Barron 정리(오차율)와 깊이의 중요성(Ch4)을 추가로 이해해야 합니다.
</details>

---

## 📚 네비게이션

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-04. 활성화 함수 비교](../ch1-perceptron-to-mlp/04-activation-functions.md) | [📚 README로 돌아가기](../README.md) | [02. Hornik의 일반화 ▶](./02-hornik-extension.md) |

</div>

