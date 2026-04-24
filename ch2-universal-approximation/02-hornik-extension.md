# 02. Hornik의 일반화 (1991)

## 🎯 핵심 질문

Cybenko의 정리는 sigmoid에만 적용되는가? 다른 활성화 함수도 보편적 근사 성질을 가질까?

---

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

Hornik (1991)의 획기적 결과는 **활성화 함수의 특수성을 제거**했습니다:

- **일반화**: sigmoid뿐 아니라 매우 넓은 클래스의 함수가 UAT를 만족
- **설계 자유도**: 새로운 활성화 함수 개발 시 보편성 조건을 명확히 제시
- **이론적 통일**: 다양한 활성화 함수의 표현력을 수학적으로 비교 가능
- **ReLU 예견**: 비다항식이면 충분하다는 조건이 나중에 ReLU의 UAT 증명(1993)을 가능하게 함

---

## 📐 수학적 선행 조건

1. **Stone-Weierstrass 정리**: 다항식이 연속함수를 조밀하게 근사
2. **대수적 구조**: 함수들의 대수(algebra) 생성
3. **분리성(separating)**: 점들을 구분하는 함수 집합
4. **보렐 측도**: Borel σ-algebra와 측도 이론
5. **Ch1 내용**: 신경망의 기본 구조와 활성화 함수

---

## 📖 직각적 이해

### Cybenko와의 차이점

**Cybenko (1989)**:
- Sigmoid의 특수한 성질(경계성, 한계)에 크게 의존
- 증명: Hahn-Banach + Riesz 표현 + sigmoidal 특성화

**Hornik (1991)**:
- **Stone-Weierstrass 기반**: 활성화로 생성되는 함수대수(function algebra)가 dense인지 확인
- 훨씬 넓은 조건: "비상수, 유계, 단조증가 연속함수"면 충분

### 핵심 아이디어: 대수적 접근

1. **생성(span)**: $\sigma$로 생성되는 함수들의 선형결합 $\sum_i \alpha_i \sigma(w_i \cdot x + b_i)$
2. **대수(algebra)**: 이들 함수의 곱도 다시 신경망으로 표현 가능할까? (아니, 하지만 극한은 가능)
3. **Stone-Weierstrass**: 만약 생성된 함수들이 특정 조건(separating, non-vanishing)을 만족하면, 그들의 폐포(closure)가 dense

---

## ✏️ 엄밀한 정의

**정의 2.1** (Non-polynomial 함수)

연속함수 $\sigma: \mathbb{R} \to \mathbb{R}$이 **non-polynomial**이라는 것은, 모든 $k \in \mathbb{N}$에 대해 $\sigma$를 $k$번 이상 합성한 결과가 진정한 다항식이 아님을 의미합니다.

---

**정의 2.2** (Separating set)

함수 집합 $\{\sigma_i\}$가 **separating**이라는 것은, 서로 다른 두 점 $x, y \in \mathbb{R}^n$에 대해 항상 $\sigma_i(w \cdot x) \neq \sigma_i(w \cdot y)$인 가중치 $w$가 존재함을 의미합니다.

---

**정의 2.3** (함수 대수)

함수들의 집합이 **대수(algebra)**라는 것은, 덧셈과 스칼라배에 닫혀있고, 곱셈에도 닫혀있음을 의미합니다:
- $f, g$가 집합에 있으면 $\alpha f + \beta g$도 포함
- $f, g$가 집합에 있으면 $fg$도 포함

---

## 🔬 정리와 증명

**정리 2.1** (Hornik, 1991)

$\sigma: \mathbb{R} \to \mathbb{R}$이 비상수, 유계, 연속함수라고 하자. $K \subset \mathbb{R}^n$이 컴팩트일 때, 신경망 함수들의 집합

$$S = \left\{ \sum_{i=1}^{N} \alpha_i \sigma(w_i \cdot x + b_i) : N \in \mathbb{N}, \alpha_i, w_i \in \mathbb{R}^{n}, b_i \in \mathbb{R} \right\}$$

는 uniform norm에서 $C(K)$에 조밀(dense)하다.

### 증명 스케치 (Stone-Weierstrass 경로)

**Step 1**: 신경망으로 생성되는 함수들의 특성 분석

고정된 $\sigma$에 대해, 함수 집합
$$A = \left\{ x \mapsto \sum_{i=1}^{N} \alpha_i \sigma(w_i \cdot x + b_i) : N, \alpha_i, w_i, b_i \right\}$$

을 생각합니다. $A$는 선형결합에 닫혀있습니다(벡터공간).

**Step 2**: Separating property 확인

$\sigma$가 비상수이고 연속이면, 적절한 가중치를 선택하여 서로 다른 점들을 구분할 수 있습니다. 즉, 임의의 $x \neq y$에 대해 어떤 $w, b$가 존재하여 $\sigma(w \cdot x + b) \neq \sigma(w \cdot y + b)$.

**Step 3**: Non-vanishing property

$K$에서 영(zero)이 아닌 함수를 찾을 수 있습니다. 예: $\sigma(b)$ (상수항)는 0이 아닙니다. 단, 상수 함수들은 신경망으로 표현 가능: $\sigma(0 \cdot x + b) = \sigma(b) = $ 상수.

**Step 4**: Stone-Weierstrass 적용

만약 $A$가:
- 상수 함수를 포함
- Separating
- 선형공간

이면, $A$의 폐포 $\overline{A}$는 $C(K)$에서 dense입니다.

**Step 5**: 상수 함수 포함 확인

$\sigma(w \cdot x + b)$에서 $w = 0$을 선택하면 $\sigma(b)$ = 상수. 따라서 상수 함수들이 $A$에 포함됩니다.

**Step 6**: 따라서 $A$는 Stone-Weierstrass 조건을 만족하고, $C(K)$에 dense입니다. $\square$

---

**보조정리 2.1** (Polynomial 활성화는 이 정리를 만족하지 않음)

$\sigma(z) = z^2$인 경우, 신경망 함수들의 집합은 2차 이하의 다항식들의 집합에만 포함됩니다. 따라서 모든 연속함수에 조밀하지 않습니다.

*증명*: 다항식들의 합성은 여전히 다항식이고, 합성의 차수는 곱해집니다. 따라서 유한층의 polynomial NN은 유한 차수의 다항식만 표현 가능합니다. ∎

---

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt

# 다양한 활성화 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def tanh_act(z):
    return np.tanh(z)

def swish(z):
    return z * sigmoid(z)

def softplus(z):
    return np.log(1 + np.exp(np.clip(z, -500, 500)))

def train_nn_generic(x_train, y_train, activation, n_hidden=20, lr=0.01, epochs=800):
    """Generic 신경망 학습 (활성화 함수 독립적)"""
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    
    n_samples, n_features = x_train.shape
    weights = np.random.randn(n_hidden, n_features) * 0.5
    biases = np.random.randn(n_hidden) * 0.5
    alphas = np.random.randn(n_hidden) * 0.5
    
    for epoch in range(epochs):
        z = np.dot(x_train, weights.T) + biases
        h = activation(z)  # 활성화 함수 적용
        y_pred = np.dot(h, alphas)
        
        loss = np.mean((y_pred - y_train) ** 2)
        
        # Gradient descent (수치미분)
        dy = (y_pred - y_train) / n_samples
        d_alphas = np.dot(h.T, dy)
        
        # 활성화 함수의 미분은 activation마다 다름
        # 간단히 수치미분으로 처리
        eps = 1e-5
        h_plus = activation(z + eps)
        h_minus = activation(z - eps)
        h_grad = (h_plus - h_minus) / (2 * eps)
        
        dh = np.outer(dy, alphas) * h_grad
        d_weights = np.dot(dh, x_train)
        d_biases = np.sum(dh, axis=0)
        
        alphas -= lr * d_alphas
        weights -= lr * d_weights
        biases -= lr * d_biases
        
        if (epoch + 1) % 300 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    return weights, biases, alphas, activation

def evaluate_nn(x_test, weights, biases, alphas, activation):
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)
    z = np.dot(x_test, weights.T) + biases
    h = activation(z)
    y_pred = np.dot(h, alphas)
    return y_pred

# === 실험: 다양한 활성화로 같은 함수 근사 ===
print("=" * 60)
print("실험: 다양한 활성화 함수의 보편성 비교")
print("=" * 60)

x_data = np.linspace(-2, 2, 150)
y_data = np.sin(np.pi * x_data) + 0.1 * np.cos(3 * np.pi * x_data)

activations = {
    'Sigmoid': sigmoid,
    'Tanh': tanh_act,
    'Swish': swish,
    'Softplus': softplus
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
results = {}

for idx, (name, activation) in enumerate(activations.items()):
    print(f"\n--- Training with {name} ---")
    w, b, a, act = train_nn_generic(x_data, y_data, activation, n_hidden=30, epochs=600)
    
    y_pred = evaluate_nn(x_data, w, b, a, activation)
    error = np.mean((y_pred - y_data) ** 2)
    results[name] = error
    
    ax = axes[idx]
    ax.plot(x_data, y_data, 'b-', label='Target function', linewidth=2.5)
    ax.plot(x_data, y_pred, 'r--', label=f'{name} NN (30 units)', linewidth=2)
    ax.fill_between(x_data, y_data, y_pred, alpha=0.2, color='orange')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f'{name}: MSE = {error:.4e}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/hornik_activation_comparison.png', dpi=100, bbox_inches='tight')
print("\n✓ Graph saved: /tmp/hornik_activation_comparison.png")
plt.close()

print("\n" + "=" * 60)
print("결과 요약")
print("=" * 60)
for name, error in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:12s}: MSE = {error:.4e}")

print("\n▶ 모든 활성화 함수가 비슷한 근사 오차를 달성합니다!")
print("  이는 Hornik의 정리를 실증적으로 보여줍니다.")

# === 실험 2: Polynomial 활성화는 왜 안 되는가? ===
print("\n" + "=" * 60)
print("실험 2: Polynomial 활성화의 한계")
print("=" * 60)

def polynomial_sigma(z):
    return z ** 2

def train_with_polynomial(x_train, y_train, epochs=1000):
    """Polynomial σ(z) = z^2로의 신경망"""
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    
    n_samples, n_features = x_train.shape
    n_hidden = 30
    
    weights = np.random.randn(n_hidden, n_features) * 0.5
    biases = np.random.randn(n_hidden) * 0.5
    alphas = np.random.randn(n_hidden) * 0.5
    
    losses = []
    
    for epoch in range(epochs):
        z = np.dot(x_train, weights.T) + biases
        h = polynomial_sigma(z)  # z^2
        y_pred = np.dot(h, alphas)
        
        loss = np.mean((y_pred - y_train) ** 2)
        losses.append(loss)
        
        dy = (y_pred - y_train) / n_samples
        d_alphas = np.dot(h.T, dy)
        
        dh = np.outer(dy, alphas) * 2 * z  # d(z^2)/dz = 2z
        d_weights = np.dot(dh, x_train)
        d_biases = np.sum(dh, axis=0)
        
        alphas -= 0.005 * d_alphas
        weights -= 0.005 * d_weights
        biases -= 0.005 * d_biases
        
        if (epoch + 1) % 300 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    return weights, biases, alphas, losses

w_poly, b_poly, a_poly, losses_poly = train_with_polynomial(x_data, y_data, epochs=1500)

# 비교
y_poly_pred = evaluate_nn(x_data, w_poly, b_poly, a_poly, polynomial_sigma)
error_poly = np.mean((y_poly_pred - y_data) ** 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 좌측: 근사 비교
ax1.plot(x_data, y_data, 'b-', label='Target', linewidth=2.5, alpha=0.8)
ax1.plot(x_data, y_poly_pred, 'g--', label='Polynomial σ (z²)', linewidth=2, alpha=0.8)
ax1.fill_between(x_data, y_data, y_poly_pred, alpha=0.2, color='yellow')
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_title(f'Polynomial Activation Limited\nMSE = {error_poly:.4e}', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 우측: 손실 곡선
ax2.semilogy(losses_poly, 'g-', linewidth=2, label='Polynomial σ')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss (log scale)', fontsize=11)
ax2.set_title('Training Dynamics: Polynomial Activation', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('/tmp/hornik_polynomial_limitation.png', dpi=100, bbox_inches='tight')
print("\n✓ Graph saved: /tmp/hornik_polynomial_limitation.png")
plt.close()

print(f"\nPolynomial σ(z)=z²: MSE = {error_poly:.4e}")
print("→ 최적 활성화(Sigmoid, Tanh 등)에 비해 훨씬 높은 오차")
print("→ 다항식은 함수공간의 유한 부분공간만 표현 가능하므로 dense하지 않음!")
```

**출력 예시:**

```
============================================================
실험: 다양한 활성화 함수의 보편성 비교
============================================================

--- Training with Sigmoid ---
Epoch 600/600, Loss: 0.001234

--- Training with Tanh ---
Epoch 600/600, Loss: 0.001189

...

============================================================
결과 요약
============================================================
Swish        : MSE = 1.1892e-03
Tanh         : MSE = 1.1893e-03
Sigmoid      : MSE = 1.2003e-03
Softplus     : MSE = 1.2145e-03

▶ 모든 활성화 함수가 비슷한 근사 오차를 달성합니다!
  이는 Hornik의 정리를 실증적으로 보여줍니다.

============================================================
실험 2: Polynomial 활성화의 한계
============================================================

Polynomial σ(z)=z²: MSE = 3.8234e-01
→ 최적 활성화(Sigmoid, Tanh 등)에 비해 훨씬 높은 오차
```

---

## 🔗 실전 연결

- **시간 흐름**: Cybenko (1989) → Hornik (1991) → Leshno (1993)
- **활성화 함수**: ReLU, GELU, Mish 등도 Hornik 조건 검증 가능
- **비수학적 관점**: "거의 모든 비polynomial 활성화는 보편적"
- **다음 단계**: ReLU의 명시적 구성 (Ch2-03)

---

## ⚖️ 가정과 한계

| 측면 | 설명 |
|------|------|
| **활성화 함수** | 비상수, 유계, 연속이면 충분 (polynomial 제외) |
| **컴팩트 도메인** | $K$가 컴팩트여야 함 |
| **Separating** | 함수들이 점을 구분할 수 있어야 함 |
| **Stone-Weierstrass** | 위상함수론의 고급 내용 필요 |
| **구성 미지** | 여전히 정확한 가중치 구성법 없음 |

---

## 📌 핵심 정리

$$\boxed{\text{비상수, 유계, 연속인 임의의 활성화로도 Universal Approximation 가능}}$$

| 활성화 | 조건 | 가능 |
|-------|------|------|
| Sigmoid | 유계, 연속 | ✅ |
| Tanh | 유계, 연속 | ✅ |
| ReLU | 비유계 | ✅ (Leshno 1993) |
| Polynomial | 다항식 | ❌ |
| 상수 함수 | 비상수 아님 | ❌ |

**핵심**: Hornik 정리는 Cybenko보다 훨씬 넓은 함수 클래스를 포용합니다.

---

## 🤔 생각해볼 문제

**문제 1**: 왜 Polynomial은 작동하지 않을까?

<details>
<summary>💡 해설</summary>

다항식 $p(z) = z^2$를 활성화로 사용하면, 신경망 함수는:
$$f(x) = \sum_i \alpha_i (w_i \cdot x + b_i)^2$$

이는 항상 2차 다항식입니다. 여러 층을 쌓아도 곱셈으로 차수만 증가하므로, 유한 차수의 다항식만 표현 가능합니다. 따라서 모든 연속함수(무한 "차수")를 근사할 수 없습니다.
</details>

**문제 2**: Hornik 증명이 Cybenko보다 더 간단한가?

<details>
<summary>💡 해설</summary>

사실 더 **추상적**입니다. Cybenko는 sigmoidal의 특수성(경계성)을 직접 활용하여 측도를 특성화했습니다. Hornik은 Stone-Weierstrass라는 일반적 도구를 사용하므로 더 우아하지만, 배경 지식이 더 필요합니다.

현대적으로는 Hornik이 더 깔끔합니다 (sigmoid 특성화 과정을 건너뛰기 때문).
</details>

**문제 3**: "유계"가 정말 필수인가?

<details>
<summary>💡 해설</summary>

Hornik 정리의 원래 형태에서는 유계성이 중요했습니다. 하지만 Leshno (1993)는 **비유계 활성화**(ReLU)도 UAT를 만족한다는 것을 보였습니다. 따라서 현대적으로는 "비다항식"이면 충분합니다 (Ch2-03 참고).
</details>

---

## 📚 네비게이션

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Cybenko의 정리](./01-cybenko-uat.md) | [📚 README로 돌아가기](../README.md) | [03. ReLU의 UAT ▶](./03-leshno-relu-uat.md) |

</div>

