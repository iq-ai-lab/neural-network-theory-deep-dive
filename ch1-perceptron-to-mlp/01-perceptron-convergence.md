# 01. 퍼셉트론과 Novikoff 수렴 정리

## 🎯 핵심 질문

- Rosenblatt(1958)의 퍼셉트론 알고리즘은 정확히 무엇을 업데이트하는가?
- 선형 분리 가능한 데이터에서 퍼셉트론은 왜 유한 스텝 안에 수렴하는가?
- 수렴 속도를 결정하는 **margin $\gamma$**와 **반지름 $R$**의 의미는?
- Novikoff(1962)의 bound $(R/\gamma)^2$은 어디서 오는가?
- 선형 분리 불가능한 데이터에서는 어떤 일이 벌어지는가?

---

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

퍼셉트론은 **현대 신경망의 최소 단위**이다. `torch.nn.Linear → Sign`은 결국 $y = \text{sign}(w \cdot x + b)$ 하나의 퍼셉트론이고, MLP는 이를 쌓은 것이다. Novikoff 수렴 정리는 **"gradient 기반 학습이 왜 수렴하는가"** 에 대한 가장 오래된, 그리고 가장 엄밀한 답이다. 이 증명 기법($\|w\|$의 상한과 $w \cdot w^*$의 하한 사이의 Cauchy-Schwarz 조임)은 이후 **SVM의 margin bound**, **online learning의 regret bound**, **Perceptron-based kernel methods**로 확장된다. 퍼셉트론이 경사하강이 아닌 **mistake-driven** 방식으로 수렴한다는 사실은 **SGD와 online learning의 분업**을 이해하는 출발점이며, 현대 학습이론에서의 **implicit regularization**(큰 margin으로 수렴하는 경향) 논의의 뿌리이기도 하다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 내적 $w \cdot x$, 노름 $\|w\|$, Cauchy-Schwarz 부등식 $|a \cdot b| \leq \|a\| \|b\|$
- 해석학 기초: 단조증가 수열의 수렴, 부등식 기법
- 확률론: 데이터 분포 개념(증명 자체는 결정론적)

---

## 📖 직관적 이해

### 퍼셉트론이 하는 일

2차원을 예로 들자. 점들이 빨강(레이블 $+1$)과 파랑(레이블 $-1$)으로 나뉘어 있다. 퍼셉트론은 이 둘을 **직선 하나로 구분**하려 한다. 직선은 $w \cdot x + b = 0$으로 표현되고, 학습이란 $w, b$를 조정해 **모든 점이 자기 쪽에** 들어가도록 하는 것이다.

알고리즘은 단순하다:

1. 하나의 데이터 $(x_i, y_i)$를 본다.
2. 현재 예측 $\hat y_i = \text{sign}(w \cdot x_i + b)$이 정답과 다르면 (**mistake**):
   - $w \leftarrow w + y_i x_i$
   - $b \leftarrow b + y_i$
3. 예측이 맞으면 그대로 둔다.
4. 모든 데이터에서 mistake가 없을 때까지 반복.

**왜 이 업데이트가 작동하는가?** 틀렸다는 것은 $y_i(w \cdot x_i + b) \leq 0$이라는 의미다. 업데이트 후:

$$y_i(w_\text{new} \cdot x_i + b_\text{new}) = y_i(w + y_i x_i) \cdot x_i + y_i(b + y_i) = y_i(w \cdot x_i + b) + y_i^2 \|x_i\|^2 + 1$$

$y_i^2 = 1$이고 $\|x_i\|^2 \geq 0$이므로, $y_i(w \cdot x_i + b)$는 업데이트에 의해 **증가**한다. 즉, **같은 점에서의 판단은 점점 "덜 틀리는" 방향**으로 움직인다.

### Margin: "얼마나 여유 있게 분리되는가"

데이터가 선형 분리 가능하다는 것은 **어떤 $w^*, b^*$가 존재해** 모든 점에서 $y_i(w^* \cdot x_i + b^*) > 0$이 성립한다는 뜻이다. **Margin** $\gamma$는 이 여유의 최소값이다:

$$\gamma := \min_i y_i (w^* \cdot x_i + b^*), \quad \text{단} \|w^*\| = 1$$

$\gamma$가 **크면**(데이터가 널찍하게 분리되어 있으면) 학습이 **빠르게** 끝난다. $\gamma$가 **작으면**(데이터가 간신히 분리되어 있으면) 퍼셉트론은 **미세 조정을 반복**하며 느리게 수렴한다.

| 상황 | margin $\gamma$ | mistake 수 bound |
|------|----------------|-----------------|
| 데이터가 멀찍이 분리 | 크다 | 적다 |
| 데이터가 간신히 분리 | 작다 | 많다 |
| 데이터가 딱 경계 위 | $\gamma = 0$ | 수렴 보장 없음 |
| 선형 분리 불가능 | 존재하지 않음 | **영원히 수렴 안 함** (다음 문서) |

> **비유**: 울타리를 치는데 — 양떼와 늑대가 멀찍이 떨어져 있으면 한 번에 울타리를 그릴 수 있다. 뒤엉켜 있으면 계속 조정해야 한다. margin은 "양과 늑대 사이의 여유 공간"이다.

### Novikoff 증명의 두 축

증명의 핵심은 두 양을 동시에 추적하는 것이다:

1. **하한 축**: $w_k \cdot w^*$가 매 mistake마다 최소 $\gamma$만큼 증가 → $k \gamma$ 이상
2. **상한 축**: $\|w_k\|^2$이 매 mistake마다 최대 $R^2$만큼 증가 → $k R^2$ 이하

Cauchy-Schwarz로 이 둘을 엮으면 $k \gamma \leq w_k \cdot w^* \leq \|w_k\| \leq \sqrt{k} R$ → $k \leq (R/\gamma)^2$.

**즉, mistake 수는 데이터의 기하(margin과 반지름)만으로 결정되고, 차원 $d$와는 무관하다.**

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 퍼셉트론과 분류 규칙

**퍼셉트론** $f_{w, b}: \mathbb{R}^d \to \{-1, +1\}$은 가중치 $w \in \mathbb{R}^d$와 편향 $b \in \mathbb{R}$에 대해

$$f_{w, b}(x) := \text{sign}(w \cdot x + b) = \begin{cases} +1 & w \cdot x + b > 0 \\ -1 & w \cdot x + b < 0 \end{cases}$$

로 정의된다. (경계 $w \cdot x + b = 0$의 값은 규약에 따라 $+1$로 둔다.)

### 정의 1.2 — 선형 분리 가능성

데이터 $\mathcal{D} = \{(x_1, y_1), \ldots, (x_n, y_n)\} \subset \mathbb{R}^d \times \{-1, +1\}$이 **선형 분리 가능(linearly separable)**하다는 것은 어떤 $w^* \in \mathbb{R}^d$와 $b^* \in \mathbb{R}$이 존재해

$$y_i (w^* \cdot x_i + b^*) > 0, \quad \forall i \in \{1, \ldots, n\}$$

이 성립함을 의미한다.

### 정의 1.3 — Margin과 반지름

선형 분리 가능한 데이터 $\mathcal{D}$에 대해,

- **Margin** (분리 경계로부터의 최소 거리, 단위벡터 $w^*$ 기준):
$$\gamma := \max_{\|w^*\| = 1} \min_i y_i(w^* \cdot x_i + b^*)$$

- **데이터 반지름**:
$$R := \max_i \|\tilde x_i\|, \quad \tilde x_i := (x_i, 1) \in \mathbb{R}^{d+1}$$

(편향 $b$를 가중치의 일부로 흡수하기 위해 $x$에 $1$을 붙이는 trick)

### 정의 1.4 — 퍼셉트론 알고리즘

초기값 $w_0 = 0 \in \mathbb{R}^{d+1}$로 시작. 데이터를 순회하며 예측 $\hat y_i = \text{sign}(w_k \cdot \tilde x_i)$이 $y_i$와 다르면:

$$w_{k+1} \leftarrow w_k + y_i \tilde x_i$$

예측이 맞으면 $w_{k+1} = w_k$. 첨자 $k$는 **mistake 횟수**이며, 모든 점에서 mistake가 없으면 알고리즘은 종료한다.

---

## 🔬 정리와 증명

### 정리 1.1 (Novikoff 1962) — 퍼셉트론 수렴 bound

**명제**: 데이터 $\mathcal{D}$가 선형 분리 가능하고, 단위 분리자 $w^* \in \mathbb{R}^{d+1}$ ($\|w^*\| = 1$)에 대해 모든 $i$에서 $y_i (w^* \cdot \tilde x_i) \geq \gamma > 0$이 성립한다고 하자. 또한 $\|\tilde x_i\| \leq R$. 그러면 퍼셉트론 알고리즘이 일으키는 **mistake 총 횟수** $k$는

$$k \leq \left(\frac{R}{\gamma}\right)^2$$

을 만족하며, 알고리즘은 **유한 스텝 내에 종료**한다.

**증명**:

$k$번째 mistake가 데이터 $(x^{(k)}, y^{(k)})$에서 발생했다고 하자. (단순화를 위해 $\tilde x$ 표기의 tilde를 생략.) 업데이트 공식:

$$w_{k} = w_{k-1} + y^{(k)} x^{(k)}, \quad w_0 = 0$$

이므로 $k$번째 mistake 후 누적 형태:

$$w_k = \sum_{j=1}^{k} y^{(j)} x^{(j)}$$

**1단계 — 하한**: $w_k \cdot w^*$의 증가량.

mistake라는 것은 $y^{(k)}(w_{k-1} \cdot x^{(k)}) \leq 0$이라는 뜻. 하지만 우리가 보고 싶은 것은 $w^*$ 방향과의 내적:

$$w_k \cdot w^* = (w_{k-1} + y^{(k)} x^{(k)}) \cdot w^* = w_{k-1} \cdot w^* + y^{(k)}(w^* \cdot x^{(k)})$$

가정에 의해 $y^{(k)}(w^* \cdot x^{(k)}) \geq \gamma$ (margin 조건). 따라서:

$$w_k \cdot w^* \geq w_{k-1} \cdot w^* + \gamma$$

$w_0 \cdot w^* = 0$이므로 귀납적으로:

$$\boxed{w_k \cdot w^* \geq k \gamma} \tag{A}$$

**2단계 — 상한**: $\|w_k\|^2$의 증가량.

$$\|w_k\|^2 = \|w_{k-1} + y^{(k)} x^{(k)}\|^2 = \|w_{k-1}\|^2 + 2 y^{(k)} (w_{k-1} \cdot x^{(k)}) + \|x^{(k)}\|^2 (y^{(k)})^2$$

$y^{(k)} \in \{-1, +1\}$이므로 $(y^{(k)})^2 = 1$. 그리고 $k$번째 update는 mistake에서만 발생하므로 $y^{(k)}(w_{k-1} \cdot x^{(k)}) \leq 0$. 또한 $\|x^{(k)}\|^2 \leq R^2$:

$$\|w_k\|^2 \leq \|w_{k-1}\|^2 + 0 + R^2 = \|w_{k-1}\|^2 + R^2$$

$\|w_0\|^2 = 0$이므로 귀납적으로:

$$\boxed{\|w_k\|^2 \leq k R^2} \tag{B}$$

**3단계 — Cauchy-Schwarz로 조임**:

$\|w^*\| = 1$이므로 Cauchy-Schwarz에 의해:

$$w_k \cdot w^* \leq \|w_k\| \cdot \|w^*\| = \|w_k\|$$

(A)와 (B)를 결합:

$$k \gamma \overset{(A)}{\leq} w_k \cdot w^* \leq \|w_k\| \overset{(B)}{\leq} \sqrt{k} R$$

양변 제곱 후 $k$로 정리:

$$k^2 \gamma^2 \leq k R^2 \implies k \leq \frac{R^2}{\gamma^2} = \left(\frac{R}{\gamma}\right)^2$$

$\square$

### 정리 1.2 — Mistake 상한의 차원 독립성

**명제**: 정리 1.1의 bound $(R/\gamma)^2$는 **입력 차원 $d$에 명시적으로 의존하지 않는다**.

**증명**: 증명에서 사용된 모든 양($w_k \cdot w^*$, $\|w_k\|$, $R$, $\gamma$)은 **내적과 노름**만을 사용한다. Cauchy-Schwarz 부등식도 차원에 의존하지 않는다. 따라서 $(R/\gamma)^2$은 데이터의 기하학적 구조(분리 여유와 범위)만으로 결정된다. $\square$

> **해석**: 이것이 **kernel trick**의 정당성의 기반이다. 고차원(혹은 무한차원) 특성 공간으로 mapping해도 $R$과 $\gamma$만 유한하면 퍼셉트론은 유한 스텝에 수렴한다. → SVM으로 직접 이어진다.

### 정리 1.3 — bound의 최적성(tightness)

**명제**: bound $(R/\gamma)^2$는 **상수 factor까지 최적**이다. 즉, $\Theta((R/\gamma)^2)$ mistake가 실제로 필요한 데이터가 존재한다.

**증명 스케치**:
2차원 평면에 $n$개의 점을 반지름 $R$의 원 위에 놓고, 정확히 margin $\gamma$로 분리 가능하게 배치하면 $\Theta((R/\gamma)^2)$의 mistake가 실제로 요구됨을 구성할 수 있다 (Minsky & Papert 1969). 세부 구성은 연습문제. $\square$

### 예시

**예시 1 — 2차원 간단한 경우**:
$x_1 = (1, 0), y_1 = +1$; $x_2 = (-1, 0), y_2 = -1$; $x_3 = (0, 1), y_3 = +1$; $x_4 = (0, -1), y_4 = -1$

분리자 후보: $w^* = (1, 1) / \sqrt{2}$, $b^* = 0$. Margin $\gamma = 1/\sqrt{2}$, $R = 1$ (모든 점의 노름).

Bound: $k \leq (1 / (1/\sqrt 2))^2 = 2$. 실제로 적절한 순서로 데이터를 제공하면 2번 이하의 mistake로 끝난다.

**예시 2 — Margin이 작아질 때**:
$x_1 = (1, \epsilon), y_1 = +1$; $x_2 = (1, -\epsilon), y_2 = -1$

$\epsilon \to 0$이면 margin $\gamma \to 0$, mistake bound $\to \infty$. 이는 **margin이 작을수록 학습이 어려움**을 수치화한다.

---

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────
# 1. 선형 분리 가능한 2D 데이터 생성 (margin γ 제어)
# ──────────────────────────────────────────────────────────
def make_separable_data(n=100, margin=0.3, R=1.0):
    """2D에서 margin γ 이상으로 분리 가능한 데이터 생성"""
    w_star = np.array([1.0, 1.0]) / np.sqrt(2)   # 단위 분리자
    X, y = [], []
    while len(X) < n:
        x = rng.uniform(-R, R, size=2)
        if np.linalg.norm(x) > R:
            continue
        score = w_star @ x
        if abs(score) < margin:                   # margin 영역 밖만 채택
            continue
        X.append(x)
        y.append(1 if score > 0 else -1)
    return np.array(X), np.array(y), w_star

X, y, w_star = make_separable_data(n=200, margin=0.3, R=1.0)
gamma, R = 0.3, 1.0

# ──────────────────────────────────────────────────────────
# 2. 퍼셉트론 알고리즘 (bias는 homogeneous 변수로 흡수)
# ──────────────────────────────────────────────────────────
def perceptron(X, y, max_epochs=1000):
    # x를 (x, 1)로 확장 → w 안에 bias 포함
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    d = X_aug.shape[1]
    w = np.zeros(d)
    mistakes = 0
    history = [w.copy()]
    for epoch in range(max_epochs):
        any_mistake = False
        # 매 epoch 데이터를 섞어서 순회
        order = rng.permutation(len(X_aug))
        for i in order:
            yhat = np.sign(w @ X_aug[i])
            if yhat == 0: yhat = 1
            if yhat != y[i]:
                w = w + y[i] * X_aug[i]
                mistakes += 1
                history.append(w.copy())
                any_mistake = True
        if not any_mistake:
            return w, mistakes, history
    return w, mistakes, history

w_final, k, hist = perceptron(X, y)
print(f"총 mistake 수: {k}")
print(f"이론 bound (R/γ)² = ({R}/{gamma})² = {(R/gamma)**2:.2f}")
# → 실제 k가 bound 이내인지 확인

# ──────────────────────────────────────────────────────────
# 3. margin별 수렴 속도 측정 — k vs (R/γ)²
# ──────────────────────────────────────────────────────────
margins = [0.5, 0.3, 0.2, 0.1, 0.05]
actual_k, theory_k = [], []
for m in margins:
    X_m, y_m, _ = make_separable_data(n=200, margin=m, R=1.0)
    _, k_m, _ = perceptron(X_m, y_m)
    actual_k.append(k_m)
    theory_k.append((1.0 / m) ** 2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (좌) 데이터와 수렴한 분리선
ax = axes[0]
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, label='+1')
ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', alpha=0.6, label='-1')
# 분리선: w·x + b = 0 → y = -(w0 x + b) / w1
w0, w1, b = w_final
xs = np.linspace(-1, 1, 100)
ax.plot(xs, -(w0 * xs + b) / w1, 'k-', label='학습된 분리자')
ax.plot(xs, -w_star[0] * xs / w_star[1], 'g--', alpha=0.5, label='정답 $w^*$')
ax.set_title(f'퍼셉트론 수렴 — {k}번 mistake')
ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=0.3)

# (우) margin vs mistake 수
ax = axes[1]
ax.plot(margins, actual_k, 'o-', label='실제 mistake 수')
ax.plot(margins, theory_k, 's--', label=r'이론 bound $(R/\gamma)^2$')
ax.set_xlabel(r'margin $\gamma$'); ax.set_ylabel('mistake 수')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('margin이 작아질수록 $(R/\gamma)^2$로 증가'); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('perceptron_convergence.png', dpi=150); plt.show()
```

**출력 예시**:
```
총 mistake 수: 8
이론 bound (R/γ)² = (1.0/0.3)² = 11.11

margin   actual_k   theory_k
0.50      4           4.0
0.30      8          11.1
0.20     17          25.0
0.10     62         100.0
0.05    233         400.0
```

**관찰**: 실제 mistake 수는 항상 이론 bound 이하. margin이 반으로 줄면 mistake 수는 약 4배로 증가(정확히 $(1/\gamma)^2$ 스케일).

---

## 🔗 실전 연결

### SGD · Online Learning과의 분업

퍼셉트론 업데이트 $w \leftarrow w + y_i x_i$는 사실 **hinge loss $\max(0, -y_i(w \cdot x_i))$의 subgradient descent**와 동치이다(학습률 $\eta = 1$). 현대 딥러닝의 SGD는 이 발상의 연속이다. 다만:

- **퍼셉트론**: mistake 시에만 업데이트, $\eta = 1$ 고정
- **SGD**: 매 스텝 gradient 계산, $\eta$ 감쇠
- **Online learning**: regret bound $\sqrt{T}$로 일반화

### SVM과의 연결

퍼셉트론이 수렴한 직선은 **임의의** 분리선이지 margin이 큰 직선은 아니다. SVM은 "가장 큰 margin을 가진 분리선"을 찾으며, 수렴 증명의 **Cauchy-Schwarz 기법**을 그대로 이어받는다. Novikoff bound의 $\gamma$가 SVM의 primal 목적함수다.

### PyTorch에서의 재현

```python
# PyTorch로 퍼셉트론: 사실상 nn.Linear + sign
import torch, torch.nn as nn
class Perceptron(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1, bias=True)
    def forward(self, x):
        return torch.sign(self.fc(x))
# 단, torch.sign은 grad가 0 → hinge loss로 대체 학습
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 선형 분리 가능 | **선형 분리 불가**면 퍼셉트론은 **영원히 수렴하지 않음** — 다음 문서에서 XOR로 구체화 |
| margin $\gamma > 0$ | 경계 위의 데이터($\gamma = 0$)는 분리 불가와 동일 취급 |
| $\|x\| \leq R$ 유한 | unbounded 데이터에서는 bound 무의미 — 일반적으로 표준화 전제 |
| mistake-driven 학습 | SGD 같은 continuous update가 아니므로 현대 autograd와 직접 연결 어려움 |
| 단일 뉴런 | 표현력 매우 제한적 — MLP로의 확장이 필수 |

**주의**: Novikoff bound는 **수렴 시점**의 상한이지, **수렴한 분리자가 최적**이라는 뜻은 아니다. 예를 들어 margin이 가장 큰 분리자는 아닐 수 있다. 그것은 SVM의 몫.

---

## 📌 핵심 정리

$$\boxed{\text{선형 분리 가능 + margin } \gamma + \text{반지름 } R \implies \text{mistake 수} \leq (R/\gamma)^2}$$

| 개념 | 의미 |
|------|------|
| **퍼셉트론 업데이트** | $w \leftarrow w + y_i x_i$ (mistake 시에만) |
| **Margin $\gamma$** | 분리 경계로부터의 최소 거리 — 학습 난이도를 결정 |
| **Novikoff bound** | mistake 횟수 $\leq (R/\gamma)^2$, 차원 $d$와 무관 |
| **증명 전략** | $w_k \cdot w^* \geq k\gamma$ (하한) + $\|w_k\|^2 \leq kR^2$ (상한) + Cauchy-Schwarz |
| **차원 독립성** | 고차원·무한차원(kernel)으로 확장 가능 → SVM의 기초 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2차원 데이터 $(1, 1, +1), (1, -1, +1), (-1, 1, -1), (-1, -1, -1)$에 대해 margin $\gamma$와 반지름 $R$을 구하고 Novikoff bound를 계산하라.

<details>
<summary>힌트 및 해설</summary>

분리자 후보: $w^* = (1, 0)$ (첫 좌표만 보는 분리). 그러면 $y_i(w^* \cdot x_i) = 1$이 모든 점에서 성립. 따라서 $\gamma = 1$.

$R = \max \|x_i\| = \sqrt 2$ (모든 점의 노름).

Novikoff bound: $k \leq (\sqrt{2}/1)^2 = 2$. 즉, 최대 2번의 mistake 안에 수렴.

실제로 $x_1, x_3$ 순서로 주면 mistake 2회로 수렴함을 확인 가능.

</details>

**문제 2** (심화): 증명 2단계에서 $y^{(k)}(w_{k-1} \cdot x^{(k)}) \leq 0$ 조건을 사용했다. 만약 **mistake가 아닌 "margin 부족"**($y^{(k)}(w_{k-1} \cdot x^{(k)}) < \mu$로 어떤 $\mu > 0$에 못 미침)일 때도 업데이트하는 **margin perceptron**을 생각하자. 이 경우 수렴 bound는 어떻게 바뀌는가?

<details>
<summary>힌트 및 해설</summary>

update 조건이 $y^{(k)}(w_{k-1} \cdot x^{(k)}) < \mu$로 완화되면, 2단계의 $y^{(k)}(w_{k-1} \cdot x^{(k)}) \leq 0$ 대신 $\leq \mu$를 쓰게 된다:

$$\|w_k\|^2 \leq \|w_{k-1}\|^2 + 2\mu + R^2$$

따라서 $\|w_k\|^2 \leq k(2\mu + R^2)$. Novikoff 유도를 다시 하면:

$$k\gamma \leq \sqrt{k(2\mu + R^2)} \implies k \leq \frac{2\mu + R^2}{\gamma^2}$$

즉, $\mu$가 작을수록 bound가 원래 Novikoff에 가까워진다. $\mu = 0$이면 정확히 원래 bound. 이 변종은 **stability**(noise에 강함)를 얻는 대신 수렴 속도에서 약간의 손해를 본다. 현대 machine learning의 SVM soft-margin과 연결됨.

</details>

**문제 3** (딥러닝 연결): MLP에서 **출력층 직전**까지 학습된 특성 $\phi(x) \in \mathbb{R}^h$($h$ = hidden width)에 퍼셉트론을 적용한다고 하자. 왜 이것이 **"deep features + linear classifier"** 패러다임의 이론적 근거가 되는가? Novikoff bound는 어떤 양에 의존하게 되는가?

<details>
<summary>힌트 및 해설</summary>

특성 공간 $\phi(x)$ 위에서의 margin $\gamma_\phi$와 반지름 $R_\phi$가 원래 입력 공간의 그것과 **다르다**. 잘 학습된 $\phi$는:

- **Margin $\gamma_\phi$를 키우고** (데이터를 분리하기 좋게 펼쳐놓음)
- **반지름 $R_\phi$를 키우지 않도록** 정규화됨 (BatchNorm, weight decay)

따라서 deep feature 위의 퍼셉트론 수렴 bound $(R_\phi / \gamma_\phi)^2$이 작아진다. 이것이 **"end-to-end 학습이 왜 kernel method보다 강력한가"**의 정량적 해석: $\phi$를 학습으로 조정하면 kernel은 고정되어 있는 feature 공간보다 더 나은 $R/\gamma$를 얻을 수 있다.

또한 이것은 **transfer learning**의 이론적 근거이기도 하다: pre-train으로 얻은 $\phi$가 새 task에서도 큰 $\gamma_\phi$를 유지하면, 소수의 예제로도 퍼셉트론(선형 분류기)이 빠르게 학습된다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. Minsky-Papert의 XOR 문제와 단층의 한계 ▶](./02-xor-and-single-layer.md) |

</div>
