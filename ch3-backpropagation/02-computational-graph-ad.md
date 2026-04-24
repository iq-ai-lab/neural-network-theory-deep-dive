# 02. Computational Graph와 Automatic Differentiation

## 🎯 핵심 질문

- **연산을 어떻게 구조화하면 미분이 체계적으로 가능할까?**
  - 신경망의 forward pass를 그래프로 표현하면?
  - 각 노드가 저장해야 할 정보는 무엇인가?

- **자동 미분(Automatic Differentiation, AD)의 두 가지 방식은 무엇인가?**
  - Forward-mode AD: 입력부터 출력으로 (top-down 미분)
  - Reverse-mode AD: 출력부터 입력으로 (bottom-up 미분) = 역전파

- **신경망에서는 왜 reverse-mode가 필수적인가?**
  - 입력 차원 $n \sim 10^6$, 출력(손실) $m = 1$일 때 계산 복잡도는?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

PyTorch, TensorFlow, JAX 같은 모든 현대 딥러닝 프레임워크의 핵심은 **자동 미분(AD)**입니다:

- **Computational graph** = 연산의 DAG(방향성 비순환 그래프)로 표현
  - 각 연산을 노드로, 데이터 흐름을 엣지로 기록
  - Forward pass 중 이 그래프 자동 구성

- **Forward-mode AD**
  - 입력의 미소 변화를 추적하며 출력까지 전파
  - 입력 하나당 forward pass 한 번 필요 → $O(n)$ passes, 입력 많을 때 비효율

- **Reverse-mode AD (= Backpropagation)**
  - 손실에서 시작해서 역방향으로 출력 → 입력까지 미분 전파
  - 출력 하나(손실)당 backward pass 한 번 필요 → $O(1)$ pass, **모든 파라미터 gradient 동시 계산**
  - $10^6$개 파라미터: forward-mode는 $10^6$배 느림, reverse-mode는 $1$번

따라서 **AD의 원리 이해 = 현대 신경망 학습의 근본 메커니즘 이해**입니다.

## 📐 수학적 선행 조건

- 편미분과 Jacobian (01번 문서의 내용)
- 연쇄법칙: $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$
- Dual number: $x + \dot{x}\epsilon$, $\epsilon^2 = 0$ (형식적 확장)
- 행렬 곱셈의 시간 복잡도: $O(n \cdot m \cdot p)$ for $[n \times m] \cdot [m \times p]$

## 📖 직관적 이해

### Forward-Mode AD의 직관

신경망을 일련의 연산으로 본다:
$$x_0 \xrightarrow{f_1} x_1 \xrightarrow{f_2} \cdots \xrightarrow{f_L} x_L = L$$

**Forward-mode:**
- 입력 $x_0$에 대한 미분만 관심
- $\dot{x}_0$를 "기초 방향"으로 설정 (예: $\dot{x}_0 = e_i$, $i$번째 입력 축)
- Forward pass를 수행하면서 동시에 미분도 추적: $\dot{x}_k = J_{f_k}(x_{k-1}) \dot{x}_{k-1}$
- 결과: 특정 입력 방향에 대한 모든 중간값과 손실의 미분

**문제**: 입력이 100만 개면 100만 번 forward pass 필요!

### Reverse-Mode AD의 직관

**Reverse-mode:**
- 손실 $L$에서 시작: $\bar{L} = 1$
- 역방향으로 upstream gradient 전파: $\bar{x}_{k-1} = (J_{f_k}(x_{k-1}))^T \bar{x}_k$
- 결과: 모든 입력에 대한 미분을 **단 한 번**의 backward pass로 계산!

**장점**: 손실은 스칼라 $m=1$ → 한 번의 backward로 모든 파라미터 gradient 획득

---

## ✏️ 엄밀한 정의

### 1. Computational Graph

**정의:** 신경망의 연산을 방향성 비순환 그래프(DAG)로 표현
$$G = (V, E)$$
- **노드** $V$: 각 연산 또는 변수
  - 입력 노드, 중간 연산 노드, 출력 노드
- **엣지** $E$: 데이터 흐름
  - $u \to v$: $u$의 출력이 $v$의 입력으로 사용됨

**각 노드가 저장하는 정보:**
- Forward pass에서: $x_k$ (연산의 출력값)
- Backward pass를 위해: 자신의 Jacobian $J_k$ 또는 그 계산에 필요한 정보

### 2. Forward-Mode Automatic Differentiation

**정의:** 입력에서 출력으로 미분을 전파하는 방식

**Dual Number 표현:**
임의의 변수를 $x + \dot{x}\epsilon$로 표현, 여기서 $\epsilon^2 = 0$ (형식적 기호)

$$\begin{align}
(a + \dot{a}\epsilon) + (b + \dot{b}\epsilon) &= (a+b) + (\dot{a}+\dot{b})\epsilon \\
(a + \dot{a}\epsilon) \cdot (b + \dot{b}\epsilon) &= ab + (a\dot{b} + b\dot{a})\epsilon \quad (\text{곱의 미분})
\end{align}$$

**알고리즘:**
입력 $x = [x_1, \ldots, x_n]^T$에 대한 출력 $y = f(x)$의 Jacobian $J \in \mathbb{R}^{m \times n}$을 구하려면:

```
for i = 1 to n:
    x̊ ← x + e_i * ε      // e_i = i번째 표준 기저 벡터
    ẙ ← forward_pass(x̊)  // dual number로 계산
    J[:, i] ← ẏ         // y의 비표준 부분
```

**복잡도:** $m \times n$ Jacobian을 구하는 데 $O(n \times T)$ (여기서 $T$ = primitive 연산 수)

### 3. Reverse-Mode Automatic Differentiation (Backpropagation)

**정의:** 출력에서 입력으로 미분을 전파하는 방식

**Adjoint (보수 변수) 표현:**
각 변수 $x_k$에 대해 "adjoint" $\bar{x}_k := \frac{\partial L}{\partial x_k}$를 정의

**알고리즘:**

```
1. Forward pass: x₀ → x₁ → ... → x_L (모든 x_k 저장)
2. Initialize: x̄_L ← 1 (손실은 스칼라)
3. Backward pass (l = L-1, ..., 0):
   x̄_l ← (∂f_{l+1}/∂x_l)^T · x̄_{l+1}
```

**VJP (Vector-Jacobian Product) 해석:**
$$\bar{x}_l = J_{f_{l+1}}(x_l)^T \bar{x}_{l+1}$$

**복잡도:** $m \times n$ Jacobian을 구하는 데 $O(m \times T)$

### 4. JVP vs VJP

| 연산 | Forward-Mode | Reverse-Mode |
|------|-------------|-------------|
| **JVP** (Jacobian-Vector Product) | $J \mathbf{v}$ | - |
| **VJP** (Vector-Jacobian Product) | - | $\mathbf{u}^T J$ |
| 입력이 $n$ 차원, 출력이 $m$ 차원일 때 비용 | $O(n)$ passes | $O(m)$ passes |
| **신경망** ($n \sim 10^6$, $m = 1$) | 매우 비효율 | 매우 효율적 |

## 🔬 정리와 증명

**정리 2.1 (Forward-Mode AD의 정확성)**

Primitive 연산들이 미분가능하면, forward-mode AD로 계산된 $\dot{y}$는 정확히:
$$\dot{y} = J_f(x) \dot{x}$$
를 만족합니다.

**증명 스케치:**
Dual number 연산의 정의에 의해, 각 primitive 연산 $g(a, b)$에 대해:
$$g(a + \dot{a}\epsilon, b + \dot{b}\epsilon) = g(a,b) + \left(\frac{\partial g}{\partial a}\dot{a} + \frac{\partial g}{\partial b}\dot{b}\right)\epsilon$$

연쇄 계산으로 전체 함수 $f$에 대해:
$$f(x + \dot{x}\epsilon) = f(x) + J_f(x)\dot{x} \epsilon$$

따라서 dual part를 추출하면 정확한 미분을 얻습니다. $\square$

**정리 2.2 (Reverse-Mode AD의 정확성)**

Primitive 연산들이 미분가능하고, backward pass에서 각 단계의 VJP를 정확히 계산하면:
$$\bar{x}_l = \frac{\partial L}{\partial x_l}$$

**증명 스케치:**
귀납법으로, $\bar{x}_L = \frac{\partial L}{\partial x_L} = 1$ (초기값)에서:
$$\bar{x}_l = \frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial x_l} = \bar{x}_{l+1} \cdot J_{f_{l+1}}(x_l)$$

즉, adjoint의 연쇄 계산이 정확한 편미분을 구합니다. $\square$

**정리 2.3 (복잡도 비교)**

$f: \mathbb{R}^n \to \mathbb{R}^m$를 $T$개의 primitive 연산으로 구성할 때:

- **Forward-mode**: $O(n \cdot T)$ (입력 $n$ 각각에 대해 forward pass)
- **Reverse-mode**: $O(m \cdot T)$ (출력 $m$ 각각에 대해 backward pass)
- **신경망 ($m = 1$ 손실)**: Reverse-mode가 forward-mode보다 $\frac{n}{1} = n$배 빠름

## 💻 NumPy로 바닥부터 구현

### Dual Number 클래스 (Forward-Mode)

```python
import numpy as np

class DualNumber:
    """Dual number: x + ẋ*ε, ε² = 0"""
    def __init__(self, real, dual):
        self.real = np.asarray(real, dtype=np.float32)
        self.dual = np.asarray(dual, dtype=np.float32)
    
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, 
                            self.dual + other.dual)
        return DualNumber(self.real + other, self.dual)
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real,
                            self.real * other.dual + self.dual * other.real)
        return DualNumber(self.real * other, self.dual * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real / other.real,
                            (self.dual * other.real - self.real * other.dual) / (other.real**2))
        return DualNumber(self.real / other, self.dual / other)
    
    def sin(self):
        return DualNumber(np.sin(self.real), 
                         np.cos(self.real) * self.dual)
    
    def exp(self):
        exp_real = np.exp(self.real)
        return DualNumber(exp_real, exp_real * self.dual)
    
    def __repr__(self):
        return f"Dual({self.real}, {self.dual}*ε)"

# 예제: f(x₁, x₂) = sin(x₁) + x₁*x₂
def f_forward_mode(x1, x2):
    return DualNumber(np.sin(x1.real) + x1.real * x2.real,
                     np.cos(x1.real) * x1.dual + x1.dual * x2.real + x1.real * x2.dual)

# Forward-mode 자동 미분
x1_val = 1.5
x2_val = 2.0

# J[1] = ∂f/∂x₁
x1_forward = DualNumber(x1_val, 1.0)
x2_forward = DualNumber(x2_val, 0.0)
result1 = f_forward_mode(x1_forward, x2_forward)
print("=== Forward-Mode AD ===")
print(f"∂f/∂x₁ at (x₁={x1_val}, x₂={x2_val}): {result1.dual:.6f}")

# J[2] = ∂f/∂x₂
x1_forward = DualNumber(x1_val, 0.0)
x2_forward = DualNumber(x2_val, 1.0)
result2 = f_forward_mode(x1_forward, x2_forward)
print(f"∂f/∂x₂ at (x₁={x1_val}, x₂={x2_val}): {result2.dual:.6f}")
```

### Reverse-Mode AD (Backpropagation 수동 구현)

```python
class Variable:
    """계산 그래프의 노드"""
    def __init__(self, value, children=None, op_name=""):
        self.value = value
        self.children = children or []  # (parent_var, jacobian) 튜플
        self.op_name = op_name
        self.grad = None
    
    def backward(self, grad=None):
        """역전파: VJP 계산"""
        if grad is None:
            grad = 1.0  # 손실은 스칼라
        
        self.grad = grad
        
        for parent, jvp_func in self.children:
            parent_grad = jvp_func(grad)  # VJP: u^T J
            if parent.grad is None:
                parent.grad = parent_grad
            else:
                parent.grad += parent_grad

def add_variable(a, b):
    """덧셈: z = a + b"""
    if isinstance(a, (int, float)):
        a = Variable(a)
    if isinstance(b, (int, float)):
        b = Variable(b)
    
    z = Variable(a.value + b.value, op_name="add")
    z.children.append((a, lambda grad: grad))
    z.children.append((b, lambda grad: grad))
    return z

def mul_variable(a, b):
    """곱셈: z = a * b"""
    if isinstance(a, (int, float)):
        a = Variable(a)
    if isinstance(b, (int, float)):
        b = Variable(b)
    
    z = Variable(a.value * b.value, op_name="mul")
    z.children.append((a, lambda grad, b_val=b.value: grad * b_val))
    z.children.append((b, lambda grad, a_val=a.value: grad * a_val))
    return z

def sin_variable(a):
    """sin 함수: z = sin(a)"""
    z = Variable(np.sin(a.value), op_name="sin")
    z.children.append((a, lambda grad, a_val=a.value: grad * np.cos(a_val)))
    return z

# 예제: f(x₁, x₂) = sin(x₁) + x₁*x₂
print("\n=== Reverse-Mode AD (Backpropagation) ===")
x1 = Variable(1.5, op_name="x1")
x2 = Variable(2.0, op_name="x2")

z1 = sin_variable(x1)      # z₁ = sin(x₁)
z2 = mul_variable(x1, x2)  # z₂ = x₁*x₂
f = add_variable(z1, z2)   # f = z₁ + z₂

f.backward()  # 역전파

print(f"f({x1.value}, {x2.value}) = {f.value:.6f}")
print(f"∂f/∂x₁ = {x1.grad:.6f}")
print(f"∂f/∂x₂ = {x2.grad:.6f}")

# 해석적 확인
analytical_df_dx1 = np.cos(1.5) + 2.0
analytical_df_dx2 = 1.5
print(f"\nAnalytical ∂f/∂x₁ = {analytical_df_dx1:.6f}")
print(f"Analytical ∂f/∂x₂ = {analytical_df_dx2:.6f}")
```

### MLP Forward + Backward 비교

```python
# 간단한 신경망: y = σ(Wx + b)
def forward_mlp(x, W, b, sigma):
    z = W @ x + b
    a = sigma(z)
    return a, z

def reverse_mode_gradient(x, W, b, loss_grad):
    """손실 gradient가 주어졌을 때, x와 W에 대한 gradient 계산"""
    # Forward
    z = W @ x + b
    a = 1 / (1 + np.exp(-z))  # sigmoid
    
    # Backward
    dL_da = loss_grad
    dL_dz = dL_da * a * (1 - a)  # σ' = σ(1-σ)
    dL_dW = np.outer(dL_dz, x)   # ∂L/∂W = (∂L/∂z) x^T
    dL_db = dL_dz
    dL_dx = W.T @ dL_dz
    
    return dL_dW, dL_db, dL_dx

# 테스트
np.random.seed(42)
x = np.array([1.0, 2.0, 3.0])
W = np.random.randn(2, 3)
b = np.random.randn(2)
loss_grad = np.array([0.1, -0.05])

dL_dW, dL_db, dL_dx = reverse_mode_gradient(x, W, b, loss_grad)

print("\n=== MLP Reverse-Mode Gradient ===")
print(f"∂L/∂W shape: {dL_dW.shape}")
print(f"∂L/∂b: {dL_db}")
print(f"∂L/∂x: {dL_dx}")

# 수치 미분으로 검증
def numerical_gradient_W(x, W, b, loss_grad, eps=1e-5):
    dL_dW = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[i, j] += eps
            z_plus = W_plus @ x + b
            a_plus = 1 / (1 + np.exp(-z_plus))
            loss_plus = np.sum(loss_grad * a_plus)
            
            W_minus = W.copy()
            W_minus[i, j] -= eps
            z_minus = W_minus @ x + b
            a_minus = 1 / (1 + np.exp(-z_minus))
            loss_minus = np.sum(loss_grad * a_minus)
            
            dL_dW[i, j] = (loss_plus - loss_minus) / (2 * eps)
    return dL_dW

dL_dW_numerical = numerical_gradient_W(x, W, b, loss_grad)
print(f"\n수치 미분 오차 (max abs diff): {np.max(np.abs(dL_dW - dL_dW_numerical)):.2e}")
```

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 연쇄법칙과 Jacobian](./01-chain-rule-jacobian.md) | [📚 README로 돌아가기](../README.md) | [03. Reverse-Mode AD = Backpropagation ▶](./03-reverse-mode-backprop.md) |

</div>
