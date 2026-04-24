# 01. 연쇄법칙과 Jacobian: 벡터 미분의 기초

## 🎯 핵심 질문

- **다변수 함수의 미분을 어떻게 정의하고 표기할까?**
  - 스칼라, 벡터, 행렬 간의 미분 표기법의 차이점은?
  - Jacobian 행렬이란 무엇이고, 왜 역전파의 핵심인가?
  
- **연쇄법칙은 벡터/행렬 함수에서 어떻게 일반화되나?**
  - 합성함수 $h = g \circ f$의 도함수를 행렬 곱셈으로 계산할 수 있는가?
  
- **신경망 가중치 미분 $\partial L / \partial W$를 행렬 형태로 표현하려면?**

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

역전파(Backpropagation)는 **연쇄법칙의 행렬 버전**입니다. 신경망의 손실함수 $L$에서 수백만 개의 가중치 $W, b$에 대한 미분을 계산할 때:

- **스칼라 미분**으로는 불충분: $\frac{dL}{dx}$는 1차원에서만 명확
- **Jacobian** 행렬이 필요: 벡터 입출력을 추적하기 위해 행렬 구조 필수
- **연쇄법칙의 행렬 곱셈 형태**: $J_h = J_g \cdot J_f$ → GPU에서 최적화된 GEMM(행렬 곱셈)으로 구현 가능

따라서 **정확한 표기법과 Jacobian 이해 = 역전파 알고리즘의 완전한 이해**입니다.

## 📐 수학적 선행 조건

- 편미분(partial derivative): $\frac{\partial f}{\partial x_i}$의 정의
- 벡터 표기: $\mathbf{x} = (x_1, \ldots, x_n)^T \in \mathbb{R}^n$
- 행렬 표기: $W \in \mathbb{R}^{m \times n}$ 지수와 차원
- Gradient 벡터: $\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$

## 📖 직관적 이해

**Jacobian의 직관:**
- 함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$는 $n$차원 입력을 $m$차원 출력으로 변환
- Jacobian $J_{\mathbf{f}}(x)$는 각 입력의 미소 변화가 각 출력에 미치는 영향을 행렬로 기록
- **행 하나** = 한 출력이 모든 입력에 대해 얼마나 민감한지
- **열 하나** = 한 입력이 모든 출력에 미치는 효과

**신경망 맥락:**
- 역전파는 출력의 손실 $L$에서 역으로 입력까지 Jacobian을 곱하며 내려옴
- 각 계층의 Jacobian을 알면, 그 계층의 가중치에 대한 gradient를 계산 가능

## ✏️ 엄밀한 정의

### 1. 편미분과 Gradient

스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$에 대해:
- 편미분: $\frac{\partial f}{\partial x_i}(x)$ = $x_i$ 방향 변화율
- Gradient 벡터: 
$$\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1}(x) \\ \vdots \\ \frac{\partial f}{\partial x_n}(x) \end{pmatrix} \in \mathbb{R}^n$$

### 2. Jacobian 행렬 (Magnus/혼합 표기법)

벡터 함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, $\mathbf{f}(x) = (f_1(x), \ldots, f_m(x))^T$에 대해:

**분자 레이아웃 (Numerator Layout, 신경망 커뮤니티 표준):**
$$J_{\mathbf{f}}(x) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix} \in \mathbb{R}^{m \times n}$$

즉, $(J_{\mathbf{f}})_{ij} = \frac{\partial f_i}{\partial x_j}$

### 3. 합성함수의 연쇄법칙

함수 합성 $h = g \circ f$, 즉 $\mathbf{h}(x) = \mathbf{g}(\mathbf{f}(x))$에 대해:
- $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^p$
- $\mathbf{g}: \mathbb{R}^p \to \mathbb{R}^m$
- $\mathbf{h}: \mathbb{R}^n \to \mathbb{R}^m$

**행렬 연쇄법칙:**
$$J_{\mathbf{h}}(x) = J_{\mathbf{g}}(\mathbf{f}(x)) \cdot J_{\mathbf{f}}(x)$$

**차원 확인:** $[m \times p] \cdot [p \times n] = [m \times n]$ ✓

### 4. 신경망 가중치 미분의 행렬 형식

선형 변환 $\mathbf{y} = W \mathbf{x} + b$에 대해, 스칼라 손실 $L(\mathbf{y})$이 있을 때:

**가중치 미분:**
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial W}$$

행렬 형태로:
$$\frac{\partial L}{\partial W} = \left(\nabla_{\mathbf{y}} L\right) \cdot \mathbf{x}^T \in \mathbb{R}^{m \times n}$$

이때 $\nabla_{\mathbf{y}} L \in \mathbb{R}^m$ (벡터)이고, 이를 $\mathbf{x}^T \in \mathbb{R}^{1 \times n}$과 행렬곱하면 $\mathbb{R}^{m \times n}$ 획득.

### 5. Kronecker 곱과 벡터화

가중치를 벡터로 변환하여 다루려면:
- $\text{vec}(W) \in \mathbb{R}^{mn}$: $W$의 열들을 순서대로 쌓은 벡터
- **Kronecker 곱**: $A \otimes B$
- **성질**: $\text{vec}(AXB) = (B^T \otimes A) \text{vec}(X)$

예시: $\frac{\partial L}{\partial \text{vec}(W)}^T = (\mathbf{x}^T \otimes \nabla_{\mathbf{y}} L)$

## 🔬 정리와 증명

**정리 1.1 (벡터 연쇄법칙)**

$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^p$, $\mathbf{g}: \mathbb{R}^p \to \mathbb{R}^m$이 미분가능하면, $\mathbf{h} = \mathbf{g} \circ \mathbf{f}$의 Jacobian은:
$$J_{\mathbf{h}}(x) = J_{\mathbf{g}}(\mathbf{f}(x)) \cdot J_{\mathbf{f}}(x)$$

**증명:**
$\mathbf{h}_i(x) = g_i(\mathbf{f}(x))$이므로,

$$\frac{\partial h_i}{\partial x_j} = \sum_{k=1}^{p} \frac{\partial g_i}{\partial f_k} \frac{\partial f_k}{\partial x_j}$$

이는 행렬 곱셈의 정의 $(AB)_{ij} = \sum_k A_{ik} B_{kj}$와 정확히 일치합니다. $\square$

**정리 1.2 (신경망 가중치 미분)**

입력 $\mathbf{x}$, 가중치 $W$, 편향 $\mathbf{b}$에 대해:
$$\mathbf{y} = W \mathbf{x} + \mathbf{b}$$
$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial \mathbf{y}}\right) \mathbf{x}^T$$

**증명:** $y_i = \sum_j W_{ij} x_j + b_i$이므로,
$$\frac{\partial y_i}{\partial W_{kl}} = \delta_{ik} x_l$$

연쇄법칙:
$$\frac{\partial L}{\partial W_{kl}} = \sum_i \frac{\partial L}{\partial y_i} \frac{\partial y_i}{\partial W_{kl}} = \frac{\partial L}{\partial y_k} x_l$$

따라서 $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^T$ (차원: $[m \times 1] \cdot [1 \times n] = [m \times n]$). $\square$

## 💻 NumPy로 바닥부터 구현

```python
import numpy as np
from scipy.optimize import approx_fprime

# 1. Jacobian 수치 미분으로 계산
def numerical_jacobian(f, x, eps=1e-5):
    """
    f: R^n -> R^m 함수
    x: 입력 점
    반환: Jacobian 행렬 [m x n]
    """
    m = f(x).shape[0] if f(x).ndim > 0 else 1
    n = x.shape[0]
    J = np.zeros((m, n))
    
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        x_minus = x.copy()
        x_minus[j] -= eps
        
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        J[:, j] = (f_plus - f_minus) / (2 * eps)
    
    return J

# 2. 예제: 벡터 함수 정의
def f_example(x):
    """f(x) = [x[0]^2 + x[1], x[0] * x[1], sin(x[0])]"""
    return np.array([
        x[0]**2 + x[1],
        x[0] * x[1],
        np.sin(x[0])
    ])

# 3. 해석적 Jacobian 계산 (이 예제)
def jacobian_analytical(x):
    """손으로 계산한 Jacobian"""
    return np.array([
        [2*x[0], 1],
        [x[1], x[0]],
        [np.cos(x[0]), 0]
    ])

# 4. 검증
x_test = np.array([1.5, 2.0])
J_num = numerical_jacobian(f_example, x_test)
J_ana = jacobian_analytical(x_test)

print("=== Jacobian 비교 ===")
print(f"Numerical:\n{J_num}")
print(f"Analytical:\n{J_ana}")
print(f"오차 (max abs diff): {np.max(np.abs(J_num - J_ana)):.2e}")

# 5. 연쇄법칙 검증: h = g ∘ f
def g_example(y):
    """g(y) = [y[0] + y[1], y[0]^2]"""
    return np.array([
        y[0] + y[1],
        y[0]**2
    ])

def h_example(x):
    return g_example(f_example(x))

def jacobian_g_analytical(y):
    return np.array([
        [1, 1, 0],
        [2*y[0], 0, 0]
    ])

# 연쇄법칙: J_h = J_g(f(x)) * J_f(x)
J_f = jacobian_analytical(x_test)
f_x = f_example(x_test)
J_g = jacobian_g_analytical(f_x)
J_h_chain = J_g @ J_f  # 행렬 곱셈

J_h_numerical = numerical_jacobian(h_example, x_test)

print("\n=== 연쇄법칙 검증 ===")
print(f"J_h (chain rule): J_g * J_f =\n{J_h_chain}")
print(f"J_h (numerical):\n{J_h_numerical}")
print(f"오차: {np.max(np.abs(J_h_chain - J_h_numerical)):.2e}")

# 6. 신경망 가중치 미분: y = Wx + b
def test_linear_jacobian():
    W = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    b = np.array([0.5, -0.5])
    x = np.array([1.0, 2.0])
    
    # y = Wx + b
    y = W @ x + b
    
    # Jacobian: ∂y/∂W[i,j] = δ_{ki} x_j
    # 즉, 행렬 형태: J = [x^T; x^T; ...] (각 출력마다 x^T)
    J = np.zeros((2, 4))  # 2 outputs, 4 weights (2x2 matrix)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                if k == i:
                    J[k, i*2 + j] = x[j]
    
    # 수치 미분으로 검증
    def linear_func(W_vec):
        W_temp = W_vec.reshape((2, 2))
        return W_temp @ x + b
    
    W_vec = W.flatten()
    J_num = numerical_jacobian(linear_func, W_vec)
    
    print("\n=== 선형 변환의 가중치 Jacobian ===")
    print(f"Analytical J:\n{J}")
    print(f"Numerical J:\n{J_num}")
    print(f"오차: {np.max(np.abs(J - J_num)):.2e}")

test_linear_jacobian()
```

**실행 결과 예상:**
```
=== Jacobian 비교 ===
Numerical:
[[ 3.  1. ]
 [ 2.  1.5]
 [ 0.07 0. ]]
Analytical:
[[ 3.  1. ]
 [ 2.  1.5]
 [ 0.07 0. ]]
오차 (max abs diff): 1.23e-08

=== 연쇄법칙 검증 ===
J_h (chain rule): J_g * J_f =
[[...]]
오차: 2.45e-07

=== 선형 변환의 가중치 Jacobian ===
오차: 5.67e-09
```

## 🔗 실전 연결

1. **자동 미분 엔진 (PyTorch, TensorFlow):**
   - 연산 그래프의 각 노드가 자신의 Jacobian을 저장
   - 역전파 = 출력에서 입력으로 Jacobian을 chain rule로 곱하는 것

2. **신경망 층 구현:**
   ```python
   # 선형층: y = Wx + b
   # ∇_W L = (∇_y L) ⊗ x^T (외적)
   # ∇_x L = W^T ∇_y L
   ```

3. **수치 안정성:**
   - Gradient checking: 해석적 미분과 수치 미분이 1e-5 이내 일치 확인
   - 행렬 미분의 trace trick으로 계산 복잡도 감소

## ⚖️ 가정과 한계

- **미분가능성 가정**: 모든 함수가 미분가능해야 함 (ReLU처럼 꺾이는 함수는 거의 모든 점에서 미분가능)
- **표기법 선택**: 이 문서는 **분자 레이아웃(numerator layout)**을 따름 (신경망 커뮤니티 표준)
- **계산 복잡도**: 수치 Jacobian은 $O(n \cdot \text{forward evaluations})$ → 역전파 방식이 훨씬 효율적
- **행렬 순서**: 신경망에서 $W \in \mathbb{R}^{m \times n}$일 때, forward는 $y = Wx$ (행 기준), backward는 $\partial L / \partial W = (\partial L / \partial y) x^T$ (행렬 곱셈 규칙)

## 📌 핵심 정리

| 개념 | 정의 | 신경망에서의 역할 |
|------|------|------------------|
| **Jacobian** | $J_f \in \mathbb{R}^{m \times n}$, $(J_f)_{ij} = \partial f_i / \partial x_j$ | 각 계층의 입출력 민감도 추적 |
| **연쇄법칙** | $J_{g \circ f} = J_g(f) \cdot J_f$ | 역전파의 수학적 기초 |
| **가중치 미분** | $\partial L/\partial W = (\partial L/\partial y) x^T$ | 경사 하강법 업데이트 계산 |
| **벡터화** | $\text{vec}(AXB) = (B^T \otimes A) \text{vec}(X)$ | 배치 처리 최적화 |

**핵심 수식:**
$$\boxed{J_{\mathbf{h}}(x) = J_{\mathbf{g}}(\mathbf{f}(x)) \cdot J_{\mathbf{f}}(x), \quad \frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^T}$$

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1:</b> ReLU 함수 $\sigma(z) = \max(0, z)$의 Jacobian을 구하시오.</summary>

**해답:**
ReLU는 구간별 선형 함수입니다:
$$\sigma(z) = \begin{cases} z & z > 0 \\ 0 & z \leq 0 \end{cases}$$

따라서 (z ≠ 0에서):
$$\frac{\partial \sigma}{\partial z_i} = \begin{cases} 1 & z_i > 0 \\ 0 & z_i < 0 \end{cases}$$

벡터 입력 $z = (z_1, \ldots, z_n)$에 대해, Jacobian은 **대각 행렬**:
$$J_\sigma(z) = \text{diag}(\mathbb{1}_{z > 0}) = \begin{pmatrix} \mathbb{1}_{z_1 > 0} & & \\ & \ddots & \\ & & \mathbb{1}_{z_n > 0} \end{pmatrix}$$

역전파에서 upstream gradient $\delta$가 들어오면:
$$\delta_{\text{input}} = \delta \odot \mathbb{1}_{z > 0}$$
(여기서 $\odot$는 element-wise 곱)
</details>

<details>
<summary><b>문제 2:</b> $\text{vec}(ABC)$ 형태의 미분을 Kronecker 곱으로 표현하시오. 차원도 함께 확인하시오.</summary>

**해답:**
$A \in \mathbb{R}^{p \times q}$, $B \in \mathbb{R}^{q \times r}$, $C \in \mathbb{R}^{r \times s}$일 때:

$$\text{vec}(ABC) = (C^T \otimes A) \text{vec}(B)$$

**증명:** $\text{vec}(AXB) = (B^T \otimes A) \text{vec}(X)$ 공식을 $X = BC$에 적용하면:
$$\text{vec}(A(BC)) = (C^T \otimes A) \text{vec}(B)$$

**차원 확인:**
- $\text{vec}(ABC) \in \mathbb{R}^{ps}$ (왜냐하면 $ABC \in \mathbb{R}^{p \times s}$)
- $C^T \in \mathbb{R}^{s \times r}$, $A \in \mathbb{R}^{p \times q}$
- $C^T \otimes A \in \mathbb{R}^{ps \times qr}$
- $\text{vec}(B) \in \mathbb{R}^{qr}$
- $(ps \times qr) \cdot (qr) = (ps)$ ✓
</details>

<details>
<summary><b>문제 3:</b> 배치 처리에서 $Z = XW^T + b$ (X ∈ ℝ^{B×n}, W ∈ ℝ^{m×n})일 때, ∂L/∂W를 구하시오. (행렬 표기)</summary>

**해답:**
각 샘플 $i$에 대해 $z_i = x_i W^T + b$이고, 손실은 모든 샘플의 합 $L = \sum_i L_i(z_i)$입니다.

$$\frac{\partial L}{\partial W} = \sum_{i=1}^{B} \frac{\partial L_i}{\partial z_i} x_i$$

벡터 형태로 (모든 샘플을 한번에):
- $\Delta \in \mathbb{R}^{B \times m}$: $\Delta_{ij} = \frac{\partial L_i}{\partial z_{ij}}$ (batch의 각 샘플, 각 출력)
- $X \in \mathbb{R}^{B \times n}$

$$\boxed{\frac{\partial L}{\partial W} = \Delta^T X \in \mathbb{R}^{m \times n}}$$

**검증:** $\Delta^T \in [m \times B]$, $X \in [B \times n]$ → $[m \times n]$ ✓

이는 BLAS의 GEMM(General Matrix Multiply)으로 GPU 최적화 가능합니다.
</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch2-05. Barron 근사율](../ch2-universal-approximation/05-barron-rate.md) | [📚 README로 돌아가기](../README.md) | [02. Computational Graph와 AD ▶](./02-computational-graph-ad.md) |

</div>
