# 02. Vanishing/Exploding Gradient의 수학적 분석

## 🎯 핵심 질문

- RNN에서 **시간이 길어질수록 역전파된 기울기가 0으로 수렴**하거나 **폭발**하는 이유는 무엇인가?
- 이것이 LSTM/GRU와 같은 고급 구조를 만들게 한 **핵심 동기**인가?
- 스펙트럼 반지름(spectral radius) $\rho(W_{hh})$가 어떻게 기울기 크기를 지배하는가?
- Gradient Clipping, Weight Normalization 등의 실전 기법은 이론적으로 어떻게 정당화되는가?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

Vanishing/Exploding Gradient는:

1. **RNN 실패의 근본 원인**: 20년간 RNN을 쓸모없게 만들었던 문제의 수학적 원인을 설명한다
2. **LSTM/GRU 설계의 근거**: 왜 "skip connection" 같은 구조가 필요한지 이론적으로 정당화한다
3. **정규화 기법의 선택**: Gradient Clipping vs. Layer Normalization vs. Spectral Normalization의 차이를 이해할 수 있다
4. **하이퍼파라미터 선택**: $W_{hh}$의 초기 스펙트럼 반지름을 어떻게 설정할지 결정할 근거를 제공한다

## 📐 수학적 선행 조건

- **행렬 노름(Matrix Norm)**: Frobenius norm $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$, spectral norm $\|A\|_2 = \max_{\|x\|=1} \|Ax\|$
- **고유값(Eigenvalue)과 스펙트럼**: 행렬 $A$의 스펙트럼 반지름 $\rho(A) = \max_i |\lambda_i(A)|$
- **Jordan Canonical Form**: 행렬을 Jordan block으로 분해하는 이론
- **연쇄법칙과 행렬 곱 미분**: Jacobian 행렬의 곱 성질

## 📖 직관적 이해

**Simple Example**:

RNN에서 숨은 상태는 매 시간 $h_t = \tanh(W_{hh} h_{t-1} + \cdots)$로 업데이트된다.

손실이 시간 1에서의 입력과 가중치에 어떻게 의존하는지 보려면:

$$\frac{\partial L}{\partial h_0} \propto \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}}$$

만약 각 항 $\frac{\partial h_t}{\partial h_{t-1}} = 0.5$라면:
- $T=5$: 기울기는 $(0.5)^5 = 0.03$으로 **거의 0에 가까워진다** (Vanishing)

반대로 각 항이 $\frac{\partial h_t}{\partial h_{t-1}} = 2$라면:
- $T=5$: 기울기는 $2^5 = 32$로 **폭발한다** (Exploding)

**핵심 직관**: $\frac{\partial h_t}{\partial h_{t-1}}$의 크기가 1보다 작으면 작을수록, 또는 1보다 크면 클수록, 곱해지면서 지수적으로 증폭된다.

이 도함수는 $W_{hh}$와 활성화 함수에 의해 결정되므로, $W_{hh}$의 크기(스펙트럼 반지름)가 중요하다.

## ✏️ 엄밀한 정의

**정의 (스펙트럼 반지름)**:

행렬 $A \in \mathbb{R}^{n \times n}$의 스펙트럼 반지름은:

$$\rho(A) = \max_{i=1,\ldots,n} |\lambda_i|$$

여기서 $\lambda_i$는 $A$의 고유값이다.

**정의 (Gradient Norm)**:

시간 0에서 시간 $t$로의 기울기 전파를 나타내는 Jacobian 곱:

$$\frac{\partial h_t}{\partial h_0} = \prod_{k=1}^t \frac{\partial h_k}{\partial h_{k-1}}$$

이의 노름(spectral norm):

$$\left\| \frac{\partial h_t}{\partial h_0} \right\|_2 = \text{largest singular value of the product}$$

**정의 (Vanishing/Exploding Gradient)**:

- **Vanishing**: $\lim_{T \to \infty} \left\| \frac{\partial L}{\partial h_0} \right\| = 0$ (또는 매우 빠르게 감소)
- **Exploding**: $\lim_{T \to \infty} \left\| \frac{\partial L}{\partial h_0} \right\| = \infty$ (또는 매우 빠르게 증가)

## 🔬 정리와 증명

**정리 1 (Hochreiter et al., 2001; Pascanu et al., 2013)**:

Vanilla RNN에서 $h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$라 하면, 손실 $L_t$에 대해:

$$\frac{\partial h_t}{\partial h_0} = \prod_{k=1}^t W_{hh}^T \text{diag}(\sigma'(z_k))$$

여기서 $z_k = W_{hh}h_{k-1} + W_{xh}x_k + b_h$.

따라서:

$$\left\| \frac{\partial h_t}{\partial h_0} \right\| \leq \prod_{k=1}^t \|W_{hh}^T\| \cdot \|\text{diag}(\sigma'(z_k))\|$$

$\|W_{hh}^T\| = \|W_{hh}\|_2 = \rho(W_{hh})$이고, $\|\text{diag}(\sigma'(z_k))\| \leq \max_z |\sigma'(z)|$이므로:

$$\left\| \frac{\partial h_t}{\partial h_0} \right\| \leq \left(\rho(W_{hh}) \cdot \max_z |\sigma'(z)|\right)^t$$

따라서:
- $\rho(W_{hh}) \cdot \max |\sigma'| < 1$ ⟹ **Vanishing** ($\propto \gamma^t$, $\gamma < 1$)
- $\rho(W_{hh}) \cdot \max |\sigma'| > 1$ ⟹ **Exploding** ($\propto \gamma^t$, $\gamma > 1$)

**증명**:

$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$에서:

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\sigma'(z_t)) \cdot W_{hh}$$

(여기서 $\sigma'(z_t)$는 벡터이고, diag(·)는 대각 행렬화)

따라서:

$$\frac{\partial h_t}{\partial h_0} = \frac{\partial h_t}{\partial h_{t-1}} \cdots \frac{\partial h_1}{\partial h_0}$$

$$= \text{diag}(\sigma'(z_t)) W_{hh} \text{diag}(\sigma'(z_{t-1})) W_{hh} \cdots \text{diag}(\sigma'(z_1)) W_{hh}$$

각 행렬의 노름을 계산하면:

$$\|AB\|_2 \leq \|A\|_2 \|B\|_2$$

따라서:

$$\left\|\frac{\partial h_t}{\partial h_0}\right\|_2 \leq \prod_{k=1}^t \left\|\text{diag}(\sigma'(z_k))\right\|_2 \cdot \|W_{hh}\|_2$$

$$\leq \prod_{k=1}^t \max_k |\sigma'(z_k)| \cdot \rho(W_{hh})$$

$$\leq \left(\rho(W_{hh}) \cdot \max_z |\sigma'(z)|\right)^t$$

$\square$

**정리 2 (활성화 함수별 Critical Point)**:

- **Sigmoid**: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$
  - Critical: $\rho(W_{hh}) > 4$ ⟹ Exploding
  - Safe: $\rho(W_{hh}) < 4$ (tanh 함께 사용 시 더 엄격)

- **Tanh**: $\sigma'(z) = 1 - \tanh^2(z) \leq 1$
  - Critical: $\rho(W_{hh}) > 1$ ⟹ Exploding
  - Safe: $\rho(W_{hh}) < 1$

- **ReLU**: $\sigma'(z) = \mathbb{1}[z > 0]$ (0 또는 1)
  - 이론적으로 극단적이지만, dead ReLU 때문에 실전에서 성능 저하

**정리 3 (추가 bounded)**:

만약 $\rho(W_{hh}) < \frac{1}{\max_z |\sigma'(z)|}$라면, 기울기는 기하급수적으로 감소한다:

$$\left\|\frac{\partial L}{\partial h_t}\right\| \lesssim \lambda^t, \quad \lambda = \rho(W_{hh}) \cdot \max |\sigma'| < 1$$

## 💻 NumPy로 바닥부터 구현

**Spectral Radius와 Gradient Norm의 관계 실험**:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def compute_gradient_flow(W_hh, X, T=50, activation='tanh'):
    """
    RNN의 기울기 흐름을 시뮬레이션
    
    Args:
        W_hh: (d_h, d_h) 순환 가중치 행렬
        X: (T, d_x) 입력 시퀀스
        T: 시간 단계
        activation: 활성화 함수 종류
    
    Returns:
        gradient_norms: 각 시간 단계에서의 기울기 노름
    """
    d_h = W_hh.shape[0]
    d_x = X.shape[1]
    
    # W_xh는 무시하고 W_hh의 효과만 계산
    h = np.random.randn(d_h, 1)
    gradient_norms = []
    
    # Forward pass
    h_history = [h]
    z_history = []
    
    for t in range(T):
        x = X[t:t+1].T
        # 간단히: z = W_hh @ h + x를 input으로 봄
        z = W_hh @ h + x[:1]  # x의 첫 차원만
        
        if activation == 'tanh':
            h = np.tanh(z)
        elif activation == 'sigmoid':
            h = sigmoid(z)
        else:
            h = np.maximum(0, z)  # ReLU
        
        h_history.append(h)
        z_history.append(z)
    
    # Backward pass: 각 시간에서 기울기 Jacobian 곱 계산
    for t in range(T):
        # 시간 0에서 시간 t로의 기울기 Jacobian
        jacobian = np.eye(d_h)
        
        for k in range(t, 0, -1):
            if activation == 'tanh':
                sigma_prime = tanh_derivative(z_history[k-1])
            elif activation == 'sigmoid':
                sigma_prime = sigmoid_derivative(z_history[k-1])
            else:
                sigma_prime = (z_history[k-1] > 0).astype(float)
            
            # diag(σ') @ W_hh
            jacobian = (sigma_prime * jacobian.T).T @ W_hh
        
        norm = np.linalg.norm(jacobian, ord=2)  # Spectral norm
        gradient_norms.append(norm)
    
    return np.array(gradient_norms)


def experiment_spectral_radius():
    """스펙트럼 반지름 변화에 따른 기울기 노름 분석"""
    
    d_h = 32
    d_x = 10
    T = 50
    
    # 다양한 스펙트럼 반지름
    spectral_radii = [0.5, 1.0, 1.5, 2.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for activation, ax in zip(['tanh', 'sigmoid', 'relu'], axes):
        for rho_target in spectral_radii:
            # 스펙트럼 반지름이 rho_target인 행렬 생성
            # 방법: 고유값을 rho_target으로 설정하고 정규화
            eigenvalues = np.ones(d_h) * rho_target
            Q = np.linalg.qr(np.random.randn(d_h, d_h))[0]
            W_hh = Q @ np.diag(eigenvalues) @ Q.T
            
            # 스펙트럼 반지름 확인
            rho_actual = np.max(np.abs(np.linalg.eigvals(W_hh)))
            
            # 입력 생성
            X = np.random.randn(T, d_x)
            
            # 기울기 흐름 계산
            gradient_norms = compute_gradient_flow(W_hh, X, T, activation)
            
            ax.semilogy(range(T), gradient_norms, 
                       label=f'ρ={rho_actual:.2f}', marker='o', markersize=3)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('||∂h_t/∂h_0|| (log scale)')
        ax.set_title(f'{activation.upper()} Activation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/gradient_flow.png', dpi=100)
    print("Gradient flow visualization saved to /tmp/gradient_flow.png")
    plt.close()


def analyze_critical_points():
    """활성화 함수별 임계값 분석"""
    
    print("\n=== 활성화 함수별 임계 스펙트럼 반지름 ===\n")
    
    # Sigmoid
    z_values = np.linspace(-5, 5, 1000)
    sigmoid_grad = sigmoid_derivative(z_values)
    max_sigmoid_grad = np.max(sigmoid_grad)
    critical_rho_sigmoid = 1.0 / max_sigmoid_grad
    
    print(f"Sigmoid:")
    print(f"  max σ'(z) = {max_sigmoid_grad:.4f}")
    print(f"  Critical ρ = {critical_rho_sigmoid:.4f}")
    print(f"  Vanishing if ρ < {critical_rho_sigmoid:.4f}")
    
    # Tanh
    tanh_grad = tanh_derivative(z_values)
    max_tanh_grad = np.max(tanh_grad)
    critical_rho_tanh = 1.0 / max_tanh_grad
    
    print(f"\nTanh:")
    print(f"  max σ'(z) = {max_tanh_grad:.4f}")
    print(f"  Critical ρ = {critical_rho_tanh:.4f}")
    print(f"  Vanishing if ρ < {critical_rho_tanh:.4f}")
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(z_values, sigmoid_grad, label='Sigmoid σ\'(z)', linewidth=2)
    ax.plot(z_values, tanh_grad, label='Tanh σ\'(z)', linewidth=2)
    ax.axhline(y=max_sigmoid_grad, color='blue', linestyle='--', alpha=0.5, 
               label=f'Max Sigmoid = {max_sigmoid_grad:.3f}')
    ax.axhline(y=max_tanh_grad, color='orange', linestyle='--', alpha=0.5,
               label=f'Max Tanh = {max_tanh_grad:.3f}')
    
    ax.set_xlabel('z')
    ax.set_ylabel('σ\'(z)')
    ax.set_title('활성화 함수 도함수 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/activation_derivatives.png', dpi=100)
    print("\nActivation function derivatives saved to /tmp/activation_derivatives.png")
    plt.close()


def gradient_clipping_effect():
    """Gradient Clipping의 효과"""
    
    print("\n=== Gradient Clipping 효과 ===\n")
    
    d_h = 32
    T = 50
    
    # 스펙트럼 반지름이 큰 행렬 (Exploding 위험)
    eigenvalues = np.ones(d_h) * 2.0
    Q = np.linalg.qr(np.random.randn(d_h, d_h))[0]
    W_hh = Q @ np.diag(eigenvalues) @ Q.T
    
    X = np.random.randn(T, 10)
    
    unclipped = compute_gradient_flow(W_hh, X, T, 'tanh')
    
    # Gradient clipping 시뮬레이션
    clipped_norms = np.clip(unclipped, 0, 5.0)  # threshold = 5.0
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.semilogy(range(T), unclipped, 'o-', label='Unclipped', linewidth=2, markersize=4)
    ax.semilogy(range(T), clipped_norms, 's--', label='Clipped (threshold=5.0)', 
               linewidth=2, markersize=4)
    
    ax.axhline(y=5.0, color='red', linestyle=':', alpha=0.7, label='Clipping threshold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Gradient Norm (log scale)')
    ax.set_title('Gradient Clipping Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/gradient_clipping.png', dpi=100)
    print("Gradient clipping effect saved to /tmp/gradient_clipping.png")
    plt.close()


if __name__ == "__main__":
    print("=== Vanishing/Exploding Gradient 분석 ===\n")
    
    experiment_spectral_radius()
    analyze_critical_points()
    gradient_clipping_effect()
    
    print("\n분석 완료!")
```

**구현의 핵심 포인트**:

1. **Jacobian 곱 계산**: 시간 역순으로 $\text{diag}(\sigma') \cdot W_{hh}$를 곱해나감
2. **Spectral Norm**: Numpy의 `np.linalg.norm(..., ord=2)`로 계산
3. **지수 증감**: 스펙트럼 반지름에 따라 기울기가 지수적으로 감소/증가하는 것을 관찰 가능

## 🔗 실전 연결

**Gradient Clipping** (Hochreiter & Schmidhuber, 2001):

현대 PyTorch/TensorFlow에서 표준:

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)

# TensorFlow
tf.clip_by_global_norm(grads, clip_norm=5.0)
```

**Weight Initialization** (Uniform initialization to control $\rho$):

```python
# Spectral radius를 특정 값으로 초기화
def init_rnn_weight(d_h, target_rho=0.9):
    W = np.random.randn(d_h, d_h) / np.sqrt(d_h)
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (target_rho / rho)  # 정규화
    return W
```

**Layer Normalization** (Ba et al., 2016):

```python
# 각 시간 단계에서 hidden state 정규화
h_normalized = (h - h.mean()) / (h.std() + eps)
```

## ⚖️ 가정과 한계

1. **상한(Upper Bound)**: 정리 1의 상한은 tight하지 않을 수 있다. 실제 기울기는 더 작을 수 있다.

2. **Activation Function의 다양성**: ReLU, GELU, Swish 등 최신 활성화 함수는 도함수의 최댓값이 명확하지 않을 수 있다.

3. **Sequence 길이의 가변성**: 고정 길이 $T$가 아닌 variable-length sequences에서는 분석이 더 복잡하다.

4. **Coupled Dynamics**: 실제 RNN에서는 $W_{xh}$도 역할을 하므로, $W_{hh}$만의 효과는 부분적이다.

## 📌 핵심 정리

| 개념 | 수식 | 의미 |
|------|------|------|
| **기울기 전파** | $\frac{\partial h_t}{\partial h_0} = \prod_k W_{hh}^T \text{diag}(\sigma'(z_k))$ | 시간 축 연쇄 곱 |
| **상한(Upper Bound)** | $\|\frac{\partial h_t}{\partial h_0}\| \leq (\rho(W_{hh}) \cdot \max \|\sigma'\|)^t$ | 기울기의 지수적 증감 |
| **Vanishing 조건** | $\rho(W_{hh}) \cdot \max \|\sigma'\| < 1$ | $t \to \infty$에서 기울기 → 0 |
| **Exploding 조건** | $\rho(W_{hh}) \cdot \max \|\sigma'\| > 1$ | $t \to \infty$에서 기울기 → ∞ |
| **Tanh Critical** | $\rho(W_{hh}) \approx 1$ | Tanh의 최대 도함수가 1 |
| **Sigmoid Critical** | $\rho(W_{hh}) \approx 4$ | Sigmoid의 최대 도함수가 0.25 |

## 🤔 생각해볼 문제

1. 스펙트럼 반지름이 정확히 1.0일 때, 기울기가 **steady state**에 도달한다고 볼 수 있을까? 이 경우 long-term dependency를 학습할 수 있을까?

2. **Residual Connection** ($h_t = h_{t-1} + f(h_{t-1})$)은 이 분석에서 어떻게 다를까? skip connection이 vanishing gradient를 완화하는 이유를 수학적으로 설명하면?

3. **Batch Normalization** vs **Layer Normalization** 중 RNN에서는 왜 Layer Normalization이 더 효과적일까?

4. Gradient Clipping의 threshold를 너무 작게 설정하면 어떤 문제가 생길까? 반대로 너무 크게 설정하면?

5. **Gradient Clipping이 바이어스 없는 기울기를 제공할까?** 아니면 $\rho(W_{hh})$의 정보를 손상시킬까?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. RNN의 정의와 BPTT 유도](./01-rnn-bptt.md) | [📚 README로 돌아가기](../README.md) | [03. LSTM과 GRU의 이론적 근거 ▶](./03-lstm-gru-theory.md) |

</div>
