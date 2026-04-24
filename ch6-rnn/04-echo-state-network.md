# 04. Echo State Network과 Reservoir Computing

## 🎯 핵심 질문

- **Reservoir Computing**이란 무엇인가? 왜 순환 가중치를 학습하지 않는가?
- **Echo State Property (ESP)**는 언제 보장되는가? 이론적 조건은 무엇인가?
- ESN이 LSTM/GRU보다 **빠르고 단순**하면서도 왜 덜 사용될까?
- **신경과학적 영감**은 무엇인가? Liquid State Machine과의 연결은?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

Echo State Network은:

1. **학습 이론의 다른 관점**: RNN을 모두 학습하지 않고 일부만 학습하는 패러다임
2. **계산 효율성**: 비선형 차원 축소 + 선형 회귀로 $O(h^3)$의 ridge regression만 필요
3. **신경과학 영감**: Fusi et al., 바이러스 대뇌의 plasticity와 연결
4. **Liquid State Machine과 뇌 계산**: Spiking neural network의 이론적 근거

## 📐 수학적 선행 조건

- **행렬 스펙트럼 반지름** $\rho(A)$ (Chapter 2에서 학습)
- **Ridge Regression**: $w^* = (X^T X + \lambda I)^{-1} X^T y$
- **Spectral Analysis**: 고유값 분해
- **Dynamical Systems**: 시스템의 안정성, Lyapunov exponent

## 📖 직관적 이해

**기존 RNN의 문제**:

모든 가중치를 학습해야 하므로:
- 계산량: BPTT로 $O(T \cdot d_h^2)$ 반복 계산
- 학습 불안정: vanishing/exploding gradient
- 메모리: 전체 계산 그래프 유지

**Reservoir Computing의 아이디어**:

```
입력 x_t → [Random Recurrent Network] → h_t (풍부한 dynamics)
                                      ↓
                                   [선형 회귀]
                                      ↓
                                    출력 y_t
```

1. **Reservoir** (순환 부분): **고정된 random 가중치**
2. **Readout** (출력층): **선형 회귀**로만 학습

**핵심 직관**:

Random network도 충분히 복잡한 입력-상태 매핑을 만들 수 있다!

수학적으로, random $W_{hh}$는:
- 입력 신호를 **고차원 공간**으로 사영
- 충분히 복잡한 비선형 특징을 자동 생성
- 나머지는 선형 조합으로 해결

## ✏️ 엄밀한 정의

**Echo State Network의 정의**:

상태 업데이트:
$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

출력:
$$y_t = W_{hy} h_t + b_y$$

여기서:
- $W_{hh}$: **고정된 random 행렬** (초기화 후 학습 X)
- $W_{xh}$: **고정된 random 행렬** (초기화 후 학습 X)
- $W_{hy}$: **학습 가능한 선형 가중치** (ridge regression)
- $\sigma(\cdot)$: 보통 tanh

**Echo State Property (ESP)**:

시스템이 ESP를 만족한다는 것은:

$$\lim_{t \to \infty} |h_t(u) - h_t(v)| = 0 \quad \text{for any initial conditions } h_0(u), h_0(v)$$

초기 조건과 무관하게 모든 궤적이 같은 **attractor**로 수렴한다.

이는 시스템의 출력이 **현재 입력과 최근 입력 히스토리에만 의존**함을 보장한다.

**정리 1 (ESP의 충분 조건, Jaeger 2001)**:

만약 $\rho(W_{hh}) < 1$이면, tanh 활성화에서 ESP가 만족된다.

직관: 순환 가중치의 고유값이 1보다 작으면, 숨은 상태가 안정적으로 감소하므로, 과거 정보가 지수적으로 망각된다.

## 🔬 정리와 증명

**정리 1 (ESP와 Spectral Radius)**:

명제: $\rho(W_{hh}) < 1$이면, tanh 활성화를 사용하는 ESN에서 ESP가 성립한다.

증명 스케치:

$h_t$의 업데이트를 분석하면:

$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

Tanh는 1-Lipschitz 함수이므로:

$$|h_t(u) - h_t(v)| \leq |W_{hh}(h_{t-1}(u) - h_{t-1}(v))|$$

$$\leq \|W_{hh}\|_2 \cdot |h_{t-1}(u) - h_{t-1}(v)|$$

$$\leq \rho(W_{hh}) \cdot |h_{t-1}(u) - h_{t-1}(v)|$$

따라서:

$$|h_t(u) - h_t(v)| \leq \rho(W_{hh})^t \cdot |h_0(u) - h_0(v)|$$

$\rho(W_{hh}) < 1$이면:

$$\lim_{t \to \infty} |h_t(u) - h_t(v)| = 0$$

따라서 모든 궤적이 수렴한다. $\square$

**정리 2 (필요충분 조건, Bounded 입력)**:

Bounded 입력 $\|x_t\| \leq B$에 대해, tanh 활성화를 사용하면:

$$\rho(W_{hh}) < 1 \text{ (충분 조건)}$$

$$\rho(W_{hh}) \leq \frac{1}{1 + \|W_{xh}\| B} \text{ (더 정밀한 경계)}$$

실제로는, $\rho(W_{hh}) < 1$이 ESP의 거의 필요충분조건이다.

**정리 3 (정보량과 Spectral Radius)**:

더 낮은 $\rho(W_{hh})$:
- ESP는 더 빨리 수렴 → 과거 정보 빠르게 망각
- 현재 입력에만 의존 → 장기 의존성 학습 불가능

더 높은 $\rho(W_{hh})$ (단, < 1):
- 유용한 정보를 더 오래 유지
- 과거 입력의 영향이 더 오래 남음

**최적점**: $\rho(W_{hh}) \approx 0.9$ 정도가 많은 태스크에서 성능 최적

**정리 4 (Complexity of Training)**:

ESN의 학습 복잡도:

1. Reservoir forward pass: $O(T \cdot d_h^2)$ (비학습, 한 번)
2. Ridge regression: $O(d_h^3)$ (행렬 역 계산)

총 복잡도: $O(T \cdot d_h^2 + d_h^3)$

반면 BPTT: $O(T \cdot d_h^2)$ 반복 계산 (epoch 수 만큼)

따라서 ESN은 **한 번의 forward pass + 한 번의 회귀**로 완료된다!

**정리 5 (Liquid State Machine과의 연결)**:

Maass et al. (2002)의 Liquid State Machine:

```
입력 x(t) → [Spiking Neural Network] → s(t) (스파이크 상태)
                                      ↓
                                   [선형 분류기]
                                      ↓
                                  출력 y(t)
```

- Spiking neurons: 비연속 출력 (0 또는 1)
- Reservoir: liquid의 비유 (입력에 반응하는 동적계)

ESN은 LSM의 **연속 버전**으로 볼 수 있다.

## 💻 NumPy로 바닥부터 구현

**Echo State Network 직접 구현**:

```python
import numpy as np
from scipy.linalg import solve

class EchoStateNetwork:
    def __init__(self, n_input, n_reservoir, n_output, 
                 spectral_radius=0.9, input_scale=1.0, 
                 regularization=1e-6):
        """
        Args:
            n_input: 입력 차원
            n_reservoir: reservoir 숨은 상태 차원
            n_output: 출력 차원
            spectral_radius: W_hh의 스펙트럼 반지름 (< 1)
            input_scale: 입력 스케일링
            regularization: Ridge regression의 정규화 계수
        """
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.regularization = regularization
        
        # 1. Random recurrent 가중치 생성
        # Sparse 행렬 (sparsity = 0.1)
        sparsity = 0.1
        self.W_hh = np.random.randn(n_reservoir, n_reservoir)
        mask = np.random.rand(n_reservoir, n_reservoir) > sparsity
        self.W_hh *= mask
        
        # 스펙트럼 반지름으로 정규화
        eig_vals = np.linalg.eigvals(self.W_hh)
        rho = np.max(np.abs(eig_vals))
        self.W_hh *= spectral_radius / rho
        
        # 2. Random 입력 가중치
        self.W_in = (np.random.rand(n_reservoir, n_input) - 0.5) * 2
        self.W_in *= input_scale
        
        # 3. 학습 가능한 출력 가중치 (초기화)
        self.W_out = np.random.randn(n_output, n_reservoir) * 0.1
        self.b_out = np.zeros((n_output, 1))
    
    def forward(self, X, return_states=False):
        """
        Forward pass (상태 계산만)
        
        Args:
            X: (T, n_input) 입력 시퀀스
            return_states: True이면 모든 숨은 상태 반환
        
        Returns:
            h_states: (T, n_reservoir) 또는 (T+1, n_reservoir)
        """
        T, d_in = X.shape
        assert d_in == self.n_input
        
        h = np.zeros((self.n_reservoir, 1))
        h_states = [h.ravel()]
        
        for t in range(T):
            x = X[t:t+1].T  # (n_input, 1)
            
            # 상태 업데이트: h_{t+1} = tanh(W_hh @ h_t + W_in @ x_t)
            h = np.tanh(self.W_hh @ h + self.W_in @ x)
            
            if return_states:
                h_states.append(h.ravel())
        
        if return_states:
            return np.vstack(h_states)  # (T+1, n_reservoir)
        else:
            return np.vstack(h_states)  # (T, n_reservoir)
    
    def train(self, X_train, y_train, transient=100):
        """
        Ridge regression으로 출력 가중치 학습
        
        Args:
            X_train: (T, n_input)
            y_train: (T, n_output)
            transient: 처음 transient 샘플은 버림 (시스템 안정화)
        """
        T, _ = X_train.shape
        
        # 상태 계산
        H = self.forward(X_train, return_states=True)  # (T+1, n_reservoir)
        
        # Transient 제거
        H = H[transient:T, :]
        y = y_train[transient-1:T-1, :]
        
        # Ridge regression: W_out = (H^T H + lambda I)^{-1} H^T y
        # 수치 안정성을 위해 더 큰 행렬 형태로
        HTH = H.T @ H
        HTy = H.T @ y
        
        # Regularization (lambda * I)
        reg_matrix = self.regularization * np.eye(self.n_reservoir)
        
        # 선형 시스템 풀기
        self.W_out = solve(HTH + reg_matrix, HTy).T
    
    def predict(self, X_test):
        """
        예측
        
        Args:
            X_test: (T, n_input)
        
        Returns:
            y_pred: (T, n_output)
        """
        H = self.forward(X_test, return_states=True)
        
        # 첫 상태 제거 (초기 상태)
        H = H[1:, :]
        
        y_pred = H @ self.W_out.T
        return y_pred


def mackey_glass_experiment():
    """
    Mackey-Glass 시계열 예측 (비선형 시계열 예측의 벤치마크)
    
    방정식: x(t) = 0.9 * x(t-τ) / (1 + x(t-τ)^10) + 0.1 * x(t)
    """
    
    def generate_mackey_glass(length=3000, tau=17, initial=0.1):
        """Mackey-Glass 시계열 생성"""
        x = np.zeros(length)
        x[0] = initial
        
        for t in range(tau, length):
            x[t] = 0.9 * x[t - tau] / (1 + x[t - tau]**10) + 0.1 * x[t - 1]
        
        return x
    
    print("\n=== Mackey-Glass 시계열 예측 (ESN) ===\n")
    
    # 데이터 생성
    mg_series = generate_mackey_glass(length=2500)
    
    # 정규화
    mg_series = (mg_series - mg_series.mean()) / mg_series.std()
    
    # 시퀀스 형태로 변환 (look-back = 10)
    lookback = 10
    X = np.array([mg_series[i:i+lookback] for i in range(len(mg_series) - lookback)])
    y = mg_series[lookback:].reshape(-1, 1)
    
    # Train/Test 분할
    train_size = 1500
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 여러 spectral radius로 실험
    spectral_radii = [0.5, 0.8, 0.9, 0.95, 0.99]
    results = {}
    
    for rho in spectral_radii:
        esn = EchoStateNetwork(
            n_input=lookback,
            n_reservoir=300,
            n_output=1,
            spectral_radius=rho,
            input_scale=1.0,
            regularization=1e-6
        )
        
        # 학습
        esn.train(X_train, y_train, transient=50)
        
        # 예측
        y_pred_train = esn.predict(X_train)
        y_pred_test = esn.predict(X_test)
        
        # RMSE 계산
        rmse_train = np.sqrt(np.mean((y_train[50:] - y_pred_train[50:])**2))
        rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
        
        results[rho] = {'rmse_train': rmse_train, 'rmse_test': rmse_test}
        
        print(f"ρ = {rho:.2f}: Train RMSE = {rmse_train:.4f}, Test RMSE = {rmse_test:.4f}")
    
    # 시각화
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Spectral radius vs RMSE
    ax = axes[0, 0]
    rhos = list(results.keys())
    train_rmses = [results[rho]['rmse_train'] for rho in rhos]
    test_rmses = [results[rho]['rmse_test'] for rho in rhos]
    
    ax.plot(rhos, train_rmses, 'o-', label='Train', linewidth=2)
    ax.plot(rhos, test_rmses, 's-', label='Test', linewidth=2)
    ax.set_xlabel('Spectral Radius ρ')
    ax.set_ylabel('RMSE')
    ax.set_title('ESN Performance vs Spectral Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 최적 모델 (ρ = 0.9)의 예측
    ax = axes[0, 1]
    esn_best = EchoStateNetwork(
        n_input=lookback,
        n_reservoir=300,
        n_output=1,
        spectral_radius=0.9,
        input_scale=1.0,
        regularization=1e-6
    )
    esn_best.train(X_train, y_train, transient=50)
    y_pred_best = esn_best.predict(X_test)
    
    ax.plot(y_test[:200], label='True', linewidth=1, alpha=0.7)
    ax.plot(y_pred_best[:200], label='Predicted', linewidth=1, alpha=0.7, linestyle='--')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Best ESN Prediction (ρ=0.9, first 200 test points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Spectral radius 분포
    ax = axes[1, 0]
    esn_high = EchoStateNetwork(
        n_input=lookback,
        n_reservoir=100,
        n_output=1,
        spectral_radius=0.9
    )
    eigs = np.abs(np.linalg.eigvals(esn_high.W_hh))
    ax.hist(eigs, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=0.9, color='red', linestyle='--', label='Target ρ = 0.9')
    ax.set_xlabel('|Eigenvalue|')
    ax.set_ylabel('Frequency')
    ax.set_title('Eigenvalue Distribution of W_hh')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 에러 시각화
    ax = axes[1, 1]
    error = np.abs(y_test - y_pred_best)
    ax.semilogy(error[:200], linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Absolute Error (log scale)')
    ax.set_title('Prediction Error Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/esn_mackey_glass.png', dpi=100)
    print("\nPlot saved to /tmp/esn_mackey_glass.png")
    plt.close()


def esn_vs_lstm_complexity():
    """ESN과 LSTM의 학습 복잡도 비교"""
    
    print("\n=== ESN vs LSTM 복잡도 비교 ===\n")
    
    d_h_values = np.array([32, 64, 128, 256])
    T = 1000
    
    print("Dimension | ESN Train (ops) | LSTM Train (ops) | Ratio")
    print("-" * 60)
    
    for d_h in d_h_values:
        # ESN: 한 번의 forward + ridge regression
        esn_ops = T * d_h**2 + d_h**3
        
        # LSTM: BPTT (assuming 10 epochs)
        lstm_ops = 10 * T * d_h**2
        
        ratio = lstm_ops / esn_ops
        
        print(f"{d_h:^9d} | {esn_ops:^15.0e} | {lstm_ops:^16.0e} | {ratio:^5.1f}x")
    
    print("\n결론: ESN은 고차원에서 LSTM보다 훨씬 빠를 수 있다.")


if __name__ == "__main__":
    mackey_glass_experiment()
    esn_vs_lstm_complexity()
```

**구현의 핵심**:

1. **Spectral radius 제어**: 고유값의 최댓값이 < 1 되도록 정규화
2. **Ridge regression**: numpy의 선형 시스템 solver 사용
3. **Transient 제거**: 시스템 안정화 기간 후 학습

## 🔗 실전 연결

**Reservoir Computing의 응용**:

1. **시계열 예측**: 가장 흔한 응용 (Mackey-Glass, sunspot activity)
2. **채널 등화**: 통신 신호 처리
3. **음성 인식**: 초기 RNN을 대체 (2000년대)
4. **Spiking Neural Networks**: Neuromorphic 칩에서의 뇌 영감 컴퓨팅

**Liquid State Machine** (신경과학 버전):

```
Spiking Neural Network 형태의 reservoir
↓
선형 분류기로 읽기
```

## ⚖️ 가정과 한계

1. **선형 읽기의 한계**: 출력이 선형이므로, 매우 복잡한 함수 매핑은 어렵다.
   - 해결책: 비선형 readout (ridge regression 대신 SVM, neural network)

2. **메모리의 한계**: Spectral radius가 1에 가까워질수록 메모리가 증가하지만, 1을 초과할 수 없다.
   - Echo State Property 보장 불가능

3. **데이터 효율성**: Reservoir는 고정되어 있으므로, 특정 태스크에 최적화되지 않는다.
   - 종종 LSTM보다 더 많은 reservoir가 필요

4. **Hyperparameter 선택**: Spectral radius, input scale, regularization 등을 조정해야 한다.

## 📌 핵심 정리

| 항목 | RNN/LSTM | ESN |
|------|----------|-----|
| **구조** | 모든 가중치 학습 | Random reservoir + 선형 readout |
| **학습 복잡도** | $O(T \cdot d_h^2)$ × epoch 수 | $O(T \cdot d_h^2 + d_h^3)$ (한 번) |
| **학습 안정성** | 낮음 (vanishing gradient) | 높음 (고정된 reservoir) |
| **메모리 효율** | 전체 그래프 필요 | Forward pass만 필요 |
| **표현력** | 높음 (모든 가중치 최적화) | 중간 (비선형 features + 선형 조합) |
| **실전 성능** | 매우 우수 (특히 NLP) | 시계열 예측에서 좋음 |

| 개념 | 설명 |
|------|------|
| **Echo State Property** | 초기 조건과 무관하게 모든 궤적이 attractor로 수렴 |
| **Spectral Radius** | $\rho(W_{hh})$가 ESP의 핵심 제어 파라미터 |
| **Reservoir** | 고정된 비선형 사영 (dynamics engine) |
| **Readout** | 선형 회귀로 출력 계산 |

## 🤔 생각해볼 문제

1. **왜 ESN은 현대에 덜 사용되는가?** Transformer가 더 높은 성능을 제공하기 때문인가? 아니면 이론적/구현의 어려움?

2. **Nonlinear Readout**: Readout을 비선형으로 만들면 (예: 신경망) 성능이 향상될까? 이 경우 장점은?

3. **Liquid State Machine의 신경과학적 의미**: 실제 뇌의 회로도 "고정된 임의적 연결"과 "선택적 학습 출력"으로 동작할까?

4. **Spectral Radius의 역설**: 높은 스펙트럼 반지름은 더 많은 정보를 유지하지만, ESP가 깨진다. 어떻게 절충할까?

5. **Deep Reservoir**: Reservoir를 여러 층으로 쌓으면 (hierarchical)? 각 층의 spectral radius는 어떻게 설정할까?

6. **Feedback Connections in Readout**: Readout 층에서 다시 reservoir로 피드백을 주면? (Closed-loop)

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. LSTM과 GRU의 이론적 근거](./03-lstm-gru-theory.md) | [📚 README로 돌아가기](../README.md) | [Ch7-01. Self-Attention과 √d_k ▶](../ch7-transformer/01-self-attention.md) |

</div>
