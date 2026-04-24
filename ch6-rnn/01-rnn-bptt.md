# 01. RNN의 정의와 역전파 시간(BPTT) 유도

## 🎯 핵심 질문

- Vanilla RNN은 어떻게 **시간 축에서 가중치를 공유**하면서 순환 구조를 만드는가?
- BPTT는 CNN의 역전파와 어떻게 다르며, 왜 **展開된 계산 그래프(unrolled computational graph)**에서 연쇄법칙을 적용하는가?
- 공유 가중치 $W_{hh}$를 학습할 때, 각 시점의 기여도는 어떻게 합산되는가?
- Truncated BPTT는 언제, 어떤 대안으로 사용되는가?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

RNN은 시계열, 자연어, 음성 등 **순차 데이터의 표준 아키텍처**였으며, BPTT의 원리를 이해하면:

1. **Transformer 이전의 "동작하는 RNN"** 을 설계할 수 있다
2. **가중치 공유의 수학적 의미**를 파악하여 정규화 기법을 선택할 수 있다
3. **계산 그래프의 복잡성**이 왜 시퀀스 길이에 선형 증가하는지 설명할 수 있다
4. Truncated BPTT, Gradient Checkpointing 등의 **메모리 절약 기법의 근거**를 이해한다

## 📐 수학적 선행 조건

- **행렬의 미분**: $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$, Jacobian 행렬 (shape tracking)
- **연쇄법칙(Chain Rule)**: $\frac{\partial f(g(x))}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$
- **행렬 곱의 Hadamard 곱** ($\odot$)과 벡터화(vectorization)
- **계산 그래프(Computational Graph)**: 각 연산을 노드로 표현, 역전파는 그래프를 역방향 순회

## 📖 직관적 이해

**Vanilla RNN의 구조**는 이렇게 생각할 수 있다:

```
시간 1:  x₁ → [RNN Cell: h₀]  → h₁ → [Linear] → y₁
         ↓         ↑
         └─────────┘ (W_hh 공유)

시간 2:  x₂ → [RNN Cell: h₁]  → h₂ → [Linear] → y₂
         ↓         ↑
         └─────────┘ (같은 W_hh)

시간 T:  xₜ → [RNN Cell: hₜ₋₁] → hₜ → [Linear] → yₜ
```

**BPTT의 핵심 아이디어**:
- RNN을 시간 축을 따라 "펼친다(unroll)"
- 마치 T개의 깊은 신경망처럼 역전파를 수행한다
- 단, **같은 가중치 $W_{hh}$에 대한 기울기**는 모든 시점에서 온 것을 합산해야 한다

예를 들어, 시간 3에서 역전파:
- $\frac{\partial L}{\partial W_{hh}}$는 직접 영향: 시간 3 → 시간 2의 경로
- 간접 영향: 시간 3 → 시간 2 → 시간 1의 경로 등

이 모든 경로를 합산하는 것이 BPTT의 본질이다.

## ✏️ 엄밀한 정의

**Vanilla RNN Forward Pass**:

$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

여기서:
- $x_t \in \mathbb{R}^{d_x}$: 시간 $t$의 입력
- $h_t \in \mathbb{R}^{d_h}$: 숨은 상태 (hidden state)
- $y_t \in \mathbb{R}^{d_y}$: 시간 $t$의 출력
- $\sigma(\cdot)$: 활성화 함수 (보통 tanh)
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$, $W_{xh} \in \mathbb{R}^{d_h \times d_x}$, $W_{hy} \in \mathbb{R}^{d_y \times d_h}$

**손실 함수** (시계열 전체):

$$L = \sum_{t=1}^T L_t(y_t, y_t^*)$$

여기서 $L_t$는 보통 MSE 또는 cross-entropy.

**Unrolled Computational Graph**:

시간 축을 전개하면, 시간 $t$에서의 $h_t$는 $h_{t-1}, h_{t-2}, \ldots, h_0$의 함수이다. 이를 명시하면:

$$h_t = f(h_{t-1}, x_t, W_{hh}, W_{xh}, b_h)$$

연쇄적으로:

$$h_2 = f(f(h_0, x_1, \ldots), x_2, \ldots)$$

## 🔬 정리와 증명

**정리 1 (BPTT의 기울기 계산)**:

손실 $L = \sum_{t=1}^T L_t(y_t, y_t^*)$에 대해, $W_{hh}$의 기울기는:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L_t}{\partial W_{hh}}$$

각 시점의 기여도는:

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \left(\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}$$

**증명**:

$L_t = L_t(y_t, y_t^*)$이고 $y_t = W_{hy}h_t$이므로:

$$\frac{\partial L_t}{\partial h_t} = \frac{\partial L_t}{\partial y_t} \cdot W_{hy}^T$$

$h_t$는 $h_{t-1}$을 통해 $W_{hh}$와 연결되고, $h_{t-1}$은 $h_{t-2}$를 통해, 최종적으로 $h_1$까지 연쇄된다. 따라서:

$$\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_{1}} \cdot \frac{\partial h_1}{\partial W_{hh}} + \cdots + \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_{2}} \cdot \frac{\partial h_2}{\partial W_{hh}}$$

재귀 구조에서, $k < t$일 때:

$$\frac{\partial h_t}{\partial h_k} = \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial h_{t-2}} \cdots \frac{\partial h_{k+1}}{\partial h_k}$$

$h_k = \sigma(W_{hh}h_{k-1} + W_{xh}x_k + b_h)$이므로:

$$\frac{\partial h_k}{\partial W_{hh}} = \text{diag}(\sigma'(z_k)) h_{k-1}$$

여기서 $z_k = W_{hh}h_{k-1} + W_{xh}x_k + b_h$.

따라서:

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial h_t} \left(\prod_{j=k+1}^t \text{diag}(\sigma'(z_j))\right) \text{diag}(\sigma'(z_k)) h_{k-1}^T$$

전체 손실에 대해서는 모든 $t$에서 합산한다. $\square$

**Truncated BPTT**:

실전에서는 계산 비용을 절약하기 위해 **마지막 $k$-step**만 역전파한다:

$$\frac{\partial L_t}{\partial W_{hh}}^{(k)} = \sum_{j=\max(1, t-k)}^t \left(\prod_{i=j+1}^t \text{diag}(\sigma'(z_i))\right) \text{diag}(\sigma'(z_j)) h_{j-1}^T$$

이는 long-term dependency를 완전히 학습하지 못하는 대신, **메모리와 속도를 크게 절약**한다.

## 💻 NumPy로 바닥부터 구현

**Vanilla RNN과 BPTT 직접 구현** (autograd 없이):

```python
import numpy as np

class VanillaRNN:
    def __init__(self, d_x, d_h, d_y, learning_rate=0.01):
        # 초기화: Xavier initialization
        self.W_xh = np.random.randn(d_h, d_x) / np.sqrt(d_x)
        self.W_hh = np.random.randn(d_h, d_h) / np.sqrt(d_h)
        self.W_hy = np.random.randn(d_y, d_h) / np.sqrt(d_h)
        self.b_h = np.zeros((d_h, 1))
        self.b_y = np.zeros((d_y, 1))
        
        self.lr = learning_rate
        self.h_prev = np.zeros((d_h, 1))
    
    def forward(self, X):
        # X: (T, d_x) 형태의 시퀀스
        T, d_x = X.shape
        d_h, d_y = self.W_hy.shape[1], self.W_hy.shape[0]
        
        # 캐시: forward pass 값들
        h_cache = [self.h_prev]
        z_cache = []
        y_cache = []
        
        h = self.h_prev
        for t in range(T):
            x = X[t:t+1].T  # (d_x, 1)
            
            # Pre-activation
            z = self.W_hh @ h + self.W_xh @ x + self.b_h
            z_cache.append(z)
            
            # Activation
            h = np.tanh(z)
            h_cache.append(h)
            
            # Output
            y = self.W_hy @ h + self.b_y
            y_cache.append(y)
        
        return y_cache, h_cache, z_cache, X
    
    def backward(self, y_pred, y_true, cache, truncation_length=None):
        # y_true: (T, d_y)
        T = len(y_pred)
        d_h = self.W_hh.shape[0]
        
        y_cache, h_cache, z_cache, X = cache
        
        # 기울기 초기화
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 초기 dh = 0
        dh_next = np.zeros((d_h, 1))
        
        # 역시간 역전파
        trunc = truncation_length if truncation_length else T
        
        for t in reversed(range(T)):
            # 출력층 기울기
            dy = y_pred[t] - y_true[t:t+1].T
            
            dW_hy += dy @ h_cache[t+1].T
            db_y += dy
            
            # 숨은층으로 역전파
            dh = self.W_hy.T @ dy + dh_next
            
            # Pre-activation에서의 기울기
            dz = dh * (1 - np.tanh(z_cache[t])**2)
            
            # 가중치 기울기 (truncation 고려)
            if truncation_length is None or (T - 1 - t) < truncation_length:
                dW_xh += dz @ X[t:t+1]
                dW_hh += dz @ h_cache[t].T
                db_h += dz
            
            # 이전 시간으로 역전파
            dh_next = self.W_hh.T @ dz
        
        # 기울기 clipping
        for d in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(d, -5, 5, out=d)
        
        return dW_xh, dW_hh, dW_hy, db_h, db_y
    
    def update(self, grads):
        dW_xh, dW_hh, dW_hy, db_h, db_y = grads
        
        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.W_hy -= self.lr * dW_hy
        self.b_h -= self.lr * db_h
        self.b_y -= self.lr * db_y
    
    def train_step(self, X, y_true, truncation_length=None):
        y_pred, h_cache, z_cache, _ = self.forward(X)
        grads = self.backward(y_pred, y_true, 
                             (y_pred, h_cache, z_cache, X),
                             truncation_length)
        self.update(grads)
        
        # Loss 계산
        loss = 0.5 * np.sum((np.array([y.ravel() for y in y_pred]) - y_true)**2)
        return loss / len(X)


# Character-level 예측 예제
def char_rnn_experiment():
    text = "hello world " * 10
    char_to_idx = {c: i for i, c in enumerate(sorted(set(text)))}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    d_x = len(char_to_idx)
    d_h = 16
    d_y = d_x
    
    rnn = VanillaRNN(d_x, d_h, d_y, learning_rate=0.01)
    
    # 시퀀스 길이 10으로 자르기
    seq_len = 10
    sequences = []
    targets = []
    
    for i in range(len(text) - seq_len):
        seq = np.zeros((seq_len, d_x))
        for j in range(seq_len):
            seq[j, char_to_idx[text[i+j]]] = 1
        sequences.append(seq)
        
        target = np.zeros((1, d_y))
        target[0, char_to_idx[text[i+seq_len]]] = 1
        targets.append(target)
    
    # 학습 (truncated BPTT, k=5)
    for epoch in range(50):
        total_loss = 0
        for seq, target in zip(sequences, targets):
            loss = rnn.train_step(seq, target, truncation_length=5)
            total_loss += loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(sequences):.4f}")


if __name__ == "__main__":
    char_rnn_experiment()
```

**주요 구현 포인트**:

1. **Forward pass**: 시간 축을 따라 순차 계산, 각 단계의 값 캐싱
2. **Backward pass**: 역시간 순회, $\frac{\partial h_t}{\partial W_{hh}}$ 합산
3. **Truncation**: truncation_length 변수로 제어
4. **기울기 Clipping**: exploding gradient 방지

## 🔗 실전 연결

**현대 PyTorch/TensorFlow에서**:

```python
import torch
import torch.nn as nn

# PyTorch의 기본 RNN은 BPTT를 자동으로 구현
rnn = nn.RNN(input_size=d_x, hidden_size=d_h, batch_first=True)
output, h_n = rnn(X)  # autograd가 BPTT를 자동 계산
```

**BPTT의 한계와 극복**:

| 문제 | 해결책 |
|------|--------|
| Vanishing Gradient | LSTM, GRU 사용 |
| Exploding Gradient | Gradient Clipping, Normalization |
| 계산 비용 (O(T·h²)) | Truncated BPTT, Gradient Checkpointing |
| Long-term Dependency 학습 불가 | Attention, Transformer 도입 |

## ⚖️ 가정과 한계

1. **미분 가능성**: RNN의 모든 연산이 미분 가능해야 하므로, ReLU 같은 비매끄러운 함수는 이론적으로 문제가 될 수 있다 (실전에서는 문제 없음).

2. **Truncated BPTT의 편향**: 긴 의존성을 학습할 수 없으므로, long-range 패턴이 있는 데이터에서 성능이 떨어진다.

3. **계산 비용**: $O(T \cdot d_h^2)$이므로, 길이 $T$가 크면 매우 느려진다.

4. **메모리**: 전체 계산 그래프를 메모리에 유지해야 하므로, $T$가 매우 크면 GPU 메모리 부족.

## 📌 핵심 정리

| 개념 | 핵심 포인트 |
|------|-----------|
| **Vanilla RNN** | $h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$, 순환 연결로 메모리 구현 |
| **Unrolled Graph** | 시간 축을 펼쳐서 깊은 네트워크 구조로 변환 |
| **BPTT** | 연쇄법칙을 역시간으로 적용, 같은 가중치의 기울기를 모든 시점에서 합산 |
| **기울기 식** | $\frac{\partial L}{\partial W_{hh}} = \sum_t \sum_{k=1}^t (\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}) \frac{\partial h_k}{\partial W_{hh}}$ |
| **Truncated BPTT** | 계산 절약을 위해 최근 $k$-step만 역전파 |
| **구현** | NumPy에서 직접 구현 가능, forward/backward 분리 |

## 🤔 생각해볼 문제

1. Vanilla RNN에서 $W_{hh}$가 **모든 시간 단계에서 공유**되는 이유는 무엇인가? 각 시간 단계마다 다른 가중치를 사용하면 어떻게 될까?

2. BPTT의 기울기 식에서, $\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t (\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}) \frac{\partial h_k}{\partial W_{hh}}$는 왜 각 $k$에 대해 **Jacobian 곱**의 형태인가?

3. Truncated BPTT에서 truncation length를 줄이면 학습 속도는 빨라지지만, 어떤 종류의 패턴을 놓치게 될까?

4. $W_{hh}$의 초기값 크기가 RNN의 학습에 미치는 영향은? (다음 장의 vanishing/exploding gradient와 연결)

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch5-04. CNN 아키텍처](../ch5-cnn/04-cnn-architectures.md) | [📚 README로 돌아가기](../README.md) | [02. Vanishing/Exploding Gradient의 수학적 분석 ▶](./02-vanishing-exploding.md) |

</div>
