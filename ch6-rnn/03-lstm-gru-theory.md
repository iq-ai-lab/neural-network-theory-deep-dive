# 03. LSTM과 GRU의 이론적 근거

## 🎯 핵심 질문

- LSTM이 **Constant Error Carousel (CEC)**를 통해 vanishing gradient를 어떻게 극복하는가?
- **Cell State의 덧셈 업데이트**가 곱셈 업데이트와 어떻게 다른가?
- GRU는 LSTM을 단순화한 것인데, 왜 성능이 유사할까? 어떤 부분을 단순화했는가?
- **Gate 메커니즘의 수학적 정체성**은 무엇인가?

## 🔍 왜 이 이론이 현대 딥러닝에 필수인가

LSTM/GRU는:

1. **Transformer 이전의 SOTA 순차 모델**: 2015-2017년간 모든 NLP 태스크의 기본
2. **Gating 메커니즘의 원형**: Attention, Transformer의 선택 메커니즘의 개념적 선구자
3. **설계 철학의 귀감**: 문제를 정확히 이해한 후 구조로 해결하는 방식
4. **여전히 활용**: Transformer와 함께 하이브리드 모델로 사용되며, 작은 모델에서는 여전히 SOTA

## 📐 수학적 선행 조건

- **벡터 연산**: Hadamard 곱(element-wise), broadcasting
- **Sigmoid와 Tanh의 성질**: 출력 범위, 도함수
- **행렬 미분**: chain rule in matrix form
- **Recurrent Gradient**: 이전 장의 $\frac{\partial h_t}{\partial h_{t-1}}$ 개념

## 📖 직관적 이해

**Vanilla RNN의 문제**:

$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$h_t$는 **완전히 덮어씌워진다**. 이전 상태 $h_{t-1}$의 정보는 거의 손실된다.

시간이 지날수록:
- 정보 손실 누적
- 과거 신호의 영향 지수적 감소

**LSTM의 아이디어**:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Cell state** $c_t$는:
1. 이전 cell state의 **일부를 유지** ($f_t \odot c_{t-1}$)
2. **새로운 정보를 추가** ($i_t \odot \tilde{c}_t$)

**곱하기가 아닌 더하기**: 덧셈 때문에 기울기 흐름이 끊기지 않는다!

예를 들어:
$$\frac{\partial c_t}{\partial c_{t-1}} = f_t + \text{(다른 항들)}$$

만약 $f_t \approx 1$ (forget gate가 "모두 유지")이면, $\frac{\partial c_t}{\partial c_{t-1}} \approx 1$이고, 시간 역전파에서:

$$\prod_{k=1}^T \frac{\partial c_k}{\partial c_{k-1}} \approx 1^T = 1$$

기울기가 유지된다! (Constant Error Carousel)

## ✏️ 엄밀한 정의

**LSTM의 수식**:

Forget Gate:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Input Gate:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

Candidate Cell State:
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

Cell State Update:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Output Gate:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

Hidden State:
$$h_t = o_t \odot \tanh(c_t)$$

여기서:
- $[h_{t-1}, x_t]$는 concatenation
- $\odot$는 element-wise 곱(Hadamard 곱)
- $\sigma(\cdot) = \text{sigmoid}$, 출력 범위 $[0, 1]$ (gate로 작동)
- $\tanh(\cdot)$, 출력 범위 $[-1, 1]$ (정보로 작동)

**GRU (Gated Recurrent Unit)**:

Update Gate:
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

Reset Gate:
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

Candidate Hidden State:
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

Hidden State Update:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**주요 차이**:
- LSTM: 3개 gate (f, i, o) + 별도 cell state
- GRU: 2개 gate (z, r) + hidden state 직접 사용

## 🔬 정리와 증명

**정리 1 (LSTM의 Gradient Flow)**:

LSTM에서 cell state $c_t$의 역전파 기울기:

$$\frac{\partial c_t}{\partial c_0} = \prod_{k=1}^t f_k + \text{(다른 기여도)}$$

증명:

Chain rule에서:
$$\frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial}{\partial c_{t-1}}(f_t \odot c_{t-1} + i_t \odot \tilde{c}_t)$$

$$= f_t + \text{diag}(c_{t-1}^T) \frac{\partial f_t}{\partial c_{t-1}} + \text{diag}(i_t^T) \frac{\partial \tilde{c}_t}{\partial c_{t-1}}$$

첫 번째 항 $f_t$가 주요 경로이다. $f_t \in [0, 1]$이므로:

$$\prod_{k=1}^t f_k \in [0, 1]$$

**중요**: Vanilla RNN과 달리, $\prod_k f_k$는 $W_{hh}$의 고유값이 아니라, **forget gate가 학습한 값**이다!

따라서:
- Forget gate가 중요한 정보를 "기억하도록" 학습 ($f_k \approx 1$) → 기울기 유지
- Forget gate가 불필요한 정보를 "잊도록" 학습 ($f_k \approx 0$) → 신호 감소

이는 **self-adaptive** 한 해결책이다.

**정리 2 (CEC의 기울기 안정성)**:

만약 forget gate가 평균적으로 $\bar{f} \approx 0.9$라면:

$$\prod_{k=1}^{100} f_k \approx (0.9)^{100} \approx 0.0027$$

이는 Vanilla RNN에서 $\rho(W_{hh}) = 0.9$일 때:

$$\rho(W_{hh})^{100} = (0.9)^{100} \approx 0.0027$$

과 같은 크기이지만, **차이**는:

1. LSTM: forget gate가 **시간에 따라 다르게 설정** 가능
2. Vanilla RNN: 고정된 $W_{hh}$

따라서 LSTM은 각 시간 단계에서 **선택적 정보 유지**가 가능하다.

**정리 3 (Output Gate의 역할)**:

Hidden state는:
$$h_t = o_t \odot \tanh(c_t)$$

Output gate $o_t$는 cell state의 어느 부분을 외부에 노출할지 제어한다.

- Cell state $c_t$: 내부 메모리 (potentially unbounded)
- Hidden state $h_t$: 외부 출력 (bounded by tanh)

이는 **메모리와 출력의 분리**를 가능하게 한다.

**정리 4 (GRU vs LSTM 표현력)**:

GRU:
$$h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$$

Update gate $z_t$가 **interpolation weight** 역할을 한다.

LSTM:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

차이:
- GRU: additive update at output level
- LSTM: additive update at cell state level

**정리**: Empirically 유사하지만, LSTM이 약간 더 복잡한 패턴을 포착할 수 있다. GRU는 계산량이 적으므로 빠르다.

## 💻 NumPy로 바닥부터 구현

**LSTM Cell 직접 구현**:

```python
import numpy as np

class LSTMCell:
    def __init__(self, d_x, d_h):
        """
        Args:
            d_x: 입력 차원
            d_h: 숨은 상태 차원
        """
        self.d_x = d_x
        self.d_h = d_h
        
        # Xavier initialization
        std = np.sqrt(1.0 / (d_x + d_h))
        
        # Forget gate
        self.W_f = np.random.randn(d_h, d_x + d_h) * std
        self.b_f = np.zeros((d_h, 1))
        
        # Input gate
        self.W_i = np.random.randn(d_h, d_x + d_h) * std
        self.b_i = np.zeros((d_h, 1))
        
        # Cell candidate
        self.W_c = np.random.randn(d_h, d_x + d_h) * std
        self.b_c = np.zeros((d_h, 1))
        
        # Output gate
        self.W_o = np.random.randn(d_h, d_x + d_h) * std
        self.b_o = np.zeros((d_h, 1))
    
    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t: (d_x, 1) 입력 벡터
            h_prev: (d_h, 1) 이전 숨은 상태
            c_prev: (d_h, 1) 이전 cell state
        
        Returns:
            h_t: (d_h, 1) 현재 숨은 상태
            c_t: (d_h, 1) 현재 cell state
            cache: 역전파용 캐시
        """
        # Concatenate
        hx = np.vstack([h_prev, x_t])
        
        # Forget gate
        z_f = self.W_f @ hx + self.b_f
        f_t = self._sigmoid(z_f)
        
        # Input gate
        z_i = self.W_i @ hx + self.b_i
        i_t = self._sigmoid(z_i)
        
        # Cell candidate
        z_c = self.W_c @ hx + self.b_c
        c_tilde = np.tanh(z_c)
        
        # Cell state update
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate
        z_o = self.W_o @ hx + self.b_o
        o_t = self._sigmoid(z_o)
        
        # Hidden state
        h_t = o_t * np.tanh(c_t)
        
        # Cache for backward
        cache = (x_t, h_prev, c_prev, hx, z_f, f_t, z_i, i_t, 
                 z_c, c_tilde, c_t, z_o, o_t, h_t)
        
        return h_t, c_t, cache
    
    def backward(self, dh_t, dc_t, cache):
        """
        Args:
            dh_t: (d_h, 1) hidden state로부터의 기울기
            dc_t: (d_h, 1) cell state로부터의 기울기 (역시간에서 누적)
            cache: forward에서 저장한 값들
        
        Returns:
            기울기들 및 이전 시간의 기울기 (dc_prev, dh_prev)
        """
        (x_t, h_prev, c_prev, hx, z_f, f_t, z_i, i_t, 
         z_c, c_tilde, c_t, z_o, o_t, h_t) = cache
        
        # Output gate로부터의 기울기
        d_tanh_c = dh_t * o_t * (1 - np.tanh(c_t)**2)
        dc_t_cell = d_tanh_c  # Cell state로부터
        
        # Cell state 전체 기울기 (역시간 누적 포함)
        # dc_t += d_tanh_c (이미 추가됨)
        dc_t = dc_t + d_tanh_c
        
        # Output gate 기울기
        do_t = dh_t * np.tanh(c_t)
        dz_o = do_t * o_t * (1 - o_t)
        
        # Cell candidate 기울기
        dc_tilde = dc_t * i_t
        dz_c = dc_tilde * (1 - c_tilde**2)
        
        # Input gate 기울기
        di_t = dc_t * c_tilde
        dz_i = di_t * i_t * (1 - i_t)
        
        # Forget gate 기울기
        df_t = dc_t * c_prev
        dz_f = df_t * f_t * (1 - f_t)
        
        # 이전 cell state로의 기울기
        dc_prev = dc_t * f_t
        
        # 입력 및 이전 hidden으로의 기울기
        dhx = (self.W_f.T @ dz_f + self.W_i.T @ dz_i + 
               self.W_c.T @ dz_c + self.W_o.T @ dz_o)
        
        dh_prev = dhx[:self.d_h]
        dx_t = dhx[self.d_h:]
        
        # 가중치 기울기
        dW_f = dz_f @ hx.T
        db_f = dz_f
        
        dW_i = dz_i @ hx.T
        db_i = dz_i
        
        dW_c = dz_c @ hx.T
        db_c = dz_c
        
        dW_o = dz_o @ hx.T
        db_o = dz_o
        
        grads = {
            'W_f': dW_f, 'b_f': db_f,
            'W_i': dW_i, 'b_i': db_i,
            'W_c': dW_c, 'b_c': db_c,
            'W_o': dW_o, 'b_o': db_o,
        }
        
        return dh_prev, dc_prev, dx_t, grads
    
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTM:
    def __init__(self, d_x, d_h, d_y, learning_rate=0.001):
        self.d_x = d_x
        self.d_h = d_h
        self.d_y = d_y
        self.lr = learning_rate
        
        self.lstm_cell = LSTMCell(d_x, d_h)
        
        # Output layer
        self.W_hy = np.random.randn(d_y, d_h) / np.sqrt(d_h)
        self.b_y = np.zeros((d_y, 1))
    
    def forward(self, X):
        """
        X: (T, d_x) 시퀀스
        """
        T, d_x = X.shape
        assert d_x == self.d_x
        
        h = np.zeros((self.d_h, 1))
        c = np.zeros((self.d_h, 1))
        
        h_cache = [h]
        c_cache = [c]
        cell_cache = []
        y_list = []
        
        for t in range(T):
            x_t = X[t:t+1].T
            h, c, cache = self.lstm_cell.forward(x_t, h, c)
            
            h_cache.append(h)
            c_cache.append(c)
            cell_cache.append(cache)
            
            # Output
            y_t = self.W_hy @ h + self.b_y
            y_list.append(y_t)
        
        return y_list, (h_cache, c_cache, cell_cache)
    
    def backward(self, y_pred, y_true, cache):
        """
        y_true: (T, d_y)
        """
        T, d_y = y_true.shape
        h_cache, c_cache, cell_cache = cache
        
        # 기울기 초기화
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        
        lstm_grads = {
            'W_f': np.zeros_like(self.lstm_cell.W_f),
            'b_f': np.zeros_like(self.lstm_cell.b_f),
            'W_i': np.zeros_like(self.lstm_cell.W_i),
            'b_i': np.zeros_like(self.lstm_cell.b_i),
            'W_c': np.zeros_like(self.lstm_cell.W_c),
            'b_c': np.zeros_like(self.lstm_cell.b_c),
            'W_o': np.zeros_like(self.lstm_cell.W_o),
            'b_o': np.zeros_like(self.lstm_cell.b_o),
        }
        
        dh = np.zeros((self.d_h, 1))
        dc = np.zeros((self.d_h, 1))
        
        for t in reversed(range(T)):
            # 출력층 기울기
            dy = y_pred[t] - y_true[t:t+1].T
            dW_hy += dy @ h_cache[t+1].T
            db_y += dy
            
            # Hidden state로의 기울기
            dh += self.W_hy.T @ dy
            
            # LSTM cell 역전파
            dh, dc, dx, cell_grad = self.lstm_cell.backward(dh, dc, cell_cache[t])
            
            for key in cell_grad:
                lstm_grads[key] += cell_grad[key]
        
        # Gradient clipping
        clip_val = 5.0
        for key in lstm_grads:
            lstm_grads[key] = np.clip(lstm_grads[key], -clip_val, clip_val)
        
        dW_hy = np.clip(dW_hy, -clip_val, clip_val)
        db_y = np.clip(db_y, -clip_val, clip_val)
        
        return dW_hy, db_y, lstm_grads
    
    def update(self, dW_hy, db_y, lstm_grads):
        self.W_hy -= self.lr * dW_hy
        self.b_y -= self.lr * db_y
        
        self.lstm_cell.W_f -= self.lr * lstm_grads['W_f']
        self.lstm_cell.b_f -= self.lr * lstm_grads['b_f']
        self.lstm_cell.W_i -= self.lr * lstm_grads['W_i']
        self.lstm_cell.b_i -= self.lr * lstm_grads['b_i']
        self.lstm_cell.W_c -= self.lr * lstm_grads['W_c']
        self.lstm_cell.b_c -= self.lr * lstm_grads['b_c']
        self.lstm_cell.W_o -= self.lr * lstm_grads['W_o']
        self.lstm_cell.b_o -= self.lr * lstm_grads['b_o']


def copy_task():
    """
    Copy Task: 입력 시퀀스 후반의 일부 심볼을 출력으로 복사
    Long-term dependency 학습의 난이도 테스트
    """
    np.random.seed(42)
    
    # 파라미터
    seq_len = 20  # 시퀀스 길이
    vocab_size = 10  # 심볼 수
    d_h_rnn = 8
    d_h_lstm = 8
    
    # 데이터 생성
    def generate_copy_data(n_samples=100):
        X = []
        y = []
        
        for _ in range(n_samples):
            # 입력: 첫 10개 토큰 + 10개 dummy + 시작 신호
            seq = np.random.randint(0, vocab_size, size=10)
            dummy = np.zeros(10, dtype=int)
            marker = vocab_size  # 시작 신호
            
            input_seq = np.concatenate([seq, dummy, [marker]])
            target_seq = np.concatenate([dummy, [marker], seq])  # 출력: 시작 신호 후 원본
            
            X.append(input_seq)
            y.append(target_seq)
        
        return np.array(X), np.array(y)
    
    X, y = generate_copy_data(100)
    
    # LSTM 학습
    print("\n=== Copy Task with LSTM ===")
    lstm = LSTM(d_x=vocab_size+1, d_h=d_h_lstm, d_y=vocab_size+1, learning_rate=0.01)
    
    losses = []
    for epoch in range(100):
        total_loss = 0
        for i in range(len(X)):
            X_seq = np.eye(vocab_size+1)[X[i]]  # One-hot encoding
            y_seq = np.eye(vocab_size+1)[y[i]]
            
            y_pred, cache = lstm.forward(X_seq)
            dW_hy, db_y, lstm_grads = lstm.backward(y_pred, y_seq, cache)
            lstm.update(dW_hy, db_y, lstm_grads)
            
            # Loss
            loss = 0.5 * np.sum((np.array([yp.ravel() for yp in y_pred]) - y_seq)**2)
            total_loss += loss
        
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.4f}")
    
    # 학습 곡선
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Copy Task: LSTM Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/lstm_copy_task.png', dpi=100)
    print("\nPlot saved to /tmp/lstm_copy_task.png")
    plt.close()


if __name__ == "__main__":
    copy_task()
```

**구현의 핵심**:

1. **Gate 계산**: Sigmoid 활성화로 [0,1] 범위 유지
2. **Cell State 업데이트**: 덧셈으로 정보 유지
3. **기울기 흐름**: LSTM의 차이점 명확히 관찰 가능

## 🔗 실전 연결

**PyTorch의 LSTM**:

```python
import torch
import torch.nn as nn

class SimpleSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# 사용
model = SimpleSeq2Seq(10, 128, 10)
```

**Bidirectional LSTM** (양방향):

```python
self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, 
                    batch_first=True)
# 출력: (batch, seq_len, 2*hidden_size)
```

## ⚖️ 가정과 한계

1. **Gate의 선형성**: 각 gate는 입력의 선형 함수이므로, 복잡한 선택 로직은 표현하기 어렵다.

2. **메모리 용량**: Cell state의 차원이 메모리 용량을 결정하므로, 아주 긴 시퀀스는 여전히 어렵다.

3. **학습 초기 상태**: Cell state의 초기값과 forget gate의 초기 편향(bias)이 학습에 큰 영향을 미친다. (보통 forget bias를 1로 초기화)

4. **Computational Cost**: LSTM은 Vanilla RNN의 4배 계산량 필요.

## 📌 핵심 정리

| 항목 | Vanilla RNN | LSTM | GRU |
|------|------------|------|-----|
| **업데이트 메커니즘** | $h_t = \sigma(W_{hh}h_{t-1} + \cdots)$ | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ |
| **Gate 개수** | 0 | 3 (f, i, o) | 2 (z, r) |
| **가중치 수** | $d_h^2 + d_h \cdot d_x$ | $4d_h^2 + 4d_h \cdot d_x$ | $3d_h^2 + 3d_h \cdot d_x$ |
| **Gradient Flow** | 곱셈 (vanishing) | 덧셈 via CEC | 덧셈 (interpolation) |
| **Long-term Memory** | 취약 | 강함 | 중간 |
| **학습 속도** | 빠름 | 느림 | 중간 |

## 🤔 생각해볼 문제

1. **Forget Gate 초기 편향**: 왜 forget gate의 bias를 1로 초기화하면 더 잘 학습될까?

2. **Output Gate의 필요성**: Output gate를 제거하고 $h_t = \tanh(c_t)$만 사용하면 어떻게 될까?

3. **Cell State vs Hidden State**: 내부 메모리와 외부 출력을 분리하는 이유는? Transformer에서는 이런 분리가 없는데?

4. **GRU의 Reset Gate**: Reset gate가 없다면 ($r_t = 1$), GRU는 어떻게 될까?

5. **Peephole Connections**: Jozefowicz et al. (2015)의 peephole LSTM은 cell state를 gate의 입력으로 추가하는데, 왜 도움이 될까?

6. **Gating vs Attention**: LSTM의 gate 메커니즘과 Attention의 개념적 유사성은? 둘 다 "선택"하는 메커니즘인가?

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Vanishing/Exploding Gradient의 수학적 분석](./02-vanishing-exploding.md) | [📚 README로 돌아가기](../README.md) | [04. Echo State Network과 Reservoir Computing ▶](./04-echo-state-network.md) |

</div>
