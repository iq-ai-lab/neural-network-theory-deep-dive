# 5. ResNet과 Residual Connection의 Gradient Flow

## 🎯 핵심 질문

Residual Connection이 왜 깊은 신경망 훈련에 필수적인가? 일반 네트워크와 ResNet의 그래디언트 흐름이 구체적으로 어떻게 다른가? 그리고 ResNet의 여러 변형들(Pre-Activation, Bottleneck, DenseNet)은 어떤 개선을 제공하는가?

## 🔍 필요성

2015년 이전, 깊은 신경망(50+ 층)은 훈련이 거의 불가능했습니다. Vanishing gradient 문제로 인해, 역전파 중 그래디언트가 초기 층으로 전달되지 못했습니다. ResNet (He et al., 2015)의 도입으로, 152층, 심지어 1000층 이상의 네트워크도 훈련 가능하게 되었습니다. 이는 현대 깊은 모델(Transformer, Vision Transformer 등)의 기초가 되었습니다.

## 📐 선행 지식

- 역전파와 연쇄법칙 (Ch3)
- Batch Normalization (Ch4)
- 깊은 신경망의 훈련 어려움
- 행렬의 고유값(eigenvalue)

## 📖 직관

**Residual Connection의 핵심 아이디어**

일반적인 신경망:
$$h_{l+1} = F_l(h_l)$$

블록별로, 각 층은 이전 층의 출력을 변환합니다. 깊으면 깊을수록, 변환이 쌓입니다. 역전파 시:

$$\frac{\partial L}{\partial h_l} = \frac{\partial L}{\partial h_{l+1}} \cdot \frac{\partial F_l}{\partial h_l} \cdot \frac{\partial F_{l-1}}{\partial h_{l-2}} \cdots$$

곱셈이 많아지면 그래디언트가 지수적으로 감쇠합니다 (vanishing) 또는 폭발합니다 (exploding).

**ResNet의 해결책**

$$h_{l+1} = h_l + F_l(h_l)$$

역전파 시:

$$\frac{\partial L}{\partial h_l} = \frac{\partial L}{\partial h_{l+1}} \left( 1 + \frac{\partial F_l}{\partial h_l} \right)$$

**핵심**: $1$ 항이 있으므로, $\frac{\partial F_l}{\partial h_l}$이 작아도 그래디언트가 직접 전달됩니다. 이를 "gradient highway(그래디언트 고속도로)"라고 부릅니다.

예를 들어, $\frac{\partial F_l}{\partial h_l} \approx 0.5$ 라면:
- 일반망: $0.5 \cdot 0.5 \cdot 0.5 \cdots = 0.5^L$ (지수적 감쇠)
- ResNet: $1 + 0.5 \approx 1.5$ (각 층에서 독립적)

## ✏️ 정의

**Residual Block**

$$y = x + F(x)$$

여기서:
- $x$: 입력 (skip connection)
- $F(x)$: 학습 가능한 함수 (일반적으로 2-3개 합성층)
- $y$: 출력

**일반적 구성**:
$$F(x) = \text{Conv}_1 \to \text{BN} \to \text{ReLU} \to \text{Conv}_2 \to \text{BN}$$

출력 후 ReLU는 skip connection 이후:
$$y = \text{ReLU}(x + F(x))$$

**형상 일치(Shape Matching)**

$x$와 $F(x)$의 형상이 같아야 더할 수 있습니다. 해상도나 채널이 다르면:

$$y = x + F_{\text{project}}(x)$$

여기서 $F_{\text{project}}$는 1×1 컨볼루션 등으로 차원을 맞춥니다.

## 🔬 증명

### 정리 1: ResNet의 Backward Pass 분석

**명제**: Residual block에서 그래디언트는 skip connection을 통해 직접 전달됩니다.

**증명**:

Forward pass:
$$y = x + F(x; \theta)$$

손실 $L$에 대한 역전파:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

연쇄법칙:
$$\frac{\partial y}{\partial x} = \frac{\partial}{\partial x}(x + F(x; \theta)) = I + \frac{\partial F}{\partial x}$$

따라서:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left( I + \frac{\partial F}{\partial x} \right)$$

분리하면:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot I + \frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}$$

첫 항 $\frac{\partial L}{\partial y} \cdot I = \frac{\partial L}{\partial y}$ 는 $\frac{\partial F}{\partial x}$와 무관하게 그대로 전달됩니다.

따라서 $F$의 가중치가 아무리 작아도, 그래디언트가 "직접 경로"를 통해 전달됩니다. $\square$

### 정리 2: 깊은 Plain Network와 ResNet의 비교

**명제**: 깊은 네트워크에서 ResNet이 일반 네트워크보다 훈련이 쉽습니다.

**증명 스케치**:

Plain network $L$개 층:
$$h_l = F_l(h_{l-1})$$

그래디언트:
$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_L} \prod_{l=1}^L \frac{\partial F_l}{\partial h_{l-1}}$$

각 $\left\|\frac{\partial F_l}{\partial h_{l-1}}\right\|$ 의 고유값이 모두 1보다 작다면:
$$\left\| \frac{\partial L}{\partial h_0} \right\| \lesssim \prod_{l=1}^L \rho < \rho^L$$

여기서 $\rho < 1$은 최대 고유값입니다. $L$이 크면 지수적으로 감쇠합니다.

ResNet:
$$h_l = h_{l-1} + F_l(h_{l-1})$$

그래디언트:
$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_L} \prod_{l=1}^L (I + \frac{\partial F_l}{\partial h_{l-1}})$$

각 항의 최소 고유값이 1 이상이므로 (1이 추가됨):
$$\left\|\frac{\partial L}{\partial h_0}\right\| \gtrsim \left\|\frac{\partial L}{\partial h_L}\right\|$$

그래디언트가 감쇠하지 않습니다.

### 정리 3: Pre-Activation (He et al., 2016)

**명제**: BN과 ReLU를 skip connection 전에 적용하는 "Pre-Activation" 구조가 더 좋은 그래디언트 흐름을 제공합니다.

**기존 (Post-Activation)**:
$$y = \text{ReLU}(x + F(x))$$

문제: ReLU가 음수를 0으로 만들어, skip connection도 영향을 받습니다.

**Pre-Activation**:
$$y = x + F(x)$$

여기서 $F(x) = \text{ReLU}(\text{BN}(\text{Conv}_2(\text{ReLU}(\text{BN}(\text{Conv}_1(x))))))$

개선: skip connection은 순수한 항등성을 유지하고, $F$의 비선형성은 독립적으로 작용합니다.

## 💻 NumPy 구현

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. ResNet 블록 (간단한 구현)
class ResidualBlock:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # 가중치 초기화
        self.W1 = np.random.randn(input_dim, 64) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, output_dim) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(output_dim)
        
        # Skip connection (차원 일치)
        if input_dim != output_dim:
            self.W_skip = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
            self.b_skip = np.zeros(output_dim)
        else:
            self.W_skip = None
    
    def forward(self, x):
        # F(x) 계산
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        self.z2 = self.a1 @ self.W2 + self.b2
        
        # Skip connection
        if self.W_skip is not None:
            self.x_skip = x @ self.W_skip + self.b_skip
        else:
            self.x_skip = x
        
        # 출력
        self.y = self.x_skip + self.z2
        self.a_out = np.maximum(0, self.y)  # 최종 ReLU
        
        return self.a_out
    
    def backward(self, grad_output):
        # 최종 ReLU 그래디언트
        grad_y = grad_output * (self.y > 0)
        
        # W2, b2 그래디언트
        grad_W2 = self.a1.T @ grad_y / grad_y.shape[0]
        grad_b2 = np.sum(grad_y, axis=0) / grad_y.shape[0]
        
        # skip connection 포함 역전파
        grad_skip = grad_y
        grad_a1 = grad_y @ self.W2.T
        
        # ReLU 그래디언트
        grad_z1 = grad_a1 * (self.z1 > 0)
        
        # W1, b1 그래디언트
        grad_W1 = self.x @ self.W1.T @ grad_z1.T
        grad_b1 = np.sum(grad_z1, axis=0)
        
        # 입력 그래디언트
        if self.W_skip is not None:
            grad_x_skip = grad_skip @ self.W_skip.T
            grad_x = (grad_z1 @ self.W1.T) + grad_x_skip
        else:
            grad_x = grad_skip + (grad_z1 @ self.W1.T)
        
        return grad_x, grad_W1, grad_b1, grad_W2, grad_b2

# 2. 깊은 네트워크 비교
def deep_network_gradient_flow_analysis(depth=10, width=32):
    """
    깊은 plain network vs ResNet의 그래디언트 흐름 비교
    """
    np.random.seed(42)
    
    # 더미 입력과 출력
    batch_size = 32
    X = np.random.randn(batch_size, width)
    y_true = np.random.randn(batch_size, width)
    
    # 1. Plain Network (각 층은 단순 선형 + ReLU)
    plain_grads = []
    h = X.copy()
    weights_plain = []
    
    for l in range(depth):
        W = np.random.randn(width, width) * 0.1
        weights_plain.append(W)
        h = np.maximum(0, h @ W)  # ReLU
    
    # 역전파 (간단화)
    grad = (h - y_true) / batch_size
    for l in range(depth - 1, -1, -1):
        # ReLU 마스크
        h_before = X if l == 0 else h
        
        # 그래디언트 추적
        plain_grads.append(np.linalg.norm(grad))
        
        # 다음 층으로 역전파
        grad = grad @ weights_plain[l].T
        grad = grad * (h_before > 0)
    
    plain_grads = np.array(plain_grads[::-1])
    
    # 2. ResNet-like (매 2층마다 skip connection)
    resnet_grads = []
    h = X.copy()
    weights_resnet = []
    h_skip_memory = [X]
    
    for l in range(depth):
        W = np.random.randn(width, width) * 0.1
        weights_resnet.append(W)
        
        # Residual block (2층마다)
        if l % 2 == 0:
            h_in = h_skip_memory[-1]
        
        h = np.maximum(0, h @ W)
        
        if l % 2 == 1:
            h = h + h_in  # Skip connection
            h_skip_memory.append(h)
    
    # 역전파
    grad = (h - y_true) / batch_size
    for l in range(depth - 1, -1, -1):
        resnet_grads.append(np.linalg.norm(grad))
        grad = grad @ weights_resnet[l].T
        grad = grad * (h > 0)
    
    resnet_grads = np.array(resnet_grads[::-1])
    
    return plain_grads, resnet_grads

# 실험 실행
depths = [10, 20, 50, 100]
results = {'plain': [], 'resnet': []}

print("="*60)
print("깊이에 따른 그래디언트 흐름 분석")
print("="*60)

for depth in depths:
    plain_grads, resnet_grads = deep_network_gradient_flow_analysis(depth=depth)
    
    # 초기층 그래디언트 (vanishing의 지표)
    plain_ratio = plain_grads[0] / plain_grads[-1] if plain_grads[-1] > 1e-10 else 0
    resnet_ratio = resnet_grads[0] / resnet_grads[-1] if resnet_grads[-1] > 1e-10 else 0
    
    results['plain'].append(plain_ratio)
    results['resnet'].append(resnet_ratio)
    
    print(f"\n깊이: {depth}")
    print(f"  Plain:  초층/최종 그래디언트 비율 = {plain_ratio:.2e}")
    print(f"  ResNet: 초층/최종 그래디언트 비율 = {resnet_ratio:.2e}")
    print(f"  개선: {resnet_ratio / (plain_ratio + 1e-10):.1f}배")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 비율 비교 (로그 스케일)
axes[0].semilogy(depths, results['plain'], 'o-', label='Plain Network', linewidth=2, markersize=8)
axes[0].semilogy(depths, results['resnet'], 's-', label='ResNet', linewidth=2, markersize=8)
axes[0].set_xlabel('네트워크 깊이', fontsize=12)
axes[0].set_ylabel('초층/최종층 그래디언트 비율', fontsize=12)
axes[0].set_title('깊이에 따른 그래디언트 감쇠', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 특정 깊이에서 층별 그래디언트
depth = 50
plain_grads, resnet_grads = deep_network_gradient_flow_analysis(depth=depth)

axes[1].plot(range(depth), plain_grads, 'o-', label='Plain Network', alpha=0.7, linewidth=2)
axes[1].plot(range(depth), resnet_grads, 's-', label='ResNet', alpha=0.7, linewidth=2)
axes[1].set_xlabel('층 번호', fontsize=12)
axes[1].set_ylabel('그래디언트 규범', fontsize=12)
axes[1].set_title(f'깊이 {depth}에서 층별 그래디언트', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('/tmp/resnet_gradient_flow.png', dpi=100, bbox_inches='tight')
print("\n그래프 저장: /tmp/resnet_gradient_flow.png")

```

## 🔗 실전

Residual Connection의 활용:

1. **ResNet 계열** (He et al., 2015):
   - ResNet-50/101/152: ImageNet의 표준
   - 변종: Wide ResNet, ResNeXt

2. **Transformer의 Skip Connection**:
   - MultiHead Attention에서: $z = x + \text{Attention}(x)$
   - FFN에서: $y = z + \text{FFN}(z)$

3. **DenseNet** (Huang et al., 2016):
   - 더 강한 버전: 모든 이전 층과 연결
   - $h_l = F_l([h_0, h_1, \ldots, h_{l-1}])$
   - 메모리 효율성 개선

4. **Highway Networks** (Srivastava et al., 2015):
   - $h_{l+1} = T_l \odot F_l(h_l) + (1 - T_l) \odot h_l$
   - 학습 가능한 게이트로 skip 강도 조절

## ⚖️ 한계

1. **초기화에 민감**: $F(x)$가 0에 가까워야 처음에는 거의 항등함수
   - Fixup Initialization 필요 (Ch4-05 참조)

2. **메모리 오버헤드**: 역전파를 위해 중간 활성화를 모두 저장해야 함

3. **최적화 landscape 변화**: Skip connection이 있어도 깊은 네트워크는 여전히 어려울 수 있음

4. **깊이의 한계**: 아무리 skip connection이 있어도 극도로 깊으면 (1000+층) 훈련 어려움

## 📌 핵심 정리

| 항목 | 일반 네트워크 | ResNet |
|------|-------------|--------|
| **Forward** | $h_{l+1} = F_l(h_l)$ | $h_{l+1} = h_l + F_l(h_l)$ |
| **Backward** | $\frac{\partial L}{\partial h_0} = \prod \frac{\partial F_l}{\partial h_{l-1}}$ | $\frac{\partial L}{\partial h_0} \propto \prod (I + \frac{\partial F_l}{\partial h_{l-1}})$ |
| **그래디언트** | 지수 감쇠 (깊으면 소실) | 안정적 전달 (깊어도 유지) |
| **최대 깊이** | 20-30층 가능 | 100+ 층 가능 |
| **변형** | - | Pre-Activation, Bottleneck, DenseNet |

## 🤔 문제

1. **문제 5.1**: Skip connection 없이 50층 네트워크를 훈련시키고, ResNet과 그래디언트를 비교하세요 (제공된 NumPy 코드 활용).

2. **문제 5.2**: Pre-Activation vs Post-Activation의 그래디언트 흐름 차이를 실험으로 보이세요.

3. **문제 5.3**: DenseNet 구조를 NumPy로 구현하고, ResNet과 메모리 사용량을 비교하세요.

4. **문제 5.4**: Fixup Initialization (Ch4-05)를 ResNet에 적용하고, 표준 Xavier initialization과 비교하세요.

---

<div align="center">

| | |
|---|---|
| [◀ 04. Transformer의 범용성](./04-transformer-uat.md) | [📚 README로 돌아가기](../README.md) |

</div>
