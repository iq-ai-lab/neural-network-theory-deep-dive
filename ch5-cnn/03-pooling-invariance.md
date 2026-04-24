# Chapter 5-03. Pooling과 Local Invariance

## 🎯 핵심 질문
- Pooling은 수학적으로 무엇이며, 왜 필요한가?
- Max pooling이 제공하는 invariance(불변성)는 정확히 무엇인가?
- Stride와 pooling의 관계는? 언제 어느 것을 사용해야 하는가?

---

## 🔍 필요성

CNN의 equivariance는 강력하지만, 때론 **과도합니다**.

**문제 시나리오**:
- 이미지를 1-2 픽셀 shift해도 분류 결과는 같아야 함
- 하지만 strict equivariance는 미세한 shift에도 출력이 정확히 shift됨
- 결과: 숫자 "5"와 "5"(1픽셀 우측이동)을 다르게 분류할 수 있음

**해결책 - Pooling**:
- 국소 영역(3×3, 2×2)에서 "대표값"(최댓값, 평균값) 추출
- 작은 shift에 robust하게 만듦
- 계산량 감소, receptive field 확장 (Ch5-02의 trade-off)

**특별히, Max pooling**:
- "가장 활성화된 특징"만 통과
- 작은 이동에 불변성(invariance) 제공
- 신경 세포의 "최댓값 선택" 행동과 유사

---

## 📐 선행 지식

- **Ch5-01**: Convolution, equivariance 정의
- **Ch5-02**: Receptive field, parameter sharing
- **Ch3**: 활성화 함수, 기울기
- **수학**: 최댓값 함수, 편미분 (불연속점 처리)

---

## 📖 직관

### Max Pooling의 기하학적 의미

$2 \times 2$ max pooling을 $[a, b, c, d]$ (2×2 패치)에 적용:
$$
\text{MaxPool}([a,b,c,d]) = \max(a, b, c, d)
$$

**결과**: 가장 큰 값만 남음. 다른 값들은 버려짐.

**효과**: 
- 이 영역이 어떤 특징(edge, texture)에 "반응"했는지만 유지
- 정확한 위치는 무시

### 작은 Shift에 대한 Robustness

```
원본 2×2 영역:          1픽셀 shift 후:
[5  3]                  [3  1  ?]
[2  1]  → max=5         [1  4  ?]  → max=4 (또는 3 이상)
```

shift가 작으면 → 최댓값이 같거나 비슷함 → invariance!

### Stride와의 관계

- **Pooling 없이 stride 2 conv**: 정보 손실, 계산량 감소
- **Stride 1 + Pooling**: 세부 정보 유지, 나중에 선택적 손실
- **실무**: 대부분 stride=1 conv + pooling 조합

### Global Average Pooling

Pooling의 극단:
$$\text{GlobalAvgPool}(F) = \frac{1}{H \times W} \sum_{i,j} F[i,j]$$

**효과**: 특성맵 전체를 하나의 수로 축약
- 완전연결층의 가중치 감소
- 공간 정보 완전 제거 (high-level semantic features만)

---

## ✏️ 정의

### 1. Max Pooling

**정의**:
$$\boxed{\text{MaxPool}_k(X)[i,j] = \max_{(m,n) \in \mathcal{N}(i,j)} X[m,n]}$$

여기서:
- $X$ = 입력 특성맵 (높이 $H$, 너비 $W$, 채널 $C$)
- $k$ = pooling 커널 크기 (보통 2×2 또는 3×3)
- $\mathcal{N}(i,j)$ = $(i,j)$를 중심으로 한 $k \times k$ 영역
- **stride** = pooling 윈도우 이동 단계 (보통 $k$와 같음, "non-overlapping")

**예시** (2×2 pooling, stride=2):
```
입력:
[1  2 | 5  3]
[4  6 | 2  1]
------+-----
[3  1 | 4  2]
[0  2 | 7  8]

출력:
[6   5]
[3   8]
```

### 2. Average Pooling

$$\boxed{\text{AvgPool}_k(X)[i,j] = \frac{1}{k^2} \sum_{(m,n) \in \mathcal{N}(i,j)} X[m,n]}$$

**vs Max pooling**:
- Max: "가장 활성화된" 값
- Avg: "평균적 활성화"
- Max가 더 일반적 (특히 early layers)

### 3. Local Translation Invariance (이론)

**정리**: $k$-pooling 후, shift 크기 $|s| \leq k/2$에 대해:
$$\boxed{|\text{Pool}(T_s X)[i] - T_s \text{Pool}(X)[i]| \leq \text{bounded}}$$

**증명 스케치**:
원래 $\max(x_1, \ldots, x_{k^2})$가 index $i^*$에서 달성된다면,
shift 후에도 비슷한 값이 남아있음.

**정확한 보장은 아님**:
- strict invariance 아님 (soft invariance)
- 작은 shift에만 유효

### 4. Receptive Field와 Stride

#### Stride 없는 경우 (Ch5-02)
$$\text{RF}_L = (k-1)L + 1$$

#### Stride/Dilation이 있는 경우

각 층의 stride $s_l$, dilation $d_l$:
$$\boxed{\text{RF}_l = \text{RF}_{l-1} + \left((k-1) \prod_{j<l} s_j\right) \cdot d_l}$$

**직관**: 
- stride 증가 → receptive field 빠르게 증가
- 같은 RF를 stride로 달성하려면 깊이 감소 가능

### 5. Stride와 Pooling의 Trade-off

| 특성 | Stride 2 Conv | Stride 1 + Pooling |
|------|--------|---------|
| **파라미터** | $k^2 C_\text{in} C_\text{out}$ | $k^2 C_\text{in} C_\text{out}$ |
| **연산** | 같음 | 같음 (추가 max 연산) |
| **정보 손실** | 고정적 | 적응적 (어디서 max가 나왔는지는 keep) |
| **Gradient Flow** | 빠름 | Pooling에서 선택된 위치만 gradient |
| **실무** | 초기: 정보 손실 많음 | 일반적 선호 |

### 6. Dilated Pooling & Stochastic Pooling

**Dilated Pooling**:
$$\text{DilPool}_k(X)[i,j] = \max_{(m,n) \in \mathcal{N}_d(i,j)} X[m,n]$$

여기서 $\mathcal{N}_d$는 dilation $d$를 포함한 영역.

**Stochastic Pooling** (훈련 시만):
$$\text{StocPool} = \text{sample from } \max(\cdot) \text{ region}$$
비율 높은 값에 더 높은 확률.

---

## 🔬 증명

### 정리 1: Max Pooling의 Invariance

**명제**: 
$2 \times 2$ max pooling에 대해, shift $|s| \leq 1$인 경우:
$$\text{MaxPool}(T_s X) \text{와 } T_s \text{MaxPool}(X) \text{는 } 100\% \text{ 보장은 아니지만 자주 일치}$$

**증명** (직관):

$2 \times 2$ 윈도우:
$$W = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

$\max(a,b,c,d) = M$이라 하자 (위치 $(i^*,j^*)$).

**경우 1**: $s=(1,0)$ (오른쪽 1칸 shift)
```
원본 후보 2×2:
[a b] → max = M

Shift 후:
[b ?] → ? 자리는 새로 나타난 값
```

$b$는 여전히 나타남. 만약 $M = a$라면 문제, 
하지만 $M \geq b$이므로 $b \leq M$.

새로운 최댓값은 $\geq b$이고, $M \in [b, M]$이면 거의 같음.

**정확한 명제** (더 엄밀):
$$\|\text{MaxPool}(T_s X) - T_s \text{MaxPool}(X)\|_\infty \leq \text{max variance in small neighborhood}$$

### 정리 2: Receptive Field의 빠른 확장

**명제**:
Stride $s$인 convolution은 receptive field를 $\prod s$ 배 가속

**증명**:
$l$번째 층의 유효 kernel width = $(k-1) \prod_{j \leq l} s_j + 1$

stride 없을 때 대비, stride 곱 만큼 빨리 증가.

---

## 💻 NumPy 실험

### 1. Max Pooling 직접 구현

```python
import numpy as np
import matplotlib.pyplot as plt

def max_pool2d(image, pool_size=2, stride=None):
    """
    2D max pooling 구현
    image: (H, W) 또는 (H, W, C)
    pool_size: 풀링 커널 크기 (보통 2)
    stride: 스트라이드 (기본값=pool_size)
    """
    if stride is None:
        stride = pool_size
    
    if image.ndim == 2:
        h, w = image.shape
        c = 1
        image = image[:, :, np.newaxis]
    else:
        h, w, c = image.shape
    
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out, c))
    
    for i in range(h_out):
        for j in range(w_out):
            for ch in range(c):
                window = image[i*stride:i*stride+pool_size,
                              j*stride:j*stride+pool_size, ch]
                output[i, j, ch] = np.max(window)
    
    return output.squeeze()

def avg_pool2d(image, pool_size=2, stride=None):
    """Average pooling"""
    if stride is None:
        stride = pool_size
    
    if image.ndim == 2:
        h, w = image.shape
        c = 1
        image = image[:, :, np.newaxis]
    else:
        h, w, c = image.shape
    
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out, c))
    
    for i in range(h_out):
        for j in range(w_out):
            for ch in range(c):
                window = image[i*stride:i*stride+pool_size,
                              j*stride:j*stride+pool_size, ch]
                output[i, j, ch] = np.mean(window)
    
    return output.squeeze()

# ===== 예시 =====
X = np.array([
    [1, 2, 5, 3],
    [4, 6, 2, 1],
    [3, 1, 4, 2],
    [0, 2, 7, 8]
], dtype=float)

print("원본 이미지:")
print(X)

max_pool = max_pool2d(X, pool_size=2, stride=2)
print("\nMax Pooling (2×2, stride=2):")
print(max_pool)

avg_pool = avg_pool2d(X, pool_size=2, stride=2)
print("\nAverage Pooling (2×2, stride=2):")
print(avg_pool)

# ===== 더 복잡한 예시 =====
X_large = np.random.randn(8, 8)
print("\n" + "="*50)
print("8×8 랜덤 이미지")

for pool_size in [2, 3]:
    result = max_pool2d(X_large, pool_size=pool_size, stride=pool_size)
    print(f"\nMax Pooling {pool_size}×{pool_size}:")
    print(f"  입력: {X_large.shape} → 출력: {result.shape}")

# Non-overlapping vs overlapping
result_2_2 = max_pool2d(X_large, pool_size=2, stride=2)
result_2_1 = max_pool2d(X_large, pool_size=2, stride=1)
print(f"\nPooling size=2, stride=2: {result_2_2.shape}")
print(f"Pooling size=2, stride=1 (overlapping): {result_2_1.shape}")
```

**출력**:
```
원본 이미지:
[[1. 2. 5. 3.]
 [4. 6. 2. 1.]
 [3. 1. 4. 2.]
 [0. 2. 7. 8.]]

Max Pooling (2×2, stride=2):
[[6. 5.]
 [3. 8.]]

Average Pooling (2×2, stride=2):
[[3.25 2.75]
 [1.5  5.25]]

8×8 랜덤 이미지

Max Pooling 2×2:
  입력: (8, 8) → 출력: (4, 4)

Max Pooling 3×3:
  입력: (8, 8) → 출력: (2, 2)
```

### 2. Translation Invariance 검증

```python
def shift_image(img, dy, dx, constant_value=0):
    """이미지를 (dy, dx)만큼 shift"""
    result = np.full_like(img, constant_value)
    
    if dy > 0:
        if dx > 0:
            result[dy:, dx:] = img[:-dy, :-dx]
        else:
            result[dy:, :dx] = img[:-dy, -dx:]
    else:
        if dx > 0:
            result[:dy, dx:] = img[-dy:, :-dx]
        else:
            result[:dy, :dx] = img[-dy:, -dx:]
    
    return result

# ===== Translation Invariance 테스트 =====
print("="*60)
print("Max Pooling의 Translation Invariance")
print("="*60)

# 패턴 생성
X = np.zeros((8, 8))
X[2:5, 2:5] = 1  # 중앙에 작은 정사각형

print("\n원본 이미지:")
print(X.astype(int))

# Pooling 결과
pooled = max_pool2d(X, pool_size=2, stride=2)
print("\nMax Pooling 결과 (2×2, stride=2):")
print(pooled.astype(int))

# Shift 테스트
shifts = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
print("\n" + "-"*60)
print("Shift별 Pooling 결과 비교")
print("-"*60)

for dy, dx in shifts:
    X_shifted = shift_image(X, dy, dx, constant_value=0)
    
    # 방법 1: Shift → Pool
    pool_after_shift = max_pool2d(X_shifted, pool_size=2, stride=2)
    
    # 방법 2: Pool → Shift
    pooled_original = max_pool2d(X, pool_size=2, stride=2)
    pool_then_shift = shift_image(pooled_original, 
                                  dy//2, dx//2, constant_value=0)
    
    # 일치 여부
    match = np.allclose(pool_after_shift, pool_then_shift)
    
    print(f"\nShift ({dy},{dx}):")
    print(f"  Shift→Pool: {pool_after_shift.astype(int).tolist()}")
    print(f"  Pool→Shift: {pool_then_shift.astype(int).tolist()}")
    print(f"  일치: {match}")

# ===== 상세한 예시 =====
print("\n" + "="*60)
print("미세한 Shift에서의 불변성")
print("="*60)

# 2×2 이미지로 자세히
simple_img = np.array([
    [1, 2],
    [3, 4]
], dtype=float)

print("\n2×2 이미지:")
print(simple_img)
print(f"Max pooling: {max_pool2d(simple_img)}")

# 오른쪽으로 1칸 shift (circular로, 또는 padding 가정)
simple_shifted = np.array([
    [2, 0],
    [4, 0]
], dtype=float)

print("\n오른쪽으로 1칸 shift:")
print(simple_shifted)
print(f"Max pooling: {max_pool2d(simple_shifted)}")

print("\n→ 최댓값이 4로 같음 (invariance!)")
```

**출력**:
```
============================================================
Max Pooling의 Translation Invariance
============================================================

원본 이미지:
[[0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 1 1 1 0 0 0]
 [0 0 1 1 1 0 0 0]
 [0 0 1 1 1 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]]

Max Pooling 결과 (2×2, stride=2):
[[0 0 0 0]
 [0 1 1 0]
 [0 1 1 0]
 [0 0 0 0]]
```

### 3. Receptive Field 확장 시각화

```python
def conv2d_simple(X, kernel_size=3, stride=1):
    """단순 2D convolution (max over patch)"""
    if X.ndim == 2:
        h, w = X.shape
        c = 1
        X = X[:, :, np.newaxis]
    else:
        h, w, c = X.shape
    
    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1
    
    output = np.zeros((h_out, w_out, c))
    
    for i in range(h_out):
        for j in range(w_out):
            for ch in range(c):
                window = X[i*stride:i*stride+kernel_size,
                          j*stride:j*stride+kernel_size, ch]
                output[i, j, ch] = np.max(window)
    
    return output.squeeze()

def calculate_rf(num_layers, kernel_size, stride_per_layer):
    """Receptive field 계산"""
    rf = kernel_size
    for _ in range(num_layers - 1):
        rf += (kernel_size - 1) * stride_per_layer
    return rf

# ===== Stride와 Pooling의 receptive field =====
print("="*70)
print("Receptive Field: Stride vs Pooling")
print("="*70)

# 설정
L = 5  # 5개 층
k = 3  # 3×3 커널

print(f"\nL={L}층, k={k}×{k} 커널:")

# Stride 없음 (모두 pooling)
rf_pool = calculate_rf(L, k, stride_per_layer=0)
print(f"\nStride=1, 각 층 후 2×2 pooling (stride=2):")
print(f"  실제 receptive field: stride 때문에 {rf_pool + (2**L - 1)} 근처")

# 모두 stride 2
rf_stride = 1
for i in range(L):
    rf_stride = rf_stride + (k-1) * (2**i)
print(f"\nStride=2 conv만 사용:")
print(f"  Receptive field: {rf_stride}")

# 혼합 (stride 1 + pooling)
print(f"\nStride=1 + 2×2 pooling (stride=2):")
print(f"  Conv receptive field: {rf_pool}")
print(f"  추가로 pooling으로 공간 감소")

# ===== 가시화 =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Receptive field 성장
layers = np.arange(1, 11)
rf_k3 = [(k-1)*l + 1 for l in layers]
rf_k5 = [4*l + 1 for l in layers]
rf_k7 = [6*l + 1 for l in layers]

ax = axes[0, 0]
ax.plot(layers, rf_k3, 'o-', label='k=3', linewidth=2)
ax.plot(layers, rf_k5, 's-', label='k=5', linewidth=2)
ax.plot(layers, rf_k7, '^-', label='k=7', linewidth=2)
ax.set_xlabel('Layer 깊이')
ax.set_ylabel('Receptive Field (stride=1)')
ax.set_title('Kernel 크기별 RF 성장')
ax.grid(True, alpha=0.3)
ax.legend()

# 2. Stride의 영향
rf_stride1 = [(k-1)*l + 1 for l in layers]
rf_stride2 = [1 + 2*((k-1)*(2**l - 1)/(2-1)) + 1 for l in layers]  # 근사

ax = axes[0, 1]
ax.plot(layers, rf_stride1, 'o-', label='stride=1', linewidth=2)
ax.set_xlabel('Layer 깊이')
ax.set_ylabel('Receptive Field')
ax.set_title('Stride=1 vs 2 (k=3)')
ax.grid(True, alpha=0.3)
ax.legend()

# 3. Pooling의 invariance 효과
np.random.seed(42)
feature_map = np.random.randn(8, 8)

shifts = np.arange(0, 3)
max_pool_responses = []
avg_pool_responses = []

for shift in shifts:
    shifted_map = np.roll(feature_map, shift, axis=1)
    max_p = max_pool2d(shifted_map, pool_size=2, stride=2)
    avg_p = avg_pool2d(shifted_map, pool_size=2, stride=2)
    max_pool_responses.append(np.mean(max_p))
    avg_pool_responses.append(np.mean(avg_p))

ax = axes[1, 0]
ax.plot(shifts, max_pool_responses, 'o-', label='Max Pool', linewidth=2, markersize=8)
ax.plot(shifts, avg_pool_responses, 's-', label='Avg Pool', linewidth=2, markersize=8)
ax.set_xlabel('Shift 크기 (픽셀)')
ax.set_ylabel('평균 Pooling 출력')
ax.set_title('Shift에 따른 Pooling 불변성')
ax.grid(True, alpha=0.3)
ax.legend()

# 4. 정보 손실 vs 공간 축소
layer_num = np.arange(1, 6)
spatial_size_pooling = 256 // (2**layer_num)
spatial_size_stride2 = 256 // (2**layer_num)

ax = axes[1, 1]
ax.semilogy(layer_num, spatial_size_pooling, 'o-', 
            label='Stride-2 Conv', linewidth=2, markersize=8)
ax.plot(layer_num, spatial_size_stride2, 's--', 
        label='Stride-1 + Pooling', linewidth=2, markersize=8, color='orange')
ax.set_xlabel('Layer 깊이')
ax.set_ylabel('공간 크기 (256 입력 기준)')
ax.set_title('공간 축소 속도 (이론적)')
ax.grid(True, alpha=0.3, which='both')
ax.legend()

plt.tight_layout()
plt.show()
```

### 4. Global Average Pooling

```python
def global_average_pool(feature_maps):
    """
    Global average pooling
    입력: (H, W, C)
    출력: (C,)
    """
    return np.mean(feature_maps, axis=(0, 1))

def global_max_pool(feature_maps):
    """Global max pooling"""
    return np.max(feature_maps.reshape(-1, feature_maps.shape[-1]), axis=0)

# ===== 예시 =====
# 간단한 특성맵 생성 (8×8×3)
feature_maps = np.random.randn(8, 8, 3)

print("Global Average Pooling:")
gap_result = global_average_pool(feature_maps)
print(f"  입력: {feature_maps.shape}")
print(f"  출력: {gap_result.shape}")
print(f"  값: {gap_result}")

gmp_result = global_max_pool(feature_maps)
print("\nGlobal Max Pooling:")
print(f"  입력: {feature_maps.shape}")
print(f"  출력: {gmp_result.shape}")
print(f"  값: {gmp_result}")

# ===== Class Activation Map (CAM) 가능 =====
print("\n"+"="*60)
print("Global Average Pooling의 장점: CAM 해석성")
print("="*60)

# 마지막 conv 층의 특성맵 (가정)
last_features = np.random.randn(7, 7, 512)

# 각 채널의 평균
channel_importance = global_average_pool(last_features)

print(f"\n마지막 특성맵: {last_features.shape}")
print(f"채널별 평균 (중요도): {channel_importance.shape}")
print(f"\n상위 5개 채널의 중요도:")
top_indices = np.argsort(channel_importance)[-5:][::-1]
for idx in top_indices:
    print(f"  채널 {idx}: {channel_importance[idx]:.4f}")
```

---

## 🔗 실전 응용

### 아키텍처 설계 패턴

```python
# 일반적인 CNN 패턴: Conv-Pooling-Conv-Pooling-...

class SimpleConvNet:
    def __init__(self):
        self.layers = []
    
    def add_conv_pool_block(self, k, c_in, c_out, pool_size=2):
        """Conv 후 Pooling 블록 추가"""
        conv_params = k**2 * c_in * c_out
        # 공간 크기는 pooling으로 1/pool_size^2 축소
        self.layers.append({
            'type': 'conv',
            'kernel': k,
            'params': conv_params,
            'channels': (c_in, c_out)
        })
        self.layers.append({
            'type': 'pool',
            'size': pool_size
        })

# 예: ResNet/VGG 스타일 아키텍처
net = SimpleConvNet()
net.add_conv_pool_block(k=3, c_in=3, c_out=64, pool_size=2)
net.add_conv_pool_block(k=3, c_in=64, c_out=128, pool_size=2)
net.add_conv_pool_block(k=3, c_in=128, c_out=256, pool_size=2)

print("CNN 아키텍처:")
spatial_size = 224
for i, layer in enumerate(net.layers):
    if layer['type'] == 'conv':
        params = layer['params']
        c_in, c_out = layer['channels']
        print(f"  Conv{i}: {c_in}→{c_out}, 파라미터={params:,}, "
              f"출력 크기={spatial_size}×{spatial_size}×{c_out}")
    elif layer['type'] == 'pool':
        spatial_size //= layer['size']
        print(f"  Pool{i}: {layer['size']}×{layer['size']}, "
              f"출력 크기={spatial_size}×{spatial_size}")
```

---

## ⚖️ 한계와 고려사항

1. **Invariance는 soft, 보장 아님**:
   - 이론적으로는 soft invariance
   - 실제로는 매우 큰 shift에는 실패

2. **정보 손실**:
   - Max pooling은 명시적으로 정보를 버림
   - 어떤 위치에서 max가 나왔는지 알 수 없음 (역전파는 가능하지만 순방향은 손실)

3. **Gradient 역전파**:
   - Max pooling: "winner take all" — 최댓값 위치만 gradient 통과
   - 나머지 위치의 gradient는 0
   - 장점: 기울기 선택도 → 단점: 모든 정보 활용 못함

4. **Stride vs Pooling의 선택**:
   - 초기 층: pooling 선호 (정보 유지 중요)
   - 깊은 층: stride 사용 가능 (이미 특징 추출됨)

5. **Alternative: Strided Convolution**:
   - 일부 아키텍처(ResNet)은 pooling 대신 stride 2 conv 사용
   - 모든 위치를 고려하면서 공간 축소

---

## 📌 핵심 정리

| 개념 | 수식 | 효과 | 비용 |
|------|------|------|------|
| **Max Pooling** | $\max(k \times k)$ | Invariance + RF 확장 | 정보 손실 |
| **Avg Pooling** | $\text{mean}(k \times k)$ | 부드러운 요약 | 덜 선택적 |
| **Global Avg Pool** | $\text{mean(all)}$ | 공간 정보 제거 | 매우 압축적 |
| **Stride 2 Conv** | 한 층에서 공간 2배 축소 | 동시에 학습 | 정보 손실, 파라미터 증가 |

---

## 🤔 문제

**1번**: $8 \times 8$ 이미지에 2×2 max pooling (stride=2)을 적용한 후 다시 적용하면 출력 크기는?

**2번**: $2 \times 2$ 이미지에 다음 max pooling을 수행하세요:
```
[3  1]
[2  4]
```

**3번**: Max pooling의 역전파(backward pass)를 설명하세요. Gradient는 어디로 흐르는가?

**4번**: Stride 2 convolution과 Stride 1 convolution + Max Pooling의 receptive field 차이를 비교하세요. (같은 깊이에서)

**5번**: Global average pooling을 왜 fully-connected 층의 대안으로 사용하는가? (파라미터, 과적합 관점)

---

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 파라미터 공유와 VC 이론](./02-parameter-sharing-vc.md) | [📚 README로 돌아가기](../README.md) | [04. CNN 아키텍처 이론 ▶](./04-cnn-architectures.md) |

</div>

---

*마지막 업데이트: 2025-04-24 | 난이도: ★★★☆☆ | 소요 시간: 85분*
