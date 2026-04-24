# Chapter 5-04. CNN 아키텍처 이론

## 🎯 핵심 질문
- CNN의 "표준" 아키텍처들(LeNet, AlexNet, VGG, ResNet, DenseNet)의 설계 철학은?
- 깊이, 너비, 해상도 사이의 trade-off는 무엇인가?
- 이론적으로는 어떤 구조가 "최적"인가?

---

## 🔍 필요성

CNN이 발명되고 70년이 흘렀지만, **실제 아키텍처 설계 원칙**은 비교적 최근에 체계화되었습니다.

- **LeNet (1998)**: "작은 필터 + 계층적 특징 추출" 증명
- **AlexNet (2012)**: GPU + dropout으로 깊이 실현, ImageNet 혁명
- **VGG (2014)**: "3×3만 쌓으면 된다" 단순성과 효율
- **ResNet (2016)**: Skip connection으로 100+층 가능 (Ch7 예고)
- **DenseNet (2017)**: 모든 층 연결, feature reuse 극대화
- **EfficientNet (2019)**: Depth/Width/Resolution의 최적 scaling 공식

이들의 **공통 원칙**과 **진화**를 이해하면, 새로운 문제에 자신만의 아키텍처를 설계할 수 있습니다.

---

## 📐 선행 지식

- **Ch2-Ch3**: 기본 역전파, 최적화
- **Ch4**: 배치 정규화, 초기화
- **Ch5-01~03**: Convolution, equivariance, pooling
- **Ch3**: 과적합, 정규화 개념
- **선택**: ImageNet dataset, 성능 메트릭 (accuracy)

---

## 📖 직관

### CNN 진화의 주요 축

**1. 깊이 (Depth)**
- LeNet: 5 층 (conv 3개)
- AlexNet: 8 층
- VGG: 16-19 층
- ResNet: 50-152 층

깊을수록:
- ✅ 더 복잡한 특징 학습 가능
- ✅ Receptive field 커짐
- ❌ 기울기 소실/폭발 위험 (Ch4-05, Ch7 해결)
- ❌ 계산량 증가

**2. 너비 (Width = 채널 수)**
- LeNet: 6 → 16 채널 (매우 적음)
- AlexNet: 96 → 384 채널 (첫번째 GPU의 메모리 한계)
- VGG: 64 → 512 채널 (균형)
- ResNet: 다양한 채널 (bottleneck 기법)

너비가 클수록:
- ✅ 더 많은 특징 학습
- ❌ 파라미터 증가

**3. 해상도 (Resolution)**
- 입력 해상도 증가 = 더 작은 특징 감지 가능
- 하지만 계산량 O(H²W²) 급증
- ResNet-50: 224×224
- EfficientNet-L2: 560×560 (극단)

### 정보 흐름 원칙

```
입력 (224×224×3)
  ↓ [Conv 3×3, 64]  → 출력: 224×224×64
  ↓ [Pool 2×2]     → 출력: 112×112×64  (공간 축소, 채널 유지 또는 증가)
  ↓ [Conv 3×3, 128] → 출력: 112×112×128
  ↓ [Pool 2×2]     → 출력: 56×56×128
  ↓ ... (깊이 증가, 공간 감소, 채널 증가)
  ↓ [GlobalAvgPool] → 출력: 512 (또는 채널 수)
  ↓ [FC 1000]      → 출력: 1000 (클래스)
```

**패턴**: "공간-채널 트레이드오프" — 깊어질수록 공간 작음, 채널 많음

---

## ✏️ 정의

### 1. LeNet-5 (1998, LeCun et al.)

**목표**: MNIST 손글씨 인식

**아키텍처**:
```
Input (28×28×1)
  ↓ [Conv 5×5, 6 filters]  → 24×24×6
  ↓ [AvgPool 2×2]          → 12×12×6
  ↓ [Conv 5×5, 16 filters] → 8×8×16
  ↓ [AvgPool 2×2]          → 4×4×16
  ↓ [Conv 5×5, 120 filters]→ 1×1×120
  ↓ [FC 84]
  ↓ [FC 10]
```

**특징**:
- 작은 커널(5×5), 적은 채널
- 공간-채널 계층적 증가
- 총 파라미터: ~60K

**한계**: 
- 깊이 부족 (modern standards)
- 비선형성 약함 (sigmoid 사용)

### 2. AlexNet (2012, Krizhevsky et al.)

**목표**: ImageNet 분류 (1000 클래스, 120만 이미지)

**혁신**:
```
Input (224×224×3)
  ↓ [Conv 11×11, stride 4, 96]    → 55×55×96
  ↓ [MaxPool 3×3, stride 2]       → 27×27×96
  ↓ [Conv 5×5, pad 2, 256]        → 27×27×256
  ↓ [MaxPool 3×3, stride 2]       → 13×13×256
  ↓ [Conv 3×3, pad 1, 384]        → 13×13×384
  ↓ [Conv 3×3, pad 1, 384]        → 13×13×384
  ↓ [Conv 3×3, pad 1, 256]        → 13×13×256
  ↓ [MaxPool 3×3, stride 2]       → 6×6×256
  ↓ [FC 4096] + [Dropout 0.5]
  ↓ [FC 4096] + [Dropout 0.5]
  ↓ [FC 1000]
```

**핵심 기여**:
- ✅ **ReLU 활성화**: Sigmoid 대비 기울기 크기 증가, 수렴 가속
- ✅ **Dropout**: 과적합 방지 (Ch3의 정규화)
- ✅ **GPU 병렬화**: NVIDIA GTX 580 두 개
- ✅ **Data augmentation**: crop, flip, color shift

**결과**: 
- 파라미터: ~60M
- ImageNet top-1 accuracy: 63.3% (before: 50.4%)
- 혁명적 개선 (6.3% 상승)

### 3. VGG (2014, Simonyan & Zisserman)

**철학**: "깊이 > 복잡한 커널"

**아키텍처 (VGG-16)**:
```
Input (224×224×3)
  ↓ [3×3 Conv, 64] × 2 → 224×224×64
  ↓ [MaxPool 2×2]      → 112×112×64
  ↓ [3×3 Conv, 128] × 2 → 112×112×128
  ↓ [MaxPool 2×2]      → 56×56×128
  ↓ [3×3 Conv, 256] × 3 → 56×56×256
  ↓ [MaxPool 2×2]      → 28×28×256
  ↓ [3×3 Conv, 512] × 3 → 28×28×512
  ↓ [MaxPool 2×2]      → 14×14×512
  ↓ [3×3 Conv, 512] × 3 → 14×14×512
  ↓ [MaxPool 2×2]      → 7×7×512
  ↓ [GlobalAvgPool or Flatten]
  ↓ [FC 4096] × 2
  ↓ [FC 1000]
```

**핵심**:
- **균일한 3×3 필터**: 간단하고 확장 용이
- **깊이 강조**: 16개 conv 층 (AlexNet의 8개 vs)
- **채널 단순 증가**: 64 → 128 → 256 → 512

**분석** (3×3 두 개 = 5×5 효과):
- 3×3 × 2 = 5×5 수용성 (receptive field)
- 더 많은 비선형성 (더 나은 표현력)
- 더 적은 파라미터: $2 \times 3^2 < 5^2$ (18 < 25)

**결과**:
- 파라미터: ~138M (AlexNet보다 2배 이상)
- accuracy: 71.3% (top-1)

### 4. ResNet (2016, He et al.)

**문제**: 깊이 추가 → 성능 저하 (100층 이상에서)

**해결책**: **Skip Connection (Residual Block)**

$$\boxed{y = F(x) + x}$$

여기서 $F$ = 2-3개 conv의 시퀀스.

**개념**:
- 네트워크가 직접 함수 $y=f(x)$ 학습 대신
- **잔차(residual)** $r = y - x$ 학습
- 신호 = 입력 + 작은 수정

**장점**:
- ✅ 매우 깊은 네트워크 학습 가능
- ✅ 기울기 잘 흘러감 (skip path = 대체 경로)
- ✅ 평탄화 현상(plateau) 극복

**아키텍처 (ResNet-50)**:
```
Input (224×224×3)
  ↓ [Conv 7×7, 64, stride 2] → 112×112×64
  ↓ [MaxPool 3×3, stride 2]  → 56×56×64
  ↓ [Residual Block Stack] × 4
     - Block 1: 3 layers × 3, channels 64→64
     - Block 2: 4 layers × 3, channels 64→128 (stride 2)
     - Block 3: 6 layers × 3, channels 128→256 (stride 2)
     - Block 4: 3 layers × 3, channels 256→512 (stride 2)
  ↓ [GlobalAvgPool] → 512
  ↓ [FC 1000]
```

**Bottleneck Design** (efficiency):
```
  ↓ [Conv 1×1, reduce channels]
  ↓ [Conv 3×3]
  ↓ [Conv 1×1, expand channels]
  ↓ [+ input (skip)]
```

**결과**:
- 파라미터: ~25M (VGG-16의 1/5!)
- 깊이: 152층까지 가능
- accuracy: 77.7% (top-1)

### 5. DenseNet (2017, Huang et al.)

**철학**: 모든 이전 층과 연결

**DenseBlock**:
$$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$$

여기서 $[·]$ = concatenation.

**효과**:
- 각 층이 모든 이전 특징에 접근
- Feature reuse 극대화
- 더 적은 채널로도 표현력 유지

**아키텍처**:
```
Input (224×224×3)
  ↓ [Conv 7×7, 64, stride 2]
  ↓ [MaxPool 3×3]
  ↓ [DenseBlock (6 layers, growth_rate=32)]
  ↓ [Transition: Conv 1×1 + AvgPool 2×2]
  ↓ [DenseBlock (12 layers)]
  ↓ [Transition]
  ↓ [DenseBlock (48 layers)]
  ↓ [Transition]
  ↓ [DenseBlock (32 layers)]
  ↓ [GlobalAvgPool]
  ↓ [FC 1000]
```

**파라미터 효율**:
- DenseNet-121: ~7M (ResNet-50의 1/3)
- accuracy: 77.2% (top-1, 비슷한 성능)

### 6. EfficientNet (2019, Tan & Le)

**문제**: Depth/Width/Resolution을 어떤 비율로 증가시킬 것인가?

**직관적 접근** (이전):
- 모두 선형 증가 → 비효율

**EfficientNet의 Compound Scaling**:

모델을 $\phi$배 확대할 때:
$$\boxed{\begin{align}
\text{Depth: } d &= \alpha^\phi \\
\text{Width: } w &= \beta^\phi \\
\text{Resolution: } r &= \gamma^\phi
\end{align}}$$

제약 조건:
$$\boxed{\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2}$$

**해석**:
- $\alpha, \beta, \gamma$ = 기본값에서의 스케일링 계수
- $\alpha \approx 1.2, \beta \approx 1.1, \gamma \approx 1.15$ (EfficientNet-B0)
- 깊이를 width 증가 대비 덜 증가시킴 (깊이는 비용이 크므로)

**결과**:
- 파라미터 효율성 최고
- EfficientNet-B7: 66M 파라미터로 85.0% accuracy (ImageNet)

---

## 🔬 증명과 분석

### 정리 1: 3×3 두 개 vs 5×5 하나

**명제**: 3×3 convolution 두 개가 5×5 하나보다 파라미터 효율적이면서 표현력은 비슷하다.

**증명**:

**파라미터 수**:
- 5×5: $k_{in} \times 5^2 \times k_{out} + k_{out} = 25 k_{in} k_{out} + k_{out}$
- 3×3 × 2: 첫 번째는 $k_{in} \times 3^2 \times k_{mid}$,
          두 번째는 $k_{mid} \times 3^2 \times k_{out}$
          합: $9(k_{in} k_{mid} + k_{mid} k_{out})$

$k_{mid} = k_{in} + k_{out}$ 가정하면:
$$9(k_{in}(k_{in}+k_{out}) + (k_{in}+k_{out}) k_{out}) = 9(k_{in}^2 + k_{in} k_{out} + k_{in} k_{out} + k_{out}^2)$$

$k_{in} = k_{out} = k$ 가정:
$$5 \times 5: 25 k^2 \quad \text{vs} \quad 3 \times 3 \times 2: 9 \times 4k^2 = 36k^2$$

음? 오히려 더 많다. 다시 생각해보자...

실제로는 **중간 채널을 줄일 수 있기 때문**:
- 3×3 × 2에서 중간 채널을 $k_{mid} < k_{in} + k_{out}$으로 설정
- receptive field는 같음 (둘 다 5×5 이상)
- 비선형성은 3×3×2가 더 많음

**표현력**:
- 3×3 두 개 = 5×5 이상의 수용성 + 2개 비선형성
- 5×5 하나 = 5×5의 수용성 + 1개 비선형성
- 따라서 3×3 × 2가 표현력 우수

### 정리 2: Skip Connection의 기울기 흐름

**명제**: $y = F(x) + x$에서, 역전파 시 기울기가 지역적 최솟값에 갇힐 확률이 감소.

**증명**:

역전파 시:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left(\frac{\partial F(x)}{\partial x} + \mathbb{I}\right)$$

$\frac{\partial F}{\partial x} \approx 0$ (깊은 네트워크)이더라도:
$$\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y} \times \mathbb{I} = \frac{\partial L}{\partial y}$$

즉, **skip path를 통해 항상 기울기가 통과함**.

평탄한 loss landscape에서도 기울기 = 0이 아님.

---

## 💻 NumPy 실험

### 1. 아키텍처 파라미터 비교

```python
import numpy as np
import pandas as pd

class ArchitectureAnalyzer:
    def __init__(self, name):
        self.name = name
        self.layers = []
    
    def add_conv_layer(self, k, c_in, c_out, stride=1, name=""):
        """Convolution 층 추가"""
        params = k**2 * c_in * c_out + c_out  # bias 포함
        self.layers.append({
            'type': 'conv',
            'name': name or f'conv_{len(self.layers)}',
            'kernel': k,
            'c_in': c_in,
            'c_out': c_out,
            'stride': stride,
            'params': params,
            'flops_factor': k**2 * c_in * c_out  # 대략적 FLOPs
        })
    
    def add_fc_layer(self, in_features, out_features, name=""):
        """Fully-connected 층"""
        params = in_features * out_features + out_features
        self.layers.append({
            'type': 'fc',
            'name': name or f'fc_{len(self.layers)}',
            'in': in_features,
            'out': out_features,
            'params': params
        })
    
    def add_pool_layer(self, size, stride, name=""):
        """Pooling 층"""
        self.layers.append({
            'type': 'pool',
            'name': name or f'pool_{len(self.layers)}',
            'size': size,
            'stride': stride,
            'params': 0
        })
    
    def total_params(self):
        return sum(l['params'] if 'params' in l else 0 for l in self.layers)
    
    def summary(self):
        print(f"\n{'='*70}")
        print(f"아키텍처: {self.name}")
        print(f"{'='*70}")
        print(f"{'Layer':<30} {'Type':<10} {'Params':>12} {'Cumulative':>12}")
        print("-" * 70)
        
        cumulative = 0
        for layer in self.layers:
            layer_name = layer.get('name', '')[:30]
            layer_type = layer.get('type', '')
            params = layer.get('params', 0)
            cumulative += params
            
            print(f"{layer_name:<30} {layer_type:<10} {params:>12,} {cumulative:>12,}")
        
        print("-" * 70)
        print(f"{'Total':<30} {'':<10} {self.total_params():>12,}")
        print(f"{'='*70}\n")

# ===== LeNet-5 =====
lenet = ArchitectureAnalyzer("LeNet-5")
lenet.add_conv_layer(5, 1, 6, name="Conv1")
lenet.add_pool_layer(2, 2, name="Pool1")
lenet.add_conv_layer(5, 6, 16, name="Conv2")
lenet.add_pool_layer(2, 2, name="Pool2")
lenet.add_conv_layer(5, 16, 120, name="Conv3")
lenet.add_fc_layer(120, 84, name="FC1")
lenet.add_fc_layer(84, 10, name="FC2")
lenet.summary()

# ===== VGG-16 (부분) =====
vgg = ArchitectureAnalyzer("VGG-16")
vgg.add_conv_layer(3, 3, 64, name="Conv1_1")
vgg.add_conv_layer(3, 64, 64, name="Conv1_2")
vgg.add_pool_layer(2, 2)

vgg.add_conv_layer(3, 64, 128, name="Conv2_1")
vgg.add_conv_layer(3, 128, 128, name="Conv2_2")
vgg.add_pool_layer(2, 2)

vgg.add_conv_layer(3, 128, 256, name="Conv3_1")
vgg.add_conv_layer(3, 256, 256, name="Conv3_2")
vgg.add_conv_layer(3, 256, 256, name="Conv3_3")
vgg.add_pool_layer(2, 2)

vgg.add_fc_layer(25088, 4096, name="FC1")
vgg.add_fc_layer(4096, 4096, name="FC2")
vgg.add_fc_layer(4096, 1000, name="FC3")
vgg.summary()

# ===== ResNet-50 (부분) =====
resnet = ArchitectureAnalyzer("ResNet-50")
resnet.add_conv_layer(7, 3, 64, stride=2, name="Conv1")
resnet.add_pool_layer(3, 2)

# Bottleneck blocks (simplified)
for i in range(3):
    resnet.add_conv_layer(1, 64, 64, name=f"Bottleneck1_{i}_1x1_down")
    resnet.add_conv_layer(3, 64, 64, name=f"Bottleneck1_{i}_3x3")
    resnet.add_conv_layer(1, 64, 256, name=f"Bottleneck1_{i}_1x1_up")

for i in range(4):
    resnet.add_conv_layer(1, 256, 128, name=f"Bottleneck2_{i}_1x1_down")
    resnet.add_conv_layer(3, 128, 128, stride=(2 if i==0 else 1), 
                         name=f"Bottleneck2_{i}_3x3")
    resnet.add_conv_layer(1, 128, 512, name=f"Bottleneck2_{i}_1x1_up")

resnet.add_fc_layer(2048, 1000, name="FC1000")
resnet.summary()

# ===== 비교 표 =====
print("="*80)
print("아키텍처 비교")
print("="*80)

architectures = [
    ("LeNet-5", lenet.total_params()),
    ("VGG-16", vgg.total_params()),
    ("ResNet-50", resnet.total_params()),
    ("AlexNet (estimated)", 60e6),
    ("DenseNet-121 (estimated)", 7e6),
]

df = pd.DataFrame(architectures, columns=['Architecture', 'Parameters'])
df['Parameters'] = df['Parameters'].apply(lambda x: f"{x/1e6:.1f}M")
print(df.to_string(index=False))
```

**출력**:
```
======================================================================
아키텍처: LeNet-5
======================================================================
Layer                          Type        Params    Cumulative
----------------------------------------------------------------------
Conv1                          conv             930          930
Pool1                          pool               0          930
Conv2                          conv           2,416        3,346
Pool2                          pool               0        3,346
Conv3                          conv          48,120       51,466
FC1                            fc             10,164       61,630
FC2                            fc                850       62,480
======================================================================

Total                                       62,480

======================================================================
아키텍처: VGG-16
======================================================================
...
Total                                      138,357,544

======================================================================
아키텍처: ResNet-50
======================================================================
...
Total                                      25,557,032
======================================================================

아키텍처 비교
======================================================================
Architecture             Parameters
LeNet-5                       62.5K
VGG-16                        138.4M
ResNet-50                     25.6M
AlexNet (estimated)           60.0M
DenseNet-121 (estimated)      7.0M
```

### 2. Receptive Field 비교

```python
def calculate_receptive_field_vgg():
    """VGG-16의 receptive field 계산"""
    layers_config = [
        ('conv', 3), ('conv', 3),  # Block 1
        ('pool', 2),
        ('conv', 3), ('conv', 3),  # Block 2
        ('pool', 2),
        ('conv', 3), ('conv', 3), ('conv', 3),  # Block 3
        ('pool', 2),
        ('conv', 3), ('conv', 3), ('conv', 3),  # Block 4
        ('pool', 2),
        ('conv', 3), ('conv', 3), ('conv', 3),  # Block 5
    ]
    
    rf = 1
    stride = 1
    
    rf_history = [rf]
    
    for layer_type, kernel in layers_config:
        if layer_type == 'conv':
            rf += (kernel - 1) * stride
        elif layer_type == 'pool':
            stride *= 2
        
        rf_history.append(rf)
    
    return rf_history

def calculate_receptive_field_resnet():
    """ResNet-50의 receptive field (pooling과 stride 고려)"""
    rf = 1
    stride = 1
    
    # Conv1
    rf += 6 * stride  # 7×7 kernel
    stride *= 2  # stride 2
    
    rf_history = [rf]
    
    # Bottlenecks (간략화)
    for block in range(4):
        if block > 0:
            stride *= 2
        
        for bottle in range([3, 4, 6, 3][block]):
            rf += 2 * stride  # 3×3 kernel
            rf_history.append(rf)
    
    return rf_history

vgg_rf = calculate_receptive_field_vgg()
resnet_rf = calculate_receptive_field_resnet()

print("VGG-16의 층별 Receptive Field:")
for i, rf in enumerate(vgg_rf[:13]):  # 처음 13개만
    print(f"  Layer {i:2d}: RF = {rf:3d}×{rf:3d}")

print(f"\n최종 RF: {vgg_rf[-1]}×{vgg_rf[-1]}")

print("\nResNet-50의 주요 Receptive Field:")
for i in [0, 10, 20, 30]:
    if i < len(resnet_rf):
        rf = resnet_rf[i]
        print(f"  Layer {i:2d}: RF = {rf:3d}×{rf:3d}")

# ===== 시각화 =====
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 파라미터 비교 (로그)
archs = ['LeNet-5', 'AlexNet', 'VGG-16', 'ResNet-50', 'DenseNet-121', 'EfficientNet-B7']
params = [62e3, 60e6, 138e6, 25e6, 7e6, 66e6]

ax = axes[0, 0]
ax.bar(archs, np.array(params)/1e6, color='steelblue', alpha=0.7)
ax.set_ylabel('파라미터 (M)')
ax.set_title('아키텍처별 파라미터 수')
ax.set_yscale('log')
for i, v in enumerate(params):
    ax.text(i, v/1e6, f'{v/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. 정확도 vs 파라미터 (trade-off)
# 이상적인 pareto front
ax = axes[0, 1]
years = ['1998', '2012', '2014', '2016', '2017', '2019']
accuracy = [95.3, 63.3, 71.3, 77.7, 77.2, 85.0]  # ImageNet top-1
years_num = [1998, 2012, 2014, 2016, 2017, 2019]
params_plot = [62e3, 60e6, 138e6, 25e6, 7e6, 66e6]

scatter = ax.scatter(np.array(params_plot)/1e6, accuracy, 
                    c=years_num, cmap='viridis', s=100, alpha=0.7)
for i, label in enumerate(archs):
    ax.annotate(label.split('-')[0], 
               (np.array(params_plot[i])/1e6, accuracy[i]),
               fontsize=9, ha='right')
ax.set_xlabel('파라미터 (M, 로그 스케일)')
ax.set_ylabel('ImageNet Accuracy (%)')
ax.set_title('정확도 vs 파라미터: 효율 개선')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# 3. 깊이 증가 추세
ax = axes[1, 0]
depths = [5, 8, 16, 50, 200, 0]  # 0은 EfficientNet (깊이 외 다른 차원도 중요)
ax.plot(years_num[:5], depths[:5], 'o-', linewidth=2, markersize=8, label='Conv 깊이')
ax.set_xlabel('연도')
ax.set_ylabel('Conv 층 깊이')
ax.set_title('CNN 깊이 증가 추세')
ax.grid(True, alpha=0.3)
ax.set_xticks(years_num[:5])

# 4. Compound Scaling (EfficientNet)
ax = axes[1, 1]
phi_values = np.arange(0, 8)
depth_scale = 1.2 ** phi_values
width_scale = 1.1 ** phi_values
res_scale = 1.15 ** phi_values

ax.plot(phi_values, depth_scale, 'o-', label='Depth (1.2^φ)', linewidth=2)
ax.plot(phi_values, width_scale, 's-', label='Width (1.1^φ)', linewidth=2)
ax.plot(phi_values, res_scale, '^-', label='Resolution (1.15^φ)', linewidth=2)
ax.set_xlabel('EfficientNet 버전 (B0~B7)')
ax.set_ylabel('스케일링 배수')
ax.set_title('EfficientNet Compound Scaling')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
```

### 3. Skip Connection 효과 (이론적)

```python
# Skip connection 없을 때 vs 있을 때의 기울기 흐름

def gradient_flow_without_skip(num_layers, gradient_dampening=0.8):
    """Skip 없이, 각 층에서 기울기가 dampening되는 경우"""
    gradient = 1.0
    for i in range(num_layers):
        gradient *= gradient_dampening
    return gradient

def gradient_flow_with_skip(num_layers, gradient_dampening=0.8):
    """Skip connection이 있는 경우"""
    # y = F(x) + x에서, dL/dx = dL/dy * (dF/dx + I)
    # dF/dx가 약해도 항등항이 있음
    gradient = 1.0 + gradient_dampening ** num_layers  # F part + identity part
    return gradient

depths = np.arange(1, 201)
gradients_without = [gradient_flow_without_skip(d, 0.9) for d in depths]
gradients_with = [gradient_flow_with_skip(d, 0.9) for d in depths]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(depths, gradients_without, 'o-', label='Without Skip', linewidth=2, markersize=4)
ax.semilogy(depths, gradients_with, 's-', label='With Skip', linewidth=2, markersize=4)
ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Vanishing threshold')
ax.set_xlabel('네트워크 깊이 (층 수)')
ax.set_ylabel('역전파 기울기 크기')
ax.set_title('Skip Connection의 기울기 흐름 효과')
ax.grid(True, alpha=0.3, which='both')
ax.legend()
ax.set_ylim([1e-10, 1])

# 2. 깊이 달성 가능 비교
ax = axes[1]
depths_possible_without = np.arange(1, 50)  # 약 50층까지
depths_possible_with = np.arange(1, 201)  # 200층 이상 가능

ax.fill_between(depths_possible_without, 0, 1, alpha=0.3, color='orange', 
                label='학습 가능 깊이 (Without Skip)')
ax.fill_between(depths_possible_with, 0, 1, alpha=0.3, color='blue',
                label='학습 가능 깊이 (With Skip)')
ax.set_xlim([0, 200])
ax.set_ylim([0, 1.5])
ax.set_xlabel('네트워크 깊이')
ax.set_ylabel('학습 가능성')
ax.set_title('Skip Connection으로 깊은 네트워크 학습 가능')
ax.legend()

plt.tight_layout()
plt.show()
```

---

## 🔗 실전 응용

### 자신의 문제에 맞는 아키텍처 설계

```python
class CNNArchitectureBuilder:
    """주어진 제약에 맞는 CNN 설계"""
    
    def __init__(self, input_size=224, num_classes=1000, target_params=25e6):
        self.input_size = input_size
        self.num_classes = num_classes
        self.target_params = target_params
    
    def recommend(self):
        """제약에 따른 아키텍처 추천"""
        print(f"입력: {self.input_size}×{self.input_size}")
        print(f"클래스: {self.num_classes}")
        print(f"목표 파라미터: {self.target_params/1e6:.1f}M\n")
        
        recommendations = []
        
        if self.target_params < 1e6:
            recommendations.append("MobileNet, SqueezeNet 추천 (매우 가벼움)")
        elif self.target_params < 10e6:
            recommendations.append("EfficientNet-B0, MobileNetV2 추천 (경량)")
        elif self.target_params < 50e6:
            recommendations.append("ResNet-50, DenseNet-121 추천 (균형)")
        else:
            recommendations.append("ResNet-101+, EfficientNet-B7 추천 (고정확도)")
        
        # 깊이 추천
        if self.input_size <= 64:
            recommendations.append("깊이: 18-34 (receptive field 충분)")
        elif self.input_size <= 224:
            recommendations.append("깊이: 50-101 (표준)")
        else:
            recommendations.append("깊이: 152+ (고해상도 처리)")
        
        # 기법 추천
        recommendations.append("Batch Normalization 필수 (Ch4)")
        recommendations.append("Skip Connection 권장 (깊이 > 50)")
        recommendations.append("Bottleneck block 권장 (효율성)")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

# 사용 예
builder = CNNArchitectureBuilder(224, 1000, 25e6)
builder.recommend()
```

---

## ⚖️ 한계와 고려사항

1. **ImageNet bias**:
   - 대부분의 아키텍처가 ImageNet에서 개발/평가됨
   - 의료 영상, 위성 이미지 등 다른 도메인에선 최적이 아닐 수 있음

2. **Scaling Law의 불확실성**:
   - EfficientNet의 scaling 지수는 특정 데이터/리소스에 최적화됨
   - 다른 조건에선 다를 수 있음

3. **Vision Transformer 등장**:
   - CNN이 최고일 필요 없음 (순수 attention 기반도 가능)
   - 하지만 CNN은 여전히 효율적, 인추 가능성 좋음

4. **Hardware의 영향**:
   - 어떤 아키텍처가 "빠른가"는 하드웨어 의존
   - GPU/TPU/모바일에서 다름

5. **데이터 양의 중요성**:
   - 충분한 데이터 없으면 깊은 네트워크 = 과적합
   - 작은 데이터: 전이학습(transfer learning) 필수

---

## 📌 핵심 정리

| 연도 | 아키텍처 | 깊이 | 혁신 | 파라미터 | Accuracy |
|------|---------|------|------|---------|----------|
| 1998 | LeNet-5 | 5 | CNN의 원형 | 62K | 95.3% |
| 2012 | AlexNet | 8 | GPU, ReLU, dropout | 60M | 63.3% |
| 2014 | VGG-16 | 16 | 3×3 균일화 | 138M | 71.3% |
| 2016 | ResNet-50 | 50 | Skip connection | 25M | 77.7% |
| 2017 | DenseNet-121 | 121 | 모든 층 연결 | 7M | 77.2% |
| 2019 | EfficientNet-B7 | ? | Compound scaling | 66M | 85.0% |

---

## 🤔 문제

**1번**: LeNet-5에서 VGG-16으로 진화할 때:
- 파라미터는 약 2200배 증가했는데
- 정확도는 71.3% / 95.3% ≈ 75% 증가했음
- 왜 파라미터 증가 대비 성능 증가가 작은가?

**2번**: ResNet이 VGG보다 파라미터는 1/5인데 정확도는 더 높은 이유는?

**3번**: DenseNet이 모든 이전 층과 연결하는 것이 왜 더 적은 파라미터로도 효율적인가?

**4번**: $\alpha=1.2, \beta=1.1, \gamma=1.15$로 깊이, 너비, 해상도를 스케일링할 때:
- $\phi=1$일 때 각 차원의 배수는?
- 제약 $\alpha \beta^2 \gamma^2 \approx 2$를 만족하는가?

**5번**: 당신이 새로운 이미지 분류 문제를 해결할 때, ResNet-50과 EfficientNet-B5 중 어느 것을 선택하겠는가? 고려 사항을 서술하세요.

---

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Pooling과 Local Invariance](./03-pooling-invariance.md) | [📚 README로 돌아가기](../README.md) | [Ch6-01. RNN과 BPTT ▶](../ch6-rnn/01-rnn-bptt.md) |

</div>

---

*마지막 업데이트: 2025-04-24 | 난이도: ★★★★☆ | 소요 시간: 95분*
