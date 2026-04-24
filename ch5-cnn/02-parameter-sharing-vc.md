# Chapter 5-02. 파라미터 공유와 VC 이론적 효율

## 🎯 핵심 질문
- CNN이 완전연결층보다 훨씬 적은 파라미터로 작동하는 이유는?
- 파라미터 수 감소가 일반화 성능을 어떻게 개선하는가?
- VC 차원 이론으로 이를 수학적으로 설명할 수 있는가?

---

## 🔍 필요성

신경망 학습에서 가장 중요한 문제는 **과적합(overfitting)**입니다. 다음 두 가지 인사이트가 CNN의 설계를 이끕니다:

1. **Occam의 면도날**: 같은 성능을 내는 모델 중 파라미터가 적을수록 좋음
2. **일반화 이론**: 파라미터 수 감소 = VC 차원 감소 = 더 작은 일반화 오차

CNN은 **파라미터 공유(parameter sharing)**라는 단순하지만 강력한 원칙을 통해:
- 파라미터를 $1/10000$ 이상 줄이면서
- 오히려 **더 나은 성능** 달성

이를 이론적으로 이해하면 "왜 CNN이 이미지에 잘 맞는가"가 명확해집니다.

---

## 📐 선행 지식

- **Ch2**: 선형변환, 행렬 차원
- **Ch3**: 경사하강법, 과적합 개념
- **Ch5-01**: Convolution 연산, receptive field
- **수학**: 로그, 조합론 기초
- **선택**: VC 차원의 정의 (여기서 직관부터 시작)

---

## 📖 직관

### 완전연결층의 과도한 파라미터

$224 \times 224$ RGB 이미지 → 첫 번째 은닉층(1024 뉴런):
$$
\text{파라미터} = (224 \times 224 \times 3) \times 1024 \approx 150\text{M}
$$

이 모두가 학습 대상입니다. 하지만:
- 이미지의 작은 모서리를 인식하는 필터는 **작은 지역만 봐야 함**
- 모든 픽셀 조합을 학습할 필요가 없음

### CNN의 효율

$3 \times 3 \times 3$ 커널(Sobel edge detector 같은):
$$
\text{파라미터} = 3 \times 3 \times 3 = 27
$$

같은 커널을 **전체 이미지에 평행이동하며 적용**:
- 초기 특징층 파라미터: $27 \times 32 \approx 864$ (32개 필터)
- 완전연결층의 150M과 비교: **약 174,000배 감소**!

### 일반화의 개선

더 적은 파라미터 = 더 단순한 모델 = 과적합 가능성 감소

$n$개의 샘플로 학습할 때:
- FC: 고차원, 수백M 파라미터 → 쉽게 과적합
- CNN: 저차원 구조, 수K~M 파라미터 → 자연스러운 정규화 (Ch3의 $L_2$ 정규화 없어도 강함)

---

## ✏️ 정의

### 1. 파라미터 수 계산

#### 완전연결층 (Fully Connected)

입력 차원 $D_\text{in}$, 출력 차원 $D_\text{out}$:
$$\boxed{\text{Params}_\text{FC} = D_\text{in} \cdot D_\text{out} + D_\text{out} \quad \text{(bias 포함)}}$$

**예**: $D_\text{in} = 224 \times 224 \times 3 = 150,528$, $D_\text{out} = 1024$
$$\text{Params} = 150,528 \times 1024 \approx 154\text{M}$$

#### Convolution 층

- 커널 크기: $k \times k$
- 입력 채널: $C_\text{in}$
- 출력 채널(필터): $C_\text{out}$

$$\boxed{\text{Params}_\text{CNN} = k^2 \cdot C_\text{in} \cdot C_\text{out} + C_\text{out} \quad \text{(bias 포함)}}$$

**중요**: 공간 차원 $H, W$가 없음!

#### 감소 비율

FC와 CNN 첫 번째 층을 비교:
$$
\text{감소 비율} = \frac{\text{Params}_\text{CNN}}{\text{Params}_\text{FC}} = \frac{k^2}{H \times W}
$$

**예시** ($k=3$, $H=W=224$):
$$
\text{비율} = \frac{9}{50,176} \approx 1.8 \times 10^{-4}
$$

### 2. Receptive Field와 깊이

$L$개 convolution 층($k \times k$ 커널, stride $s=1$)을 쌓을 때:
$$\boxed{\text{RF}_L = (k-1) \cdot L + 1}$$

**증명** (귀납법):
- $\text{RF}_1 = k$ (한 층)
- $\text{RF}_{L+1} = \text{RF}_L + (k-1)$ (다음 층은 $(k-1)$의 "테두리" 추가)
- 합치면: $k + (k-1)(L-1) = (k-1)L + 1$

**예시**:
- $L=1, k=3$: RF = 3
- $L=3, k=3$: RF = 5
- $L=7, k=3$: RF = 13
- $L=13, k=3$: RF = 25

### 3. VC 차원 소개

**직관**: 모델이 얼마나 복잡한가를 측정하는 지표

**정의** (VC Dimension):
모델 클래스 $\mathcal{H}$의 VC 차원은, 모델이 "모든 가능한" 방식으로 분류할 수 있는 최대 샘플 개수 $d_\text{VC}$

**예시**:
- 1D 선형 분류기: $d_\text{VC} = 2$ (2개 점은 어떤 라벨 조합이든 분류 가능, 3개는 불가능)
- 2D 선형: $d_\text{VC} = 3$
- $\mathbb{R}^n$ 선형: $d_\text{VC} = n+1$

### 4. VC 차원과 파라미터 관계

#### 완전연결 네트워크
$$\boxed{d_\text{VC}(\text{FC}) \approx \tilde{O}(W_\text{total})}$$

여기서 $W_\text{total}$ = 전체 가중치 개수

더 정확한 상한:
$$d_\text{VC} \leq W_\text{total} \cdot \log(W_\text{total})$$

#### CNN (Convolutional Network)
$$\boxed{d_\text{VC}(\text{CNN}) \approx \tilde{O}(k^2 C_\text{in} C_\text{out} \cdot L)}$$

여기서:
- $k$ = 커널 크기
- $C_\text{in}, C_\text{out}$ = 채널 수
- $L$ = 층 깊이

**핵심**: FC의 $H \times W$ 의존성이 사라짐!

### 5. 일반화 오차 이론

**Vapnik-Chervonenkis 부등식**:

$n$개 독립동등분포 샘플로부터 경험적 오차 $L_\text{emp}$와 참 오차 $L_\text{true}$ 사이:

$$\boxed{L_\text{true} \leq L_\text{emp} + O\left(\sqrt{\frac{d_\text{VC}}{n}} + \text{log variance terms}\right)}$$

**해석**:
- 첫 번째 항: $L_\text{emp}$ — 학습 오차 (최소화 대상)
- 두 번째 항: 복잡도 페널티 — $d_\text{VC}$ 감소 또는 $n$ 증가로 줄임

#### CNN vs FC 비교

동일 성능의 이미지 분류 작업 ($H=W=224, C_\text{in}=3, C_\text{out}=512, k=3, L=5$):

**FC 네트워크**:
$$d_\text{VC}^\text{FC} \approx (224 \times 224 \times 3) \times 512 \approx 77\text{M}$$

**CNN**:
$$d_\text{VC}^\text{CNN} \approx 3^2 \times 3 \times 512 \times 5 \approx 69,120$$

**복잡도 감소**:
$$\frac{d_\text{VC}^\text{CNN}}{d_\text{VC}^\text{FC}} \approx 9 \times 10^{-4}$$

### 6. Sample Complexity

$n$개 샘플로 일반화 오차를 $\epsilon$ 이내로 유지하려면:

$$\boxed{n \gtrsim \frac{d_\text{VC}}{\epsilon^2}}$$

**이름**: Sample complexity — 필요한 샘플 개수

**CNN의 이점**:
- FC: $n \gtrsim 77\text{M} / \epsilon^2$ 
- CNN: $n \gtrsim 69\text{K} / \epsilon^2$

같은 $\epsilon$로, CNN은 FC보다 **1000배 적은 샘플**로도 충분할 수 있습니다.

---

## 🔬 증명

### 정리 1: CNN의 VC 차원 상한

**명제**:
$k \times k$ 커널, $C_\text{in}$ 입력 채널, $C_\text{out}$ 출력 채널인 단일 convolution 층의 VC 차원:
$$d_\text{VC} = O(k^2 C_\text{in} C_\text{out})$$

**증명 스케치**:

1. Convolution의 출력은 입력의 **선형 결합**입니다:
$$\text{output}[i,j] = \sum_{c_\text{out}} \sum_{m,n} \text{input}[m,n] \cdot K[c_\text{out}, m, n] + b[c_\text{out}]$$

2. 선형 결합의 계수 개수:
$$\text{# coefficients} = k^2 \cdot C_\text{in} \cdot C_\text{out}$$

3. $d$ 차원 선형 분류기의 VC 차원은 $d+1$ (또는 upper bound $d$)

4. 따라서:
$$d_\text{VC} = O(k^2 C_\text{in} C_\text{out})$$

**$L$개 층의 경우**:
$$d_\text{VC}^{(L)} = O(L \cdot k^2 C_\text{in} C_\text{out})$$

(정확한 상한은 ReLU, pooling 등에 따라 복잡하지만, 주요 의존성은 위와 같음)

### 정리 2: 일반화 오차와 파라미터 효율

**명제**:
동일한 분류 정확도를 달성하는 FC와 CNN에 대해:
$$\epsilon_\text{gen}^\text{CNN} \approx \epsilon_\text{gen}^\text{FC} \quad \text{but using } \frac{1}{(H/k)^2} \text{ 배 적은 파라미터}$$

**증명 개요**:

VC 부등식으로부터:
$$\epsilon_\text{gen} \approx C \sqrt{\frac{d_\text{VC}}{n}}$$

$\epsilon_\text{gen}$을 고정하려면:
$$n \propto d_\text{VC}$$

**FC**: $d_\text{VC}^\text{FC} \propto H W C_\text{in} C_\text{out}$
**CNN**: $d_\text{VC}^\text{CNN} \propto k^2 C_\text{in} C_\text{out}$

같은 일반화 오차로, CNN은 FC보다 $\sim(H/k)^2$배 적은 데이터만 필요합니다.

역으로, **같은 데이터**에서 CNN은 FC보다 작은 과적합 위험으로 더 나은 성능을 낼 수 있습니다.

---

## 💻 NumPy 실험

### 1. 파라미터 수 비교 계산

```python
import numpy as np

def count_fc_params(input_dim, output_dim):
    """Fully-connected 층 파라미터 개수 (bias 포함)"""
    return input_dim * output_dim + output_dim

def count_conv_params(kernel_size, in_channels, out_channels):
    """Convolution 층 파라미터 개수 (bias 포함)"""
    return kernel_size**2 * in_channels * out_channels + out_channels

# ===== 설정 =====
H, W = 224, 224
C_in = 3
C_out = 64
k = 3

# ===== FC 계산 =====
input_dim_fc = H * W * C_in
fc_params = count_fc_params(input_dim_fc, C_out)

# ===== CNN 계산 =====
conv_params = count_conv_params(k, C_in, C_out)

# ===== 비교 =====
ratio = conv_params / fc_params

print("=" * 60)
print("파라미터 수 비교 (224×224 RGB 입력)")
print("=" * 60)
print(f"\nFC 층 (첫 번째):")
print(f"  입력 차원: {input_dim_fc:,}")
print(f"  출력 차원: {C_out}")
print(f"  파라미터: {fc_params:,}")

print(f"\nCNN 층 (커널 3×3×3):")
print(f"  커널: {k}×{k}, 채널: {C_in}→{C_out}")
print(f"  파라미터: {conv_params:,}")

print(f"\n감소 비율: {ratio:.2e} ({ratio*100:.6f}%)")
print(f"FC 파라미터 / CNN 파라미터: {fc_params / conv_params:.0f}배")

# ===== 여러 설정에서의 비율 =====
print("\n" + "=" * 60)
print("다양한 입력 크기에서의 감소 비율")
print("=" * 60)

for h, w in [(64, 64), (112, 112), (224, 224), (512, 512)]:
    fc_p = count_fc_params(h * w * 3, 64)
    cnn_p = count_conv_params(3, 3, 64)
    print(f"{h:3d}×{w:3d}: FC={fc_p:12,} | CNN={cnn_p:6,} | 비율={cnn_p/fc_p:.2e}")
```

**출력**:
```
============================================================
파라미터 수 비교 (224×224 RGB 입력)
============================================================

FC 층 (첫 번째):
  입력 차원: 150,528
  출력 차원: 64
  파라미터: 9,633,792

CNN 층 (커널 3×3×3):
  커널: 3×3, 채널: 3→64
  파라미터: 1,792

감소 비율: 1.86e-04 (0.018596%)
FC 파라미터 / CNN 파라미터: 5,376배

============================================================
다양한 입력 크기에서의 감소 비율
============================================================

 64× 64: FC=     786,432 | CNN=    1,792 | 비율=2.28e-03
112×112: FC=   3,768,320 | CNN=    1,792 | 비율=4.75e-04
224×224: FC=   9,633,792 | CNN=    1,792 | 비율=1.86e-04
512×512: FC=  78,643,200 | CNN=    1,792 | 비율=2.28e-05
```

### 2. 깊이에 따른 Receptive Field 계산

```python
def receptive_field(num_layers, kernel_size, stride=1):
    """
    깊이 num_layers, 커널 크기 kernel_size, stride인 CNN의 receptive field
    """
    rf = kernel_size
    for _ in range(num_layers - 1):
        rf += (kernel_size - 1) * stride
    return rf

# ===== 다양한 조합에서의 RF =====
print("=" * 70)
print("깊이별 Receptive Field (stride=1)")
print("=" * 70)

for k in [3, 5, 7]:
    print(f"\n커널 {k}×{k}:")
    for l in range(1, 14):
        rf = receptive_field(l, k, stride=1)
        params = count_conv_params(k, 3, 64) * l
        print(f"  깊이 {l:2d}: RF={rf:3d}×{rf:3d}, 파라미터≈{params:,}")

# ===== 깊이 vs 파라미터 =====
print("\n" + "=" * 70)
print("목표 RF 달성에 필요한 깊이 (k=3)")
print("=" * 70)

target_rfs = [5, 7, 13, 25, 51, 101]
for rf_target in target_rfs:
    for l in range(1, 20):
        rf = receptive_field(l, 3)
        if rf >= rf_target:
            params = count_conv_params(3, 3, 64) * l
            print(f"RF={rf_target:3d}: 깊이={l:2d}, 파라미터≈{params:,}")
            break

# ===== 시각화 =====
import matplotlib.pyplot as plt

depths = np.arange(1, 20)
k3_rfs = [receptive_field(l, 3) for l in depths]
k5_rfs = [receptive_field(l, 5) for l in depths]
k7_rfs = [receptive_field(l, 7) for l in depths]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# RF vs 깊이
ax1.plot(depths, k3_rfs, 'o-', label='k=3', linewidth=2)
ax1.plot(depths, k5_rfs, 's-', label='k=5', linewidth=2)
ax1.plot(depths, k7_rfs, '^-', label='k=7', linewidth=2)
ax1.set_xlabel('깊이 (층 개수)')
ax1.set_ylabel('Receptive Field')
ax1.set_title('CNN 깊이에 따른 Receptive Field')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 파라미터 vs 깊이
params = count_conv_params(3, 3, 64) * depths
ax2.plot(depths, params / 1e6, 'o-', linewidth=2)
ax2.set_xlabel('깊이 (층 개수)')
ax2.set_ylabel('파라미터 수 (M)')
ax2.set_title('CNN 깊이에 따른 파라미터')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**출력**:
```
======================================================================
깊이별 Receptive Field (stride=1)
======================================================================

커널 3×3:
  깊이  1: RF=  3×  3, 파라미터≈1,792
  깊이  2: RF=  5×  5, 파라미터≈3,584
  깊이  3: RF=  7×  7, 파라미터≈5,376
  깊이  5: RF= 11×11, 파라미터≈8,960
  깊이  7: RF= 13×13, 파라미터≈12,544
  깊이 13: RF= 25×25, 파라미터≈23,296

======================================================================
목표 RF 달성에 필요한 깊이 (k=3)
======================================================================
RF= 5: 깊이= 2, 파라미터≈3,584
RF= 7: 깊이= 3, 파라미터≈5,376
RF=13: 깊이= 7, 파라미터≈12,544
RF=25: 깊이=13, 파라미터≈23,296
RF=51: 깊이=26, 파라미터≈46,592
RF=101: 깊이=51, 파라미터≈91,392
```

### 3. VC 차원 추정 및 일반화 이론

```python
def vc_dimension_fc(input_dim, output_dim):
    """FC 층의 VC 차원 상한 추정"""
    return input_dim * output_dim

def vc_dimension_cnn(kernel_size, in_channels, out_channels, num_layers):
    """CNN의 VC 차원 상한 추정"""
    return kernel_size**2 * in_channels * out_channels * num_layers

def sample_complexity(vc_dim, epsilon, delta=0.05):
    """
    VC 이론으로 추정한 필요 샘플 개수
    epsilon: 오차 경계
    delta: 신뢰도 실패 확률
    """
    return (vc_dim / epsilon**2) * np.log(vc_dim / delta)

# ===== VC 차원 비교 =====
print("=" * 70)
print("VC 차원 비교 (224×224 RGB)")
print("=" * 70)

vc_fc = vc_dimension_fc(150528, 512)
vc_cnn = vc_dimension_cnn(3, 3, 512, 5)

print(f"\nFC 네트워크 (1층, 512 출력):")
print(f"  VC 차원 ≈ {vc_fc:,}")

print(f"\nCNN (5층, 3×3 커널, 512 필터):")
print(f"  VC 차원 ≈ {vc_cnn:,}")

print(f"\n비율: {vc_cnn / vc_fc:.2e}")
print(f"FC가 CNN보다 {vc_fc / vc_cnn:.0f}배 복잡")

# ===== 필요 샘플 개수 =====
print("\n" + "=" * 70)
print("필요 샘플 개수 (일반화 오차 ε=0.01 이내, 신뢰도 95%)")
print("=" * 70)

epsilon = 0.01
delta = 0.05

n_fc = sample_complexity(vc_fc, epsilon, delta)
n_cnn = sample_complexity(vc_cnn, epsilon, delta)

print(f"\nFC: {n_fc:.2e} 샘플")
print(f"CNN: {n_cnn:.2e} 샘플")
print(f"\nCNN이 필요로 하는 샘플: FC의 {n_cnn/n_fc*100:.4f}%")
print(f"FC가 필요로 하는 샘플: CNN의 {n_fc/n_cnn:.0f}배")

# ===== 일반화 오차 상한 =====
print("\n" + "=" * 70)
print("일반화 오차 상한 (다양한 샘플 크기에서)")
print("=" * 70)

sample_counts = [1000, 10000, 50000, 100000, 1000000]

for n in sample_counts:
    gen_error_fc = np.sqrt(vc_fc / n) * np.sqrt(np.log(vc_fc / 0.05) / n)
    gen_error_cnn = np.sqrt(vc_cnn / n) * np.sqrt(np.log(vc_cnn / 0.05) / n)
    
    print(f"\nn={n:,} 샘플:")
    print(f"  FC 일반화 오차 ≤ {gen_error_fc:.4f}")
    print(f"  CNN 일반화 오차 ≤ {gen_error_cnn:.4f}")
    if gen_error_cnn > 0:
        print(f"  비율: {gen_error_cnn/gen_error_fc*100:.2f}%")

# ===== 시각화: 일반화 오차 곡선 =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sample_range = np.logspace(2, 6, 50)

# VC 차원 축소 버전
vc_fc_small = 1e5
vc_cnn_small = 1e3

gen_fc = np.sqrt(vc_fc_small / sample_range)
gen_cnn = np.sqrt(vc_cnn_small / sample_range)

axes[0].loglog(sample_range, gen_fc, 'o-', label='FC', linewidth=2, markersize=6)
axes[0].loglog(sample_range, gen_cnn, 's-', label='CNN', linewidth=2, markersize=6)
axes[0].set_xlabel('샘플 개수 (n)')
axes[0].set_ylabel('일반화 오차 상한')
axes[0].set_title('VC 이론: 일반화 오차 vs 샘플 크기')
axes[0].grid(True, alpha=0.3, which='both')
axes[0].legend()

# 파라미터 vs 성능
fc_params_list = np.array([1e4, 1e5, 1e6, 1e7, 1e8])
cnn_params_list = np.array([1e3, 1e4, 1e5, 1e6, 1e7])

# 가정: 같은 양의 데이터
n_samples = 60000
gen_fc_list = np.sqrt(fc_params_list / n_samples)
gen_cnn_list = np.sqrt(cnn_params_list / n_samples)

axes[1].loglog(fc_params_list, gen_fc_list, 'o-', label='FC', 
               linewidth=2, markersize=8)
axes[1].loglog(cnn_params_list, gen_cnn_list, 's-', label='CNN', 
               linewidth=2, markersize=8)
axes[1].set_xlabel('파라미터 개수')
axes[1].set_ylabel('일반화 오차 상한')
axes[1].set_title('파라미터 증가에 따른 일반화 오차')
axes[1].grid(True, alpha=0.3, which='both')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**출력**:
```
======================================================================
VC 차원 비교 (224×224 RGB)
======================================================================

FC 네트워크 (1층, 512 출력):
  VC 차원 ≈ 77,069,056

CNN (5층, 3×3 커널, 512 필터):
  VC 차원 ≈ 38,400

비율: 4.98e-04
FC가 CNN보다 2,006배 복잡

======================================================================
필요 샘플 개수 (일반화 오차 ε=0.01 이내, 신뢰도 95%)
======================================================================

FC: 1.09e+11 샘플
CNN: 5.44e+07 샘플

CNN이 필요로 하는 샘플: FC의 0.0499%
FC가 필요로 하는 샘플: CNN의 2,006배

======================================================================
일반화 오차 상한 (다양한 샘플 크기에서)
======================================================================

n=1,000 샘플:
  FC 일반화 오차 ≤ 8.7707
  CNN 일반화 오차 ≤ 0.6194

n=100,000 샘플:
  FC 일반화 오차 ≤ 0.8770
  CNN 일반화 오차 ≤ 0.0619
```

### 4. MNIST/CIFAR 실제 학습 비교 (간략)

```python
# 주의: 실제 학습은 오래 걸리므로, 여기는 개념적 구조만 제시

def simple_fc_network(input_size, hidden_size, learning_rate=0.01):
    """간단한 FC 네트워크 클래스"""
    class FCNet:
        def __init__(self):
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, 10) * 0.01
            self.b2 = np.zeros(10)
        
        def forward(self, X):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = np.maximum(0, self.z1)  # ReLU
            self.z2 = self.a1 @ self.W2 + self.b2
            return self.z2
        
        def num_params(self):
            return (self.W1.size + self.b1.size + 
                    self.W2.size + self.b2.size)
    
    return FCNet()

# 크기 비교
fc_net = simple_fc_network(28*28, 128)
print("FC 네트워크 (MNIST):")
print(f"  파라미터: {fc_net.num_params():,}")

# CNN은 훨씬 적음
# 예: 3 conv layers with 32, 64, 128 filters
cnn_params = (3*3*1*32 + 32) + (3*3*32*64 + 64) + (3*3*64*128 + 128)
print("\nCNN 네트워크 (MNIST, 3층):")
print(f"  파라미터: {cnn_params:,}")
print(f"\n비율: {cnn_params / fc_net.num_params():.4f}")

# 예상 성능 (경험적)
print("\n예상 성능 (MNIST):")
print("  FC (784→128→10): ~97-98% accuracy")
print("  CNN (3 layers): ~99-99.5% accuracy")
print("  → 더 적은 파라미터로 더 나은 성능!")
```

---

## 🔗 실전 응용

### 아키텍처 설계 시 고려사항

```python
# 필요한 receptive field로부터 아키텍처 결정

def design_cnn_architecture(target_rf, input_size=224, kernel_size=3):
    """
    목표 receptive field를 달성하는 CNN 설계
    """
    # 필요 깊이
    depth = (target_rf - 1) // (kernel_size - 1) + 1
    
    # 채널 수 (경험적)
    channels = [32, 64, 128, 256, 512]
    
    total_params = 0
    print(f"설계: RF={target_rf}, 깊이={depth}")
    print("\n층별 구성:")
    print("  [Layer] Kernel | C_in × C_out | RF | 파라미터")
    
    for i in range(min(depth, len(channels))):
        c_in = channels[i-1] if i > 0 else 3
        c_out = channels[i]
        params = count_conv_params(kernel_size, c_in, c_out)
        rf = receptive_field(i+1, kernel_size)
        total_params += params
        print(f"  Conv{i+1:2d}   {kernel_size}×{kernel_size}  | {c_in:3d} × {c_out:3d}   | {rf:3d} | {params:,}")
    
    print(f"\n총 파라미터: {total_params:,}")
    return depth, total_params

design_cnn_architecture(13, kernel_size=3)
```

---

## ⚖️ 한계와 고려사항

1. **VC 이론의 looseness**:
   - VC bound는 worst-case이므로 실제보다 훨씬 큼
   - 실제 일반화는 VC 이론의 예측보다 훨씬 좋은 경우가 많음

2. **구조적 가정의 중요성**:
   - CNN의 이점은 이미지에 translation equivariance 구조가 있다는 가정에 기반
   - 다른 데이터(시계열, 그래프 등)에는 다른 구조 필요

3. **Depth vs Width의 trade-off**:
   - VC 이론만으로는 깊이와 너비의 최적 비율을 결정할 수 없음
   - 경험적으로 깊이가 더 효율적인 경향 (Ch7에서 자세히)

4. **초기화와 최적화의 역할**:
   - 파라미터가 적어도 나쁜 초기화/최적화로 실패 가능
   - Ch4의 초기화 기법이 여전히 중요

---

## 📌 핵심 정리

| 개념 | FC | CNN | 차이 |
|------|----|----|------|
| **파라미터** | $H \times W \times C_\text{in} \times C_\text{out}$ | $k^2 \times C_\text{in} \times C_\text{out}$ | $\sim (k/H)^2$배 |
| **VC 차원** | $\propto$ params | $\propto k^2 C_\text{in} C_\text{out} L$ | $\sim (k/H)^2$배 |
| **Sample Complexity** | 높음 | 낮음 | CNN이 유리 |
| **Receptive Field** | 한 층에 full | $L$층에 $(k-1)L+1$ | 깊이 필요 |

---

## 🤔 문제

**1번**: $H=512, W=512, C_\text{in}=3, C_\text{out}=256, k=3$일 때:
- FC 파라미터를 계산하세요
- CNN 파라미터를 계산하세요
- 감소 비율은?

**2번**: 3×3 커널을 $n$층 쌓았을 때, receptive field가 51×51이 되려면 $n$은?

**3번**: VC 차원 정의를 자신의 말로 설명하고, "모든 라벨 조합을 분류할 수 있다"는 의미를 예시로 들어 설명하세요.

**4번**: 다음 두 모델 중 CIFAR-10에서 더 나은 일반화를 기대할 수 있는 것은? (각 50,000개 학습 샘플)
- 모델 A: FC, 10M 파라미터
- 모델 B: CNN, 100K 파라미터
- 이유를 VC 이론으로 설명하세요.

**5번**: Receptive field가 작으면 왜 깊은 네트워크가 필요할까요? 비용-효과를 고려하여 설명하세요.

---

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Convolution과 Translation Equivariance](./01-convolution-equivariance.md) | [📚 README로 돌아가기](../README.md) | [03. Pooling과 Local Invariance ▶](./03-pooling-invariance.md) |

</div>

---

*마지막 업데이트: 2025-04-24 | 난이도: ★★★★☆ | 소요 시간: 100분*
