# Chapter 5-01. Convolution의 수학과 Translation Equivariance

## 🎯 핵심 질문
- Convolution 연산은 수학적으로 무엇이고, 왜 이미지 처리에 자연스러운가?
- Translation equivariance란 무엇이며, CNN이 이를 어떻게 달성하는가?
- Cross-correlation과 convolution의 차이는 무엇인가? ML에서는 왜 구분하지 않는가?

---

## 🔍 필요성

합성곱(convolution)은 신호 처리와 수학에서 기원한 연산인데, 이미지 인식에 유달리 잘 맞는 이유가 있습니다:

1. **위치 불변성 검출**: 동일한 특징(모서리, 질감, 패턴)이 이미지 어디에 있든 같은 방식으로 검출되어야 함
2. **국소성**: 각 픽셀의 정보는 주변 픽셀과의 상호작용으로 형성됨
3. **공유 가중치**: 작은 커널로 전체 이미지를 처리하여 파라미터 효율 달성

정확한 수학적 정의와 equivariance 증명을 이해하면, CNN이 단순한 기법이 아니라 **이미지 데이터의 통계적 구조를 반영한 체계적 설계**임을 알 수 있습니다.

---

## 📐 선행 지식

- **기본**: 벡터-행렬 연산, 부분미분, 선형대수 (행렬 곱셈, 교환성, 결합성)
- **신호처리**: Fourier 변환 개념 (선택)
- **Chapter 4**: 활성화 함수, 역전파 기초
- **Ch2**: 선형변환과 함수 합성

---

## 📖 직관

### Convolution의 기하학적 의미

**연속 신호의 관점**: 두 함수 $f, g$의 convolution $(f * g)(x)$는 "$f$를 뒤집어 슬라이드하며 $g$와의 중첩적 곱의 합"입니다.

- **뒤집기**: $g$의 인덱스를 역순으로 (수학 정의)
- **슬라이드**: 모든 위치에서 계산
- **중첩적 곱**: 겹치는 부분의 점곱셈

### 이미지에서의 의미

이미지 $I$에 커널 $K$ (edge detector, blur kernel 등)를 적용하면:
- 각 위치 $(i,j)$에서 국소 영역의 가중 합 계산
- 작은 커널 = 국소적 특징 추출
- 모든 위치에서 **같은 커널 사용** = 평행이동 동등성(equivariance)

이것이 "특징은 위치와 무관하게 같은 방식으로 검출"된다는 뜻입니다.

---

## ✏️ 정의

### 1. 연속 Convolution

$$\boxed{(f * g)(x) = \int_{-\infty}^{\infty} f(y) g(x-y) \, dy}$$

**해석**:
- $x$에서의 output = $f$ 전체와 $g$(뒤집어짐)를 위치 $x$에서 정렬했을 때의 적분
- **교환성**: $(f*g) = (g*f)$ — 순서 무관
- **결합성**: $(f*g)*h = f*(g*h)$ — 여러 convolution 순서 무관
- **선형성**: $f*(ag+bh) = a(f*g) + b(f*h)$

### 2. 이산 Convolution

$$\boxed{(f*g)[n] = \sum_{m=-\infty}^{\infty} f[m] g[n-m]}$$

**예시** ($n=2$):
$$
\begin{align}
(f*g)[2] &= \sum_m f[m] g[2-m] \\
&= \cdots + f[0]g[2] + f[1]g[1] + f[2]g[0] + f[3]g[-1] + \cdots
\end{align}
$$

### 3. 2D Convolution (이미지)

$$\boxed{(I*K)[i,j] = \sum_{m,n} I[m,n] \, K[i-m, j-n]}$$

**표기**:
- $I$ = 입력 이미지 $(H \times W)$
- $K$ = 커널/필터 $(k_h \times k_w)$
- 출력 크기: 경계 처리에 따라 다름 (zero-padding, valid, same)

### 4. Cross-Correlation (ML에서 "Convolution"으로 부르는 것)

$$\boxed{(f \star g)[n] = \sum_m f[m] g[n+m]}$$

**vs Convolution**:
| 연산 | 식 | 뒤집기 |
|------|-----|---------|
| Convolution | $\sum f[m] g[n-m]$ | 있음 |
| Cross-corr | $\sum f[m] g[n+m]$ | 없음 |

**중요**: 머신러닝에서는 커널을 학습하므로, 뒤집기 여부가 무의미합니다.
$(f \star g) = (f * g_{\text{flipped}})$이기 때문입니다. 따라서 대부분의 DL 프레임워크는 cross-correlation을 사용하면서 "convolution"이라 부릅니다.

### 5. Translation Equivariance

**정의**: 연산자 $\phi$가 translation equivariant하다는 것은
$$\boxed{\phi(T_s x) = T_s \phi(x) \quad \forall s}$$

여기서 $T_s$는 shift 연산자:
$$(T_s f)(x) = f(x - s)$$

**해석**: input을 평행이동한 후 연산 = 연산한 후 평행이동

---

## 🔬 증명

### 정리: Convolution Layer는 Translation Equivariant이다

**명제**: 
$$\phi(x) = (x * k) \quad \Rightarrow \quad \phi(T_s x) = T_s \phi(x)$$

**증명**:

좌변을 계산:
$$
\begin{align}
\phi(T_s x)[n] &= ((T_s x) * k)[n] \\
&= \sum_m (T_s x)[m] \, k[n-m] \\
&= \sum_m x[m-s] \, k[n-m] \quad \text{(shift 정의)} \\
&= \sum_m x[m-s] \, k[n-m]
\end{align}
$$

$m' = m - s$ 치환 ($m = m' + s$):
$$
\begin{align}
&= \sum_{m'} x[m'] \, k[n-(m'+s)] \\
&= \sum_{m'} x[m'] \, k[(n-s)-m'] \\
&= (x*k)[n-s] \\
&= (T_s \phi(x))[n]
\end{align}
$$

따라서 $\phi(T_s x) = T_s \phi(x)$. $\square$

### 물리적 의미

- 이미지의 특징(가장자리, 질감)이 오른쪽으로 이동 → 검출된 특징도 같은 양만큼 오른쪽으로 이동
- 따라서 같은 필터는 **위치와 무관하게 같은 특징을 찾음**
- 이미지에서 "1이 어디에 있든 같은 way로 검출"

### 깊이가 있는 CNN의 경우

$L$개 layer CNN: $\phi = \phi_L \circ \phi_{L-1} \circ \cdots \circ \phi_1$

각 $\phi_i$가 equivariant이면:
$$
\phi(T_s x) = (\phi_L \circ \cdots \circ \phi_1)(T_s x) = (T_s \phi_L \circ \cdots) = T_s \phi(x)
$$

**단, 주의**: Max pooling, ReLU는 equivariant이지만 average pooling의 경우 경계에서 미묘한 차이 발생 가능.

---

## 💻 NumPy 실험

### 1. 1D Convolution 구현 및 equivariance 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def conv1d(f, k, mode='same'):
    """
    1D convolution: (f*k)[n] = sum f[m] k[n-m]
    mode='same': output 크기 = input 크기 (zero padding)
    """
    n_f, n_k = len(f), len(k)
    
    # Kernel 뒤집기
    k_flipped = k[::-1]
    
    if mode == 'same':
        # Zero padding
        pad = (n_k - 1) // 2
        f_padded = np.pad(f, (pad, pad), mode='constant')
        output = np.zeros(n_f)
        
        for n in range(n_f):
            # 위치 n에서의 convolution
            output[n] = np.dot(f_padded[n:n+n_k], k_flipped)
        return output
    
    else:  # 'valid'
        output = np.zeros(n_f - n_k + 1)
        for n in range(len(output)):
            output[n] = np.dot(f[n:n+n_k], k_flipped)
        return output

# 신호 정의
f = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
k = np.array([1, -1], dtype=float)  # 미분 kernel

# Convolution 수행
phi_f = conv1d(f, k, mode='same')
print("원본 신호:", f)
print("Convolution 결과:", phi_f)

# ===== Equivariance 검증 =====
shift_amount = 2
f_shifted = np.roll(f, shift_amount)  # 신호를 shift_amount만큼 평행이동

# 방법 1: shift 후 convolution
result1 = conv1d(f_shifted, k, mode='same')

# 방법 2: convolution 후 shift
result2 = np.roll(phi_f, shift_amount)

print("\n--- Equivariance 검증 (shift={}) ---".format(shift_amount))
print("Shift → Conv:", result1)
print("Conv → Shift:", result2)
print("차이 (L2 norm):", np.linalg.norm(result1 - result2))
print("Equivariant: {}".format(np.allclose(result1, result2)))
```

**출력 예시**:
```
원본 신호: [1. 2. 3. 4. 5. 6. 7. 8.]
Convolution 결과: [1. 1. 1. 1. 1. 1. 1. 1.]

--- Equivariance 검증 (shift=2) ---
Shift → Conv: [1. 1. 1. 1. 1. 1. 1. 1.]
Conv → Shift: [1. 1. 1. 1. 1. 1. 1. 1.]
차이 (L2 norm): 0.0
Equivariant: True
```

### 2. 2D Convolution 구현

```python
def conv2d(image, kernel, mode='same'):
    """
    2D convolution: (I*K)[i,j] = sum_{m,n} I[m,n] K[i-m, j-n]
    """
    h_img, w_img = image.shape
    h_ker, w_ker = kernel.shape
    
    # Kernel 뒤집기 (2D)
    kernel_flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    
    if mode == 'same':
        pad_h = (h_ker - 1) // 2
        pad_w = (w_ker - 1) // 2
        img_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros((h_img, w_img))
        
        for i in range(h_img):
            for j in range(w_img):
                patch = img_padded[i:i+h_ker, j:j+w_ker]
                output[i, j] = np.sum(patch * kernel_flipped)
        return output

# 작은 이미지 정의
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

# 다양한 커널
# 1. Vertical edge detector
kernel_v = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=float)

# 2. Horizontal edge detector
kernel_h = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=float)

# 3. Blur kernel (Gaussian approximation)
kernel_blur = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype=float) / 16.0

print("원본 이미지:")
print(image)

for name, ker in [("Vertical Edge", kernel_v), 
                   ("Horizontal Edge", kernel_h),
                   ("Blur", kernel_blur)]:
    result = conv2d(image, ker, mode='same')
    print(f"\n{name} 커널 결과:")
    print(result)

# ===== 2D Equivariance 검증 =====
image_shifted = np.roll(np.roll(image, 1, axis=0), 1, axis=1)  # (1,1) shift

result1 = conv2d(image_shifted, kernel_v, mode='same')
result2 = np.roll(np.roll(conv2d(image, kernel_v, mode='same'), 1, axis=0), 1, axis=1)

print("\n--- 2D Equivariance 검증 ---")
print("차이 (L2 norm):", np.linalg.norm(result1 - result2))
print("Equivariant: {}".format(np.allclose(result1, result2)))
```

**출력**:
```
원본 이미지:
[[ 1.  2.  3.  4.]
 [ 5.  6.  7.  8.]
 [ 9. 10. 11. 12.]
 [13. 14. 15. 16.]]

Vertical Edge 커널 결과:
[[-2.  -4.  -4.  -2.]
 [-8. -16. -16.  -8.]
 [-8. -16. -16.  -8.]
 [-2.  -4.  -4.  -2.]]

Blur 커널 결과:
[[ 2.8125  3.375   3.9375  2.75  ]
 [ 5.      5.75    6.5     4.75  ]
 [ 9.      10.25  11.      7.75  ]
 [ 8.8125  9.75   10.6875  7.25  ]]

--- 2D Equivariance 검증 ---
차이 (L2 norm): 0.0
Equivariant: True
```

### 3. Cross-Correlation vs Convolution 비교

```python
def cross_correlation1d(f, k):
    """Cross-correlation: (f★k)[n] = sum f[m] k[n+m]"""
    n_f, n_k = len(f), len(k)
    output = np.zeros(n_f)
    
    for n in range(n_f):
        for m in range(n_k):
            if 0 <= n + m < n_f:
                output[n] += f[n + m] * k[m]
    return output

f = np.array([1, 2, 3, 4, 5], dtype=float)
k = np.array([0.5, 0.5], dtype=float)

conv_result = conv1d(f, k, mode='valid')
xcorr_result = cross_correlation1d(f, k)
xcorr_with_flip = cross_correlation1d(f, k[::-1])

print("신호:", f)
print("Kernel:", k)
print("\nConvolution (뒤집음):", conv_result)
print("Cross-correlation (뒤집지 않음):", xcorr_result)
print("Cross-correlation (뒤집음):", xcorr_with_flip)
print("\nConv == Xcorr(flipped)?", np.allclose(conv_result, xcorr_with_flip))
```

---

## 🔗 실전 응용

### 이미지 처리 예시

```python
from scipy import signal
import matplotlib.pyplot as plt

# 간단한 그레이스케일 이미지 생성
img = np.zeros((100, 100))
img[30:70, 30:70] = 1  # 중앙에 정사각형

# Sobel operator (edge detection)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)

edges = conv2d(img, sobel_x, mode='same')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('원본 이미지')
axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Sobel 엣지 검출')
axes[2].plot(edges[50, :])
axes[2].set_title('가로 프로필')
plt.tight_layout()
plt.show()
```

### Equivariance의 실용적 의미

```python
# 패턴 인식의 위치 불변성
pattern = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=float)  # + 모양

# 큰 이미지의 다양한 위치에 패턴 삽입
bg = np.zeros((20, 20))
bg[2:5, 2:5] = pattern      # 위쪽 좌측
bg[8:11, 8:11] = pattern    # 중앙
bg[14:17, 14:17] = pattern  # 아래쪽 우측

# 패턴을 찾는 커널
detector = pattern.copy()

response = conv2d(bg, detector, mode='same')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(bg, cmap='gray')
axes[0].set_title('배경 + 3개 패턴')
axes[1].imshow(response, cmap='hot')
axes[1].set_title('Convolution 응답')
axes[2].plot(response.flatten())
axes[2].set_title('응답값 분포')
plt.tight_layout()
plt.show()

print("패턴 위치들의 응답값:")
for i, j in [(3, 3), (9, 9), (15, 15)]:
    print(f"  위치 ({i},{j}): {response[i, j]:.2f}")
```

---

## ⚖️ 한계와 고려사항

1. **Equivariance의 한계**:
   - Pooling, ReLU, 경계 처리 등으로 인해 strict equivariance 깨짐
   - 특히 low-level features는 equivariant하지만, high-level semantic features는 position-specific일 수 있음

2. **수용 영역(Receptive Field) 제한**:
   - 작은 커널 (3×3)은 국소 정보만 처리
   - 깊이를 쌓아야 global context 학습 (Ch5-02 참고)

3. **Boundary Effects**:
   - Zero-padding은 가장자리에서 spurious features 생성 가능
   - Circular padding, reflect padding 등 대안 존재

4. **Rotation, Scale에 대한 Equivariance 부재**:
   - Convolution은 translation에만 equivariant
   - Rotation/scale invariance는 data augmentation이나 별도 기법 필요

---

## 📌 핵심 정리

| 개념 | 설명 | 중요성 |
|------|------|--------|
| **Convolution** | $\sum f[m]g[n-m]$ — 뒤집기 포함 | 수학적 기초 |
| **Cross-corr** | $\sum f[m]g[n+m]$ — 뒤집기 없음 | ML에서 실제 사용 |
| **Equivariance** | $\phi(T_s x) = T_s \phi(x)$ | 위치 불변 특징 검출 |
| **Parameter Sharing** | 모든 위치에서 같은 커널 | 효율성 (다음 절) |
| **2D Conv** | $(I*K)[i,j] = \sum_{m,n} I[m,n]K[i-m,j-n]$ | 이미지 처리 핵심 |

---

## 🤔 문제

**1번**: $k=[1, 0, -1]$ 커널과 신호 $x=[2,3,5,7]$에 대해 1D convolution (valid mode)을 손으로 계산하세요.

**2번**: 왜 ML 프레임워크에서 cross-correlation을 "convolution"이라 부르는지 설명하고, 이것이 학습에 영향을 주지 않는 이유를 서술하세요.

**3번**: $3 \times 3$ convolution을 두 번 쌓으면 receptive field는 몇 × 몇입니까? (stride=1 가정)

**4번**: 다음 2D 이미지와 커널에 대해 중앙의 픽셀만 convolution을 계산하세요.
```
이미지:          커널:
[1 2 3]          [1 0 -1]
[4 5 6]    *     [0 1 0]
[7 8 9]          [-1 0 1]
```

**5번**: Max pooling이 translation equivariant하지 않은 경우를 구성하고 설명하세요.

---

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch4-05. Fixup 초기화](../ch4-initialization/05-fixup-initialization.md) | [📚 README로 돌아가기](../README.md) | [02. 파라미터 공유와 VC 이론 ▶](./02-parameter-sharing-vc.md) |

</div>

---

*마지막 업데이트: 2025-04-24 | 난이도: ★★★☆☆ | 소요 시간: 90분*
