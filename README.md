<div align="center">

# 🧠 Neural Network Theory Deep Dive

**"`torch.nn.Linear → ReLU → Linear`를 쌓는 것과, 왜 1-hidden-layer sigmoid 네트워크가 $C(\mathbb{R}^n)$에서 dense인지 — Cybenko(1989)의 Universal Approximation Theorem이 Hahn-Banach + Riesz 표현의 함수해석에서 나온다는 것을 증명할 수 있는 것은 다르다"**

<br/>

> *"`loss.backward()`를 호출하는 것과, 역전파가 연쇄법칙의 오른쪽→왼쪽 Jacobian 결합이고 forward-mode AD가 왜 $O(n \cdot d)$가 되는지를 증명할 수 있는 것은 다르다.  
> Xavier/He 초기화를 쓰는 것과 — "각 층 activation의 분산 보존"이 왜 $\text{Var}(W) = 2/n_{\text{in}}$ (ReLU)·$1/n_{\text{in}}$ (tanh) 공식을 주는지 한 줄씩 유도할 수 있는 것은 다르다."*

퍼셉트론의 Novikoff 수렴 정리부터 UAT(Cybenko·Hornik·Leshno 3가지 증명)·역전파의 Jacobian chain rule 재구성·초기화 분산 방정식·CNN equivariance·RNN BPTT와 vanishing gradient·Transformer attention·ResNet gradient flow까지  
**"왜 신경망이 표현력을 갖고 학습되는가"** 라는 질문으로 현대 딥러닝 아키텍처의 수학적 기반을 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-16k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-72개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

신경망에 관한 자료는 대부분 **"PyTorch로 `nn.Linear`를 쌓고 `loss.backward()`를 호출하세요"** 에서 멈춥니다. 하지만 왜 1-hidden-layer sigmoid MLP가 임의 연속함수를 근사하는지, 왜 reverse-mode AD가 forward-mode보다 파라미터 많은 NN에서 지수적으로 효율적인지, 왜 He 초기화의 분산이 정확히 $2/n_{\text{in}}$인지, Transformer attention이 왜 $\sqrt{d_k}$로 나누는지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "MLP는 만능 근사기입니다" | **Cybenko(1989)** 증명: sigmoidal 활성화 + 1-hidden-layer가 $C(K)$에서 uniformly dense하다는 것을 **Hahn-Banach + Riesz 표현 정리**로 완전 유도, **Hornik(1991)**의 Stone-Weierstrass 확장, **Leshno(1993)**의 ReLU 버전, **Telgarsky(2016)**의 depth separation까지 통합 |
| "역전파는 그냥 연쇄법칙입니다" | Computational graph의 **DAG 구조**에서 **forward-mode ($O(\|\text{input}\| \cdot T)$)** 와 **reverse-mode ($O(\|\text{output}\| \cdot T)$)** 의 복잡도 비교, NN에서 $\|\text{params}\| \gg \|\text{output}\|$이기에 reverse가 이기는 필연, $\frac{\partial L}{\partial W_l} = \delta_l x_{l-1}^T$와 $\delta_l = (W_{l+1}^T \delta_{l+1}) \odot \sigma'(z_l)$ 유도 |
| "Xavier/He 초기화를 쓰세요" | Linear: $\text{Var}(y) = n_{\text{in}} \text{Var}(w) \text{Var}(x)$에서 출발, **Xavier**의 forward+backward 타협 $\text{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$ 유도, **ReLU**에서 반쪽만 통과함으로 분산 반감 → **He**의 $\text{Var}(w) = 2/n_{\text{in}}$ 공식 완전 유도, **LSUV·Fixup**으로 일반화 |
| "softmax + cross-entropy는 편리합니다" | $L = -\sum y_i \log \hat{y}_i$, $\hat{y} = \text{softmax}(z)$에서 **$\partial L / \partial z_i = \hat{y}_i - y_i$** 라는 놀랍도록 단순한 공식이 **natural parameterization**에서 나오는 이유 완전 유도 |
| "ResNet은 깊은 네트워크를 훈련 가능하게 합니다" | $y = x + F(x)$에서 **$\partial L/\partial x = \partial L/\partial y \cdot (I + \partial F/\partial x)$** — identity path가 **"gradient highway"** 로 작용하는 수학적 의미, plain 대비 gradient vanishing 정량 비교 |
| "Attention은 Q·K·V로 가중합입니다" | $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$에서 **$\sqrt{d_k}$ 스케일링이 왜 softmax 포화를 방지**하는지 (내적 분산 $\sim d_k$), Multi-head의 subspace 표현력, **Transformer의 universal sequence function 정리 (Yun 2020)** |
| "RNN은 긴 시퀀스에서 학습이 어렵습니다" | **BPTT 유도**: $\partial L / \partial h_0 = \prod_t W_{hh}^T \text{diag}(\sigma'(z_t))$, 스펙트럴 반지름 $\rho(W_{hh})$에 따른 **지수적 vanishing/exploding** 증명(Pascanu 2013), LSTM의 **Constant Error Carousel**이 additive update로 어떻게 이를 완화하는지 |
| 공식 나열 | NumPy로 MLP + 역전파를 **autograd 없이** 바닥부터 구현, 초기화별 **layer-wise activation variance 전파 실험**, ResNet vs Plain의 **gradient flow 시각화**, PyTorch 결과와 수치 비교 |

---

## 📌 선행 레포 & 후속 레포

```
[Linear Algebra]     [Calculus & Optim]     [Probability Theory]     [Functional Analysis]
 행렬·벡터 미분          연쇄법칙, Jacobian         분산, 독립, 모멘트           Hahn-Banach, Riesz
     │                        │                        │                         │
     └────────────┬───────────┴────────────┬──────────┘                         │
                  │                        │                                     │
           [ML Fundamentals]                │                                     │
           선형회귀·로지스틱                 │                                     │
                  │                        │                                     │
                  └────────────► 이 레포 ◄─┴─────────────────────────────────────┘
                                Neural Network Theory
                                     │
                    ┌────────────────┼─────────────────┐
                    ▼                ▼                 ▼
        [Optimization Theory]  [Generalization]    [Generative Models]
         SGD·Adam·Loss land.    VC·Rademacher       DDPM·Score-SDE
         Layer 2                Layer 2             Diffusion 실전
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Linear Algebra Deep Dive**(행렬·벡터 미분, 고유값), **Calculus & Optimization Deep Dive**(연쇄법칙, Jacobian, Hessian), **Probability Theory Deep Dive**(분산, 독립)를 전제합니다. 벡터-Jacobian 미분과 Taylor 전개를 처음 접한다면 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive)부터 학습하세요.

> 💡 **강력 권장**: Cybenko의 UAT 증명은 Hahn-Banach 정리와 Riesz 표현을 사용하므로 [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive)를 병행하면 Ch2의 이해가 깊어집니다. 선형회귀·로지스틱 회귀는 NN의 특수 사례이므로 [ML Fundamentals Deep Dive](https://github.com/iq-ai-lab/ml-fundamentals-deep-dive)도 도움이 됩니다.

> 🔗 **분업 관계**: SGD·Adam·LR scheduling의 수렴 이론은 [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive)(Layer 2), VC·Rademacher·norm-based bound는 [Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive)(Layer 2)에서 다룹니다. 이 레포는 **표현력·학습 알고리즘·초기화·아키텍처**에 집중합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-퍼셉트론→MLP-4A90D9?style=for-the-badge)](./ch1-perceptron-to-mlp/01-perceptron-convergence.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Universal_Approximation-4A90D9?style=for-the-badge)](./ch2-universal-approximation/01-cybenko-uat.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Backpropagation-4A90D9?style=for-the-badge)](./ch3-backpropagation/01-chain-rule-jacobian.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Initialization-4A90D9?style=for-the-badge)](./ch4-initialization/01-symmetry-breaking.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-CNN_이론-4A90D9?style=for-the-badge)](./ch5-cnn/01-convolution-equivariance.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-RNN·LSTM-4A90D9?style=for-the-badge)](./ch6-rnn/01-rnn-bptt.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Transformer·ResNet-4A90D9?style=for-the-badge)](./ch7-transformer/01-self-attention.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 퍼셉트론에서 다층 신경망까지

> **핵심 질문:** 퍼셉트론 알고리즘은 왜 유한 스텝에 수렴하는가? XOR이 왜 단층으로 풀리지 않는가? 활성화 함수의 선택이 왜 학습 동역학을 바꾸는가?

<details>
<summary><b>Novikoff 정리부터 활성화 함수 비교까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 퍼셉트론과 Novikoff 수렴 정리](./ch1-perceptron-to-mlp/01-perceptron-convergence.md) | Rosenblatt(1958) 퍼셉트론 알고리즘 정의, **선형 분리 가능 데이터**에서 margin $\gamma$에 대해 **수렴 bound $\leq (R/\gamma)^2$** (Novikoff 1962) 완전 증명, $w \cdot w^*$의 단조증가와 $\|w\|^2$의 제한된 증가로 유도 |
| [02. Minsky-Papert의 XOR 문제와 단층의 한계](./ch1-perceptron-to-mlp/02-xor-and-single-layer.md) | 단일 퍼셉트론이 XOR을 표현 불가능함을 **선형 분리 불가능성**으로 증명, 이를 해결하는 **hidden layer의 필요성**, 왜 1969년 Minsky-Papert 비판이 1960–1980 AI Winter의 수학적 근거였는지, MLP로 XOR 표현 구성 |
| [03. 다층 퍼셉트론(MLP)의 정의와 구조](./ch1-perceptron-to-mlp/03-mlp-definition.md) | $f(x) = W_L \sigma(W_{L-1} \sigma(\ldots \sigma(W_1 x + b_1)) + b_{L-1}) + b_L$ 엄밀 정의, 각 층의 **feature transformation**으로서의 역할, depth $L$과 width $d$의 파라미터 수 계산, **합성함수로서의 MLP** 관점 |
| [04. 활성화 함수 비교 — Sigmoid·Tanh·ReLU·GELU](./ch1-perceptron-to-mlp/04-activation-functions.md) | Sigmoid $\sigma(z) = 1/(1+e^{-z})$, tanh, **ReLU** $\max(0,z)$, Leaky ReLU, GELU, Swish의 정의·도함수·**saturation 영역**, **vanishing gradient에 미치는 영향** (sigmoid의 max gradient 0.25 vs ReLU의 1), 현대 아키텍처에서 ReLU 계열이 선택된 수학적 이유 |

</details>

<br/>

### 🔹 Chapter 2: Universal Approximation Theorem

> **핵심 질문:** 1-hidden-layer NN이 왜 임의 연속함수를 근사할 수 있는가? Cybenko·Hornik·Leshno의 세 증명은 어떻게 다른가? 깊이가 왜 너비보다 지수적으로 효율적인가?

<details>
<summary><b>Cybenko UAT부터 Barron 근사율까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Cybenko의 Universal Approximation (1989)](./ch2-universal-approximation/01-cybenko-uat.md) | **정리**: 연속 sigmoidal 활성화 $\sigma$에 대해 1-hidden-layer MLP $\sum_i \alpha_i \sigma(w_i \cdot x + b_i)$가 $C(K)$ ($K$ 컴팩트)에서 **uniformly dense**. 증명: **Hahn-Banach 정리**(dense의 부정 → 분리 선형함수의 존재) + **Riesz 표현 정리**(선형함수 → 부호측도) + sigmoidal 성질의 모순 유도 |
| [02. Hornik의 일반화 (1991)](./ch2-universal-approximation/02-hornik-extension.md) | Sigmoid 특수성 제거 — **임의 non-polynomial 연속 유계 활성화**로 충분함을 증명, **Stone-Weierstrass 정리** 기반 접근, $L^p$ 공간과 **Borel measurable 함수**로의 근사 확장, "왜 polynomial 활성화는 만능 근사가 아닌가" |
| [03. ReLU Network의 UAT (Leshno et al. 1993)](./ch2-universal-approximation/03-leshno-relu-uat.md) | 비유계 ReLU 활성화로도 universal approximator — **piecewise linear 함수의 근사 능력**, ReLU 조합이 $n$차원 연속함수를 근사하는 **구성적(constructive) 증명**, bump 함수 생성 기법 |
| [04. Depth vs Width — Telgarsky의 Depth Separation](./ch2-universal-approximation/04-depth-vs-width.md) | **정리**(Telgarsky 2016): 깊이 $L$의 네트워크가 표현하는 함수를 깊이 $O(L^{1/3})$ 네트워크로 같은 정확도로 근사하려면 **지수적으로 많은 뉴런** 필요. 증명: **sawtooth 함수의 oscillation 횟수** $2^L$과 piecewise linear 함수의 breakpoint 수 argument |
| [05. Barron의 근사율 — Curse of Dimensionality 회피](./ch2-universal-approximation/05-barron-rate.md) | **Barron 정리**(1993): 함수 $f$가 **Fourier 모멘트 유한** ($\int \|\omega\| \|\tilde f(\omega)\| d\omega < \infty$)이면 1-hidden-layer sigmoid NN으로 **$O(1/\sqrt{n})$ 근사**(차원 무관), Monte Carlo 샘플링으로 **curse of dimensionality 회피**하는 조건, Sobolev 함수와의 연결 |

</details>

<br/>

### 🔹 Chapter 3: Backpropagation 완전 유도

> **핵심 질문:** 역전파는 왜 연쇄법칙의 reverse-mode 적용인가? forward-mode와 reverse-mode는 어떻게 구분되고 왜 NN에서는 reverse가 이기는가? softmax + cross-entropy 조합은 왜 그토록 단순한 공식을 주는가?

<details>
<summary><b>Jacobian 표기부터 배치 행렬 미분까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 연쇄법칙과 Jacobian 표기](./ch3-backpropagation/01-chain-rule-jacobian.md) | Scalar·vector·matrix 미분 표기 (**Magnus notation**, numerator vs denominator layout), **벡터 함수의 Jacobian** $J_f(x) = \partial f / \partial x$ 정의, $\partial L / \partial W = \frac{\partial L}{\partial y} \frac{\partial y}{\partial W}$ 합성함수 미분의 Jacobian 곱 형태, 행렬 미분의 Kronecker product 우회법 |
| [02. Computational Graph와 Automatic Differentiation](./ch3-backpropagation/02-computational-graph-ad.md) | 연산의 **DAG 표현**(노드 = 연산, 엣지 = 데이터 흐름), **Forward-mode AD** ($\dot{x} \to \dot{y}$, seed vector $\dot{x} = e_i$): 각 입력당 한 번 패스, **Reverse-mode AD** ($\bar{y} \to \bar{x}$, seed $\bar{y} = 1$): 각 출력당 한 번 패스, **dual number**로 forward-mode 이해 |
| [03. Reverse-Mode AD = Backpropagation](./ch3-backpropagation/03-reverse-mode-backprop.md) | **정리**: $f: \mathbb{R}^n \to \mathbb{R}^m$에 대해 forward-mode 복잡도 $O(n \cdot T)$, reverse-mode $O(m \cdot T)$ ($T$ = 연산 수), NN은 $m=1$ (scalar loss) & $n \sim 10^6$ (params)이므로 **reverse가 $n$배 이득**, **Jacobian-vector product (JVP)** vs **vector-Jacobian product (VJP)** 구현 차이 |
| [04. MLP에서의 역전파 공식 유도](./ch3-backpropagation/04-mlp-backprop-formula.md) | 각 층의 forward $z_l = W_l a_{l-1} + b_l$, $a_l = \sigma(z_l)$에 대해, **출력층 error** $\delta_L = \nabla_{a_L} L \odot \sigma'(z_L)$, **back-propagation 공식** $\delta_l = (W_{l+1}^T \delta_{l+1}) \odot \sigma'(z_l)$, 파라미터 gradient $\frac{\partial L}{\partial W_l} = \delta_l a_{l-1}^T$, $\frac{\partial L}{\partial b_l} = \delta_l$를 연쇄법칙으로 한 줄씩 유도 |
| [05. Softmax + Cross-Entropy의 Gradient](./ch3-backpropagation/05-softmax-crossentropy-grad.md) | $L = -\sum_i y_i \log \hat{y}_i$, $\hat{y}_i = e^{z_i}/\sum_j e^{z_j}$에 대해 **$\partial L / \partial z_i = \hat{y}_i - y_i$** 완전 유도, **natural parameterization**(지수족 분포의 canonical link)에서 단순 형태가 나오는 이유, **log-sum-exp 수치 안정성** 트릭 |
| [06. Batched Computation과 Matrix 미분](./ch3-backpropagation/06-batched-matrix-backprop.md) | 배치 차원 $B$를 추가한 forward $Z = XW^T + b^T$의 gradient가 $\frac{\partial L}{\partial W} = \Delta^T X$ (= $\sum_b \delta_b x_b^T$)임을 유도, **행렬 미분의 trace trick** $d(\text{tr}(AB)) = \text{tr}(B\,dA) = \text{tr}(A\,dB)$, 실전 GPU 구현에서 BLAS 행렬 곱으로의 매핑 |

</details>

<br/>

### 🔹 Chapter 4: 초기화 이론

> **핵심 질문:** 왜 가중치를 0으로 초기화하면 실패하는가? Xavier와 He 초기화의 분산 공식은 어떤 가정에서 유도되는가? LSUV·Fixup 같은 현대 초기화는 왜 필요한가?

<details>
<summary><b>Symmetry Breaking부터 Fixup까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 초기화의 중요성과 Symmetry Breaking](./ch4-initialization/01-symmetry-breaking.md) | **$W = 0$ 초기화가 실패하는 이유**: 모든 hidden unit이 **같은 gradient**를 받아 업데이트 후에도 동일 뉴런으로 남음(대칭성), "effective width = 1" 증명, 작은 랜덤 값의 필요성, 분산이 너무 크면 saturation, 너무 작으면 gradient vanishing — **Goldilocks zone** |
| [02. Xavier/Glorot 초기화의 유도](./ch4-initialization/02-xavier-derivation.md) | Linear 활성화 가정에서 **forward 분산 방정식** $\text{Var}(y) = n_{\text{in}} \text{Var}(w) \text{Var}(x)$, 층별 보존 조건 $\text{Var}(w) = 1/n_{\text{in}}$, **backward 분산 방정식** $\text{Var}(\delta_l) = n_{\text{out}} \text{Var}(w) \text{Var}(\delta_{l+1})$, 두 조건의 타협으로 **$\text{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$** 유도 (Glorot & Bengio 2010) |
| [03. He/Kaiming 초기화 유도 (He et al. 2015)](./ch4-initialization/03-he-derivation.md) | **ReLU가 반쪽만 통과**시키므로 $\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2}\mathbb{E}[z^2]$ (정규 분포 가정), 분산이 **절반으로 감소** → 보정으로 $\text{Var}(w) = 2/n_{\text{in}}$, 이것이 **각 층에서 activation variance를 정확히 보존**함을 증명, 30-layer MLP에서 Xavier vs He의 분산 전파 실험 |
| [04. LSUV와 Orthogonal Initialization](./ch4-initialization/04-lsuv-orthogonal.md) | **Layer-Sequential Unit-Variance (LSUV)**: 한 층씩 forward 후 activation의 실측 분산으로 $W$ 스케일 조정(Mishkin & Matas 2015), **Orthogonal initialization**: RNN에서 $W_{hh}$를 직교 행렬로 — **spectral norm이 정확히 1** → gradient가 폭발/소멸하지 않음(Saxe et al. 2014), dynamical isometry |
| [05. Fixup Initialization — BN 없이 깊은 ResNet](./ch4-initialization/05-fixup-initialization.md) | **Fixup**(Zhang et al. 2019): ResNet에서 BatchNorm 없이 학습 가능한 초기화. 각 residual block의 마지막 conv를 **0으로 초기화** + 첫 conv를 $L^{-1/(2m-2)}$로 scale down, 이것이 **초기 variance explosion 방지**하는 이유, Figure of depth $\sim 10{,}000$ 훈련 가능성 |

</details>

<br/>

### 🔹 Chapter 5: Convolutional Neural Networks의 이론

> **핵심 질문:** CNN은 왜 이미지 처리에 최적인가? Translation equivariance가 수학적으로 무엇을 의미하는가? 파라미터 공유는 왜 VC 차원을 줄이는가?

<details>
<summary><b>Convolution의 수학부터 CNN 아키텍처까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Convolution의 수학과 Translation Equivariance](./ch5-cnn/01-convolution-equivariance.md) | Convolution $(f*g)(x) = \int f(y) g(x-y) dy$의 엄밀 정의, **이산 convolution** $(f*g)[n] = \sum_m f[m] g[n-m]$ (cross-correlation과의 차이), **정리**: CNN 층 $\phi$는 **translation equivariant** $\phi(T_s x) = T_s \phi(x)$, 이것이 이미지의 shift 불변 특성과 맞는 이유 |
| [02. 파라미터 공유와 VC 이론적 효율](./ch5-cnn/02-parameter-sharing-vc.md) | Fully-connected layer 파라미터 $O(H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}})$ vs CNN $O(k^2 \cdot C_{\text{in}} \cdot C_{\text{out}})$ ($k$ = kernel size), **파라미터 감소 비율** $\sim (k/(HW))^2$, VC 차원 감소 $\to$ **일반화 오차 경계 개선**, 동일 **receptive field**의 효율적 학습 |
| [03. Pooling과 Local Invariance](./ch5-cnn/03-pooling-invariance.md) | **Max/Average pooling**이 제공하는 **local translation invariance**, **stride**와 **dilation**의 receptive field 확장 ($\text{RF}_l = \text{RF}_{l-1} + (k-1)\prod_j s_j$), pooling vs strided convolution의 trade-off, **attention pooling**으로의 일반화 |
| [04. CNN 아키텍처 이론 — LeNet → ResNet → EfficientNet](./ch5-cnn/04-cnn-architectures.md) | LeNet(1998) → AlexNet(2012) → VGG → **ResNet**(skip connection) → **DenseNet** → **EfficientNet**(compound scaling, Tan & Le 2019)의 설계 원리, **depth vs width vs resolution** trade-off, GNN(Graph Neural Network)과의 일반화 관계 |

</details>

<br/>

### 🔹 Chapter 6: RNN과 Sequence Model의 이론

> **핵심 질문:** RNN의 BPTT는 어떻게 유도되는가? Vanishing/exploding gradient는 왜 발생하고 LSTM의 gating은 어떻게 이를 완화하는가? Echo State Network의 "에코 성질"은 무엇인가?

<details>
<summary><b>RNN의 정의부터 Reservoir Computing까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. RNN의 정의와 BPTT 유도](./ch6-rnn/01-rnn-bptt.md) | $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)$, $y_t = W_{hy} h_t$ 정의, **Backpropagation Through Time (BPTT)**을 unrolled computational graph에서 유도, $\frac{\partial L}{\partial W_{hh}} = \sum_t \frac{\partial L_t}{\partial h_t} \sum_{k \leq t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$ 형태 |
| [02. Vanishing/Exploding Gradient의 수학적 분석](./ch6-rnn/02-vanishing-exploding.md) | **정리**(Pascanu et al. 2013): $\frac{\partial h_t}{\partial h_0} = \prod_{k=1}^t W_{hh}^T \text{diag}(\sigma'(z_k))$의 norm이 **스펙트럴 반지름 $\rho(W_{hh})$와 $\max \|\sigma'\|$의 곱**에 의해 **지수적으로 vanishing/exploding**, 증명: Jordan canonical form + eigenvalue argument |
| [03. LSTM과 GRU의 이론적 근거](./ch6-rnn/03-lstm-gru-theory.md) | **LSTM cell state** $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$는 **additive update**(multiplicative 대비) → **Constant Error Carousel**: gradient가 $\prod_t f_t$로 전파, $f_t \approx 1$이면 vanishing 완화, **GRU**의 reset/update gate가 LSTM을 단순화하면서도 유사한 성질 보존함을 증명 |
| [04. Echo State Network과 Reservoir Computing](./ch6-rnn/04-echo-state-network.md) | 무작위 $W_{hh}$ 고정 + **output layer $W_{hy}$만 학습**(ridge regression), **Echo State Property (ESP)**: 초기 상태 의존성이 시간에 따라 사라지는 조건, 충분조건 $\rho(W_{hh}) < 1$ (Jaeger 2001), 컴퓨팅 비용 없이 훈련, 신경과학적 연결(liquid state machine) |

</details>

<br/>

### 🔹 Chapter 7: Transformer와 Attention의 이론

> **핵심 질문:** Self-attention은 수학적으로 무엇을 계산하는가? 왜 $\sqrt{d_k}$로 나누는가? ResNet의 residual connection이 왜 gradient highway인가? Transformer는 어떤 함수 클래스를 근사할 수 있는가?

<details>
<summary><b>Self-Attention부터 Transformer UAT까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Self-Attention의 수학과 $\sqrt{d_k}$ 스케일링](./ch7-transformer/01-self-attention.md) | $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$ 정의, 각 query가 key들과의 **scaled dot-product 유사도**로 value를 가중합, **왜 $\sqrt{d_k}$?**: $Q, K \sim \mathcal{N}(0, 1)$ 가정 시 $(QK^T)_{ij} \sim \mathcal{N}(0, d_k)$ → 정규화로 softmax 포화(argmax 같은 행동) 방지 증명 |
| [02. Multi-Head Attention과 표현력](./ch7-transformer/02-multi-head-attention.md) | $h$개의 head가 **서로 다른 subspace에서 attention**: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$, 각 head의 $d_{\text{model}}/h$ 차원에서 local 패턴 포착, **Michel et al. (2019)**의 head pruning 실험에서 다양성의 중요성 — 왜 single-head가 표현력 부족인가 |
| [03. Positional Encoding의 필요성과 설계](./ch7-transformer/03-positional-encoding.md) | Self-attention의 **permutation-invariance** 증명 ($\text{Attn}(P X) = P \text{Attn}(X)$) → 위치 정보가 **반드시** 주입되어야 함, **Sinusoidal PE** $\text{PE}(p, 2i) = \sin(p / 10000^{2i/d})$의 수학적 성질(상대 위치 선형 표현), **Learned PE vs RoPE vs ALiBi**의 비교 |
| [04. Transformer의 표현력 — Universal Sequence Function (Yun et al. 2020)](./ch7-transformer/04-transformer-uat.md) | **정리**: Transformer가 **임의 continuous seq-to-seq 함수**를 uniformly approximate 가능 — multi-head attention + FFN + residual의 조합이 충분조건, **depth vs head 수의 trade-off** (더 많은 head로 더 얕은 표현 가능), compact domain에서의 dense 증명 |
| [05. ResNet과 Residual Connection의 Gradient Flow](./ch7-transformer/05-resnet-gradient-flow.md) | $y = x + F(x)$에서 **$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}(I + \frac{\partial F}{\partial x})$** — identity 항 덕분에 gradient가 **절대 0이 되지 않음**, plain network의 $\prod \frac{\partial F_l}{\partial h_{l-1}}$와 비교하여 **gradient vanishing 완화** 정량 분석, "gradient highway" 수학적 의미, DenseNet과 Highway Net로의 일반화 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 72개 정리 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **Novikoff 수렴 정리** | 선형 분리 margin $\gamma$ + 반지름 $R$의 데이터에서 퍼셉트론 알고리즘은 **$(R/\gamma)^2$ 번 이내** 수렴 | [Ch1-01](./ch1-perceptron-to-mlp/01-perceptron-convergence.md) |
| **XOR 선형 분리 불가능성** | 단일 퍼셉트론은 XOR 함수를 표현할 수 없다 — 선형 분리 불가 증명 | [Ch1-02](./ch1-perceptron-to-mlp/02-xor-and-single-layer.md) |
| **Cybenko UAT (1989)** | 연속 sigmoidal $\sigma$, 1-hidden-layer NN은 $C(K)$에서 uniformly dense — Hahn-Banach + Riesz | [Ch2-01](./ch2-universal-approximation/01-cybenko-uat.md) |
| **Hornik UAT (1991)** | 임의 non-polynomial 활성화로 universal — Stone-Weierstrass 기반 | [Ch2-02](./ch2-universal-approximation/02-hornik-extension.md) |
| **Leshno ReLU UAT (1993)** | ReLU 활성화로도 universal — piecewise linear 근사 구성 | [Ch2-03](./ch2-universal-approximation/03-leshno-relu-uat.md) |
| **Telgarsky Depth Separation (2016)** | 깊이 $L$ 표현 함수는 얕은 NN에 **지수적 width**를 요구 — sawtooth oscillation | [Ch2-04](./ch2-universal-approximation/04-depth-vs-width.md) |
| **Barron 근사율 (1993)** | 유한 Fourier 모멘트 함수는 $O(1/\sqrt{n})$로 근사 가능 — curse of dimensionality 회피 | [Ch2-05](./ch2-universal-approximation/05-barron-rate.md) |
| **Reverse-mode AD 복잡도** | $f: \mathbb{R}^n \to \mathbb{R}^m$, reverse-mode $O(m T)$ vs forward-mode $O(n T)$ | [Ch3-03](./ch3-backpropagation/03-reverse-mode-backprop.md) |
| **Backpropagation 공식** | $\delta_l = (W_{l+1}^T \delta_{l+1}) \odot \sigma'(z_l)$, $\partial L/\partial W_l = \delta_l a_{l-1}^T$ | [Ch3-04](./ch3-backpropagation/04-mlp-backprop-formula.md) |
| **Softmax-CE Gradient** | $\partial L/\partial z_i = \hat y_i - y_i$ (natural parameterization) | [Ch3-05](./ch3-backpropagation/05-softmax-crossentropy-grad.md) |
| **Xavier 분산 공식** | $\text{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$ — forward·backward 분산 보존 타협 | [Ch4-02](./ch4-initialization/02-xavier-derivation.md) |
| **He 분산 공식** | ReLU에서 $\text{Var}(w) = 2/n_{\text{in}}$ — 반쪽 통과로 인한 분산 반감 보정 | [Ch4-03](./ch4-initialization/03-he-derivation.md) |
| **Translation Equivariance** | CNN 층 $\phi$는 $\phi(T_s x) = T_s \phi(x)$ — shift 입력 → shift 출력 | [Ch5-01](./ch5-cnn/01-convolution-equivariance.md) |
| **Vanishing/Exploding Gradient** | $\|\partial h_t/\partial h_0\|$는 $\rho(W_{hh})^t$로 지수 증감 (Pascanu 2013) | [Ch6-02](./ch6-rnn/02-vanishing-exploding.md) |
| **LSTM Constant Error Carousel** | Additive cell update $c_t = f_t c_{t-1} + \ldots$로 gradient $\prod f_t$ 보존 | [Ch6-03](./ch6-rnn/03-lstm-gru-theory.md) |
| **$\sqrt{d_k}$ 스케일링의 필요성** | $QK^T$의 분산 $\sim d_k$ → 정규화로 softmax 포화 방지 | [Ch7-01](./ch7-transformer/01-self-attention.md) |
| **Transformer UAT (Yun 2020)** | Transformer는 임의 연속 seq-to-seq 함수를 uniformly 근사 | [Ch7-04](./ch7-transformer/04-transformer-uat.md) |
| **ResNet Gradient Highway** | $\partial L/\partial x = \partial L/\partial y (I + \partial F/\partial x)$ — identity로 vanishing 완화 | [Ch7-05](./ch7-transformer/05-resnet-gradient-flow.md) |

> 💡 **챕터별 총 정리 수**: Ch1(8) · Ch2(15) · Ch3(14) · Ch4(11) · Ch5(7) · Ch6(8) · Ch7(9) — 합계 **72개 정리 + 증명**, 약 **16,000+ 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다. 핵심 철학: **autograd 없이 NumPy로 바닥부터 구현**, PyTorch는 검증용.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
tqdm==4.66.0
torch==2.1.0         # 검증용 (NumPy 결과와 수치 비교)
torchvision==0.16.0  # CNN 실험 데이터 (MNIST/CIFAR)
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            tqdm==4.66.0 torch==2.1.0 torchvision==0.16.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — 바닥부터 MLP + 역전파 + 초기화 분산 전파 검증
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    """autograd 없이 직접 구현한 MLP — 역전파 공식의 수식 그대로"""
    def __init__(self, layers, init='he'):
        self.W, self.b = [], []
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            if init == 'xavier':
                W = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
            elif init == 'he':
                W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            elif init == 'zero':
                W = np.zeros((n_out, n_in))
            self.W.append(W); self.b.append(np.zeros(n_out))

    def forward(self, x):
        self.a = [x]; self.z = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ self.a[-1] + b
            a = np.maximum(0, z) if i < len(self.W) - 1 else z   # ReLU + linear output
            self.z.append(z); self.a.append(a)
        return self.a[-1]

    def backward(self, grad_output):
        """Reverse-mode AD 직접 구현 — δ_l = (W_{l+1}^T δ_{l+1}) ⊙ σ'(z_l)"""
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        delta = grad_output
        for l in reversed(range(len(self.W))):
            if l < len(self.W) - 1:
                delta = delta * (self.z[l] > 0)          # ReLU derivative
            grads_W[l] = np.outer(delta, self.a[l])
            grads_b[l] = delta
            delta = self.W[l].T @ delta
        return grads_W, grads_b

# ──────────────────────────────────────────────────────────
# 실험 1. 초기화별 activation variance 전파 (He가 보존해야 함)
# ──────────────────────────────────────────────────────────
def measure_layer_variance(init_method, depth=30, width=256, n_paths=1000):
    mlp = MLP([width] * depth, init=init_method)
    variances = []
    for _ in range(n_paths):
        x = np.random.randn(width)
        mlp.forward(x)
        variances.append([np.var(a) for a in mlp.a[1:]])
    return np.mean(variances, axis=0)   # 층별 평균 분산

plt.figure(figsize=(10, 5))
for init in ['xavier', 'he', 'zero']:
    vars_per_layer = measure_layer_variance(init)
    plt.semilogy(range(1, 31), vars_per_layer, 'o-', label=init, markersize=6)
plt.axhline(1.0, linestyle='--', color='k', alpha=0.5, label='target Var=1')
plt.xlabel('Layer depth'); plt.ylabel('Activation variance (log scale)')
plt.title('ReLU MLP: 초기화별 분산 전파 — He는 상수, Xavier는 감소, Zero는 0')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
# 결과: He는 ≈ 1 유지, Xavier는 0.5^depth로 감소, Zero는 정확히 0

# ──────────────────────────────────────────────────────────
# 실험 2. Gradient flow — ResNet vs Plain (깊이별 gradient norm)
# ──────────────────────────────────────────────────────────
# Plain: y = F(x),  ResNet: y = x + F(x)
# 깊이 L이 커질수록 plain은 ‖∂L/∂x_0‖이 지수 감소, ResNet은 보존
# → ResNet의 "gradient highway" 수치적 증거

# ──────────────────────────────────────────────────────────
# 실험 3. PyTorch 검증 — 직접 구현한 backward와 autograd 비교
# ──────────────────────────────────────────────────────────
# 동일 가중치·입력에서 numpy 버전과 torch.autograd 결과를 allclose로 비교
# → 역전파 공식이 정확히 구현되었는지 검증
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 이론이 현대 딥러닝에 필수인가** | PyTorch/TF 구현, Transformer, Diffusion과의 연결점 |
| 3 | 📐 **수학적 선행 조건** | LA, Calc, Prob, FA 레포의 어떤 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | 역전파의 chain rule, 분산 보존의 의미, attention의 soft lookup |
| 5 | ✏️ **엄밀한 정의** | UAT·Jacobian·초기화 분산 방정식의 수학적 정의 |
| 6 | 🔬 **정리와 증명** | Cybenko UAT, backprop formula, He 초기화 유도 — "자명하다" 없이 |
| 7 | 💻 **NumPy로 바닥부터 구현** | autograd 없이 forward/backward 직접, PyTorch 결과와 비교 |
| 8 | 🔗 **실전 연결** | PyTorch/TF 구현 주의점, 왜 이 설계가 선택되었는가 |
| 9 | ⚖️ **가정과 한계** | UAT는 width unlimited, 초기화는 독립 입력 가정 등 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현 문제 |

> 📚 **연습문제 총 99개**: 33문서 × 문서당 3문제(기초/심화/AI 연결), 모든 문제에 `<details>` 펼침 해설 포함. 손 계산 재현부터 Transformer/Diffusion 연결까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결되므로 순차 학습이 끊기지 않습니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 490줄(증명·코드·연습문제 포함) 기준 **약 1~1.5시간**. 전체 33문서는 약 **40~50시간** 상당.

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "PyTorch는 쓰지만 역전파·초기화를 수식으로 설명 못한다" — Backprop+Init 집중 (5일, 약 10~13시간)</b></summary>

<br/>

```
Day 1  Ch1-03, Ch1-04  MLP 정의와 활성화 함수
Day 2  Ch3-01, Ch3-02  Jacobian 표기, Computational graph와 AD
Day 3  Ch3-03, Ch3-04  Reverse-mode AD = Backprop, MLP 역전파 공식 유도
Day 4  Ch3-05, Ch3-06  Softmax-CE gradient, Batched 행렬 미분
Day 5  Ch4-01, Ch4-02, Ch4-03  Symmetry breaking, Xavier, He 유도
        → NumPy로 30-layer MLP 분산 전파 실험 재현
```

</details>

<details>
<summary><b>🟡 "Transformer를 쓰지만 수학적 기반을 모른다" — Attention·ResNet 집중 (1주, 약 12~15시간)</b></summary>

<br/>

```
Day 1  Ch3-01, Ch3-04  Jacobian 표기와 역전파 공식 (선행)
Day 2  Ch4-01, Ch4-03  초기화 기본과 He 공식
Day 3  Ch6-01, Ch6-02  RNN·BPTT와 vanishing gradient
Day 4  Ch7-05  ResNet의 gradient highway
Day 5  Ch7-01  Self-attention과 √d_k 스케일링
Day 6  Ch7-02, Ch7-03  Multi-head, Positional encoding
Day 7  Ch7-04  Transformer UAT (Yun 2020)
```

</details>

<details>
<summary><b>🔴 "신경망의 수학적 기반을 완전 정복한다" — 전체 정복 (8주, 약 40~50시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 퍼셉트론에서 MLP까지
        → Novikoff 수렴 증명을 손으로 재구성
        → XOR 분리 불가 + MLP로 표현하는 구성 이해

2주차  Chapter 2 전체 — Universal Approximation
        → Cybenko 증명의 Hahn-Banach → Riesz 흐름 숙지
        → Hornik/Leshno의 일반화 차이 파악
        → Telgarsky depth separation sawtooth argument

3주차  Chapter 3 (1~3) — AD와 reverse-mode
        → forward vs reverse 복잡도 직접 계산
        → 간단한 함수의 computational graph 그리고 VJP 수행

4주차  Chapter 3 (4~6) — MLP backprop 구현
        → NumPy로 MLP forward/backward 직접 구현
        → PyTorch autograd와 gradient allclose 검증
        → Softmax-CE의 단순한 gradient 유도 재현

5주차  Chapter 4 전체 — 초기화
        → Xavier·He 분산 방정식 한 줄씩 유도
        → 30-layer ReLU MLP 분산 전파 실험 (zero/Xavier/He 비교)
        → LSUV 구현, Fixup으로 BN 없이 100-layer ResNet

6주차  Chapter 5 전체 — CNN
        → Convolution equivariance 증명
        → MNIST/CIFAR에서 파라미터 공유의 효율 측정
        → ResNet·DenseNet의 receptive field 비교

7주차  Chapter 6 전체 — RNN·LSTM
        → BPTT 유도와 vanishing gradient 실측
        → LSTM의 CEC로 gradient 보존 실험
        → Echo State Network 미니 구현

8주차  Chapter 7 전체 — Transformer·ResNet
        → Self-attention NumPy 구현
        → √d_k 스케일링 유무에 따른 softmax 포화 실험
        → Multi-head와 positional encoding의 표현력 시각화
        → Transformer UAT 논문 읽기
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 행렬·벡터 미분, SVD, 스펙트럴 분해, 고유값 | Ch3 전체(Jacobian·Kronecker), Ch6-02(스펙트럴 반지름), Ch4-04(orthogonal init) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 다변수 미분, Taylor 전개, Hessian | Ch3 전체(연쇄법칙), Ch4-02~03(분산 방정식) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 분산, 독립, 모멘트, 극한 정리 | Ch4 전체(초기화 분산), Ch2-05(Barron 근사율) |
| [functional-analysis-deep-dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) | Hahn-Banach, Riesz 표현, Stone-Weierstrass | Ch2-01~02(UAT 증명) |
| [ml-fundamentals-deep-dive](https://github.com/iq-ai-lab/ml-fundamentals-deep-dive) | 선형회귀, 로지스틱, 경사하강 | Ch1-01(퍼셉트론의 일반화), Ch3-05(cross-entropy) |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | SGD·Adam 수렴, loss landscape | **후속 레포** — 이 레포의 역전파로 구한 gradient를 쓰는 최적화기 |
| [generalization-theory-deep-dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive) | VC, Rademacher, norm-based bound | **후속 레포** — Ch5-02(CNN의 VC 감소)의 심화 |
| [sde-deep-dive](https://github.com/iq-ai-lab/sde-deep-dive) | 이토 적분, Anderson reverse SDE, DDPM | **후속 레포** — NN을 score network로 사용하는 Diffusion 모델 |

> 💡 이 레포는 **신경망의 표현력·학습 알고리즘·초기화·아키텍처**에 집중합니다. Linear Algebra에서 벡터-행렬 미분을, Calculus에서 Taylor 전개와 Jacobian을, Probability에서 분산의 독립성을 학습한 후 오면 Ch3(backprop)과 Ch4(초기화)가 훨씬 자연스럽습니다. Ch7(Transformer)은 Ch3·Ch4의 선수가 된 후에 시작하세요.

---

## 📖 Reference

### 🏛️ 딥러닝 바이블·표준 교재
- **Deep Learning** (Goodfellow, Bengio, Courville, 2016) — 현대 DL 교과서 표준
- **Neural Networks and Deep Learning** (Nielsen, 2015, 무료 온라인) — backprop 고전 설명
- **Pattern Recognition and Machine Learning** (Bishop, 2006) — NN의 Bayesian 관점
- **Dive into Deep Learning** (Zhang, Lipton, Li, Smola, 2023) — 실습 병행 현대 교재

### 🎓 Universal Approximation Theorem
- **Approximation by Superpositions of a Sigmoidal Function** (Cybenko, 1989) — **UAT 원전**
- **Multilayer Feedforward Networks are Universal Approximators** (Hornik, 1991) — Stone-Weierstrass 확장
- **Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function** (Leshno et al., 1993) — ReLU UAT
- **Universal Approximation Bounds for Superpositions of a Sigmoidal Function** (Barron, 1993) — 근사율
- **Benefits of Depth in Neural Networks** (Telgarsky, 2016) — Depth separation

### 🔁 Backpropagation & AD
- **Learning Representations by Back-Propagating Errors** (Rumelhart, Hinton, Williams, 1986) — Backprop 재발견
- **Automatic Differentiation in Machine Learning: a Survey** (Baydin et al., 2018) — AD 종합 리뷰
- **Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation** (Griewank & Walther, 2008) — AD 교과서

### ⚙️ 초기화 이론
- **Understanding the Difficulty of Training Deep Feedforward Neural Networks** (Glorot & Bengio, 2010) — **Xavier init 원전**
- **Delving Deep into Rectifiers** (He, Zhang, Ren, Sun, 2015) — **He init 원전**
- **All You Need Is a Good Init** (Mishkin & Matas, 2015) — **LSUV**
- **Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks** (Saxe, McClelland, Ganguli, 2014) — Orthogonal init
- **Fixup Initialization: Residual Learning Without Normalization** (Zhang, Dauphin, Ma, 2019) — Fixup

### 🖼️ CNN
- **Gradient-Based Learning Applied to Document Recognition** (LeCun et al., 1998) — LeNet
- **ImageNet Classification with Deep Convolutional Neural Networks** (Krizhevsky, Sutskever, Hinton, 2012) — AlexNet
- **Deep Residual Learning for Image Recognition** (He, Zhang, Ren, Sun, 2016) — **ResNet 원전**
- **Densely Connected Convolutional Networks** (Huang et al., 2017) — DenseNet
- **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** (Tan & Le, 2019)

### 🔁 RNN·LSTM
- **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997) — **LSTM 원전**
- **On the Difficulty of Training Recurrent Neural Networks** (Pascanu, Mikolov, Bengio, 2013) — Vanishing gradient 분석
- **The "Echo State" Approach to Analysing and Training Recurrent Neural Networks** (Jaeger, 2001) — ESN
- **Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling** (Chung et al., 2014) — GRU

### 🤖 Transformer·Attention
- **Attention Is All You Need** (Vaswani et al., 2017) — **Transformer 원전**
- **Are Sixteen Heads Really Better than One?** (Michel, Levy, Neubig, 2019) — Multi-head 분석
- **Are Transformers Universal Approximators of Sequence-to-Sequence Functions?** (Yun et al., 2020) — Transformer UAT
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) — RoPE
- **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** (Press, Smith, Lewis, 2022) — ALiBi

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`nn.Linear`를 쌓는 것과 — 왜 1-hidden-layer sigmoid NN이 $C(\mathbb{R}^n)$에서 dense하고, reverse-mode AD가 forward-mode보다 파라미터 많은 NN에서 지수적으로 효율적이며, He 초기화의 $\text{Var}(W) = 2/n_{\text{in}}$이 각 층의 activation 분산을 정확히 보존하는지를 증명할 수 있는 것은 다르다"*

</div>
