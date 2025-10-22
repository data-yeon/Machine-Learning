# 로지스틱 회귀 (Logistic Regression)

로지스틱 회귀(Logistic Regression)는 **이진 분류(binary classification)** 문제를 해결하기 위한 통계적 모델이다.  
이때 종속 변수 \( y \)는 두 가지 범주 중 하나의 값만을 가진다.

일반적으로 두 범주는 다음과 같이 정의된다.

-   참(True) → `1`
-   거짓(False) → `0`

각 범주는 각각 **양성 클래스(positive class)** 와 **음성 클래스(negative class)** 로 구분된다.

---

비록 ‘회귀(regression)’라는 명칭을 사용하지만,  
로지스틱 회귀는 실제로 **분류(classification)** 문제를 다루는 모델이다.  
‘회귀’라는 용어는 통계학에서 유래된 역사적 명칭으로,  
모델의 목적이 연속형 값의 예측이 아닌, 범주형 결과의 분류임을 유의해야 한다.

---

**요약:**  
로지스틱 회귀는 종속 변수가 0 또는 1의 두 가지 범주 중 하나로 구분되는 경우에 사용되는  
이진 분류 기반 통계 모델이다.

![1_dm6ZaX5fuSmuVvM4Ds-vcg.jpg](Logistic%20Regression%2029227cc9c28281cf9d17d1d37dc05c25/1_dm6ZaX5fuSmuVvM4Ds-vcg.jpg)

# 모델의 정의 (Defining the Model)

로지스틱 회귀에서 사용하는 핵심 함수는 **시그모이드(sigmoid) 함수**이다.  
이 함수는 입력값 전체를 0과 1 사이의 값으로 변환하며,  
S자 형태의 곡선으로 표현된다.

모든 분류 문제에는 **임계값(threshold)** 이 존재하며,  
이 값은 입력 데이터가 양성(positive) 또는 음성(negative)으로 분류되는 경계를 결정한다.

---

데이터는 일반적으로 입력 공간의 중심이 0이 되도록 **정규화(normalization)** 된다.  
정규화된 입력값은 \( z \)로 표기하며, 이는 양수와 음수 값을 모두 가질 수 있다.  
이 과정을 통해 시그모이드 함수의 출력값이 항상 0과 1 사이에 위치하도록 한다.

---

시그모이드 함수는 단순히 곡선 형태를 가지는 것뿐만 아니라,  
특정 입력값이 양성일 확률 \( P \)을 나타내는 **확률 함수(probability function)** 로도 해석된다.  
이는 로지스틱 회귀가 분류 문제에서 널리 사용되는 이유이기도 하다.

정규화된 입력값을 기준으로 할 때,  
시그모이드 함수는 다음과 같이 정의된다.

$$
g(z) = \frac{1}{1+e^{-z}} = 1 - P(-z)
$$

이때, 함수의 출력값은 항상 다음의 범위 내에 존재한다.

$$
0 \lt g(z) \lt 1
$$

확률 \( P \)는 다음과 같은 로그 형태로도 표현할 수 있다.

$$
g(z) = log_e \left( \frac{P}{1-P} \right)
$$

선형 방정식(linear equation)은 다음과 같이 정의된다. $f_{\vec w,b} (\vec x) = \vec{w} \cdot \vec{x} + b$. 선형 함수와 시그모이드 함수는 형태적으로는 다르지만,  
모두 동일한 결정 경계(decision boundary)를 표현한다.  
즉, 두 함수는 서로 다른 형태를 가지더라도  
입력 데이터에 대해 동일한 모델 구조를 나타낸다.

다음 그림은 이러한 관계를 시각적으로 보여준다.

![Untitled](Logistic%20Regression%2029227cc9c28281cf9d17d1d37dc05c25/Untitled.png)

선형 함수의 결과값을 \( z \)로 정의하면 다음과 같다.

$$
z = \vec w \cdot \vec x + b
$$

이를 시그모이드 함수에 대입하면, 로지스틱 회귀의 기본 형태를 얻을 수 있다.

$$
g(z) = \frac{1}{1+e^{-z}} = \vec w \cdot \vec x + b
$$

즉, $z$ and $\vec w \cdot \vec x + b$ 는 동일한 입력 변수를 나타내므로,  
상호 대체가 가능하다.  
이를 통해 로지스틱 회귀(logistic regression)의 최종 표현식은  
시그모이드 함수를 선형 방정식에 결합한 형태로 정리된다.

### 로지스틱 회귀의 최종 식 (Final Logistic Regression Algorithm)

로지스틱 회귀의 최종 형태는 다음과 같이 정의된다.

$$
f_{\vec w,b} (\vec x) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{-(\vec w \cdot \vec x + b)}}
$$

이 식은 선형 결합 $ (\vec w \cdot \vec x + b) $ 을 시그모이드 함수 $ g(z) $ 의 입력으로 사용하는 구조를 갖는다.  
이를 통해 입력값이 어떤 경우에도 출력값은 항상 0과 1 사이의 확률 값으로 제한된다.

---

함수가 **수평축의 원점(0)** 을 기준으로 정규화(normalization)되어야 하는 이유와,  
시그모이드 함수가 실제로 어떻게 도출되는지는 직관적으로 이해하기 어렵다.  
아래의 참고 자료에서는 이러한 개념적 과정을 시각적으로 설명하고 있다.

[시그모이드 함수의 도출 및 정규화 설명 영상 (YouTube)](https://www.youtube.com/watch?v=rrnfHG_wtII)

## 출력값의 해석 (Interpreting the Output)

로지스틱 회귀 알고리즘은 실제로 0 또는 1의 값을 **정확히 예측할 수 없다.**  
따라서 모델의 출력값은 **확률(probability)** 로 해석된다.

즉, 알고리즘의 출력 $f_{\vec w, b}(\vec x)$ 는  
입력 $\vec x$ 에 대해 $y = 1$일 확률을 의미한다.  
반대로, $1 - f_{\vec w, b}(\vec x)$ 는 $y = 0$일 확률을 나타낸다.

이를 수식으로 표현하면 다음과 같다.

$$
P(y = 0) + P(y = 1) = 1
$$

---

연구 논문 및 학술 자료에서는 이러한 관계를 **보다 엄밀한 확률 표기법(probabilistic notation)** 으로 나타낸다.  
즉, 입력 $\vec x$ 와 모델 매개변수 $\vec w, b$ 가 주어졌을 때,  
$y = 1$일 조건부 확률은 다음과 같이 정의된다.

$$
f_{\vec w, b}(\vec x) = P(y = 1 \mid \vec x; \vec w, b)
$$

---

따라서 로지스틱 회귀의 출력은 단순한 이진 예측이 아니라,  
**입력 데이터가 양성 클래스(positive class)에 속할 확률을 추정하는 모델**로 해석할 수 있다.

# 결정 경계 (The Decision Boundary)

로지스틱 회귀에서 **결정 경계(decision boundary)** 는 반드시 $y$ 축 절편이 0.5에 고정될 필요는 없다.  
응용 목적에 따라 **임계값(threshold)** 을 조정하는 것은 일반적인 절차이다.

예를 들어, 종양(tumor) 탐지 알고리즘을 고려하자.  
이 경우 $g(z) \ge threshold$ 인 데이터는 종양이 있을 가능성이 있는 것으로 판단한다.  
종양을 놓치지 않기 위해서는 임계값을 낮추는 것이 바람직하며,  
이는 일부 오탐(false positive)을 감수하더라도 실제 종양을 검출하기 위한 조정이라 할 수 있다.  
이처럼 임계값 조정은 **문제의 성격과 오차 비용의 균형**에 따라 결정된다.

---

결정 결과는 $\hat y$ 로 표현되며, 다음과 같이 정의된다.

$$
\text{If } \ f_{\vec w, b}(\vec x) \ge threshold, \quad \hat y = 1 \\
\text{Otherwise, } \hat y = 0
$$

---

이제, 어떤 조건에서 이러한 두 가지 경우가 발생하는지를 살펴보자.

$$
\text{When } f_{\vec w, b}(\vec x) \ge threshold \\
g(z) \ge threshold \\
z \ge 0
$$

시그모이드 함수가 수평축의 원점(0)을 기준으로 정규화되어 있으므로,  
$z \ge 0$ 인 경우 모델은 **양성 클래스(positive class)** 로 분류한다.  
또한 $z = \vec w \cdot \vec x + b$ 이므로, 다음 관계가 성립한다.

$$
\vec w \cdot \vec x + b \ge 0 \quad \Rightarrow \quad \hat y = 1
$$

$$
\vec w \cdot \vec x + b < 0 \quad \Rightarrow \quad \hat y = 0
$$

---

$z = 0$ 인 경우는 결정 경계 자체이지만,  
이진 분류에서는 반드시 두 클래스 중 하나에 포함되어야 하므로  
일반적으로 **양성 클래스(positive class)** 로 포함시킨다.

이 논리는 **비선형(non-linear)** 또는 **다항(polynomial)** 모델에도 동일하게 적용된다.  
이 경우에도 모델의 목적은 단순히 함수가 0을 기준으로 경계를 형성하도록 학습하는 것이다.

# 비용 함수 (The Cost Function)

로지스틱 회귀(Logistic Regression)의 **비용 함수(cost function)** 는  
기본적으로 선형 회귀(Linear Regression)와 비슷한 구조를 가지고 있지만,  
선형 회귀에서 사용하던 **제곱 오차(Squared Error)** 를 그대로 사용할 수는 없다.

그 이유는 로지스틱 회귀의 출력이 **확률(0~1 사이의 값)** 이기 때문이다.  
제곱 오차를 그대로 사용할 경우,  
모델의 학습이 비선형적으로 꼬이게 되어 최적화가 제대로 이루어지지 않는다.  
따라서 로지스틱 회귀에서는 제곱 오차 대신,  
**로그 손실(Log Loss)** 또는 **이진 교차 엔트로피(Binary Cross-Entropy)** 를 사용한다.

---

### 🔹 선형 회귀의 비용 함수 복습

선형 회귀에서의 비용 함수는 다음과 같다.

$$
J(\vec w, b) = \frac{1}{2m} \sum_{i=1}^m \left( f_{\vec w,b}(\vec x^i) - y^i \right)^2
$$

여기서

-   \( f\_{\vec w,b}(\vec x^i) \): 예측값 (모델의 출력)
-   \( y^i \): 실제값
-   \( m \): 전체 데이터 개수

위 식은 **예측값과 실제값의 차이(오차)** 를 제곱한 뒤,  
모든 샘플에 대해 평균을 취한 형태이다.

---

### 🔹 로지스틱 회귀에서의 손실 함수

로지스틱 회귀의 경우, 각 데이터 샘플에 대해  
하나의 손실(loss)을 계산하는 함수를 \( L \)로 정의할 수 있다.

$$
L \left( f_{\vec w,b}(\vec x^i), y^i \right)
$$

이 식은 “예측값과 실제값이 얼마나 일치하는가”를 나타내며,  
이 손실이 작을수록 모델의 예측이 정확하다는 의미이다.

---

### 🔹 반단순화된 손실 함수 (Semi-simplified Loss Function)

로지스틱 회귀에서는 손실 함수를 다음과 같이 정의한다.

$$
L\left( f_{\vec w,b} (\vec x^i), y^i \right) =
\begin{cases}
    -\log \left( f_{\vec w,b}( \vec x^i ) \right), & \text{if } y^i = 1 \\
    -\log \left( 1 - f_{\vec w,b}( \vec x^i ) \right), & \text{if } y^i = 0
\end{cases}
$$

이 식의 의미는 다음과 같다.

-   실제 정답이 **1**인 경우, 예측값이 1에 가까울수록 손실이 작아진다.
-   실제 정답이 **0**인 경우, 예측값이 0에 가까울수록 손실이 작아진다.
-   반대로 예측이 틀릴수록 손실값은 커지며,  
    잘못된 방향(예: $y=1$인데 $f_{\vec w,b}(\vec x^i)$가 0에 가까운 경우)으로 갈수록  
    손실은 **무한대에 가까워진다.**

---

이러한 구조 덕분에,  
모델은 잘못된 예측을 할수록 강하게 “벌점(penalty)”을 받게 된다.  
따라서 학습 과정에서 모델은  
정답에 가까운 방향으로 가중치 \(\vec w\) 와 편향 \(b\)를 조정하게 된다.

---

### 🔹 손실 함수의 동작 예시

아래 그림은 예측값과 실제값의 차이에 따라  
손실 함수가 어떻게 변하는지를 시각적으로 보여준다.

![Loss Graph 1](Logistic%20Regression%2029227cc9c28281cf9d17d1d37dc05c25/Untitled%201.png)

![Loss Graph 2](Logistic%20Regression%2029227cc9c28281cf9d17d1d37dc05c25/Untitled%202.png)

---

### 🔹 핵심 요약

-   로지스틱 회귀의 손실 함수는 **로그 확률 기반(Log Loss)** 으로 구성된다.
-   예측이 정확할수록 손실은 작아지고,  
    예측이 틀릴수록 손실은 급격히 커진다.
-   이 특성 덕분에 모델은 **확률적으로 안정된 분류 경계(decision boundary)** 를 학습할 수 있다.

### 단순화된 손실 함수 (Simplified Loss Function)

로지스틱 회귀의 손실 함수는 다음과 같이 한 줄로 정리할 수 있다.

$$
L\left( f_{\vec w,b} (\vec x^i), y^i \right)
= -y^i \log \left(f_{\vec w,b} (\vec x^i)\right)
- (1 - y^i)\log \left(1 - f_{\vec w,b} (\vec x^i)\right)
$$

이 식은 다소 복잡해 보이지만, 실제로는 간단하다.  
왜냐하면 \( y \)는 **0 또는 1 중 하나의 값만** 가질 수 있기 때문이다.

즉,

-   \( y = 1 \)이면 첫 번째 항만 남고,
-   \( y = 0 \)이면 두 번째 항만 남는다.

따라서 위 식은 앞서 설명한 “반단순화된 손실 함수”와 같은 의미를 가진다.  
다만, 두 경우를 하나의 식으로 통합해 표현했기 때문에  
이 형태가 실제 계산에 더 자주 사용된다.

---

### 비용 함수와의 결합 (Combining with the Cost Function)

이제 이 손실 함수를 전체 데이터에 대해 평균 낸 형태로 확장하면  
**비용 함수(Cost Function)** 를 얻을 수 있다.  
기본 형태는 다음과 같다.

$$
J(\vec w, b) = \frac{1}{m} \sum_{i=1}^{m}
\left[
L\left( f_{\vec w,b}(\vec x^i), y^i \right)
\right]
$$

여기서 \( m \)은 전체 훈련 데이터의 개수이다.  
즉, 모든 데이터 샘플의 손실값을 더한 뒤 평균을 낸 것이 전체 비용이다.

---

### 최종 비용 함수 (Final Cost Function)

손실 함수의 정의를 위 식에 대입하고 정리하면,  
로지스틱 회귀의 최종 비용 함수는 다음과 같이 표현된다.

$$
J(\vec w,b) =
-\frac{1}{m} \sum_{i=1}^m
\left[
y^i \log \left(f_{\vec w,b}(\vec x^i)\right)
+ (1 - y^i) \log \left(1 - f_{\vec w,b}(\vec x^i)\right)
\right]
$$

이 식은 **최대우도추정(Maximum Likelihood Estimation)** 원리로부터 유도된 결과이며,  
통계적으로 매우 안정적인 특성을 가진다.  
또한 이 비용 함수는 **볼록(convex)** 형태를 가지므로  
하나의 전역 최소값(global minimum)만 존재한다.  
따라서 **경사 하강법(Gradient Descent)** 을 적용하기에 매우 적합하다.

---

# 로지스틱 회귀의 경사 하강법 (Gradient Descent for Logistic Regression)

경사 하강법은 모델의 매개변수 $ \vec{w}, b $ 를  
비용 함수 $ J(\vec{w}, b) $ 를 최소화하는 방향으로 반복적으로 갱신하는 알고리즘이다.

기본 식은 다음과 같다.

$$
w = w - \alpha \frac{\partial}{\partial w} J(\vec w, b)
$$

$$
b = b - \alpha \frac{\partial}{\partial b} J(\vec w, b)
$$

여기서 $ \alpha $ 는 학습률(learning rate)로,  
한 번의 반복에서 얼마나 크게 매개변수를 이동시킬지를 결정한다.

---

### 비용 함수의 편미분 (Gradients of the Cost Function)

비용 함수를 $ w $ 와 $ b $ 에 대해 편미분하면 다음과 같다.

$$
\frac{\partial}{\partial w} J(\vec w, b)
= \frac{1}{m} \sum_{i=1}^m
\left( f_{\vec w,b}(\vec x^i) - y^i \right) x_j^{(i)}
$$

$$
\frac{\partial}{\partial b} J(\vec w, b)
= \frac{1}{m} \sum_{i=1}^m
\left( f_{\vec w,b}(\vec x^i) - y^i \right)
$$

---

### 매개변수 갱신 (Parameter Updates)

위의 미분 결과를 경사 하강식에 대입하면  
가중치와 편향은 다음과 같이 갱신된다.

$$
w_j = w_j - \alpha
\left[
\frac{1}{m} \sum_{i=1}^m
\left( f_{\vec w,b}(\vec x^i) - y^i \right)
x_j^{(i)}
\right]
$$

$$
b = b - \alpha
\left[
\frac{1}{m} \sum_{i=1}^m
\left( f_{\vec w,b}(\vec x^i) - y^i \right)
\right]
$$

---

이 수식들은 형태상으로 **선형 회귀(Linear Regression)** 의 경사 하강법과 동일하다.  
그러나 중요한 차이점은 예측 함수에 있다.

선형 회귀에서는 단순히 선형 식 $ \vec w \cdot \vec x + b $ 를 사용하지만,  
로지스틱 회귀에서는 **시그모이드 함수 $ g(z) $** 가 추가되어  
출력이 확률 형태로 제한된다.

즉, 겉보기에는 수식이 유사하지만,  
두 모델의 **함수적 의미와 출력의 해석 방식**은 근본적으로 다르다.

# 정규화된 로지스틱 회귀 (Regularized Logistic Regression)

로지스틱 회귀에 정규화를 적용하는 방법은  
선형 회귀(Linear Regression)의 경우와 매우 유사하다.  
기존의 로지스틱 회귀 비용 함수에 **정규화 항(regularization term)** 만 추가하면 된다.

---

### 정규화가 적용된 비용 함수 (Regularized Cost Function)

로지스틱 회귀의 기본 비용 함수는 다음과 같다.

$$
J(\vec w,b) =
-\frac{1}{m}\sum_{i=1}^m
\left[
y^{i} \log \left(f_{\vec w,b}(\vec x^i)\right)
+ (1 - y^i)\log \left(1 - f_{\vec w,b}(\vec x^i)\right)
\right]
$$

이 식에 정규화 항을 추가하면 다음과 같이 수정된다.

$$
J(\vec w,b) =
-\frac{1}{m}\sum_{i=1}^m
\left[
y^{i} \log \left(f_{\vec w,b}(\vec x^i)\right)
+ (1 - y^i)\log \left(1 - f_{\vec w,b}(\vec x^i)\right)
\right]
+ \frac{\lambda}{2m} \sum_{j=1}^n w_j^2
$$

-   $ \lambda $ : 정규화 계수(regularization parameter)
-   $ w_j $ : 각 특성(feature)에 대한 가중치

정규화 항 $ \frac{\lambda}{2m} \sum w_j^2 $ 은  
가중치가 과도하게 커지는 것을 억제하여 **과적합(overfitting)** 을 방지한다.  
즉, 모델이 훈련 데이터에만 지나치게 맞춰지는 것을 막고,  
보다 **일반화된 성능(generalization)** 을 갖도록 만든다.

---

### 정규화가 적용된 경사 하강법 (Gradient Descent with Regularization)

정규화 항이 포함된 비용 함수에 대해  
경사 하강법(Gradient Descent)을 적용하면 다음과 같다.

$$
w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w},b)
$$

$$
b = b - \alpha \frac{\partial}{\partial b} J(\vec{w},b)
$$

---

### 편미분 결과 (Partial Derivatives)

비용 함수를 각각 $ w_j $ 와 $ b $ 에 대해 미분하면 다음 결과를 얻는다.

$$
\frac{\partial}{\partial w_j} J(\vec{w},b)
= \frac{1}{m} \sum_{i=1}^m
\left( f_{\vec{w},b}(\vec{x}^i) - y^i \right) x_j^{(i)}
+ \frac{\lambda}{m} w_j
$$

$$
\frac{\partial}{\partial b} J(\vec{w},b)
= \frac{1}{m} \sum_{i=1}^m
\left( f_{\vec{w},b}(\vec{x}^i) - y^i \right)
$$

이 식은 선형 회귀의 정규화 버전과 거의 동일하다.  
단, 유일한 차이점은 모델의 예측 함수가  
선형 함수 대신 **시그모이드 함수 $ f\_{\vec{w}, b}(\vec{x}) $** 라는 점이다.

---

### 최종 경사 하강법 식 (Complete Gradient Descent of Regularized Logistic Regression)

최종적으로, 정규화 항이 포함된 전체 경사 하강법 식은 다음과 같다.

$$
w_j = w_j - \alpha
\left[
\frac{1}{m} \sum_{i=1}^m
\left( f_{\vec{w},b}(\vec{x}^i) - y^i \right)x_j^{(i)}
+ \frac{\lambda}{m} w_j
\right]
$$

$$
b = b - \alpha
\frac{1}{m} \sum_{i=1}^m
\left( f_{\vec{w},b}(\vec{x}^i) - y^i \right)
$$

---

### 정규화 항의 의미 요약

-   **정규화(regularization)** 는 가중치의 크기를 제한하여 과적합을 방지함.
-   $ \lambda $ 값이 클수록, 모델은 더 단순해지고(가중치가 작아짐),  
    너무 크면 과소적합(underfitting)이 발생할 수 있음.
-   $ \lambda $ 값을 적절히 조정함으로써  
    **모델 복잡도와 예측 정확도 간의 균형(balance)** 을 맞출 수 있다.

---

요약하자면,  
정규화된 로지스틱 회귀는 기본적인 로지스틱 회귀와 동일한 구조를 가지되,  
비용 함수에 **패널티 항(regularization term)** 을 추가하여  
보다 안정적이고 일반화된 학습을 수행한다.
