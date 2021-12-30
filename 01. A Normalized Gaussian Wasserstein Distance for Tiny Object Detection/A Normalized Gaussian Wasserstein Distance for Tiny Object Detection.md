# 1. A Normalized Gaussian Wasserstein Distance for Tiny Object Detection

- 작성자 : 안종식
    - 파란색은 논문 내용과 직접적인 관계는 없지만, 중간중간 알아두면 좋을 정보를 나타냅니다.
    - 빨간색은 더 조사가 필요한 내용을 의미합니다. (의문점, 조사 필요)
    
    <aside>
    💡 콜아웃은 해석과 의견을 나타냅니다.
    </aside>

# 0. 개요

- paper : [https://arxiv.org/pdf/2110.13389.pdf](https://arxiv.org/pdf/2110.13389.pdf)
- 코드 X
- 참고 사이트 : Tiny-Object-Detection Task 정리 Github
    - [https://github.com/knhngchn/awesome-tiny-object-detection#tiny-object-detection](https://github.com/knhngchn/awesome-tiny-object-detection#tiny-object-detection)

# 1. Introduction

- 대부분의 Object Detection은 Normal Size의 Object를 Detection하는 Task를 수행하였고, Tiny Object(16x16Pixel 이하)는 외형정보가 매우 적기 때문에 Feature를 구분해내기가 어렵다.
    - 대표적인 Tiny Object Detection Dataset : AI-TOD
        - 링크 : [https://github.com/jwwangchn/AI-TOD](https://github.com/jwwangchn/AI-TOD)
        - 700,621개의 Object Instance로 구성, 8개의 Class, 28,036장의 위성 이미지로 구성됨.
        - Class List : airplane, birdge, storage-tank, ship, swimming-pool, vehicle, person, wind-mill
- 최근의 TOD(Tiny Object Detection) 논문들은 Feature를 구분해내는 능력을 키우는데 집중했고 Input Image의 Resolution을 키우거나 GAN을 써서 SR(Super Resolution)을 수행하였다. 또는 FPN(Feature Pyramid Network)를 사용해서 Scale-invariant Detector를 만들기도했지만, 이러한 방법들은 Precision을 증가시키는 대신 추가적인 cost가 발생하였다.
- Anchor base의 TOD 방법들은 BBox가 작기 때문에 Positive / Negative Sampel을 Matching하는 label-assignment 문제가 더 크게 발생할 수 있다.
    
    <aside>
    💡 Anchor Base Object Detector들은  IOU를 Threshold로 사용해서 Postive / Negative로 분류해 labl-assignment를 수행하기 때문에 문제가 IOU를 Threshold Metric으로 사용할 경우 문제가 발생할 수 있다
    
    </aside>
    

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2011.png)

- Fig 1을 보면 (a)와 (b) 모두 Translation이 같은 픽셀만큼 움직였는데 Tiny Object의 경우 IOU가 줄어드는 폭이 매우 크다.

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2012.png)

- 게다가 Fig 2는 다른 scale의 Object간에 IOU-Deviation Curve를 보여주는데 Object Size가 작아질 수록 기울기가 매우 가팔라진다 (=민감해진다)
    - Fig 2 그래프에 대한 명확한 분석 및 이해 필요
    - (Comment by 박재완)
        
        legend 의 value 는 A,B 의 두 박스 모두의 scale 을 의미. 
        
        Figure 2 는 두 가지 시나리오로 나누어 지게 되는데, 1st row 의 시나리오는 A,B box 의 비율과 크기가 동일한 상태에서 45도 각도로 deviation이 증가하게 됨.  이 경우에, deviation 이 증가하게 되면은 iou 값이 변화하는 반면 NWD 는 동일하게 유지가 되는 것을 확인. 따라서 box 의 scale 에 따른 편향을 배제할 수 있게 되었음. 반면 iou 는 box 의 scale 에 따라서 중심축의 변화 거리가 동일함에도 불구하고 iou 값이 일치하지 않는 현상이 발생하게 됨. 
        
        2nd row 의 시나리오는, B box 의 크기가 A box 보다 1/2 인 상태에서 45도 각도의 deviation 증가를 가져감. 이 경우에 확인할 수 있는 case 는, 특정 deviation 을 기준으로 0 인 IOU (다시 말해서 겹치지 않는 박스) 에 대해서도 NWD 값이 0에 수렴하지 않음을 확인 할 수 있음. 
        
        이는 논문에서 계속해서 말했던 NWD 의 장점 중 하나를 증명함.   
        
    
     
    
- ATSS와 같은 Dynacie Label Assignment Strategies는 Adaptive하게 IOU Threshold를 조정할 수 있지만 민감한 IOU는 적절한 threshold를 찾는것 또한 어렵게 한다.
    - Adative Training Sample Selection (ATSS)
        - 참고 : [https://byeongjokim.github.io/posts/Bridging-the-Gap-Between-Anchor-based-and-Anchor-free-Detection-via-Adaptive-Training-Sample-Selection/](https://byeongjokim.github.io/posts/Bridging-the-Gap-Between-Anchor-based-and-Anchor-free-Detection-via-Adaptive-Training-Sample-Selection/)
        
- 이러한 IOU Metric을 대체하기 위해 새로운 Metric으로 BBox의 Similarity를 측정할 수 있는 Wasserstein distance를 제안한다. BBox를 2D Gaussian Distribution으로 모델링하고 제안하는 NWD(Normalized Wasserstein Distance)를 사용해서 Gaussian Distribution의 유사도를 측정한다. Wasserstein Distance를 사용하면 Box가 겹치던 안겹치던 유사도를 측정할 수 있고 이는 Tiny Object에 대해서도 유사도를 측정하는데 유용하기 때문에 IOU에 비해 강인한 방법이라 할 수 있다.
    - Wasserstein Distance (정리필요)
    - (Comment by 박재완)
        
        Wasserstein ⇒ 통칭, 와서스테인 혹은 바서스테인 metric 
        
        ![KakaoTalk_20211230_010431056.jpg](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/KakaoTalk_20211230_010431056.jpg)
        
        가우시안 분포함수, 즉 정규분포 함수를 2-dimension 에 정의하게 되면은 위 그림의 (4분면 기준) 왼쪽 아래의 공식으로 나타낼 수 있다. 2-dimensional Gaussian Distribution 을 그림으로 표현해보면은 오른쪽 아래와 같은 그림으로 나타나게 되는데, x,y 과 0에 가까울수록 g(x,y)의 값이 커짐을 확인할 수 있다. 좌표를 분포로 변환시킴에 따라서 (0,0)에 가까운 값들에 자연스럽게 가중치를 가져갈 수 있다는 의미로 해석했다. ( related works 로 조사했던 PAA 에서도 앵커 확률분포에 관한 이론을 가져왔는데, anchor 의 중심에 갈수록 더욱 큰 가중치를 부여한다는 의미에 있어서 해당 논문의 저자들과 비슷한 의도를 느낄 수 있었음) 
        
        논문에서는 위의 2-dimensional Gaussian Distribution 을 그대로 사용하지 않고, 우측 상단의 확률밀도 함수 형태로 변환후에 사용하였다. 확률밀도 함수 아래의 변수 들은 box에서 얻을 수 있는 (Cx ,Cy) , w, h 를 어떻게 확률밀도함수의 변수로 사용하였나를 의미한다. (논문에서 확인가능) 
        
        그래서 GT Box 와 Proposal Box 각각에 대해서 2-dimensional Gaussian Distribution 을 적용해주고, 이 두 확률분포의 차이를 Wasserstein Distance 로 계산한다. (공식 논문참조) 
        
        Was Dist 에 대한 상세한 설명을 위해서 아래의 링크를 참조 바람.  
        
        [https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i](https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i)
        
        결국은 두 확률분포의 결합확률분포의 d(X,Y)을 가장 작게 추정한 값을 의미한다.  
        
- NWD는 1-stage, N-Stage, Anchor Based Detection에도 적용할 수 있고 단순히 Label-Assignment에 사용되는 IOU Metric을 대체하는 것이 아니라 NMS, Regression Loss에서 사용되는 IOU에도 적용할 수 있다.
- Contribution 정리
    - TOD에서 IOU의 민감성을 분석한 뒤 IOU Metric을 대체할 수 있는 강인한 NWD를 제안한다.
    - NWD를 Label Assignment, NMS, Loss Function에 적용한 Anchor-based Detector를 설계함
    - NWD는 TOD 성능을 향상시켜 AI-TOD Dataset에서 Faster RCNN대비 11.1~17.6% 향상된 성능을 보여줌.

# 2. Related Works

## 2.1 Tiny Object Detection

### Multi-scale Fetaure Learning

- 단순하게 Input Image를 Resize하거나 Feature Scaling하여 탐지. FPN, PANet, BiFPN 등등
- 이러한 방법은 좋은 성능을 나타내고 있으나 추가적인 Computational Cost가 필요함.

### Designing Better Training Strategy

- SNIP, SNIPER, SAN(Scale-Aware Network) 등 Scale Variance에 강인하게 Detector를 설계

### GAN-Based Detectors

- MT-GAN은 Image-level의 SR을 작은 ROI의 Feature에 적용함.

## 2.2 Evaluation Metric in Object Detection

- IOU / GIOU / CIOU / DIOW / GWD(Gaussian Wasserstein Distance)

## 2.3 Label Assigment Strategies

- ATSS, PAA, OTA ...
- ATSS (Adaptive Training Sample Selection) **김현진**
    - 임의의 IoU threshold를 기준으로 pos/neg를 선택하는 것이 아니라 gt특성에 따라 적절한 threshold를 추출하고 이를 기준으로 pos/neg sample 구분
    - hyperparameter가 거의 없고 (k = # of anchor box만 있음), 여러 셋팅에 대해 강인함.
    - 특징1: anchor box와 object간 center distance를 기반으로 positive sample 후보 선택 (center distance가 가까울 때 higher-quality detection이 가능하기 떄문에 closer anchor를 선택)
    - 특징2: Positive center가 object 내부에 있는 것만 사용 (center가 object 밖에 있을 경우 좋지 않은(poor) 후보이기 때문에 사용하지 않음)
    - 특징3: 평균(m), 표준편차의(v) 합을 IoU threshold로 사용
    (m↓ → low-quality → low threshold, v↓→ specific pyramid level suitable→ low threshold)
    (m↑ → high-quality → high threshold, v↑→several pyramid levels suitable → high threshold)
        
        ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2013.png)
        

# 3. Methodology

## 3.1 Gaussian Distribution Modeling for Bounding Box

- 실제 Object는 대부분 사각형이 아니기 때문에 BBox안에는 Background Pixel이 존재한다. 이 때, bbox안에서 Foreground pixel들은 bbox 중심쪽으로, Background pixel들은 bbox 외곽선 쪽으로 집중된다. 이러한 다른 종류의 pixel들의 가중치를 표현하기 위해서 bbox를 2D-Gaussian Distribution으로 표현할 수 있다.
- $\mu_x=cx, \mu_y=cy, \sigma_x=w/2, \sigma_y=h/2$일 때 bbox를 타원의 방정식으로 나타내면  아래와 같이 나타낼 수 있다

$$\frac{(x-\mu_x)^2}{\sigma^2_x} + \frac{(y-\mu_y)^2}{\sigma^2_y}$$

- 2D Gaussian Distribution 수식은 아래와 같이 나타낼 수 있다.

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2014.png)

- $x, \mu, \Sigma$ 는 좌표 (x,y), mean vector, Co-variant Matrix이고, $(x-\mu)^T\Sigma^{-1}(x-\mu)=1$ (마할라노비스 거리)일 때, 타원방정식은 2D Gaussian Distribution의 Density Contour가 된다. (Gaussian Distribution의 외곽선)
- 따라서 BBox의 정보를 가지고 2D Gaussian Distribution으로 모델링 하면 아래와 같이 평균, 공분산을 나타낸 Gaussian Distribution으로 나타낼 수 있고, 이제 두개의 Distribution의 Simiarity를 계산하면 된다.

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2015.png)

## 3.2 Normalized Gaussian Wassertein Distance

- Wasserstein Distiance를 Distribution Distance를 계산하기 위해 사용하고, $\mu_1, \mu_2$를 각각 2D Gaussian Distribution이라고 할 때, 2차 Wasserstein Distance는 아래와 같이 나타낼 수 있다.
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2016.png)
    
- 정리하면 ($||dot||_F$는 Frobenius Norm

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2017.png)

- 이를 BBox에 적용하면

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2018.png)

- Metric으로 활용되기 위해서는 IOU와 같이 0~1로 정규화가 되어야하기 때문에 normalization을 위해서 exp 연산을 활용해 NWD를 제안

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2019.png)

- C는 상수로 Dataset과 연관된 상수.
- IOU와 비교했을 때, NWD는 Scale invariance하고, Location Deviation을 Smooth하게 하며 겹치지 않아도 Similarity를 측정할 수 있다.
- Fig2에서 보면, 같은 사이즈의 Box를 위치를 이동시킬 때, 같은 Deviation Curve를 확인할 수 있다.

## 3.3 NWD-Based Detectors

- 제안하는 NWD는 어떤 Anchor Base Detector라도 IOU를 대체하도록 하기 쉽다.
- Anchor-based Detector의 대표적인 Faster R-CNN에 적용해서 NWD의 사용법을 설명

### NWD-based Label Assignment

- Faster RCNN은 RPN과 R-CNN으로 이루어져 있고, RPN과 R-CNN은 둘다 Label Assignment Process가 존재.
- RPN에서는 다른 Scale과 Ratio를 가지고 있는 Anchor들을 처음 생성하게 되고, Binary Label(Positive / Negative) 들은 Anchor에 할당되어 Classification과 Regression Head를 학습함.
- R-CNN은 RPN의 Output을 Input으로 활용하는 것이 다른 점.
- **RPN을 학습하기 위해서 Positive label은 두종류의 anchor에 할당됨. GT와 제일 높은 NWD 값을 갖고, NWD Value가 Negative Threshold 이상일 때 혹은  어떤 GT와도 NWD값이 Positive Threshold 이상일 때  Positive Label로 할당**
- Negative는 어떠한 GT와도 Negative Threshold 이하일때

### NWD-based NMS

- NMS는 score기준으로 정렬하고 제일높은 Score를 갖는 Prediction Box를 선정한다.  만약 다른 Box들과 겹치는 정도가 Threshold 이상이면 Suppress.
- 겹치는 정도를 IOU가 아닌 NWD로 적용해서 Suppress가 되지 않는 문제를 해결

### NWD-based Regression Loss

- IOU Loss는 Non-overlap이나, 하나의 Box가 완전히 다른 하나의 Box에 포함될 때 학습 Gradient를 생성할 수 없다 (학습이 안됨)
- CIOU나 DIOU를 위 상황에서 대체가능하지만 Sensitivity는 여전히 남아있고 따라서 IOU Loss대신 NWD Loss를 사용 (1-NWD)

# 4. Experiments

- AI-TOD, VisDron2019 Dataset으로 평가, Ablation은 AI-TOD로 진행. AI-TOD의 평균 Object Pixel Size는 12.8Pixel(Pascal VOC : 156.6 Pixel, MS COCO : 99.5pixels)
- 평가 Metric은 AP, AP0.5, AP0.75, AP Vt, ApT, APs, APm / APvt : 2~8pixel, t : 8~16pixel, small : 16~32, medium : 32~64pixel
- TitanX 4장에서 진행, MMDetection으로 진행, Resnet 50 FPN (Imagenet pretrained) backbone. SGD 12 epoch, 0.9 momentum, 0.0001 weight decay, 8 batch size. RPN, Faster RCNN의 Batch Size(channel) 256, 512
    - MMDetection
        - Object Detection을 Pytorch로 구현해서 Listup 해놓은 Toolbox
        - 주소 : https://github.com/open-mmlab/mmdetection
- Positive / Negative Ratio = 1:3, RPN Proposal 3000EA
- NMS IOU : 0.5

## 4.1 Comparison with Other Metrics based IOU

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2020.png)

- 여러가지 IOU Metric을 사용해서 비교한 Table

### Comparison in label assignment

### Comparison in NMS

### Comparison in loss function

## 4.2 Ablation Study

- Faster-RCNN을 base line으로 진행

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2021.png)

### Applying NWD into single module

### Applying NWD into multiple module

## 4.3 Main Result

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2022.png)

![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2023.png)

### Main results on AI-TOD

### Main results on VisDrone

# 5. Conclusion

# 6. TODO

- Wasserstein Distance에 대한 이해를 바탕으로 수식 정확하게 이해 (안종식)
- Related Works의 Label Assignment 방법들 간략하게 정리 (김현진)
    - ATSS(김현진), PAA(박재완), OTA(현청천)
- Code가 없으므로, 실제로 구현해서 비교해보기
    - Else
- Tiny Object Detection Dataset - Ai TOD VisDrone2019 (홍은기)

# **OTA: Optimal Transport Assignment for Object Detection**

- 내용
    
    # Abstract
    
    - label assignment를 global 관점에서 효과적으로 처리하기위해 Optimal Transport (OT) 방식을 도입
    - demander (anchor)와 supplier (gt)의 unit transportation cost는 classification과 regression loss의 weighted sum을 사용
    
    # 1. Introduction
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2024.png)
    
    - ambiguous anchors (anchors가 여러 gt에 동시에 positive symbol을 갖는 경우)를 처리하는 방법이 어려움
    - 각각의 gt를 독립적으로 할당하는 방식을 제거하고 이미지 내의 모든 gt에 대해서 global high confidence assignment 방식으로 변경함
    - Sinkhorn-Knopp Iteration 방법을 이용해 빠르고 효율적으로 처리
    
    # 2. Related Work
    
    ## 2.1. **Fixed Label Assignment**
    
    ## 2.2. Dynamic Label Assignment
    
    # 3. Method
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2025.png)
    
    ## 3.1. **Optimal Transport**
    
    - s: suppliers (gt) (1..m)
    - d: demanders (anchor) (1..n)
    - c: transporting cost
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2026.png)
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2027.png)
    
    ## 3.2 **OT for Label Assignment**
    
    - foreground cost
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2028.png)
    
    - background cost
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2029.png)
    
    - 각 gt에 k개의 positive anchor 할당
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2030.png)
    
    ## 3.3. **Advanced Designs**
    
    ### Center Prior.
    
    - gt에서 가까운 r^2 개의 anchor를 우선 연산
    
    ### Dynamic k Estimation
    
    ![Untitled](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/Untitled%2031.png)
    
    ---
    
    ---
    

## PAA

    (**Probabilistic Anchor Assignment with IoU Prediction for Object Detection**) 

    by 박재완 

- 개요
    
    PAA 는 제목에서 알 수 있듯이, “확률적 앵커할당 방법” 을 의미합니다. 앵커의 할당을, 관련 모델의 학습 결과(확률 분포 변환 후)에 따라서 Ground Truth 박스에 대한 pos or neg 샘플로 구분하게 됩니다. 각 앵커의 score 를 계산하기 위해서 classification score 와 location score 를 objective loss 로 가져갑니다. 
    
- Figure 1 ( Labeling sample )
    
    ![figure_01.png](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/figure_01.png)
    
- 각 앵커의 score 를 RPN layer 의 classification layer 와 location layer 를 통해서 계산해 낼 수 있다. (논문에서는 detection model 의 결과값이라고 표현)
- Anchor score 는 cls score 와 loc score 를 공식에 따라서 조합한 결과값.
- 현재 그림에서 보이는 Anchor Score 의 분포는 파란색곡선(neg) 과 빨간색곡선(pos) 두 가지 분포로 이루어져 있으며, 이러한 anchor 에 대한 label 은 detection 모델이 cls 와 loc 에 따라서 score 값을 도출해 주었을 때 이를 확률적 분포로 변환하고, 이러한 확률적 분포에 따라서 labeling 을 한 결과이다.
- (본문) This transforms the anchor assignment problem to a
maximum likelihood estimation for a probability distribution

      ⇒ 결론 : anchor assignment 문제를 최대우도법(MLE)으로 변환시켜 준다. 

- Figure 2 ( Frame Work )

![figure_02.png](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/figure_02.png)

- Proposal Method

We can define and get
Scls from the output of the classification head. 

⇒ classification head. ⇒ 정확한 layer 형태 코드필요, RPN 상의 classification layer 의 fc 부분을 의미하는 것이라고 예상중 

⇒ Here we use the Intersection-over-Union (IoU) of a predicted box with its GT box as Sloc

     location score 를 GT 와의 IOU 값으로 가져가겠다고 함. 

  

Score 값의 log 변환(아래 참조) 

![formula01.png](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/formula01.png)

![formula02.png](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/formula02.png)

![figure_03.png](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/figure_03.png)

- Figure 3

  MLE 즉, 최대우도분포를 위해서 각 분포에 대해 params 들을 최적화 하였을 때, 나눌 수 있는 각 분포들에 대한 Boundary 예시들. 

⇒ 최종 objective score 공식 

P(probability) * S(Score) 의 pi 를 최대로 만드는 파라미터 세타의 결정 ⇒ 세타의 결정에 따라 위처럼 boundary schema 가 나누어지게 됨. 

![formula03.png](1%20A%20Normalized%20Gaussian%20Wasserstein%20Distance%20for%20T%20b112aadd8d9f48e9a534572a7b6c641b/formula03.png)

⇒ 위와같은 formula 를 학습과정에 따라서 loss 로 가져갈 수 있다. 

이해를 돕기 위한 본문 문구 

> In each training iteration, after estimating Ppos and Pneg, the gradients of the loss objectives w.r.t. θ can be calculated and stochastic gradient descent can be performed.
>
