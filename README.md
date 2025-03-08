# Human Face Image Generation with GAN Models

<br><br><br>

## 작업시점
2024/12
<br><br><br>
## 참여인원
김민성 <br>
고민수 <br>
윤성현 <br>
홍영진 <br>
<br><br><br>

## Abstract
<br>
GAN을 이용하여 사람 얼굴 이미지를 학습하고, <br>
실존하지 않지만 그럴 듯한 얼굴 이미지를 생성한다. <br>
<br>
<br>

## 1. Introduction

<br><br>
GAN(Generative Adversarial Network)은 머신 러닝에서 널리 사용되는 생성 모델(Generative Model) 중 하나로 2014년에 처음 제안되었다. GAN은 서로 경쟁하는 두 개의 신경망인 생성자(Generator)와 판별자(Discriminator)로 구성된다. 이 두 네트워크는 게임 이론의 아이디어를 기반으로 학습하며, 결과적으로 매우 사실적인 데이터를 생성할 수 있다. <br><br>
GAN의 구성 요소 중 생성자는 입력으로 랜덤한 노이즈 벡터를 받고, 이를 이용해 새로운 데이터를 생성한다. 생성자의 목표는 판별자를 속일 만큼 진짜와 유사한 데이터를 생성하는 것이다. 판별자는 반면 입력으로 받은 데이터가 실제(real)인지 생성된(fake) 데이터인지 판단한다. 판별자는 진짜와 가짜 데이터를 정확히 구분하도록 학습을 진행한다. <br><br>
GAN이 작동하는 원리는 생성자와 판별자가 서로 경쟁(adversarial)하며 학습하는데 있다. 생성자는 판별자를 속이기 위해 더 사실적인 데이터를 생성하도록 학습하며, 판별자는 진짜 데이터를 가짜 데이터와 정확히 구분하도록 학습한다. 이 과정을 반복하면서 생성자는 점점 더 실제 데이터와 유사한 데이터를 생성하게 된다. <br><br>
프로젝트에서는 존재하는 GAN 모델을 바탕으로 데이터를 학습시키고 성능 지표를 도입해 데이터 및 모델의 파인 튜닝(fine-tuning) 결과를 검증하고자 한다. <br><br>

<br><br><br>

## 2. GAN Training
<br><br>

### 2.1. GAN 모델 선정
초반부에는 단순한 GAN을 선택해 학습시켜보며 이미지를 생성하는 전반적인 과정을 파악했다. <br>
이후로 WGAN(Wasserstein GAN) 모델을 사용해 본격적인 학습을 시도했다. <br>
비교군으로는 비슷한 성능을 보여준 DCGAN(Deep Convolutional GAN) 모델을 채택하여 프로젝트를 검증했다. <br>

<br><br>

### 2.1.1. Skeleton 코드 (Vanilla GAN)

<br>

![image](https://github.com/user-attachments/assets/866d7aff-d777-4185-be6d-83124e37c76b)

<br><br>

### 2.1.2. WGAN

<br><br>

WGAN은 GAN의 변형 중 하나로, GAN이 가진 여러 문제점을 개선하기 위해 제안된 모델이다. <br>
특히 원본 GAN의 학습 불안정성과 모드 붕괴 문제를 완화하려는 목적을 가진다. <br>
특징은 Wasserstein Distance(두 확률 분포의 사이의 거리를 측정)를 도입하여 생성된 데이터와 실제 데이터 간의 분포 차이를 더 안정적이고 효과적으로 측정한다는 점이다. <br>

<br>

![image](https://github.com/user-attachments/assets/0b7f65f6-e7d9-4690-8248-ef64c6ef1c8f)

<br><br>

### 2.1.3. DCGAN
<br><br>
DCGAN은 기본 GAN에 합성곱 신경망(Convolutional Neural Network, CNN)을 도입하여 이미지 데이터를 다룰 때 더욱 효과적인 생성을 목표로 만들어졌다. <br>
GAN의 기본 구조를 유지하면서도 합성곱과 풀링을 활용하여 성능을 향상시킨 모델이다. <br>

<br>
 
![image](https://github.com/user-attachments/assets/b955231f-38cc-423c-9bd4-f5348e4dd86b)

<br>
 
![image](https://github.com/user-attachments/assets/c2cd5a15-56bb-4d70-836f-2a15bdbc77d0)

<br>

## 2.2. 데이터 튜닝

<br><br>

### 2.2.1. Align & Cropped 이미지 데이터셋

<br><br>
프로젝트에 사용한 이미지 데이터셋은 <Large-scale CelebFaces Attributes, (CelebA) Dataset >이다. <br>
약 20만 개 이상의 얼굴 이미지를 보유하고 있으며 178 × 218px 크기 이미지를 표준으로 제공한다. <br>
얼굴이 이미지 중앙으로 정렬되고 주변부가 크롭된 상태로 제공되기 때문에 관련 전처리는 따로 수행하지 않았다. <br>

<br>
![image](https://github.com/user-attachments/assets/f8a2d5f4-5f87-495b-8378-a0120307fddf)

<br><br>

### 2.2.2. Front Faces 이미지 선별 전처리
<br><br>
원본 데이터셋으로부터 정면 얼굴만을 사용하도록 하는 전처리를 수행하였다. <br>
그 결과 총 202,599개 이미지가 126,113개로 수가 줄어들었다. <br>
한 에폭을 기준으로 학습 이미지 수가 반절 가까이 줄었지만 성능 지표가 유지될 수 있을지 의문이 발생하였고, 줄어든 학습량과 전처리 효과 사이의 trade-off를 예상하였다. <br>
만약 성능이 보존되거나 개선될 경우 정면 얼굴 전처리의 효과가 예상보다 컸다는 해석이 가능할 것이다. <br>
그리고 사람 얼굴에서 탐지한 랜드마크 정보를 이용해 이미지 상에서 얼굴 밀도를 높이도록 보강(확대)하였다.<br>
<br>
적용 원리는 다음과 같다. <br>
아래 그림에서 빨간 사각형이 얼굴 경계를 나타내며, 파란 점은 각각 left eye, right eye, nose, left mouse, right mouse 랜드마크를 가리킨다. <br>
정면 판단 여부는 크게 3가지 기준으로 결정한다. <br>
<br>
1) 양쪽 눈의 y좌표 차이가 작아야 한다.<br>
2) 코의 x좌표가 양쪽 눈의 중앙값과 가까워야 한다.<br>
3) 입의 좌우 점이 대칭적이어야(y좌표 차이가 작아야) 한다.<br>

<br>
![image](https://github.com/user-attachments/assets/9bf6d8db-596c-4cf0-b0c8-15a03ef72669)

<br><br>
 
![image](https://github.com/user-attachments/assets/86392fff-2b7b-4ee7-9b9f-83042ab6d8a5)

<br><br>

### 2.2.3. Without Sunglasses 이미지 선별 전처리
<br><br>
추가로 선글라스 이미지를 제거하는 전처리를 수행하였다. <br>
선글라스를 낀 이미지가 학습되면서 생성 이미지의 눈가가 어둡게 나오는 현상을 발견하였기에, 정성적 평가를 고려하여 이를 개선하고자 했다. <br>
이 과정에서 122,965개의 이미지가 남았고 3000개 가량의 데이터가 소거되었다.<br>
선글라스 착용 유무 판단 방법은 다음과 같다. <br>
<br>
1) 눈 영역 주위에 사각형을 설정. <br>
2) 이미지의 평균 밝기를 계산하여 눈 주위 사각형의 밝기가 임계 값보다 낮으면 선글라스 착용으로 판단.<br>

<br><br>
 
![image](https://github.com/user-attachments/assets/3ee45b37-d7d4-432f-a8b0-b8fb0b7dfe8b)

<br><br>
 
![image](https://github.com/user-attachments/assets/d4b681db-b635-4015-9295-53d37a4cc9ac)

<br><br><br>

## 3. Results
<br><br>
![image](https://github.com/user-attachments/assets/badb27f9-e0da-4da0-a94d-d6a56a705f73)


Inception Score, IS는 생성된 이미지의 다양성과 품질을 동시에 평가한다. <br>
사전 학습된 Inception 네트워크를 사용해 생성 이미지가 특정 클래스를 잘 표현하는지 측정한다. <br>
값이 높을수록 품질이 높고 생성 이미지가 다양함을 의미하며, 수학적으로 범위는 0점~무한대 점수까지 가능하다. <br>
<br>
실제로 적용했을 땐 1~2점 구간을 이미지가 기본 구조를 갖추지 못하는 상태, 5~8점이 매우 높은 퀄리티의 이미지 생성 구간으로 간주하는 것이 적절해 보였다(CelebA 데이터셋 기준). <br>
이 프로젝트에서는 Inception v3 모델(네트워크)을 사용해 성능을 평가했다.<br>
<br>
IS 성능 평가의 대표적인 한계점은 다음과 같다.<br>
<br>
1) 실제 데이터와 비교하지 않음.<br>
2) 스코어 평가 기준이 Inception 네트워크의 분류 능력에 한정됨. 이는 네트워크가 학습한 데이터에 직접적으로 연결돼 있다.<br>
3) 하나의 분류 카테고리 안에서는 단일한 이미지만 반복해서 생성하더라도 가려낼 수 없음(intra-class diversity를 측정 불가).<br>
<br>
만약 생성 이미지가 명확하고 학습한 데이터와 유사하다면 좋은 성능을 보이는 것이지만, IS 평가 방식은 네트워크의 학습 범주를 벗어난 데이터를 제대로 평가할 수 없다. <br>
분류할 수 없는 데이터는 마치 노이즈와 같은 낮은 점수를 얻을 것이다. <br>
IS의 단점을 개선하기 위해 개발된 성능 평가 방법은 Frechet Inception Distance, FID이다.<br>

<br>
![image](https://github.com/user-attachments/assets/1f2b1f13-5c05-440e-9dc8-02e11b6c4cae)
<br><br><br>

## 4. Conclusion
<br><br>
GAN을 이용한 이미지 생성은 2014년 처음 제안된 이후로 꾸준히 다루어진 주제다. 
DCGAN과 WGAN이 도입되며 초기 발전을 이루었으며 CycleGAN, Pix2Pix 등이 등장하며 GAN 응용 분야가 확장되었다. 
StyleGAN의 발표 이후로 고해상도 이미지 생성에서도 큰 발전을 거뒀고, GAN의 고도화 및 대중화 시대를 열어주었다. 
이 프로젝트에서는 비교적 초기 모델에 해당하는 GAN들을 사용해 GAN의 학습 원리를 이해하고, 성능 평가를 도입해 입출력 간 연관성을 확인하고자 했다.
이미지 품질을 높이는 것에 주안점을 뒀기에 정면 얼굴 위주로 학습하는 것을 시도했다. 
여러 방향이 나오는 측면 얼굴과 달리 (이미지 중앙에 정렬된)정면 얼굴 데이터가 더 안정적인 분포를 가지고 학습에 효과적일 것으로 기대했다. 
모델이 생성한 데이터의 분포가 실제 데이터 분포를 근사하는 원리를 고려한 것이었다. 
실제로는 전처리를 거칠 때마다(학습 데이터가 소거될 때마다) 성능 평가 지표가 떨어지는 모습을 보였기에 데이터셋의 총량이 학습에 더 핵심적이었다고 결론을 내렸다.
주목할 점은 이미지 업스케일링의 효과였다. 
눈에 직접 보이는 결과물보다 정량적 평가 성능이 낮게 나온다고 생각했기에 IS를 측정하기 전 업스케일링을 적용하고 성능 평가가 달라지는지 확인했다. 
생성 이미지가 작은 사이즈였기에 IS를 계산하기 위해 (299, 299) 스케일로 조정하는 과정에서 품질이 훼손된다고 추측했었는데, 실제로 업스케일링을 적용했을 때 성능 지표가 눈에 띄게 향상되었다. 
당시엔 이를 이미지 크기가 IS 입력에 적합해졌기 때문이라고 판단했다. 
그렇지만 이유를 재점검한 결과 업스케일링 된 이미지의 품질 향상이 IS 평가에 직접적인 영향을 미친 것으로 추측됐다. 
업스케일링 과정에서 기존의 뭉개졌던 특징들이 복원 및 추가되는 것을 볼 수 있었는데 이는 업스케일링에 사용된 모델(Real-ESRGAN)이 해당 유형의 데이터셋에서 강력한 복원과 추측 능력을 이미 학습했기 때문인 것으로 생각된다.

<br>

![image](https://github.com/user-attachments/assets/441c0fe5-409c-4792-9da1-ff6e6cdb4b74)

<br>
 
![image](https://github.com/user-attachments/assets/1791c539-feed-4e01-85cf-6c59dab49ef2)

<br>
![image](https://github.com/user-attachments/assets/5e4375a6-4b62-4aec-9225-6d0f9cd60384)

<br>
