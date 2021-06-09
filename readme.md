# Tensorflow 2.x를 활용한 Tensorflow Lite 프로젝트 레포지토리입니다.












Sparkfun 사의 Sparkfun Edge를 사용해서 한글 필사를 Non-Semantic한 방향으로 인식 가능하도록 시도중입니다.














현재 마주한 문제는 크게 세 가지 입니다.
















#### 1. 데이터셋의 질이 높지 않다.


데이터의 양이 각 클래스당 30개로 적고, 필사이자 각 자모음별로 떨어져있어야 하기에 augmentation이나 ttf to png를 통한 데이터셋 강화가 필요합니다. 
ttf to png를 구현해보았으나 matplotlib의 문제인지 그림 파일이 깨졌습니다. 





#### 2. 기기의 성능이 우려된다.

Sparkfun edge는 TFlite가 구동가능한 임베디드 보드입니다. 주로 교육용이나 개인용 소형 프로젝트에 쓰이기에, 그렇게까지 높은 성능을 보이지 못합니다. 학습이 아닌 단순한 모델의 구동, 
극단적으로 단순한 필터링에 가까운 구동을 하나, 해당 수준의 과제를 행한 기록이 보이지 않고, 기기의 카메라 모듈 성능이 의심이 됩니다. 해상도를 떠나 상이 제대로 맺힐지 모르겠습니다. 





#### 3. 레퍼런스가 부족하다

많은 레퍼런스가 Semantic Segmantation을 활용하였습니다. 미국의 한 대학에서 학생들이 진행한 프로젝트를 제외하고, Semantic Segmentation을 활용한 OCR이 대부분입니다. 
이미 온보드로 돌릴 수 있을지 모를 코드인데, 더욱 많은 리소스를 필요로 하는 논리 방식은 피하고 싶습니다. 배경 지식이 부족한 상태이기 때문에, 되도록 레퍼런스를 많이 참고하고 싶습니다만 현재 환경에 맞는 레퍼런스가 쉽게 보이지 않습니다. 
