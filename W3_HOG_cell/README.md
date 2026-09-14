# HOG Descriptor – With Cell

OpenCV와 C++를 사용하여 Cell을 포함한 HOG Descriptor를 직접 구현한 프로젝트입니다.

## 개발 환경

- Visual Studio
- C++
- OpenCV 3.4.5
- Release / x64

## HOG 구조

- Block 크기: `16×16`
- Cell 크기: `8×8`
- Block당 Cell: `4개`
- Cell당 Histogram: `9-bin`
- Block당 Descriptor: `36차원`
- Block 이동 간격: `8 pixel`
- 방향 범위: `0° 이상 180° 미만`

## 구현 과정

1. Prewitt 마스크로 gradient를 계산합니다.
2. Gradient 방향을 9개 bin으로 분류합니다.
3. 16×16 block을 8×8 cell 4개로 나눕니다.
4. 각 cell에서 9-bin histogram을 구합니다.
5. 4개의 histogram을 연결하여 36차원 block histogram을 만듭니다.
6. Block별로 L2 정규화합니다.
7. 동일한 descriptor index의 절댓값 차이를 평균내어 이미지를 비교합니다.

## 유사도 비교

```text
Mean Difference = sum(|A[i] - B[i]|) / N
