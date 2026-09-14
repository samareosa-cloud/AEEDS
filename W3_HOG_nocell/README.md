# HOG Descriptor – No Cell

OpenCV를 사용하여 이미지의 HOG(Histogram of Oriented Gradients) Descriptor를 직접 구현하고, 기준 이미지와 비교 이미지 사이의 유사도를 계산한 프로젝트입니다.

이번 구현에서는 16×16 block을 cell로 나누지 않고, 각 block에서 하나의 9-bin histogram을 생성합니다.

## 개발 환경

- Visual Studio
- C++
- OpenCV 3.4.5
- 실행 환경: Release / x64

## 구현 과정

1. 입력 이미지를 Grayscale로 불러옵니다.
2. 3×3 Prewitt 마스크를 사용하여 x, y 방향 gradient를 계산합니다.
3. 각 픽셀의 gradient magnitude와 orientation을 구합니다.
4. Gradient 방향을 0° 이상 180° 미만의 범위로 변환합니다.
5. 방향을 20° 간격의 9개 bin으로 분류합니다.
6. 16×16 block을 8픽셀 간격으로 이동합니다.
7. 각 block에서 9-bin histogram을 생성합니다.
8. 각 block의 histogram을 L2 정규화합니다.
9. 정규화된 block histogram을 순서대로 연결하여 HOG Descriptor를 생성합니다.
10. 동일한 histogram index의 절댓값 차이를 평균하여 이미지 유사도를 비교합니다.

## HOG 구조

- Block 크기: `16×16`
- 이동 간격: `8 pixel`
- Cell 분할: 없음
- 방향 범위: `0° 이상 180° 미만`
- Histogram bin: `9개`
- Bin 간격: `20°`
- Block당 Descriptor 차원: `9`

예를 들어 block이 총 105개라면 최종 HOG Descriptor의 차원은 다음과 같습니다.

```text
105 blocks × 9 bins = 945 dimensions
