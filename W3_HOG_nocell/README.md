# HOG Descriptor

`LectureNote_03.bmp`의 HOG Descriptor를 추출하고 `compare1.bmp`, `compare2.bmp`와 비교함.

## 개발 환경

* Visual Studio
* C++
* OpenCV 3.4.5
* Release / x64

## 구현 조건

* Block 크기: `16×16`
* Block 이동 간격: `8 pixel`
* Cell 분할: 사용하지 않음
* 방향 범위: `0°~180°`
* Histogram: `9-bin`
* Normalization: Block 단위 L2 normalization
* 비교 방법: 동일한 descriptor index의 평균 절대 차이

## 구현 과정

1. Prewitt mask로 각 픽셀의 x, y 방향 Gradient를 계산함.
2. Gradient Magnitude와 Orientation을 계산함.
3. 방향을 20° 간격의 9개 bin으로 구분함.
4. `16×16` Block 안의 Magnitude를 해당 방향 bin에 누적함.
5. 각 Block의 9-bin Histogram을 L2 정규화함.
6. Block을 8픽셀씩 이동하며 Histogram을 추출함.
7. 모든 Block의 Histogram을 연결하여 HOG Descriptor를 생성함.
8. 동일한 descriptor index의 절댓값 차이를 평균내어 이미지를 비교함.

## HOG Dimension

`64×128` 이미지에서 추출되는 Block 개수는 다음과 같음.

```text
가로 Block = (64 - 16) / 8 + 1 = 7
세로 Block = (128 - 16) / 8 + 1 = 15
전체 Block = 7 × 15 = 105
```

Block 하나에서 9개의 Histogram 값이 생성되므로 전체 HOG Descriptor는 다음과 같음.

```text
105 Blocks × 9 bins = 945 dimensions
```

## 비교 방법

```text
Mean Difference = Σ|Reference[i] - Compare[i]| / 945
```

Mean Difference가 작을수록 기준 이미지와 유사하다고 판단함.

## 출력 파일

* `LectureNote_03_blocks.csv`: 기준 이미지의 Block별 Histogram
* `compare1_blocks.csv`: Compare1의 Block별 Histogram
* `compare2_blocks.csv`: Compare2의 Block별 Histogram
* `descriptor_comparison.csv`: 동일한 descriptor index의 값과 차이
* `final_histogram.csv`: 세 이미지의 최종 9-bin Histogram
