by 김원준 교수님

구현 내용

1. 입력 이미지를 흑백으로 불러옴.
2. 3×3 Prewitt mask를 사용하여 각 픽셀의 (x), (y) 방향 gradient를 계산.
3. 다음 식으로 Gradient Magnitude를 계산.
Magnitude = sqrt(Gx² + Gy²)
4. 전체 Magnitude 값을 0~255 범위로 변환.
5. 계산 결과를 MagImage.bmp로 저장하고 화면에 출력.

Prewitt Mask
X 방향
-1  0  1
-1  0  1
-1  0  1
Y 방향
-1 -1 -1
 0  0  0
 1  1  1

실행 결과

Input Image: 입력 흑백 이미지
Gradient Magnitude: Gradient Magnitude 시각화 결과

MagImage.bmp 파일로도 저장됨.

교수님 코드에서는 Magnitude 값을 255에서 빼서 시각화하므로, Gradient Magnitude가 큰 강한 경계 부분이 어둡게 표현됨.

참고

현재 코드는 HOG Descriptor 전체 과정 중 Gradient Magnitude를 계산하는 단계에 해당.
Gradient 방향 계산, 9-bin histogram, Block 정규화 및 HOG Descriptor 생성 과정은 포함하지 않음.
