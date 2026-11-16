# W11 Face Verification

macOS + OpenCV 5 환경에 맞게 수정한 버전입니다. (OpenCV 4에서도 빌드됩니다)

## 기능

웹캠이나 사진 속 얼굴이 미리 등록한 사람과 같은 사람인지 판별합니다.

- 웹캠에서 `e`를 누르면 지금 보이는 얼굴을 등록
- 이후 화면에 나오는 얼굴이 등록한 사람이면 초록색(ACCEPT), 다른 사람이면 빨간색(REJECT)으로 표시
- 사진 여러 장을 넣으면 첫 장을 기준으로 나머지가 같은 사람인지 비교

얼굴에서 눈, 코, 입, 턱선 등 landmark 66점을 찾고, 각 점 주변 17x17 영역의 uniform LBP 히스토그램(59 bin)을 이어 붙여 3894차원 특징을 만듭니다. 등록한 얼굴과의 cosine similarity가 기준값 이상이면 같은 사람으로 판단합니다.

## 수정한 부분

SDM 라이브러리(RoboPai/sdm)가 Windows + OpenCV 2.4 기준 코드라 macOS + OpenCV 5에서 바로 빌드되지 않아 아래 내용을 수정했습니다.

- `sdm_compat.h` 추가: OpenCV 5에서 삭제되거나 다른 모듈로 이동한 상수와 함수(`CV_BGR2GRAY`, `CascadeClassifier`, `decomposeProjectionMatrix` 등)를 맞춰줌
- `getRotationMatrix2D` 대신 회전 행렬을 직접 계산 (OpenCV 5에서 geometry 모듈로 이동)
- C++11 → C++17
- `sdm/OpenCVSDK` (Windows용 OpenCV 2.4 헤더) 삭제
