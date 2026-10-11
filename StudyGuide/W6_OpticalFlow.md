# W6. Optical Flow: OFE · Lucas-Kanade · Pyramid · Farneback · Dense Trajectories

> 범위: **LectureNote 6** (Feature Extraction in Videos (2)),
> 선배 코드 `versatile0010/Week6/1014_optical_flow.cpp`, `shinjuyeop/W7/W7.cpp` (같은 실습)

---

## 1. 한눈에 보기

| 키워드 | 한 줄 요약 |
|---|---|
| Optical flow | 영상에서 **밝기 패턴의 겉보기 움직임** (실제 움직임 없이 조명 변화로도 생김) |
| 3가지 가정 | **Brightness constancy**, **Small motion**, **Spatial coherence** |
| Dense | 특징점이 아니라 **모든 픽셀**의 움직임 (u, v) |
| OFE | **`Ix·u + Iy·v + It = 0`**, u = dx/dt, v = dy/dt |
| Lucas-Kanade | **3×3 patch(9픽셀)는 같은 움직임** → 방정식 9개, 미지수 2개 → **최소제곱(LSE)** |
| LK 핵심 행렬 | `[[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]` = **Harris 의 structure tensor** |
| It | **현재 프레임 픽셀 밝기 − 이전 프레임 픽셀 밝기** |
| 풀 수 있는 조건 | structure tensor **가역**, 고유값이 **너무 작지 않을 것** |
| LK 한계 | 움직임이 **1픽셀보다 크면** 가정 깨짐 → **Pyramid (coarse-to-fine)** |
| Farneback | **Polynomial expansion**: `d = −½ A1⁻¹ (b2 − b1)` → `calcOpticalFlowFarneback` |
| 약점 | **조명 변화, 가림(occlusion), aperture problem** |

---

## 2. 개념 설명

### 2-1. Optical flow 란?
- 정의: **apparent motion of brightness patterns in the image**
- 실제 움직임이 없어도 조명이 바뀌면 생길 수 있음
  - ① 고정 조명 아래 **균일하게 회전하는 구** (무늬 없음) → 밝기 패턴이 안 바뀌어서 flow = 0
  - ② 정지한 구 + **움직이는 조명** → 밝기 패턴이 바뀌어서 flow ≠ 0
  → 실제 움직임과 optical flow 가 다를 수 있다 (슬라이드: "Same in optical flow!")
- 두 연속 프레임 사이의 motion field (u, v) 추정
- Feature point tracking(KLT, SIFT) 과 달리 **모든 픽셀**의 움직임 → Dense trajectories

**핵심 가정 3가지**
1. **Brightness constancy**: 같은 점은 모든 프레임에서 같은 밝기
2. **Small motion**: 점은 멀리 움직이지 않는다
3. **Spatial coherence**: 이웃 점들은 비슷하게 움직인다

**응용**: motion 기반 video segmentation, structure from motion(3D 복원), alignment(손떨림 보정 video stabilization, UAV 영상 → **global motion compensation**), video compression(MPEG: 원본 프레임 + 차이)

### 2-2. Optical Flow Equation (OFE) 유도
```
I(x, y, t) = I(x + dx, y + dy, t + dt)                 (밝기 불변)
테일러 1차 전개 후 dt 로 나누면
∂I/∂x · u + ∂I/∂y · v + ∂I/∂t = 0,   u = dx/dt,  v = dy/dt
즉  Ix·u + Iy·v + It = 0          ← OFE
```
- 픽셀 하나에서 **방정식 1개, 미지수 2개(u, v)** → 혼자서는 못 풂 (aperture problem)
- 그래서 "이웃 픽셀은 같은 움직임" 가정 추가 → Lucas-Kanade

### 2-3. Lucas-Kanade
3×3 patch 의 9개 픽셀이 모두 같은 (u, v) 를 가진다고 가정:
```
Ix(x1)u + Iy(x1)v = −It(x1)
Ix(x2)u + Iy(x2)v = −It(x2)          →    A z = b
   ⋮
Ix(xn)u + Iy(xn)v = −It(xn)

    | Ix(x1) Iy(x1) |          | u |           | −It(x1) |
A = |   ⋮      ⋮    |,    z =  | v |,     b =  |    ⋮    |
    | Ix(xn) Iy(xn) |                          | −It(xn) |
```
최소제곱 해 `z = (AᵀA)⁻¹ Aᵀ b`:
```
| u |   | ΣIx²    ΣIxIy |⁻¹ | −ΣIxIt |
| v | = | ΣIxIy   ΣIy²  |   | −ΣIyIt |
          └ Harris 의 structure tensor! ┘
```
- `It(xi)` = **현재 프레임 픽셀 밝기 − 이전 프레임 픽셀 밝기**
- **풀 수 있는 조건**
  1. structure tensor 가 **가역(invertible)** (det ≠ 0)
  2. 고유값이 **너무 작지 않을 것** (노이즈에 민감하지 않게)
  - 고유벡터/고유값 = 지배적인 방향과 그 크기
  - → **corner** 에서 가장 잘 풀리고, flat(λ≈0) 이나 edge(λ2≈0) 에서는 불안정 = aperture problem
- 2×2 역행렬: `[[a, b],[c, d]]⁻¹ = 1/(ad − bc) · [[d, −b],[−c, a]]`

### 2-4. LK 의 한계와 Pyramid
- 움직임이 **1픽셀보다 크면** small motion 가정이 깨짐 (실제로 자주 발생) → 움직이지 않는 곳에 엉뚱한 flow
- 최적 윈도우 크기는? (작으면 큰 움직임 못 잡고, 크면 spatial coherence 깨짐)
- **Coarse-to-fine (영상 pyramid)**
  - Lowpass filtering + resizing 으로 여러 해상도 생성
  - 원본에서 4픽셀 움직임 → 위 단계(coarse)에서는 1픽셀 움직임
  - **가장 높은(작은) 단계부터** flow 추정 → 보간(interpolate)·warp 해서 아래 단계의 **초기 flow** 로 전달 → 보정 반복
  - 결과: 큰 움직임도 잘 추정됨

### 2-5. Farneback: Polynomial expansion (Farnebäck, SCIA 2003)
각 이웃 영역을 2차 다항식으로 근사:
```
f1(x) = xᵀ A1 x + b1ᵀ x + c1
f2(x) = f1(x − d)                         (d 만큼 이동한 신호)
      = (x − d)ᵀ A1 (x − d) + b1ᵀ(x − d) + c1
      = xᵀ A1 x + (b1 − 2A1 d)ᵀ x + dᵀA1d − b1ᵀd + c1
      = xᵀ A2 x + b2ᵀ x + c2
계수 비교:  A2 = A1,   b2 = b1 − 2A1 d
⇒ 2A1 d = −(b2 − b1)
⇒ d = −½ A1⁻¹ (b2 − b1)          (A1 이 non-singular 일 때)
```
- 왜 쓰나? ① **separable convolution** 으로 구현 가능 (매우 효율적) ② **multi-scale** 분석으로 더 정확
- OpenCV: **`calcOpticalFlowFarneback`**

### 2-6. 약점 (Discussion)
- **조명 변화** (brightness constancy 위반)
- **Occlusion** (가려졌다 나타나는 영역은 대응점이 없음)
- **Aperture problem** (작은 창으로 보면 edge 방향 움직임은 알 수 없음)
- 착시 그림(회전 뱀): 실제 움직임 없이도 optical flow 를 "봄"

### 2-7. (Appendix) Dense Trajectories (Wang et al., CVPR 2011)
- 여러 scale 에서 특징점을 **촘촘히(dense) 샘플** → 각 scale 에서 L 프레임 동안 추적
  - 공간 간격 **W = 5 픽셀**, scale **8개** (scale factor **1/√2**)
  - 추적: `P_{t+1} = P_t + (M * w)|(x̄t, ȳt)` — flow w=(u,v) 에 **median filter** M 적용 (bilinear 보간보다 강인)
  - dense flow 는 Farneback 사용
- Trajectory shape descriptor: `S = (ΔP_t, …, ΔP_{t+L−1})`, `ΔP_t = P_{t+1} − P_t`, 정규화 `S' = S / Σ‖ΔP_j‖`
- Trajectory-aligned descriptors: volume **N×N×L**, grid **nσ × nσ × nτ** (기본 N = 32, nσ = 2, nτ = 3)
  - **HOG**: 모양(정적 gradient), **8 bin**
  - **HOF**: optical flow 방향 histogram, **9 bin** (8 + 정지 bin 1)
  - **MBH** (Motion Boundary Histogram): flow 의 u, v 각각의 gradient `(ux, uy)`, `(vx, vy)` → 또 벡터 → 방향 histogram (카메라 등 일정한 움직임 제거 효과)
  - 모두 **L2 정규화**
- **Improved DT** (Wang & Schmid, ICCV 2013): **RANSAC 으로 homography 추정 → 카메라 움직임 제거** (SURF 매칭 + optical flow), warp 한 프레임에서 HOG/HOF/MBH 추출
  - RANSAC vs Least square: LS 는 outlier 에 끌려가고 RANSAC 은 consensus 로 무시

---

## 3. 핵심 코드 (슬라이드 15~16)
```cpp
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>   // calcOpticalFlowFarneback
#include <vector>
#include <stdio.h>
#include <Windows.h>

#define Width  320
#define Height 240

using namespace cv;
using namespace std;

void main() {
    VideoCapture capture(0);
    int x, y, key;
    Mat flow, frame, prevFrame, img;

    while (1) {
        capture >> frame;
        resize(frame, img, Size(Width, Height));
        cvtColor(img, frame, CV_BGR2GRAY);           // flow 는 흑백으로 계산, 그림은 컬러 img 위에

        if (prevFrame.empty() == false) {           // 두 번째 프레임부터
            // calculate optical flow
            calcOpticalFlowFarneback(prevFrame, frame, flow, 0.4, 1, 12, 2, 8, 1.2, 0);

            // y += 5, x += 5 로 격자(grid) 지정
            for (y = 0; y < Height; y += 5) {
                for (x = 0; x < Width; x += 5) {
                    const Point2f flowatxy = flow.at<Point2f>(y, x) * 2;     // (dx, dy), 2배 확대 표시
                    line(img, Point(x, y),
                         Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)),
                         Scalar(0, 255, 0));                                  // 초록 선
                    circle(img, Point(x, y), 1, Scalar(0, 0, 0), -1);         // 검은 점 (-1 = 채움)
                }
            }
            frame.copyTo(prevFrame);
        }
        else frame.copyTo(prevFrame);               // 첫 프레임: 저장만

        imshow("Optical Flow", img);
        key = waitKey(20);
    }
}
```

**`calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)`**

| 인자 | 값 | 의미 |
|---|---|---|
| prev, next | prevFrame, frame | 이전/현재 **흑백** 프레임 |
| flow | flow | 결과, **CV_32FC2** (픽셀마다 (dx, dy)) |
| pyr_scale | 0.4 | pyramid 각 단계 축소 비율 (0.5 = 절반씩) |
| levels | 1 | pyramid 단계 수 (**1 = 추가 단계 없이 원본만**) |
| winsize | 12 | 평균 윈도우 크기 (크면 노이즈에 강하지만 흐려짐, 빠른 움직임 검출 ↑) |
| iterations | 2 | 각 단계 반복 횟수 |
| poly_n | 8 | polynomial expansion 에 쓰는 이웃 크기 |
| poly_sigma | 1.2 | 다항식 근사용 Gaussian σ |
| flags | 0 | 옵션 없음 |

- 처리 시간은 **QueryPerformance** 로 측정 (슬라이드 17)

---

## 4. 선배 코드 비교
- versatile / shinjuyeop 모두 슬라이드 코드와 거의 같고 해상도만 `320*2 × 240*2` (640×480) 로 키움 → 격자 점 수 4배, 처리 시간 증가.
- versatile 은 `circle(..., 1.5, ...)` 로 반지름에 실수 → int 로 잘려 1.
- shinjuyeop 은 이 실습이 W7 폴더에 있음 (학기 일정 차이).

---

## 5. 예상 문제

### A. 코드 빈칸

**A1.**
```cpp
#include <opencv2/video/①__________.hpp>
...
if (prevFrame.②________() == false) {
    ③______________________(prevFrame, frame, flow, 0.4, 1, 12, 2, 8, 1.2, 0);
    for (y = 0; y < Height; y += ④__) {
        for (x = 0; x < Width; x += 5) {
            const Point2f flowatxy = flow.at<⑤_______>(y, x) * 2;
            line(img, Point(x, y), Point(cvRound(x + ⑥_________), cvRound(y + flowatxy.y)), Scalar(0, 255, 0));
            circle(img, Point(x, y), 1, Scalar(0, 0, 0), ⑦___);
        }
    }
    frame.⑧______(prevFrame);
}
else frame.copyTo(prevFrame);
```

**A2.** 흑백 변환과 크기 조절
```cpp
capture >> frame;
①______(frame, img, Size(Width, Height));
②________(img, frame, CV_BGR2GRAY);
```

**A3.** Lucas-Kanade 를 직접 구현 (3×3 patch)
```cpp
for (dy = -1; dy <= 1; dy++)
  for (dx = -1; dx <= 1; dx++) {
    ix = Ix[(y+dy)*W + (x+dx)];  iy = Iy[(y+dy)*W + (x+dx)];
    it = ①__________________________________;          // 시간 미분
    Sxx += ix*ix;  Syy += iy*iy;  Sxy += ix*iy;
    Sxt += ix*it;  Syt += iy*it;
  }
det = ②____________________;
if (fabs(det) < 1e-6) continue;          // 풀 수 없음
u = ( Syy * (-Sxt) - Sxy * (-Syt)) / det;
v = ③___________________________ / det;
```

### B. 개념 / 알고리즘

**B1.** Optical flow 의 정의는? "고정 조명 아래 회전하는 무늬 없는 구" 와 "정지한 구 + 움직이는 조명" 의 optical flow 는 각각?

**B2.** Optical flow 추정의 3가지 핵심 가정은?

**B3.** OFE 를 유도하라. 왜 OFE 하나로는 (u, v) 를 구할 수 없는가?

**B4.** Lucas-Kanade 의 핵심 가정과 해법은? LSE 에 나오는 2×2 행렬은 무엇과 같은가?

**B5.** LK 가 풀리기 위한 조건 2가지는? 어떤 영역에서 잘 되고 어떤 영역에서 안 되는가?

**B6.** LK 의 한계와 pyramid(coarse-to-fine) 방식의 동작을 설명하라.

**B7.** Polynomial expansion 에서 변위 d 를 유도하라. 이 방법의 장점 2가지는?

**B8.** Optical flow 의 약점 3가지는?

**B9.** 코드에서 첫 프레임에서는 flow 를 계산하지 않는 이유는? flow 를 흑백으로 계산하는 이유는?

**B10.** Dense trajectories 의 세 가지 descriptor 는? MBH 는 무엇을 계산하는가? Improved DT 는 무엇을 추가했는가?

### C. 결과 적기 / 계산

**C1.** 한 픽셀에서 Ix = 2, Iy = 1, It = −4 이다. u = 1 이면 v 는? (OFE 만족)

**C2.** 3×3 patch 에서 (Ix, Iy, It) 가 다음 4픽셀만 0이 아니다: (1,0,−1), (1,0,−1), (0,1,2), (0,1,2). LK 로 (u, v) 를 구하라.

**C3.** ΣIx² = 4, ΣIy² = 1, ΣIxIy = 2 인 patch 는 LK 로 풀리는가? 이유는?

**C4.** 원본 영상에서 8픽셀 움직임을 1픽셀로 만들려면 pyr_scale = 0.5 pyramid 를 몇 단계 올라가야 하는가?

**C5.** Width=320, Height=240, 격자 간격 5 → 그려지는 선(점)의 개수는? 640×480 이면?

**C6.** flow.at<Point2f>(100, 50) = (1.5, −2.0) 이면 그려지는 선의 시작점과 끝점은?

**C7.** f1 의 A1 = [[1, 0],[0, 1]], b1 = (0, 0), 이동 후 b2 = (−4, 2) 이면 d 는?

**C8.** Dense trajectories 기본 파라미터(nσ=2, nτ=3)에서 HOG(8bin), HOF(9bin), MBH(x,y 각 8bin) descriptor 차원은?

---

<details>
<summary><b>정답 보기</b></summary>

**A1.** ① `tracking` ② `empty` ③ `calcOpticalFlowFarneback` ④ `5` ⑤ `Point2f` ⑥ `flowatxy.x` ⑦ `-1` (채움) ⑧ `copyTo`

**A2.** ① `resize` ② `cvtColor`

**A3.** ① `curr.at<uchar>(y+dy, x+dx) - prev.at<uchar>(y+dy, x+dx)` (현재 − 이전) ② `Sxx * Syy - Sxy * Sxy` ③ `(-Sxy * (-Sxt) + Sxx * (-Syt))` (= `Sxy*Sxt - Sxx*Syt`)

**B1.** 영상 속 밝기 패턴의 겉보기 움직임. 무늬 없는 구가 회전하면 밝기 패턴이 그대로라 **flow = 0** (실제 움직임 있음), 정지한 구에 조명이 움직이면 밝기 패턴이 변해 **flow ≠ 0** (실제 움직임 없음). → optical flow ≠ 실제 움직임.

**B2.** ① Brightness constancy ② Small motion ③ Spatial coherence

**B3.** `I(x,y,t) = I(x+dx, y+dy, t+dt)` 를 1차 테일러 전개 → `Ix dx + Iy dy + It dt = 0` → dt 로 나눔 → `Ix u + Iy v + It = 0`. 픽셀 하나당 방정식 1개, 미지수 2개라 해가 무수히 많음(직선 위 어디든) → gradient 에 수직한 성분을 알 수 없음 (aperture problem).

**B4.** 작은 patch(3×3) 의 모든 픽셀이 같은 (u, v) 로 움직인다 → 방정식 9개, 미지수 2개 → 최소제곱 `z = (AᵀA)⁻¹Aᵀb`. AᵀA = `[[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]` = **Harris corner 의 structure tensor**.

**B5.** ① structure tensor 가 가역 ② 고유값이 너무 작지 않음. **Corner(두 고유값 모두 큼)** 에서 잘 풀리고, **flat**(둘 다 ≈0, 역행렬 불안정)과 **edge**(하나 ≈0, aperture problem)에서는 잘 안 됨.

**B6.** 움직임이 1픽셀보다 크면 small motion 가정이 깨져 오추정. Pyramid: LPF + 축소로 여러 해상도를 만들고, 가장 작은(coarse) 단계에서 먼저 추정(큰 움직임이 작게 보임) → 그 결과를 확대·warp 해서 다음 단계의 초기 flow 로 주고 보정 → 원본 단계까지 반복.

**B7.** `f2(x) = f1(x−d)` 를 전개하면 `b2 = b1 − 2A1d` → `d = −½ A1⁻¹(b2 − b1)` (A1 non-singular). 장점: ① separable convolution 으로 효율적 구현 ② multi-scale 분석으로 더 정확.

**B8.** 조명 변화, occlusion(가림), aperture problem

**B9.** optical flow 는 **두 프레임**(이전, 현재) 사이에서 계산하므로 첫 프레임은 prevFrame 이 비어있어 저장만 한다. OFE 는 밝기(intensity) 기반이고 `calcOpticalFlowFarneback` 입력이 8bit 단일 채널이라 흑백 사용.

**B10.** HOG(모양), HOF(움직임 방향), MBH. MBH 는 flow 의 u, v 성분 각각에 대해 공간 gradient `(ux, uy)`, `(vx, vy)` 를 구해 방향 histogram 을 만든 것 (움직임 경계). Improved DT 는 **RANSAC 으로 homography 를 추정해 카메라 움직임을 제거**(warp)한 후 descriptor 추출.

**C1.** 2·1 + 1·v − 4 = 0 → **v = 2**

**C2.** ΣIx² = 2, ΣIy² = 2, ΣIxIy = 0, ΣIxIt = −2, ΣIyIt = 4
`[2 0; 0 2]⁻¹ · [2, −4]` → **u = 1, v = −2**
(검산: 1·1 + 0 − 1 = 0 ✓, 0 + 1·(−2) + 2 = 0 ✓)

**C3.** det = 4·1 − 2·2 = **0** → 역행렬 없음 → **못 품**. Ix : Iy = 2 : 1 로 비례 = 한 방향 gradient 만 있는 **edge** (aperture problem).

**C4.** 8 → 4 → 2 → 1 : **3단계**

**C5.** (320/5) × (240/5) = 64 × 48 = **3072개** / 640×480 → 128 × 96 = **12288개**

**C6.** at<Point2f>(y=100, x=50) 이므로 시작점 = **Point(50, 100)**, ×2 → (3.0, −4.0) → 끝점 = **Point(53, 96)**

**C7.** b2 − b1 = (−4, 2), A1⁻¹ = I → d = −½ (−4, 2) = **(2, −1)**

**C8.** 셀 수 = 2×2×3 = 12 → HOG 12×8 = **96**, HOF 12×9 = **108**, MBH = MBHx 96 + MBHy 96 = **192**

</details>
