# W5. 영상(Video) 특징: STIP · LoG · VideoCapture · FAST · ORB 매칭 · RANSAC

> 범위: **LectureNote 5** (Feature Extraction in Videos (1)),
> 선배 코드 `versatile0010/Week5`, `shinjuyeop/W5_*` (`utils.cpp/h` 포함)

---

## 1. 한눈에 보기

| 키워드 | 한 줄 요약 |
|---|---|
| STIP | Space-Time Interest Point (Laptev, IJCV 2005) = **Harris 를 시공간(x, y, t)으로 확장** |
| 시공간 Gaussian | 공간 scale σ_l², 시간 scale τ_l² 를 **따로** 둠 (separable) |
| 3×3 tensor S | Lx, Ly, Lt 로 만든 second-moment matrix |
| STIP metric | **`H = det(S) − k·trace³(S)`** = λ1λ2λ3 − k(λ1+λ2+λ3)³ |
| LoG | Laplacian of Gaussian = 멕시칸 모자. 극값 = **blob**, 크기는 **σ** 가 결정 |
| Codebook | 특징점 주변 patch → HOG/HOF → codeword histogram → SVM (**영상 길이 무관**) |
| VideoCapture | `VideoCapture capture(0); capture >> frame;` |
| ORB | **FAST corner + (rotated) BRIEF descriptor** (Rublee, ICCV 2011) |
| BRIEF | 두 점 밝기 비교로 만든 **이진 문자열** → **Hamming distance** 로 비교 |
| knnMatch k=2 | 가장 가까운 2개 후보 찾기 |
| NNDR | `dist1 < 0.8 × dist2` 이면 좋은 매칭 (ratio test) |
| RANSAC | 랜덤 샘플로 모델(homography) 추정 → **consensus(동의) 최대** 모델 채택 → inlier/outlier 분리 |

---

## 2. 개념 설명

### 2-1. Space-time Interest Point (STIP)
- 영상(volume) 데이터에서의 corner = **움직임이 갑자기 바뀌는 점** (예: 공이 벽에 부딪히는 순간, 두 물체 충돌)
- scale 에 따라 결과가 달라짐 → scale selection 필요

**(1) 시공간 Gaussian 으로 smoothing**
```
L(·; σ_l², τ_l²) = G(·; σ_l², τ_l²) * f(·)
G(x, y, t; σ_l², τ_l²) = 1 / √((2π)³ σ_l⁴ τ_l²) · exp( −(x²+y²)/(2σ_l²) − t²/(2τ_l²) )
```
- 공간과 시간 특징은 **독립(separable)** → 각각 별도 scale 파라미터
- 각 scale 이 시공간 corner 검출에 중요

**(2) Space-time second-moment matrix (3×3)**
```
        | Lx²  LxLy  LxLt |
S = G * | LxLy Ly²   LyLt |       Lx = ∂x(G*f), Ly = ∂y(G*f), Lt = ∂t(G*f)
        | LxLt LyLt  Lt²  |
```
- Harris 의 2×2 M 이 3×3 으로 확장. 고유값 λ1, λ2, λ3 사용

**(3) Harris 확장 metric**
```
H = det(S) − k·trace³(S) = λ1λ2λ3 − k(λ1+λ2+λ3)³
  = λ1³ ( αβ − k(1+α+β)³ ),    α = λ2/λ1,  β = λ3/λ1
```
- **H 의 국소적 양(+)의 시공간 최대값** 을 검출
- Harris 는 tr² (2차원), STIP 은 **tr³** (3차원) ← 차원 수와 같은 지수!

**(4) Scale selection: 2차 미분 (LoG) 사용**
- SIFT 처럼 일종의 greedy search
- Gaussian 두 번 = 분산 더하기: `G(σ²) * G(σ0²) = G(σ² + σ0²)`
- LoG 단면: 가운데 봉우리, zero-crossing 간격 = **2√2·σ**. blob 크기가 여기에 맞으면 응답 최대
- **LoG 극값 = blob**
  - Maxima: 밝은 배경 위 **어두운** blob
  - Minima: 어두운 배경 위 **밝은** blob
  - **blob 의 크기(scale) 는 LoG 필터의 σ 가 결정** (σ=2 → 작은 blob, σ=10 → 큰 blob)

**(5) Motion feature: Codebook 기반 표현**
```
특징점 추출 → 시공간 patch → 기술자(HOG: 모양, HOF: 움직임 방향) → codebook 으로 양자화
→ codeword 개수 histogram → 분류기(SVM 등)
```
- 영상 길이에 상관없이 고정 길이 벡터 → **행동 인식(action recognition)** 에 유용

### 2-2. 비디오 처리 기본 구조 (webcam)
```cpp
void main() {
    VideoCapture capture(0);          // 0번 카메라 (객체: capture)
    Mat frame;
    if (!capture.isOpened()) {
        printf("Couldn't open the web camera...\n");
        return;
    }
    while (true) {
        capture >> frame;             // 프레임 한 장 획득
        /* Our algorithm */
        imshow("Video", frame);
        if (waitKey(30) >= 0) break;  // 30ms 대기, 아무 키나 누르면 종료
    }
}
```
- Harris 함수는 **별도 .cpp 파일**(예: `cornerDetector.cpp`)로 만들고 main 에서 호출
- 매 프레임 `QueryPerformance...` 로 시간 측정 → **평균 처리 속도** 계산 (총 시간 / 프레임 수, FPS = 1 / 평균시간)
- (슬라이드 함정) `Mat cornerMap(frame.rows, frames.cols, CV_8UC3);` 는 frame 을 받기 **전**이라 rows=cols=0 (그리고 `frames` 오타). 크기는 첫 프레임 이후에 정해짐.

### 2-3. FAST → ORB
**ORB = FAST corner + BRIEF descriptor** — 전체 절차:
1. **FAST** 로 corner 검출 → 그 중 **Harris 기준 상위 N 개** 선택
2. keypoint 의 **orientation** 계산 → **밝기 가중 중심(intensity centroid)** 과 corner 점을 잇는 벡터
3. 2)의 방향으로 BRIEF 를 회전(steering) → **회전 불변(rotation-invariant) BRIEF**

**BRIEF 이진 테스트**
```
τ(p; x, y) = 1  if p(x) < p(y)
             0  otherwise
```
슬라이드 예시 (중심 29, 파란 원 35 에서 시작해 시계방향, "이웃 < 중심" 이면 1):
```
33 35 36
27 29 39      순서: 35, 36, 39, 35, 30, 24, 27, 33
24 30 35      비교: 0   0   0   0   0   1   1   0   → 00000110
```
- 이진 문자열이므로 거리 = **Hamming distance** (서로 다른 비트 수) → XOR + 비트 카운트, 매우 빠름

### 2-4. Feature matching = keypoint + descriptor
- corner 는 좋은 keypoint, 주변 내용을 기술(descriptor) → 인식용 특징
- 그래서 local descriptor 설계가 중요

### 2-5. ORB 구현 (utils.cpp/h 사용, OpenCV 함수)
**전역 파라미터 (main.cpp 맨 위)**
```cpp
int   ORB_MAX_KPTS = 1500;           // 최대 keypoint 수
float ORB_SCALE_FACTOR = 1.2;        // pyramid 축소 비율
int   ORB_PYRAMID_LEVELS = 4;        // pyramid 단계 수
float ORB_EDGE_THRESHOLD = 31.0;     // 경계 제외 크기
int   ORB_FIRST_PYRAMID_LEVEL = 0;
int   ORB_WTA_K = 2;                 // BRIEF 비교 점 개수 (2 → 비트 1개)
int   ORB_PATCH_SIZE = 31;           // BRIEF patch 크기
float MIN_H_ERROR = 2.50f;           // RANSAC: inlier 로 인정하는 최대 오차(px)
float DRATIO = 0.80f;                // NNDR ratio
```

**처리 흐름**
```
imread(…, 0) (흑백)
→ ORB::create(…, ORB::HARRIS_SCORE, …)
→ detectAndCompute(img, noArray(), kpts, desc)       // keypoint + descriptor
→ matcher_l1->knnMatch(desc1, desc2, dmatches, 2)    // BruteForce-Hamming, 후보 2개
→ matches2points_nndr(..., DRATIO)                   // ratio test
→ compute_inliers_ransac(..., MIN_H_ERROR, false)    // findHomography + RANSAC
→ 결과: keypoint 수, matches, inliers, outliers, inlier ratio, 시간(ms)
```

| matcher | 거리 | 용도 |
|---|---|---|
| `"BruteForce"` (matcher_l2) | L2 (Euclidean) | 실수형 descriptor (SIFT, SURF) |
| `"BruteForce-Hamming"` (matcher_l1) | **Hamming** | **이진 descriptor (ORB/BRIEF)** ← 사용 |

### 2-6. NNDR (Nearest Neighbor Distance Ratio, Lowe ratio test)
```cpp
if (dist1 < nndr * dist2) { /* 매칭 채택: 두 점을 연속으로 push */ }
```
- 1등과 2등 거리가 비슷하면 애매한 매칭(반복 패턴 등) → 버림
- `pmatches` 에 [ref점, tar점, ref점, tar점, …] **쌍으로** 저장 → 매칭 수 = `size() / 2`

### 2-7. RANSAC (RANdom SAmple Consensus) ★ 슬라이드 22 질문
1. 매칭 쌍 중 **최소 개수를 랜덤 샘플** (homography 는 4쌍)
2. 그 샘플로 모델(homography H) 추정
3. 모든 매칭에 H 를 적용해 오차가 임계값(MIN_H_ERROR = 2.5px) 이내인 점 수 = **consensus** 계산
4. 1~3 반복 → **consensus 가 최대인 모델** 선택
- **Inlier**: 최종 모델과 일치하는(오차 ≤ 임계값) 올바른 매칭
- **Outlier**: 모델과 맞지 않는 잘못된 매칭
- 장점: Least square 와 달리 **outlier 에 강인** (W6 부록 그림: LS 는 outlier 에 끌려감, RANSAC 은 무시)
- `MIN_H_ERROR` ↑ → inlier 기준 느슨 → inlier 수 ↑ (오매칭도 포함될 수 있음)

### 2-8. 시간 측정 (OpenCV)
```cpp
t1 = getTickCount();
/* ... */
t2 = getTickCount();
torb = 1000.0 * (t2 - t1) / getTickFrequency();   // ms
```

### 2-9. Practical example: 비디오로 확장
- 참조 영상(또는 **첫 프레임**)과 webcam 매 프레임을 ORB 매칭
- 참조 영상 특징은 **루프 밖에서 한 번만** 계산 (매 프레임 다시 할 필요 없음)

---

## 3. 핵심 코드

### 3-1. ORB 매칭 main (슬라이드 18~20)
```cpp
Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("BruteForce-Hamming");
img1 = imread("./images/1.bmp", 0);
img2 = imread("./images/2.bmp", 0);
img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

Mat img1_rgb_orb = Mat(Size(img1.cols, img1.rows), CV_8UC3);
Mat img2_rgb_orb = Mat(Size(img2.cols, img1.rows), CV_8UC3);
Mat img_com_orb  = Mat(Size(img1.cols * 2, img1.rows), CV_8UC3);   // 좌우로 붙일 캔버스

Ptr<ORB> orb = ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
                           ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K,
                           ORB::HARRIS_SCORE, ORB_PATCH_SIZE);

t1 = getTickCount();
orb->detectAndCompute(img1, noArray(), kpts1_orb, desc1_orb, false);
orb->detectAndCompute(img2, noArray(), kpts2_orb, desc2_orb, false);

matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);
matches2points_nndr(kpts1_orb, kpts2_orb, dmatches_orb, matches_orb, DRATIO);
compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);

nkpts1_orb    = kpts1_orb.size();
nkpts2_orb    = kpts2_orb.size();
nmatches_orb  = matches_orb.size() / 2;          // 쌍으로 저장되므로 /2
ninliers_orb  = inliers_orb.size() / 2;
noutliers_orb = nmatches_orb - ninliers_orb;
ratio_orb     = 100.0 * (float)ninliers_orb / (float)nmatches_orb;   // %
t2 = getTickCount();
torb = 1000.0 * (t2 - t1) / getTickFrequency();

cvtColor(img1, img1_rgb_orb, COLOR_GRAY2BGR);
cvtColor(img2, img2_rgb_orb, COLOR_GRAY2BGR);
draw_keypoints(img1_rgb_orb, kpts1_orb);
draw_keypoints(img2_rgb_orb, kpts2_orb);
draw_inliers(img1_rgb_orb, img2_rgb_orb, img_com_orb, inliers_orb, 0);
imshow("ORB", img_com_orb);
waitKey(0);
```

### 3-2. utils.cpp 핵심 함수
```cpp
void matches2points_nndr(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
                         const vector<vector<DMatch>>& matches, vector<Point2f>& pmatches, float nndr)
{
    for (size_t i = 0; i < matches.size(); i++) {
        DMatch dmatch = matches[i][0];
        float dist1 = matches[i][0].distance;    // 1등 거리
        float dist2 = matches[i][1].distance;    // 2등 거리
        if (dist1 < nndr * dist2) {
            pmatches.push_back(train[dmatch.queryIdx].pt);
            pmatches.push_back(query[dmatch.trainIdx].pt);
        }
    }
}

void compute_inliers_ransac(const vector<Point2f>& matches, vector<Point2f>& inliers,
                            float error, bool use_fund)
{
    vector<Point2f> points1, points2;
    int npoints = matches.size() / 2;
    Mat status = Mat::zeros(npoints, 1, CV_8UC1);
    for (size_t i = 0; i < matches.size(); i += 2) {
        points1.push_back(matches[i]);
        points2.push_back(matches[i + 1]);
    }
    if (npoints > 8) {
        if (use_fund) findFundamentalMat(points1, points2, FM_RANSAC, error, 0.99, status);
        else          findHomography(points1, points2, RANSAC, error, status);
        for (int i = 0; i < npoints; i++)
            if (status.at<uchar>(i) == 1) {           // 1 = inlier
                inliers.push_back(points1[i]);
                inliers.push_back(points2[i]);
            }
    }
}
```

### 3-3. Harris in video (versatile `1007_Harris_corner_video.cpp` 요약)
```cpp
VideoCapture capture(0);
LARGE_INTEGER freq, start, stop;
double diff, total = 0;  int frames = 0;
while (true) {
    capture >> frame;
    resize(frame, frame, Size(500, 500));
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    cornerMap = Harris_CornerDetect(frame, 0.04);       // cornerDetector.cpp 에 구현
    QueryPerformanceCounter(&stop);
    diff = (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart;
    total += diff;  frames++;
    printf("%f (avg %f)\n", diff, total / frames);     // 평균 처리 속도
    imshow("Corner Detection", cornerMap);
    if (waitKey(30) >= 0) break;
}
```

---

## 4. 선배 코드 주의점
- versatile `1007_ORB_feature_match.cpp`: `imread("…ref.jpg", C)` — `C` 는 오타(컴파일 에러). `0` 이어야 함.
- versatile `1007_ORB_video.cpp`: `MIN_H_ERROR = 120.5` 로 크게 키움 (주석: "얘를 조절하면 RANSAC 영향을 알 수 있음") → 거의 모든 매칭이 inlier 가 됨.
- shinjuyeop `W5_FeatureExtraction`: `DRATIO = 0.65`, `MIN_H_ERROR = 1.5` 로 더 엄격하게 → 매칭 수 ↓, 정확도 ↑. 또 knn 결과가 2개 미만인 항목 제거 (`matches[i][1]` 접근 에러 방지).
- `detectAndCompute` 에 원본 컬러 `frame` 을 넣은 versatile 코드 vs 흑백으로 넣은 shin 코드 — 슬라이드는 흑백 영상(`imread(…, 0)`) 사용.

---

## 5. 예상 문제

### A. 코드 빈칸

**A1.** 웹캠 기본 구조
```cpp
①____________ capture(0);
Mat frame;
if (!capture.②__________()) { printf("Couldn't open the web camera...\n"); return; }
while (true) {
    capture ③___ frame;
    imshow("Video", frame);
    if (④__________ >= 0) break;
}
```

**A2.** ORB 생성과 검출
```cpp
Ptr<ORB> orb = ORB::①______(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
        ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, ORB::②____________, ORB_PATCH_SIZE);
orb->③________________(img1, noArray(), kpts1_orb, desc1_orb, false);
```

**A3.** 매칭
```cpp
Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("①__________________");
matcher_l1->②__________(desc1_orb, desc2_orb, dmatches_orb, ③__);
```

**A4.** NNDR
```cpp
dist1 = matches[i][①__].distance;
dist2 = matches[i][②__].distance;
if (③__________________) { pmatches.push_back(...); pmatches.push_back(...); }
```

**A5.** 결과 통계
```cpp
nmatches_orb  = matches_orb.size() ①____;
noutliers_orb = ②_______________________;
ratio_orb     = ③_____ * (float)ninliers_orb / (float)nmatches_orb;
torb          = 1000.0 * (t2 - t1) / ④____________________;
```

**A6.** RANSAC
```cpp
H = findHomography(points1, points2, ①________, error, status);
if (status.at<unsigned char>(i) == ②__) { inliers.push_back(points1[i]); ... }
```

### B. 개념 / 알고리즘

**B1.** STIP 은 무엇을 시공간으로 확장한 것인가? 2D 와 비교해 행렬 크기와 metric 식이 어떻게 달라지는가?

**B2.** 시공간 Gaussian 에서 공간과 시간 scale 을 따로 두는 이유는?

**B3.** LoG 필터의 극값은 무엇을 의미하는가? Maxima/Minima 각각은? blob 크기를 결정하는 것은?

**B4.** Codebook 기반 표현이 행동 인식에 유용한 이유는?

**B5.** ORB 의 3단계 절차를 쓰라.

**B6.** ORB descriptor 를 비교할 때 L2 대신 Hamming distance 를 쓰는 이유는?

**B7.** knnMatch 에서 k = 2 로 하는 이유는? (NNDR 과 연관지어)

**B8.** RANSAC 이란? Inlier 와 outlier 는?

**B9.** `MIN_H_ERROR` 를 2.5 에서 120 으로 키우면 결과가 어떻게 달라지는가?

**B10.** Harris 를 비디오에 적용할 때 슬라이드가 요구한 2가지 (코드 구조, 출력) 는?

### C. 결과 적기 / 계산

**C1.** 다음 3×3 패치에서 중심과 비교한 BRIEF 비트열은? (위 가운데에서 시작, 시계방향, 이웃 < 중심 → 1)
```
50 60 70
40 55 80
30 20 90
```

**C2.** 두 이진 descriptor `10110010` 과 `10011010` 의 Hamming distance 는?

**C3.** DRATIO = 0.8 일 때 다음 매칭의 채택 여부는?
(a) dist1 = 20, dist2 = 30 (b) dist1 = 28, dist2 = 30 (c) dist1 = 40, dist2 = 50

**C4.** `matches_orb.size() = 400`, `inliers_orb.size() = 300` 이면 matches, inliers, outliers, inlier ratio 는?

**C5.** `getTickFrequency() = 1e9`, `t2 − t1 = 35,000,000` 이면 torb(ms) 는?

**C6.** 10 프레임 처리시간 합이 0.5초면 평균 처리 시간과 FPS 는?

**C7.** STIP 에서 λ1 = λ2 = λ3 = 1, k = 0.005 일 때 H 는?

---

<details>
<summary><b>정답 보기</b></summary>

**A1.** ① `VideoCapture` ② `isOpened` ③ `>>` ④ `waitKey(30)`

**A2.** ① `create` ② `HARRIS_SCORE` ③ `detectAndCompute`

**A3.** ① `BruteForce-Hamming` ② `knnMatch` ③ `2`

**A4.** ① `0` ② `1` ③ `dist1 < nndr * dist2`

**A5.** ① `/ 2` ② `nmatches_orb - ninliers_orb` ③ `100.0` ④ `getTickFrequency()`

**A6.** ① `RANSAC` ② `1`

**B1.** Harris corner detector 를 (x, y, t) 로 확장. 2×2 structure tensor → **3×3** second-moment matrix (Lx, Ly, Lt). metric: `det(M) − k·tr²(M)` → **`det(S) − k·tr³(S)`**.

**B2.** 공간 특징과 시간(움직임) 특징은 독립적이고(separable) 크기가 서로 다르므로, 각각 scale 을 조절해야 다양한 크기·속도의 시공간 corner 를 검출할 수 있다.

**B3.** LoG 극값 = **blob** 위치. Maxima: 밝은 배경 위 어두운 blob / Minima: 어두운 배경 위 밝은 blob. blob 크기는 **LoG 의 σ** 가 결정 (zero-crossing 간격 2√2σ 와 blob 크기가 맞을 때 응답 최대).

**B4.** 영상마다 길이(특징점 개수)가 달라도 codeword histogram 은 항상 codebook 크기의 **고정 길이 벡터**가 되므로 SVM 같은 분류기에 바로 넣을 수 있다.

**B5.** ① FAST 로 corner 검출 후 Harris 기준 상위 N개 선택 ② intensity centroid 와 corner 를 잇는 벡터로 orientation 계산 ③ 그 방향으로 BRIEF 를 회전시켜 rotation-invariant descriptor 계산.

**B6.** BRIEF 는 0/1 비트열이라 L2 가 의미 없고, Hamming(XOR 후 1의 개수)이 정확하면서 매우 빠름.

**B7.** 1등과 2등 후보 거리를 비교(NNDR)하려면 최근접 2개가 필요. 1등이 2등보다 충분히 가까울 때만(`d1 < 0.8·d2`) 신뢰할 수 있는 매칭으로 채택.

**B8.** 랜덤으로 최소 샘플을 뽑아 모델(homography)을 추정하고, 그 모델에 오차 임계값 안으로 들어오는 점의 수(consensus)를 세는 것을 반복해 consensus 최대 모델을 고르는 방법. Inlier = 모델과 일치하는 올바른 매칭, Outlier = 모델과 맞지 않는 잘못된 매칭. outlier 에 강인한 것이 장점.

**B9.** inlier 로 인정하는 오차 범위가 매우 커져서 거의 모든 매칭이 inlier 로 판정 → inlier ratio ↑ 이지만 잘못된 매칭도 걸러지지 않음 (RANSAC 효과가 사라짐).

**B10.** ① Harris 함수를 **별도 .cpp 파일**(cornerDetector.cpp)로 만들어 main 의 while 루프에 삽입 ② QueryPerformance 로 **처리 속도 출력**하고 **평균 처리 속도** 계산.

**C1.** 시작 60(위 가운데) → 시계방향: 60, 70, 80, 90, 20, 30, 40, 50 / 중심 55
60<55? 0 · 70? 0 · 80? 0 · 90? 0 · 20? 1 · 30? 1 · 40? 1 · 50? 1 → **00001111**

**C2.** XOR = `00101000` → **2**

**C3.** (a) 20 < 24 → **채택** (b) 28 < 24 ✗ → **버림** (c) 40 < 40 ✗ → **버림** (같으면 버림)

**C4.** matches = 200, inliers = 150, outliers = 50, ratio = **75%**

**C5.** 1000 × 35,000,000 / 1e9 = **35 ms**

**C6.** 평균 0.05초(50ms), FPS = 1/0.05 = **20**

**C7.** det = 1, tr = 3 → H = 1 − 0.005 × 27 = **0.865**

</details>
