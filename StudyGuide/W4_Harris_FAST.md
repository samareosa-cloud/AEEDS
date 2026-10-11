# W4. Corner 검출: Harris · Gaussian · NMS · FAST · Corner+HOG 매칭

> 범위: **LectureNote 4** (Feature Extraction in Images (2)), 내 코드 `W4_cornertowhite`, `W4_CornerCircle`,
> 선배 코드 `versatile0010/Week4`, `shinjuyeop/W4_*`

---

## 1. 한눈에 보기

| 키워드 | 한 줄 요약 |
|---|---|
| Corner | edge 들이 교차하는 점 → **모든 방향**으로 밝기 변화 |
| 3가지 영역 | Flat: 모든 방향 변화 없음 / Edge: edge 방향 변화 없음, 가로지르면 변화 / Corner: 모든 방향 변화 |
| SSD | `E(Δx,Δy) = Σ_W [I(x+Δx, y+Δy) − I(x,y)]²` |
| Taylor 1차 | `I(x+Δx, y+Δy) ≈ I + Ix·Δx + Iy·Δy` |
| Structure tensor M | `[[ΣIx², ΣIxIy], [ΣIxIy, ΣIy²]]` → `E ≈ [Δx Δy] M [Δx Δy]ᵀ` |
| 고유값 | Flat: λ1≈λ2≈0 / Edge: λ1≫0, λ2≈0 / **Corner: λ1≫0, λ2≫0** |
| Corner metric | **`R = det(M) − k·tr²(M)`**, det = λ1λ2, tr = λ1+λ2, **k = 0.04 ~ 0.06** |
| R 해석 | **R > 0 corner / R < 0 edge / \|R\| 작음 flat** |
| 성질 | shift·rotation **불변**, 조명 변화에 강함, **scale 변화에 취약** |
| 구현 4단계 | ① Gaussian LPF ② Ix, Iy ③ 모든 픽셀 R ④ **Non-maxima suppression** |
| FAST | 반지름 3 원 위 **16픽셀** 중 **연속 N개**(N=9, 12)가 중심보다 밝거나 어두우면 corner |

---

## 2. 개념 설명

### 2-1. 수식 유도 흐름 (서술형 대비)
1. 작은 이동 (Δx, Δy) 에 대한 SSD:
   `E(Δx,Δy) = Σ_W [I(x+Δx, y+Δy) − I(x,y)]²`  (W: 현재 픽셀 중심 윈도우)
2. 이동이 작으면 1차 Taylor 근사:
   `I(x+Δx, y+Δy) ≈ I(x,y) + [Ix Iy][Δx Δy]ᵀ`  (2D 미분 → 벡터)
3. 대입하면 `I(x,y)` 가 상쇄되고
   ```
   E(Δx,Δy) ≈ [Δx Δy] | ΣIx²   ΣIxIy | [Δx]
                      | ΣIxIy  ΣIy²  | [Δy]
                        └── Structure tensor M ──┘
   ```
4. M 의 **고유값 분석** = gradient 분포의 모양:
   - 점들이 원점에 모임 → Flat (λ1, λ2 ≈ 0)
   - 한 방향으로 길게 퍼짐(타원) → Edge (λ1 ≫ 0, λ2 ≈ 0)
   - 모든 방향으로 넓게 퍼짐(큰 원) → Corner (λ1, λ2 ≫ 0)
   - 고유벡터 = 지배적인 방향, 고유값 = 그 방향의 에너지
5. 고유값 분해는 느림 → **det, trace 로 대체 (속도 향상)**
   `det(M) = λ1λ2`, `tr(M) = λ1 + λ2` → `R = det(M) − k·tr²(M)`

### 2-2. R 로 분류
```
R > 0 (크다)       → Corner
R < 0              → Edge
|R| 작음            → Flat
```
- 내 코드(`W4_Corner.cpp`): `R ≥ 0.05·maxR` → corner(흰색), `R ≤ 0.05·minR` → edge(검정), 나머지 flat(회색)

### 2-3. Harris 성질
- **Shift / Rotation invariant**: 회전해도 고유값은 그대로 → 검출 결과가 다른 영상에서도 "반복(repeated)"됨
- 조명(illumination), affine 변화에도 꽤 강함
- **Scale 에 취약**: 확대하면 corner 가 여러 개의 edge 로 보임 (윈도우 크기 고정이므로)

### 2-4. 구현 디테일 ① Gaussian LPF
- Gaussian kernel 과 convolution ↔ 주파수 영역에서 곱 (저주파 통과 = 노이즈 제거)
```
G(x) = 1/(√(2π)·σ) · exp( −‖x − c‖² / (2σ²) )      x: 커널 내 위치, c: 커널 중심
```
- 예: 5×5 커널, σ = 1, 중심 (m,n) 에서 (m+1, n−1) 위치 → ‖x−c‖² = 1² + 1² = 2
  `G = 1/√(2π) · exp(−2/2) = 0.3989 × 0.3679 = 0.1468`
- 실제 사용 시 커널 합이 1 이 되도록 **전체 합으로 나눠 정규화**
- versatile 코드의 3×3 근사 Gaussian: `{1,2,1, 2,4,2, 1,2,1} / 16`

| ‖x−c‖² | 0 | 1 | 2 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|
| G (σ=1) | 0.3989 | 0.2420 | **0.1468** | 0.0540 | 0.0327 | 0.0073 |

### 2-5. 구현 디테일 ② Non-maxima suppression (NMS)
```
R(y) ← y = argmax_{x∈W} R(x)
```
- 윈도우 W 안에서 **최대값만 남기고 나머지는 버림** → corner 주변에 뭉쳐서 여러 개 찍히는 것(노이즈) 제거
- 결과: corner 하나당 점 하나

### 2-6. FAST (Features from Accelerated Segment Test, Rosten & Drummond ECCV 2006)
- 현재 픽셀 P 에서 **반지름 3 원 위의 16개 픽셀** 검사
- **연속으로 N 개**가 P 보다 (임계값 이상) **밝거나 / 어두우면** corner
- N = 9 (FAST-9), N = 12 (FAST-12) → 과제: Harris 와 비교
- 미분·행렬 계산이 없어서 **매우 빠름** → ORB 의 keypoint 검출기로 쓰임 (W5)

### 2-7. 처리 시간 측정 (Windows)
```cpp
#include <Windows.h>
LARGE_INTEGER freq, start, stop;
double diff;
QueryPerformanceFrequency(&freq);  // 컴퓨터 주파수 (초당 카운트)
QueryPerformanceCounter(&start);   // 시작
/* Your Algorithm */
QueryPerformanceCounter(&stop);    // 끝
diff = (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart;  // 단위: 초
printf("%f\n", diff);
```

### 2-8. Practical Example: Corner 주변 HOG 로 매칭
1. `ref.bmp`, `tar.bmp` 에서 Harris corner 추출
2. 각 corner 주변 **16×16 block, 9-bin (0~180°)** HOG 계산
3. ref 의 각 corner 에 대해 tar 에서 HOG 가 가장 비슷한(거리 최소) corner 찾기
4. 두 점을 `line()` 으로 연결 (두 영상을 가로로 붙인 결과 영상에서 tar 의 x 좌표는 **+ ref.cols**)
5. 제약: **영상 Read/Write, 결과 Display 외 OpenCV 함수 금지**
```cpp
line(resultImg, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 2, 8, 0);
//    Mat        corner(ref)    corner(tar)    선 색(파랑)   두께 lineType shift
```

---

## 3. 핵심 코드

### 3-1. Harris R 계산 (내 `W4_CornerCircle.cpp` 정리)
```cpp
// Step 2: Ix, Iy (Prewitt)
for (y = 1; y < height - 1; y++) {
    for (x = 1; x < width - 1; x++) {
        convX = 0.0;  convY = 0.0;  maskIndex = 0;
        for (yy = -1; yy <= 1; yy++) {
            for (xx = -1; xx <= 1; xx++) {
                pixel = input.at<uchar>(y + yy, x + xx);
                convX += pixel * maskX[maskIndex];
                convY += pixel * maskY[maskIndex];
                maskIndex++;
            }
        }
        Ix[y * width + x] = convX;
        Iy[y * width + x] = convY;
    }
}

// Step 3: 3x3 윈도우에서 structure tensor → R
for (y = 2; y < height - 2; y++) {          // Ix 가 1~h-2 에만 있으므로 2부터
    for (x = 2; x < width - 2; x++) {
        sumIx2 = sumIy2 = sumIxIy = 0.0;
        for (yy = -1; yy <= 1; yy++) {
            for (xx = -1; xx <= 1; xx++) {
                gx = Ix[(y + yy) * width + (x + xx)];
                gy = Iy[(y + yy) * width + (x + xx)];
                sumIx2  += gx * gx;
                sumIy2  += gy * gy;
                sumIxIy += gx * gy;
            }
        }
        det   = sumIx2 * sumIy2 - sumIxIy * sumIxIy;   // ad − bc
        trace = sumIx2 + sumIy2;
        R[y * width + x] = det - 0.04 * trace * trace;
        if (R[y * width + x] > maxR) maxR = R[y * width + x];
    }
}
```

### 3-2. Gaussian kernel 만들기 + 적용 (Step 1)
```cpp
int ksize = 5, r = ksize / 2;
double sigma = 1.0, sum = 0.0;
double kernel[5][5];
for (int i = -r; i <= r; i++)
    for (int j = -r; j <= r; j++) {
        kernel[i + r][j + r] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (sqrt(2 * CV_PI) * sigma);
        sum += kernel[i + r][j + r];
    }
for (int i = 0; i < ksize; i++)
    for (int j = 0; j < ksize; j++)
        kernel[i][j] /= sum;                 // 합이 1이 되도록 정규화

for (y = r; y < height - r; y++)
    for (x = r; x < width - r; x++) {
        double conv = 0.0;
        for (int i = -r; i <= r; i++)
            for (int j = -r; j <= r; j++)
                conv += kernel[i + r][j + r] * input.at<uchar>(y + i, x + j);
        blurred.at<uchar>(y, x) = (uchar)conv;
    }
```

### 3-3. Threshold + NMS + circle (Step 4, 3×3 local max)
```cpp
threshold = maxR * 0.05;
for (y = 1; y < height - 1; y++) {
    for (x = 1; x < width - 1; x++) {
        float v = R[y * width + x];
        if (v < threshold) continue;                // 1) 임계값
        bool isMax = true;
        for (int dy = -1; dy <= 1 && isMax; dy++)   // 2) 주변 8개보다 커야 함
            for (int dx = -1; dx <= 1; dx++)
                if (R[(y + dy) * width + (x + dx)] > v) { isMax = false; break; }
        if (isMax) {
            Scalar c;  Point pCenter;  int radius = 3;
            pCenter.x = x;  pCenter.y = y;          // Point(x, y) — x 가 먼저!
            c.val[0] = 0;  c.val[1] = 0;  c.val[2] = 255;   // BGR → 빨강
            circle(resultImg, pCenter, radius, c, 2, 8, 0);
        }
    }
}
```

### 3-4. Corner 주변 HOG + 최근접 매칭 (shinjuyeop `W4_Assignment3` 요약)
```cpp
// 한 corner (cx, cy) 주변 16x16 block, 9-bin
for (y = cy - 8; y < cy + 8; ++y)
    for (x = cx - 8; x < cx + 8; ++x) {
        float gx = gray.at<uchar>(y, x + 1) - gray.at<uchar>(y, x - 1);   // center difference
        float gy = gray.at<uchar>(y + 1, x) - gray.at<uchar>(y - 1, x);
        float mag = sqrtf(gx * gx + gy * gy);
        float ang = atan2f(gy, gx) * 180.0f / PI;
        if (ang < 0) ang += 180;
        int bin = (int)(ang / 20);  if (bin >= 9) bin = 8;
        hist[bin] += mag;
    }
// L2 정규화 후, ref 의 i 번째 corner ↔ tar 의 모든 corner 중 L2 거리 최소인 j 선택
for (i = 0; i < n_ref; ++i) {
    float best = FLT_MAX;  int bestj = -1;
    for (j = 0; j < n_tar; ++j) {
        float d = L2dist(Href + i * 9, Htar + j * 9, 9);
        if (d < best) { best = d; bestj = j; }
    }
    line(canvas, Point(ref_x[i], ref_y[i]), Point(tar_x[bestj] + ref.cols, tar_y[bestj]),
         Scalar(255, 0, 0), 2, 8, 0);
}
```

---

## 4. 세 코드 비교 / 주의점

| 항목 | 내 코드 | versatile0010 | shinjuyeop |
|---|---|---|---|
| Gaussian | X | 3×3 `{1,2,1,2,4,2,1,2,1}/16` | X |
| Gradient | Prewitt | Sobel `/9` | Prewitt (/255 또는 정수) |
| 윈도우 | 3×3 | 3×3 | 3×3 (5×5 옵션) |
| 임계값 | `0.05 × maxR` | 고정값 `500000` | `minR + (maxR−minR) × 0.4` |
| NMS | X | 시도했으나 **버그** | 비디오 버전에서 3×3 local max |

- **versatile NMS 버그**: `r_map[x][y]` 를 `x, y = 0~2` 로 돌려서 **항상 영상 왼쪽 위 3×3** 만 봄. 올바르게는 `r_map[i + y - 1][j + x - 1]`.
- versatile 은 R 계산에서 `det`, `tr` 을 `y` 루프 **안쪽**에서 계산 (마지막 반복값이 남아서 결과는 맞지만 불필요한 연산).
- 내 `W4_Corner.cpp` 는 Gaussian, NMS 가 없음 → 슬라이드 4단계 중 ②③만 구현됨 (그래서 파일명이 `HarrisStep23`). 시험엔 ①④도 나올 수 있으니 3-2, 3-3 코드 외우기.

---

## 5. 예상 문제

### A. 코드 빈칸

**A1.** Structure tensor 와 R
```cpp
sumIx2  += ①_______;
sumIy2  += gy * gy;
sumIxIy += ②_______;
...
det   = ③______________________;
trace = ④______________;
R     = det - ⑤____ * ⑥______________;
```

**A2.** 분류 (내 코드)
```cpp
if (response >= ①______________)      result = Vec3b(255,255,255);   // corner
else if (response <= ②____________)   result = Vec3b(0,0,0);         // edge
else                                   result = Vec3b(③___,___,___);  // flat (회색)
```

**A3.** 빨간 원 그리기 (슬라이드 11)
```cpp
Scalar c;  Point pCenter;  int radius = 3;
pCenter.x = ①__________;  pCenter.y = your_corner_y;
c.val[0] = ②__;  c.val[1] = 0;  c.val[2] = ③____;
④______(resultImg, pCenter, radius, c, 2, 8, 0);
```

**A4.** Gaussian kernel
```cpp
kernel[i + r][j + r] = exp( ①_______________________ ) / (sqrt(2 * CV_PI) * sigma);
sum += kernel[i + r][j + r];
...
kernel[i][j] ②____ sum;
```

**A5.** NMS (3×3)
```cpp
if (v < threshold) ①________;
for (dy = -1; dy <= 1; dy++)
  for (dx = -1; dx <= 1; dx++)
    if (R[(y + dy) * width + (x + dx)] ②___ v) isMax = false;
```

**A6.** 시간 측정
```cpp
①_________________(&freq);
QueryPerformanceCounter(&start);
HarrisCornerDetector(...);
②_________________(&stop);
diff = (double)(stop.QuadPart - start.QuadPart) / ③_____________;
```

**A7.** 매칭 선 그리기 (ref 와 tar 를 가로로 붙인 영상)
```cpp
line(canvas, Point(x1, y1), Point(①______________, y2), Scalar(②_________), 2, 8, 0);  // 파란 선
```

### B. 개념 / 알고리즘

**B1.** Flat / Edge / Corner 를 "윈도우를 움직였을 때의 밝기 변화"로 설명하라.

**B2.** Harris 수식에서 Taylor 1차 근사를 쓰는 이유와 가정은?

**B3.** Structure tensor 의 두 고유값으로 세 영역을 구분하는 기준은?

**B4.** 고유값을 직접 구하지 않고 det, trace 를 쓰는 이유는? `R = det − k·tr²` 에서 k 의 범위는?

**B5.** Harris corner 검출기 구현 4단계를 순서대로 쓰라.

**B6.** Gaussian filtering 을 먼저 하는 이유는? 공간 영역 convolution 은 주파수 영역에서 무엇과 같은가?

**B7.** Non-maxima suppression 이 필요한 이유는?

**B8.** Harris 가 강인한 변화 2가지와 취약한 변화 1가지는? 취약한 이유는?

**B9.** FAST 알고리즘을 설명하라. Harris 보다 빠른 이유는?

### C. 결과 적기 / 계산

**C1.** k = 0.04 일 때 다음 윈도우 합에 대해 R 과 판정은?
(a) ΣIx² = 10, ΣIy² = 10, ΣIxIy = 0
(b) ΣIx² = 10, ΣIy² = 0, ΣIxIy = 0
(c) ΣIx² = 0.1, ΣIy² = 0.1, ΣIxIy = 0
(d) ΣIx² = 8, ΣIy² = 2, ΣIxIy = 4

**C2.** M = [[4, 0], [0, 1]] 일 때 λ1, λ2, det, tr, R(k=0.05) 은?

**C3.** σ = 1, 5×5 커널에서 (중심 기준) (+1, −1) 위치와 (+2, 0) 위치의 G 값은? (정규화 전, 슬라이드 식)

**C4.** FAST 에서 원 위 픽셀 수와 반지름은? N=12 일 때 16개 중 연속 11개만 밝다면 corner 인가?

**C5.** QueryPerformanceFrequency 가 10,000,000 이고 start = 5,000,000, stop = 5,250,000 이면 처리 시간은?

**C6.** ref.bmp 의 가로가 320 일 때, tar 의 corner (50, 80) 은 가로로 붙인 결과 영상에서 어디에 그려야 하는가?

---

<details>
<summary><b>정답 보기</b></summary>

**A1.** ① `gx * gx` ② `gx * gy` ③ `sumIx2 * sumIy2 - sumIxIy * sumIxIy` ④ `sumIx2 + sumIy2` ⑤ `0.04` (k) ⑥ `trace * trace`

**A2.** ① `corner_threshold` (= max_R × 0.05) ② `edge_threshold` (= min_R × 0.05) ③ `128, 128, 128`

**A3.** ① `your_corner_x` ② `0` ③ `255` ④ `circle`

**A4.** ① `-(i * i + j * j) / (2 * sigma * sigma)` ② `/=`

**A5.** ① `continue` ② `>`

**A6.** ① `QueryPerformanceFrequency` ② `QueryPerformanceCounter` ③ `freq.QuadPart`

**A7.** ① `x2 + ref.cols` ② `255, 0, 0` (BGR → 파랑)

**B1.** 작은 윈도우를 여러 방향으로 움직일 때 — Flat: 어느 방향이든 변화 없음. Edge: edge 를 따라 움직이면 변화 없고, edge 를 가로지르면 변화. Corner: 모든 방향에서 큰 변화.

**B2.** SSD 식 안의 `I(x+Δx, y+Δy)` 를 Δx, Δy 의 1차식으로 바꿔 행렬(이차형식) 형태로 만들기 위해. 가정: **이동량이 매우 작다**(small movement → small difference).

**B3.** Flat: λ1 ≈ λ2 ≈ 0 / Edge: λ1 ≫ 0, λ2 ≈ 0 (한 방향만 큼) / Corner: λ1 ≫ 0, λ2 ≫ 0 (두 방향 모두 큼).

**B4.** 고유값 분해(제곱근 계산 등)가 느려서 → det = λ1λ2, tr = λ1+λ2 로 **속도 향상**. k = **0.04 ~ 0.06** (경험적).

**B5.** ① Gaussian kernel 로 LPF ② image gradient Ix, Iy 계산 ③ 모든 픽셀에서 R 계산 ④ Non-maxima suppression (그 후 circle 로 표시)

**B6.** 미분은 노이즈를 증폭시키므로 먼저 저주파 통과(노이즈 제거)를 한다. 공간 영역 convolution ↔ **주파수 영역 곱셈**.

**B7.** corner 근처의 여러 픽셀이 모두 임계값을 넘어 한 corner 에 점이 뭉쳐서 여러 개 검출됨 → 윈도우 내 최대값만 남겨 corner 당 하나만 남기고 노이즈성 결과 제거.

**B8.** 강인: **shift(이동), rotation(회전)** (+ 조명, affine 에도 비교적 강함). 취약: **scale**. 윈도우 크기가 고정이라 확대하면 같은 corner 가 완만한 곡선(여러 edge)으로 보여 corner 로 검출되지 않음.

**B9.** 현재 픽셀 P 를 중심으로 반지름 3 원 위 16개 픽셀 중 연속 N 개(9 또는 12)가 P 보다 (임계값 이상) 모두 밝거나 모두 어두우면 corner. 미분·행렬·곱셈 없이 **밝기 비교만** 하므로 매우 빠름.

**C1.** (a) det = 100 − 0 = 100, tr = 20 → R = 100 − 0.04×400 = **84 > 0 → Corner**
(b) det = 0, tr = 10 → R = 0 − 0.04×100 = **−4 < 0 → Edge**
(c) det = 0.01, tr = 0.2 → R = 0.01 − 0.04×0.04 = **0.0084 ≈ 0 → Flat**
(d) det = 16 − 16 = 0, tr = 10 → R = **−4 → Edge** (Ix, Iy 가 비례 = 한 방향 gradient)

**C2.** λ1 = 4, λ2 = 1, det = 4, tr = 5, R = 4 − 0.05×25 = **2.75 (> 0, corner 쪽)**

**C3.** (+1, −1): ‖x−c‖² = 2 → **0.1468** / (+2, 0): ‖x−c‖² = 4 → 0.3989 × e⁻² = **0.0540**

**C4.** **16개, 반지름 3**. N = 12 인데 11개만 연속 → **corner 아님** (N = 9 기준이면 corner).

**C5.** (5,250,000 − 5,000,000) / 10,000,000 = **0.025 초 (25 ms)**

**C6.** **(50 + 320, 80) = (370, 80)**

</details>
