# 종합 모의고사 (W2 ~ W6)

> 주차별 파일의 예상 문제와 **다른 숫자·다른 코드 조각**으로 만들었다. 실제 시험처럼 정답을 가리고 먼저 풀어보기.
> 구성: Part 1 코드 빈칸 (8) · Part 2 서술 (8) · Part 3 계산/결과 (12) · Part 4 버그 찾기 (3)

---

## Part 1. 코드 빈칸

**1-1. [W2] Bilinear resize**
```cpp
for (y = 0; y < re_h; y++)
  for (x = 0; x < re_w; x++) {
    pos_x = ①__________;   pos_y = y / scale;
    sx = (int)pos_x;        sy = (int)pos_y;
    p = ②__________;        q = pos_y - sy;
    v = (1-p)*(1-q)*I(sy,sx) + ③________*I(sy,sx+1)
      + (1-p)*q*I(sy+1,sx)   + p*q*I(④____, ⑤____);
  }
```

**1-2. [W3] Gradient magnitude**
```cpp
int mask_y[9] = { ①______________________ };
...
conv_y += mask_y[(yy - (y - 1)) * 3 + (xx - (x - 1))] * input.at<uchar>(②___, ③___);
...
val[y * width + x] = ④______________________;
if (min > val[y * width + x]) min = val[y * width + x];
```

**1-3. [W3] Orientation histogram**
```cpp
dir = ①______(conv_y, conv_x) * 180 / PI;
if (②________) dir += 180;
histogram[(int)(dir / ③___)] += ④____________;
```

**1-4. [W4] Harris**
```cpp
for (yy = -1; yy <= 1; yy++)
  for (xx = -1; xx <= 1; xx++) {
    gx = Ix[①__________________];   gy = Iy[(y + yy) * width + (x + xx)];
    Sxx += gx * gx;  Syy += gy * gy;  Sxy += ②_______;
  }
R[y * width + x] = (Sxx * Syy - ③_________) - k * (④_________) * (Sxx + Syy);
```

**1-5. [W4] 코너 표시**
```cpp
Scalar c(①___, ②___, ③___);          // 빨간색
circle(resultImg, Point(④___, ⑤___), 3, c, 2, 8, 0);
```

**1-6. [W5] ORB 파이프라인 (함수 이름)**
```cpp
orb->①________________(img1, noArray(), kpts1, desc1, false);
orb->detectAndCompute(img2, noArray(), kpts2, desc2, false);
matcher_l1->②__________(desc1, desc2, dmatches, 2);
③____________________(kpts1, kpts2, dmatches, matches, DRATIO);
④____________________(matches, inliers, MIN_H_ERROR, false);
```

**1-7. [W5] 매칭 통계**
```cpp
nmatches_orb  = ①________________;
ninliers_orb  = inliers_orb.size() / 2;
noutliers_orb = ②________________;
ratio_orb     = 100.0 * (float)③__________ / (float)④__________;
```

**1-8. [W6] Farneback 시각화**
```cpp
cvtColor(img, frame, ①______________);
if (prevFrame.empty() == false) {
    calcOpticalFlowFarneback(②________, ③_____, flow, 0.4, 1, 12, 2, 8, 1.2, 0);
    for (y = 0; y < Height; y += 5)
      for (x = 0; x < Width; x += 5) {
        const Point2f f = flow.at<Point2f>(④___, ⑤___) * 2;
        line(img, Point(x, y), Point(cvRound(x + f.x), cvRound(y + f.y)), Scalar(0, 255, 0));
      }
    frame.copyTo(⑥__________);
}
```

---

## Part 2. 서술

**2-1.** [W2] 영상 확대·회전에 backward mapping 을 쓰는 이유를 forward mapping 의 문제점과 함께 서술하라.

**2-2.** [W3] Gradient phase 가 edge 의 "법선 방향"이라는 것을 예를 들어 설명하라.

**2-3.** [W3] HOG 에서 cell, block, L2 정규화의 역할을 각각 쓰라.

**2-4.** [W4] Harris 에서 structure tensor 의 고유값 분석이 의미하는 바와, det/trace 로 대체하는 이유를 쓰라.

**2-5.** [W4] Harris 구현 4단계를 쓰고, Gaussian 과 NMS 가 각각 왜 필요한지 쓰라.

**2-6.** [W5] ORB 의 절차를 쓰고, BRIEF 가 Hamming distance 로 비교되는 이유를 쓰라.

**2-7.** [W5] RANSAC 의 동작과 inlier/outlier 를 설명하라. Least square 와 비교했을 때 장점은?

**2-8.** [W6] Lucas-Kanade 의 가정·해법·풀림 조건·한계(와 해결책)를 순서대로 쓰라.

---

## Part 3. 계산 / 결과 적기

**3-1.** [W2] I(0,0)=100, I(1,0)=200, I(0,1)=50, I(1,1)=150 (I(x,y)). 원본 좌표 (0.5, 0.25) 에서 NN / Average / Bilinear 값은?

**3-2.** [W2] W = H = 100, θ = 180° 로 중심 기준 backward 회전할 때, 결과 픽셀 (60, 50) 은 원본 어디를 참조하는가?

**3-3.** [W3] 다음 3×3 에서 Prewitt 로 fx, fy, magnitude, 방향(0~180), bin 번호는?
```
20 20 20
20 50 80
80 80 80
```

**3-4.** [W3] 같은 lattice 에서 center difference 로 fx, fy, magnitude, 방향은?

**3-5.** [W3] 96 × 160 (W × H) 영상, block 16, stride 8, cell 8, 9-bin → block 수, HOG 차원(cell 있음/없음)은?

**3-6.** [W4] k = 0.05 일 때 R 과 판정:
(a) ΣIx² = 6, ΣIy² = 4, ΣIxIy = 2
(b) ΣIx² = 9, ΣIy² = 1, ΣIxIy = 3

**3-7.** [W4] σ = 1 Gaussian (정규화 전, 슬라이드 식) 에서 중심 대비 (+1, 0) 와 (+2, +2) 위치의 값은?

**3-8.** [W5] BRIEF (위 가운데 시작, 시계방향, 이웃 < 중심 → 1) 비트열과, 그것과 `01100110` 의 Hamming distance 는?
```
10 90 20
80 50 30
40 60 70
```

**3-9.** [W5] `matches_orb.size() = 520`, `inliers_orb.size() = 416` 일 때 matches / inliers / outliers / ratio 는? 또한 DRATIO = 0.8 에서 (dist1, dist2) = (15, 20), (16, 20), (30, 45) 의 채택 여부는?

**3-10.** [W6] Ix = 3, Iy = −1, It = 5, v = 2 일 때 OFE 를 만족하는 u 는?

**3-11.** [W6] 3×3 patch 중 다음 3픽셀만 gradient 가 있다: (Ix, Iy, It) = (1, 0, −1), (0, 1, −2), (1, 1, −3). LK 로 (u, v) 는?

**3-12.** [W4~W6] 다음 값을 구하라.
(a) freq = 3,000,000, stop − start = 90,000 일 때 처리 시간(ms)과 FPS
(b) 400×300 영상, 격자 간격 10 → 그려지는 flow 선 개수
(c) 16픽셀 움직임을 1픽셀로 만들기 위한 pyr_scale 0.5 pyramid 단계 수
(d) STIP 에서 λ = (2, 1, 1), k = 0.01 일 때 H

---

## Part 4. 버그 찾기

**4-1.** HOG 방향 계산 코드의 오류 2개는?
```cpp
dir = atan2(conv_x, conv_y) * 180 / PI;
bin = (int)(dir / 20);
hist[bin] += mag;
```

**4-2.** Harris NMS 코드의 오류는?
```cpp
for (i = 1; i < rows - 1; i++)
  for (j = 1; j < cols - 1; j++) {
    float max = 0;
    for (x = 0; x < 3; x++)
      for (y = 0; y < 3; y++)
        if (max < r_map[x][y]) max = r_map[x][y];
    ...
  }
```

**4-3.** 웹캠 코드의 오류는?
```cpp
VideoCapture capture(0);
Mat frame;
Mat cornerMap(frame.rows, frame.cols, CV_8UC3);
while (true) {
    capture >> frame;
    HarrisCornerDetector(frame, cornerMap);
    imshow("Corner Detection", cornerMap);
}
```

---

<details>
<summary><b>정답 보기</b></summary>

### Part 1
**1-1.** ① `x / scale` ② `pos_x - sx` ③ `p*(1-q)` ④ `sy+1` ⑤ `sx+1`

**1-2.** ① `-1, -1, -1, 0, 0, 0, 1, 1, 1` ② `yy` ③ `xx` ④ `sqrt(conv_x*conv_x + conv_y*conv_y)`

**1-3.** ① `atan2` ② `dir < 0` ③ `20` ④ `mag[y*width+x]` (magnitude)

**1-4.** ① `(y + yy) * width + (x + xx)` ② `gx * gy` ③ `Sxy * Sxy` ④ `Sxx + Syy`

**1-5.** ① `0` ② `0` ③ `255` ④ `x` ⑤ `y`

**1-6.** ① `detectAndCompute` ② `knnMatch` ③ `matches2points_nndr` ④ `compute_inliers_ransac`

**1-7.** ① `matches_orb.size() / 2` ② `nmatches_orb - ninliers_orb` ③ `ninliers_orb` ④ `nmatches_orb`

**1-8.** ① `CV_BGR2GRAY` (`COLOR_BGR2GRAY`) ② `prevFrame` ③ `frame` ④ `y` ⑤ `x` ⑥ `prevFrame`

### Part 2
**2-1.** Forward 는 원본 픽셀을 결과로 보내므로 확대·회전 시 결과 좌표가 정수로 딱 떨어지지 않아 값이 안 채워지는 구멍(hole)이 생기고 겹침도 생김. Backward 는 결과 픽셀마다 원본 위치를 역변환으로 계산하고 보간하므로 모든 결과 픽셀이 빠짐없이 한 번씩 채워짐.

**2-2.** Gradient 는 밝기가 가장 급하게 변하는 방향 = 경계에 수직. 예: 왼쪽 10, 오른쪽 20 인 세로 경계면 fx > 0, fy = 0 → phase 0°(가로 방향)지만 실제 경계선은 90°(세로). 구조 방향은 +90°.

**2-3.** Cell(8×8): 작은 영역의 9-bin 방향 histogram 으로 국소 모양을 기술. Block(16×16 = cell 4개, 36-bin): 정규화 단위, 8픽셀씩 겹치며 이동. L2 정규화: block 내 크기를 맞춰 조명·대비 변화에 강인하게 만듦.

**2-4.** 고유값 = gradient 분포의 주축 방향별 에너지. 둘 다 작으면 flat, 하나만 크면 edge, 둘 다 크면 corner. 고유값 분해는 비싸므로 det = λ1λ2, tr = λ1+λ2 를 이용한 R = det − k·tr² 로 빠르게 판정.

**2-5.** ① Gaussian LPF ② Ix, Iy ③ R ④ NMS. Gaussian: 미분이 노이즈를 증폭하므로 먼저 노이즈 제거. NMS: corner 주변 여러 픽셀이 동시에 검출되는 것을 윈도우 내 최대값 하나로 줄임.

**2-6.** FAST 로 corner 검출 → Harris 점수 상위 N개 선택 → intensity centroid 로 방향 계산 → 그 방향으로 회전한 BRIEF 계산. BRIEF 는 두 점 밝기 비교 결과(0/1)의 비트열이므로 서로 다른 비트 수(Hamming)가 자연스러운 거리이고 XOR 로 매우 빠름.

**2-7.** 최소 샘플(homography 4쌍)을 랜덤으로 뽑아 모델 추정 → 전체 매칭 중 오차가 임계값 이내인 수(consensus) 계산 → 반복 → consensus 최대 모델 채택. Inlier = 그 모델에 맞는 올바른 매칭, Outlier = 맞지 않는 잘못된 매칭. LS 는 모든 점을 쓰므로 outlier 에 끌려가지만 RANSAC 은 outlier 를 무시해 강인함.

**2-8.** 가정: brightness constancy, small motion, 3×3 patch 내 같은 움직임. 해법: 9개 OFE 를 최소제곱 → `(AᵀA)⁻¹Aᵀb`, AᵀA = structure tensor. 풀림 조건: AᵀA 가역, 고유값이 너무 작지 않음(→ corner 에서 잘 됨). 한계: 1픽셀 이상 큰 움직임 → 해결: pyramid coarse-to-fine. (그 외 약점: 조명 변화, occlusion, aperture problem)

### Part 3
**3-1.** NN: (int)(0.5+0.5)=1, (int)(0.25+0.5)=0 → I(1,0) = **200**
Average: (100+200+50+150)/4 = **125**
Bilinear: p = 0.5, q = 0.25 → 위 (100+200)/2 = 150, 아래 (50+150)/2 = 100 → 150×0.75 + 100×0.25 = **137.5**

**3-2.** dx = 10, dy = 0, cos180 = −1, sin180 = 0 → pos_x = −10 + 50 = **40**, pos_y = 0 + 50 = **50** → **(40, 50)**

**3-3.** fx = (20+80+80) − (20+20+80) = **60**, fy = (80+80+80) − (20+20+20) = **180**
magnitude = √(3600 + 32400) = √36000 ≈ **189.7**, 방향 = atan2(180, 60) ≈ **71.6°** → **bin 3** (60~80)

**3-4.** fx = 80 − 20 = **60**, fy = 80 − 20 = **60**, magnitude = 60√2 ≈ **84.9**, 방향 **45°** (bin 2)

**3-5.** 가로 (96−16)/8+1 = 11, 세로 (160−16)/8+1 = 19 → **209 blocks**, cell 있음 209×36 = **7524**, 없음 209×9 = **1881**

**3-6.** (a) det = 24 − 4 = 20, tr = 10 → R = 20 − 0.05×100 = **15 → Corner**
(b) det = 9 − 9 = 0, tr = 10 → R = **−5 → Edge**

**3-7.** (+1, 0): ‖‖² = 1 → **0.2420** / (+2, +2): ‖‖² = 8 → **0.0073**

**3-8.** 순서 90, 20, 30, 70, 60, 40, 80, 10 / 중심 50 → **01100101**
`01100101` XOR `01100110` = `00000011` → **2**

**3-9.** matches **260**, inliers **208**, outliers **52**, ratio **80%**
(15,20): 15 < 16 → **채택** / (16,20): 16 < 16 ✗ → **버림** / (30,45): 30 < 36 → **채택**

**3-10.** 3u + (−1)(2) + 5 = 0 → **u = −1**

**3-11.** ΣIx² = 2, ΣIy² = 2, ΣIxIy = 1, ΣIxIt = −1 + 0 − 3 = −4, ΣIyIt = 0 − 2 − 3 = −5
`[2 1; 1 2]⁻¹ = (1/3)[2 −1; −1 2]`, b = (4, 5)
u = (2·4 − 1·5)/3 = **1**, v = (−1·4 + 2·5)/3 = **2**
(검산: 1·1 + 0 − 1 = 0 ✓, 0 + 2 − 2 = 0 ✓, 1 + 2 − 3 = 0 ✓)

**3-12.** (a) 90,000 / 3,000,000 = 0.03 s = **30 ms**, FPS ≈ **33.3**
(b) 40 × 30 = **1200**
(c) 16 → 8 → 4 → 2 → 1 : **4단계**
(d) det = 2, tr = 4 → H = 2 − 0.01 × 64 = **1.36**

### Part 4
**4-1.** ① `atan2(y, x)` 순서이므로 `atan2(conv_y, conv_x)` 여야 함 ② 음수 각도 처리(`if (dir < 0) dir += 180;`)가 없어 bin 이 음수 → 배열 범위 밖 접근. (+ 180° 정확히일 때 `bin >= 9 → 8` clamp 도 필요)

**4-2.** `r_map[x][y]` 가 x, y = 0~2 의 **절대 좌표**라 항상 영상 왼쪽 위 3×3 만 봄. `r_map[i + y - 1][j + x - 1]` 처럼 현재 (i, j) 기준 상대 좌표여야 함.

**4-3.** ① 루프 전 frame 이 비어 있어 cornerMap 크기가 0×0 (첫 프레임 이후 크기를 정해야 함) ② `isOpened()` 확인 없음 ③ `waitKey` 가 없어서 imshow 창이 갱신되지 않고 종료 조건도 없음 → `if (waitKey(30) >= 0) break;` 필요.

</details>
