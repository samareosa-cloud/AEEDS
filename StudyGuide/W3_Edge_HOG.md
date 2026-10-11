# W3. Edge 검출 · Orientation · HOG Descriptor

> 범위: **LectureNote 3** (Feature Extraction in Images (1)), 내 코드 `W3_GradientMagnitude`, `W3_HOG_nocell`, `W3_HOG_cell`,
> 선배 코드 `versatile0010/Week3`, `shinjuyeop/W3_EdgeDetection`, `shinjuyeop/W3_HOG`

---

## 1. 한눈에 보기

| 키워드 | 한 줄 요약 |
|---|---|
| Edge | 밝기(intensity) 변화가 큰 곳 → 미분값이 큰 곳 |
| 이산 미분 | Forward `f(x+1)−f(x)`, Backward `f(x)−f(x−1)`, **Center `f(x+1)−f(x−1)`** (★수업 사용) |
| Gradient | `∇f = (fx, fy)` → **벡터** → 크기(magnitude)와 방향(phase) |
| Magnitude | `√(fx² + fy²)` ≈ `|fx| + |fy|` |
| Phase | `tan⁻¹(fy / fx)` — **edge의 법선(normal) 방향**. 영상 구조 방향은 +90° |
| Prewitt mask | fx: 열 방향 `[-1 0 1]` 3줄, fy: 행 방향 `[-1 -1 -1 / 0 0 0 / 1 1 1]` |
| 9-bin histogram | 0~180° 를 20° 간격으로 나눔. 각 bin에 **magnitude 를 누적** |
| HOG | Block(16×16) = Cell(8×8) × 4 → 9bin × 4 = **36-bin**, block 단위 **L2 정규화** |
| L2 정규화 | `h_k = N_k / √(Σ N_q² + δ)` → **조명 변화에 강인** |
| 유사도 | 두 HOG 벡터의 **Euclidean distance** (작을수록 비슷) |

---

## 2. 개념 설명

### 2-1. 왜 edge인가?
- 사람은 edge만 보고도 장면을 이해할 수 있다 (gist = 구조 정보).
- **Corner** = edge 들이 만나는 점 → 주변과 구별되는(discriminative) 특징 → feature point 로 사용 (W4).

### 2-2. 이산 미분 (Δx = 1, 픽셀 간격이 1이므로)
```
∂f/∂x ≈ f(x+1) − f(x)       Forward
∂f/∂x ≈ f(x) − f(x−1)       Backward
∂f/∂x ≈ f(x+1) − f(x−1)     Center difference  ★
```

### 2-3. 2D: Gradient 벡터 → Magnitude & Phase
```
∇f = (∂f/∂x, ∂f/∂y) = (fx, fy)
|∇f| = √(fx² + fy²) ≈ |fx| + |fy|
∠∇f = tan⁻¹(fy / fx)
```

### 2-4. Convolution 으로 edge map 계산 (Prewitt)
```
Filter for fx        Filter for fy
-1  0  1            -1 -1 -1
-1  0  1             0  0  0
-1  0  1             1  1  1
```
- 3×3 영역의 픽셀과 mask를 **원소별로 곱해서 합**.
- 1D 배열로 쓰면 `mask[(yy − (y−1)) * 3 + (xx − (x−1))]` → 3×3 내부 위치를 0~8 인덱스로 변환.

### 2-5. Display 를 위한 정규화
magnitude는 0~255 범위가 아님 → **min/max 정규화** 필요:
```
out = 255 × (val − min) / (max − min)          ← edge가 흰색
out = 255 − 255 × (val − min) / (max − min)    ← 교수님 코드: 반전 → edge가 검정, 배경 흰색
```

### 2-6. Orientation 은 "법선 방향"
슬라이드 예시:
```
10 10 20
10 10 20      fx = 60, fy = 0 → ∠ = tan⁻¹(0) = 0°
10 10 20
```
영상의 경계선은 **세로(90°)** 인데 계산된 방향은 **0°** → gradient 방향은 경계선의 **법선**. 경계선 자체 방향을 원하면 +90°.

### 2-7. Orientation histogram (9-bin) 만들기 — 슬라이드 4단계
1. `atan2(y, x)` 결과(−π ~ π, 라디안)를 **degree 로 변환**: `× 180 / π`
2. **[0°, 180°]** 만 사용 → 음수면 **+180°**
3. **20° 간격으로 양자화** → 9-bin: `bin = (int)(dir / 20)`
4. 해당 bin 에 **magnitude 누적** (개수가 아니라 크기!)
   → 정규화 문제 생각해볼 것 (L2 정규화)

| 각도 | 처리 | bin |
|---|---|---|
| 35° | 그대로 | 1 (20~40) |
| −30° | +180 → 150° | 7 (140~160) |
| −180° | +180 → 0° | 0 |
| 180° (정확히) | 180/20 = 9 → 범위 밖 | **clamp → 8** (`if (bin >= 9) bin = 8`) |

### 2-8. HOG (Histograms of Oriented Gradients, Dalal & Triggs, CVPR 2005)
```
Block 16×16  ──→  Cell 8×8 × 4개
각 Cell: 9-bin histogram
Block: 4 × 9 = 36-bin  →  L2 정규화
Block 을 8픽셀(stride)씩 이동하면서 이어붙임(concatenate)
```
**L2 정규화**
```
h_k = N_k / √( Σ_{q=1}^{36} N_q² + δ ),   N_k = Σ_{D(x,y)∈k} M(x,y)
```
- D(x,y): 양자화된 방향(bin), M(x,y): edge magnitude, δ: 0으로 나누기 방지 (예: 1e-6)
- 정규화 덕분에 **조명 변화에 강인(robust to illumination)**

### 2-9. 과제: LectureNote_03.bmp (64 × 128) HOG
- 8픽셀 간격으로 16×16 block 추출
```
가로 block 수 = (64 − 16) / 8 + 1 = 7
세로 block 수 = (128 − 16) / 8 + 1 = 15
전체 block    = 7 × 15 = 105
```
| 방식 | block당 차원 | 전체 차원 |
|---|---|---|
| Cell 없음 (block 전체로 9-bin) | 9 | 105 × 9 = **945** |
| **Cell 있음 (슬라이드 정석)** | 4 × 9 = 36 | 105 × 36 = **3780** |

- 비교: `compare1.bmp`, `compare2.bmp` 와 **Euclidean distance** `√Σ(a_i − b_i)²`
- 제약: **이미지 읽기/저장 외에는 OpenCV 함수 사용 금지**

> ⚠️ **내 코드 체크**: 내 W3 코드는 평균 절대 차이 `Σ|a−b| / N` 를 썼는데, 슬라이드는 **Euclidean distance** 를 요구함.
> 내 CSV(`descriptor_comparison.csv`, cell 없음 945차원)로 계산해 보면:
>
> | 비교 | 평균 절대 차이 | Euclidean |
> |---|---|---|
> | ref vs compare1 | 0.17446 | 7.1316 |
> | ref vs compare2 | 0.17611 | **7.0535** |
>
> → 평균 절대 차이로는 compare1, **Euclidean 으로는 compare2 가 더 유사**. 지표에 따라 결론이 뒤집히므로 시험에서는 **슬라이드대로 Euclidean** 을 쓰자.

---

## 3. 핵심 코드

### 3-1. Gradient magnitude (교수님 코드 = 내 `W3_GradientMagnitudebyprof.cpp`)
```cpp
int mask_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
int mask_y[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
float* val = (float*)calloc(height * width, sizeof(float));
float min = 1000000, max = -1;

for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        conv_x = 0;  conv_y = 0;
        for (yy = y - 1; yy <= y + 1; yy++) {
            for (xx = x - 1; xx <= x + 1; xx++) {
                if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                    conv_x += mask_x[(yy - (y - 1)) * 3 + (xx - (x - 1))] * input.at<uchar>(yy, xx) / 255.0;
                    conv_y += mask_y[(yy - (y - 1)) * 3 + (xx - (x - 1))] * input.at<uchar>(yy, xx) / 255.0;
                }
            }
        }
        val[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y);   // magnitude
        if (min > val[y * width + x]) min = val[y * width + x];
        if (max < val[y * width + x]) max = val[y * width + x];
    }
}
// visualization (반전: edge = 검정)
for (y = 0; y < height; y++)
    for (x = 0; x < width; x++)
        MagImage.at<uchar>(y, x) = 255 - 255 * (val[y * width + x] - min) / (max - min);
```

### 3-2. 방향 → bin (공통)
```cpp
dir = atan2(conv_y, conv_x) * 180.0f / PI;   // rad → deg, 범위 [-180, 180]
if (dir < 0) dir += 180;                     // [0, 180]
bin = (int)(dir / 20);                       // 180/9 = 20도 간격
if (bin >= 9) bin = 8;                       // 180도 처리
histogram[bin] += magnitude;                 // magnitude 누적!
```

### 3-3. HOG with Cell (정석, 내 `W3_HOG_cell.cpp` 요약)
```cpp
nx = (width  - block) / interval + 1;        // 7
ny = (height - block) / interval + 1;        // 15
blockDim = (block / cell) * (block / cell) * nbins;   // 2*2*9 = 36

for (by = 0; by <= height - block; by += interval) {
    for (bx = 0; bx <= width - block; bx += interval) {
        fill(blockHist.begin(), blockHist.end(), 0.0f);
        cellIndex = 0;
        for (cy = 0; cy < block; cy += cell) {                 // block 안 cell 4개
            for (cx = 0; cx < block; cx += cell) {
                for (y = by + cy; y < by + cy + cell; ++y)
                    for (x = bx + cx; x < bx + cx + cell; ++x)
                        blockHist[cellIndex * nbins + binImage[y * width + x]]
                            += magnitude[y * width + x];
                ++cellIndex;
            }
        }
        // L2 정규화 (36차원 전체로)
        sumSquares = eps;
        for (i = 0; i < blockDim; ++i) sumSquares += blockHist[i] * blockHist[i];
        inverseNorm = 1.0f / sqrt(sumSquares);
        for (i = 0; i < blockDim; ++i) descriptor.push_back(blockHist[i] * inverseNorm);
    }
}
```

### 3-4. Euclidean distance (슬라이드 요구사항)
```cpp
double euclidean(const vector<float>& a, const vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return sqrt(s);       // 작을수록 유사
}
```

---

## 4. 세 코드 비교 / 주의점

| 항목 | 내 코드 | versatile0010 | shinjuyeop |
|---|---|---|---|
| Mask | Prewitt | **Sobel** (`{-1,-2,-1,0,0,0,1,2,1}`) + `/9` | Prewitt |
| 픽셀 스케일 | /255 | 그대로 | /255 |
| HOG | block 105개 × (9 또는 36) | 영상 전체 9-bin 하나만 (block X) | block 105개 × 9 = 945 |
| 비교 | 평균 절대 차이 | 없음 | **L1 + L2(Euclidean)** |

- versatile 의 `mask_x = {-1,-2,-1, 0,0,0, 1,2,1}` 는 사실 **세로 방향 미분(fy) Sobel** 모양. 이름만 x 로 바뀌어 있음 → 방향(각도)이 90° 틀어짐. 시험에서는 **수업 Prewitt** 기준으로 쓰기.
- Sobel 은 가운데 행/열에 가중치 2 → 노이즈에 조금 더 강함.

---

## 5. 예상 문제

### A. 코드 빈칸

**A1.** Prewitt mask (1D 배열)
```cpp
int mask_x[9] = { ①_________________________ };
int mask_y[9] = { ②_________________________ };
```

**A2.** 3×3 convolution 인덱스
```cpp
for (yy = y - 1; yy <= y + 1; yy++)
  for (xx = x - 1; xx <= x + 1; xx++)
    if (yy >= 0 && yy < height && xx >= 0 && xx < width)
      conv_x += mask_x[ ①_______________ * 3 + ②_______________ ] * input.at<uchar>(yy, xx);
```

**A3.** Magnitude 와 정규화 시각화
```cpp
val[y * width + x] = ①______________________;
...
MagImage.at<uchar>(y, x) = 255 - 255 * (②__________) / (③__________);
```

**A4.** 방향 histogram
```cpp
dir = atan2(①_____, ②_____) * ③_________;
if (dir < 0) dir += ④_____;
bin = (int)(dir / ⑤____);
if (bin >= 9) bin = ⑥____;
hist[bin] += ⑦__________;
```

**A5.** Block 개수와 HOG 차원
```cpp
nx = (width - ①_____) / ②_____ + 1;
ny = (height - block) / interval + 1;
dim = nx * ny * ③_____;      // cell 사용, 9-bin
```

**A6.** L2 정규화
```cpp
float s = ①_____;               // 0 나누기 방지
for (i = 0; i < 36; i++) s += ②_____________;
for (i = 0; i < 36; i++) h[i] = h[i] / ③_________;
```

### B. 개념 / 알고리즘

**B1.** Forward / Backward / Center difference 식을 쓰고, 수업에서 강조한 방식은?

**B2.** Gradient 계산 결과가 "벡터"라는 것의 의미와, 이로부터 정의되는 두 가지 양은?

**B3.** Edge orientation 이 "법선 방향"이라는 것은 무슨 뜻인가? 영상 구조 방향을 얻으려면?

**B4.** Magnitude 결과를 그대로 `uchar` 에 넣으면 안 되는 이유와 해결법은?

**B5.** HOG 에서 block 단위 정규화를 하는 이유는?

**B6.** Histogram 에 개수(count)가 아니라 magnitude 를 누적하는 이유는?

**B7.** 과제 제약 조건 "이미지 읽고 저장하는 것 외에는 OpenCV 함수 사용하지 말 것"이 의미하는 것은? (예: 쓰면 안 되는 함수)

### C. 결과 적기 / 계산

**C1.** 다음 3×3 픽셀에서 가운데 픽셀(39)에 대해 (슬라이드 5)
```
41 40 38
39 39 40
42 37 37
```
(a) Center difference 로 fx, fy, magnitude, phase
(b) Prewitt mask 로 fx, fy, magnitude, phase(0~180 변환 후), 그리고 9-bin 의 몇 번 bin?

**C2.** 다음 lattice 의 fx, fy, phase 는? (Prewitt)
```
10 10 20
10 10 20
10 10 20
```

**C3.** `atan2` 결과가 각각 −45°, 100°, 179.99°, −0.5° 일 때 bin 번호(0~8)는?

**C4.** 64×128 영상, block 16×16, stride 8, cell 8×8, 9-bin 일 때
(a) block 개수 (b) cell 없을 때 차원 (c) cell 있을 때 차원

**C5.** 128×256 영상이면 (C4와 같은 조건, cell 사용) HOG 차원은?

**C6.** 어떤 block 의 histogram 이 `[3, 4, 0, 0, 0, 0, 0, 0, 0]` 이다. δ = 0 으로 L2 정규화한 결과는?

**C7.** 두 3차원 벡터 a = (1, 2, 2), b = (1, 0, 0) 의 Euclidean distance 는?

---

<details>
<summary><b>정답 보기</b></summary>

**A1.** ① `-1, 0, 1, -1, 0, 1, -1, 0, 1` ② `-1, -1, -1, 0, 0, 0, 1, 1, 1`

**A2.** ① `(yy - (y - 1))` ② `(xx - (x - 1))`

**A3.** ① `sqrt(conv_x * conv_x + conv_y * conv_y)` ② `val[y * width + x] - min` ③ `max - min`

**A4.** ① `conv_y` ② `conv_x` (atan2(y, x) 순서!) ③ `180 / PI` ④ `180` ⑤ `20` ⑥ `8` ⑦ `magnitude` (mag 값)

**A5.** ① `block` (16) ② `interval` (8) ③ `36`

**A6.** ① `eps` (δ, 예 1e-6) ② `h[i] * h[i]` ③ `sqrt(s)`

**B1.** Forward `f(x+1)−f(x)`, Backward `f(x)−f(x−1)`, Center `f(x+1)−f(x−1)`. 수업은 **Center difference** 강조 (Prewitt mask 도 중심 차분을 3줄 합친 형태).

**B2.** 2D 영상은 x, y 두 방향으로 미분하므로 결과가 `(fx, fy)` 벡터 → **magnitude** `√(fx²+fy²)` 와 **phase(orientation)** `tan⁻¹(fy/fx)` 가 정의된다.

**B3.** Gradient 는 밝기가 가장 크게 변하는 방향 = 경계선에 **수직**인 방향. 예: 세로 경계선이면 phase = 0°. 경계선(구조) 방향을 원하면 **+90°**.

**B4.** magnitude 는 0~255 범위를 벗어나거나(최대 수백) 소수(0~1 스케일)일 수 있어 그대로 넣으면 잘리거나 거의 0이 됨 → **min/max 정규화**로 [0, 255] 변환.

**B5.** 조명(밝기/대비)이 바뀌면 gradient 크기가 전체적으로 커지거나 작아짐 → block 마다 L2 정규화하면 상대적인 방향 분포만 남아 **조명 변화에 강인**.

**B6.** 강한 edge 일수록 구조적으로 중요한 정보 → magnitude 로 가중치를 주어야 평탄 영역의 노이즈 방향이 histogram 을 지배하지 않음.

**B7.** `imread`, `imwrite`(및 표시용 `imshow`) 외 OpenCV 연산 금지 → `Sobel`, `filter2D`, `cartToPolar`, `HOGDescriptor`, `norm`, `cvtColor` 등을 쓰지 말고 반복문으로 직접 구현.

**C1.**
(a) fx = f(x+1)−f(x−1) = 40 − 39 = **1**, fy = f(y+1)−f(y−1) = 37 − 40 = **−3**
  magnitude = √(1+9) = √10 ≈ **3.16** (근사 |1|+|−3| = 4), phase = tan⁻¹(−3/1) ≈ **−71.6°**
(b) Prewitt:
  fx = (38+40+37) − (41+39+42) = 115 − 122 = **−7**
  fy = (42+37+37) − (41+40+38) = 116 − 119 = **−3**
  magnitude = √(49+9) = √58 ≈ **7.62**
  atan2(−3, −7) ≈ −156.8° → +180 → **23.2°** → **bin 1** (20~40°)

**C2.** Prewitt 그대로 계산하면 fx = (20+20+20) − (10+10+10) = **30**, fy = (10+10+20) − (10+10+20) = **0**
  → phase = tan⁻¹(0/30) = **0°** (경계선은 세로 → 법선은 가로 방향)
  ※ 슬라이드 8은 `0+0+20+0+0+20+0+0+20 = 60` 으로 왼쪽 열(−10) 항을 0으로 적어 fx = 60 으로 표기함. 시험에서 슬라이드 그대로 나오면 60, 직접 계산하라고 하면 30. **어느 쪽이든 fy = 0 이므로 phase = 0°** 는 같다.

**C3.** −45° → 135° → **6** / 100° → **5** / 179.99° → 8.9995 → **8** / −0.5° → 179.5° → **8**

**C4.** (a) 7 × 15 = **105** (b) 105 × 9 = **945** (c) 105 × 36 = **3780**

**C5.** 가로 (128−16)/8+1 = 15, 세로 (256−16)/8+1 = 31 → 465 block × 36 = **16740**

**C6.** √(9+16) = 5 → **[0.6, 0.8, 0, 0, 0, 0, 0, 0, 0]**

**C7.** √(0² + 2² + 2²) = √8 ≈ **2.83**

</details>
