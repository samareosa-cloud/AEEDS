# W2. 영상 기초: Mat · 픽셀 접근 · Resize · Rotation · 보간

> 범위: LectureNote 2 (Review 슬라이드 기준), 내 코드 `W2_colortopink`, `W2_imageresizing`, `W2_imagerotation`,
> 선배 코드 `versatile0010/Week2`, `shinjuyeop/W2*`

---

## 1. 한눈에 보기

| 키워드 | 한 줄 요약 |
|---|---|
| `Mat` | OpenCV의 영상 = 행렬. `rows`=높이, `cols`=너비 |
| `CV_8UC1` / `CV_8UC3` | 8bit unsigned, 1채널(흑백) / 3채널(컬러) |
| `at<uchar>(y, x)` | 흑백 픽셀 접근. **(행, 열) = (y, x)** 순서 |
| `at<Vec3b>(y, x)[c]` | 컬러 픽셀 접근. **c = 0:B, 1:G, 2:R** (BGR!) |
| Backward mapping | 결과 픽셀 (x,y)에서 출발해서 **원본 좌표를 역으로 계산** → 구멍(hole) 없음 |
| NN | 가장 가까운 픽셀 1개 사용 (`(int)(pos+0.5)`) |
| Average | 주변 4픽셀 단순 평균 (가중치 0.25씩) |
| Bilinear | 주변 4픽셀을 **거리에 반비례한 가중치**로 평균 |

---

## 2. 개념 설명

### 2-1. Mat 기본
```cpp
Mat img = imread("test.jpg", CV_LOAD_IMAGE_COLOR);      // = IMREAD_COLOR (3채널)
Mat gray = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE); // = IMREAD_GRAYSCALE = 0 (1채널)

int height = img.rows;   // 세로
int width  = img.cols;   // 가로

Mat result(height, width, CV_8UC3);   // Mat(rows, cols, type)  ← 행이 먼저!
Mat result2(Size(width, height), CV_8UC3); // Size(cols, rows)  ← Size는 가로가 먼저!
Mat zero = Mat::zeros(height, width, CV_8UC1); // 0으로 초기화 (회전 결과의 빈 공간 = 검정)

Mat copy = img.clone();  // 깊은 복사 (픽셀 데이터까지 복사)
Mat ref  = img;          // 얕은 복사 (같은 메모리 공유 → ref 수정 시 img도 바뀜)

imshow("name", result);  imwrite("out.bmp", result);  waitKey(0);
```

> **시험 포인트**
> - `Mat(rows, cols)` 와 `Size(cols, rows)` 는 순서가 반대다.
> - `at<>(y, x)` — y(행)가 먼저.
> - 컬러 채널 순서는 **BGR**. `Vec3b(255, 0, 255)` = B=255, R=255 → **분홍(마젠타)**.

### 2-2. 흑백 변환 (부분 영역)
```
gray = (R + G + B) / 3   → B, G, R 세 채널에 모두 gray 를 넣으면 회색이 된다
```

### 2-3. Forward vs Backward mapping
- **Forward**: 원본 (x,y) → 결과 (x',y') 계산. 확대/회전 시 결과 영상에 **빈 픽셀(hole)** 생김.
- **Backward**: 결과 (x,y) → 원본 (pos_x, pos_y) 역계산 후 **보간**. 모든 결과 픽셀이 채워짐 → 수업에서 사용하는 방식.

### 2-4. Resize (backward)
```
pos_x = x / scale,   pos_y = y / scale
re_width  = (int)(scale × width)
re_height = (int)(scale × height)
```

### 2-5. Rotation (중심 기준, backward)
결과 픽셀을 원본으로 되돌리려면 **역회전(−θ)** 을 적용:
```
dx = x − W/2,  dy = y − H/2              (중심을 원점으로)
pos_x =  cosθ·dx + sinθ·dy + W/2
pos_y = −sinθ·dx + cosθ·dy + H/2
```
- `rad = degree × π / 180` (C의 `cos`, `sin` 은 라디안!)
- 원본 범위 밖이면 0(검정) 또는 255(흰색)로 채움

### 2-6. 보간 3종 (원본 좌표 pos 가 정수가 아닐 때)
`sx = (int)pos_x`, `sy = (int)pos_y`, `p = pos_x − sx`, `q = pos_y − sy`

| 방법 | 식 |
|---|---|
| **NN** | `I(round(pos_y), round(pos_x))` — `(int)(pos+0.5)` |
| **Average** | `0.25 × [I(sy,sx) + I(sy,sx+1) + I(sy+1,sx) + I(sy+1,sx+1)]` |
| **Bilinear** | `(1−p)(1−q)·I(sy,sx) + p(1−q)·I(sy,sx+1) + (1−p)q·I(sy+1,sx) + pq·I(sy+1,sx+1)` |

- Bilinear 가중치 4개의 합 = 1 (항상)
- 품질: NN(계단현상, 가장 빠름) < Average(흐림) < Bilinear(부드러움)
- 더 좋은 방법: bi-cubic, spline (LN3 review 슬라이드)

---

## 3. 핵심 코드

### 3-1. 부분 영역 색 바꾸기 (내 코드 `colortopink.cpp`)
```cpp
for (int y = cy - BLK; y <= cy + BLK; y++) {
    for (int x = cx - BLK; x <= cx + BLK; x++) {
        if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {   // 경계 검사: x, y 로!
            int b = img.at<Vec3b>(y, x)[0];
            int g = img.at<Vec3b>(y, x)[1];
            int r = img.at<Vec3b>(y, x)[2];
            int gray = (r + g + b) / 3;
            result.at<Vec3b>(y, x)[0] = gray;   // 분홍으로 하려면 255
            result.at<Vec3b>(y, x)[1] = gray;   //                  0
            result.at<Vec3b>(y, x)[2] = gray;   //                  255
        }
    }
}
```
→ BLK=30이면 처리 영역은 (2·30+1)² = **61×61** 픽셀.

### 3-2. Resize + Bilinear (세 코드 종합 정리본)
```cpp
Mat imageResize(Mat input, float scale)
{
    int height = input.rows, width = input.cols;
    int re_h = (int)(scale * height);
    int re_w = (int)(scale * width);
    Mat result(re_h, re_w, CV_8UC3);

    for (int y = 0; y < re_h; y++) {
        for (int x = 0; x < re_w; x++) {
            float pos_x = x / scale;              // backward mapping
            float pos_y = y / scale;
            int sx = (int)pos_x, sy = (int)pos_y;
            float p = pos_x - sx, q = pos_y - sy; // 소수부 = 가중치

            if (sx >= 0 && sx < width - 1 && sy >= 0 && sy < height - 1) { // sx+1 접근하므로 width-1
                for (int c = 0; c < 3; c++) {
                    float v = (1-p)*(1-q) * input.at<Vec3b>(sy,   sx  )[c]
                            +    p *(1-q) * input.at<Vec3b>(sy,   sx+1)[c]
                            + (1-p)*   q  * input.at<Vec3b>(sy+1, sx  )[c]
                            +    p *   q  * input.at<Vec3b>(sy+1, sx+1)[c];
                    result.at<Vec3b>(y, x)[c] = (uchar)(v + 0.5);   // 반올림
                }
            }
        }
    }
    return result;
}
```

### 3-3. Rotation (내 코드 `W2_imagerotation.cpp`, 흑백 + average)
```cpp
float rad = degree * 3.141592f / 180.0f;
Mat result = Mat::zeros(height, width, CV_8UC1);

for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        pos_x =  cos(rad) * (x - width / 2.0f) + sin(rad) * (y - height / 2.0f);
        pos_y = -sin(rad) * (x - width / 2.0f) + cos(rad) * (y - height / 2.0f);
        pos_x += width / 2.0f;
        pos_y += height / 2.0f;

        sx = (int)pos_x;  sy = (int)pos_y;
        if (sx >= 0 && sx < width - 1 && sy >= 0 && sy < height - 1) {
            result.at<uchar>(y, x) = 0.25 * (input.at<uchar>(sy, sx)   + input.at<uchar>(sy, sx+1)
                                           + input.at<uchar>(sy+1, sx) + input.at<uchar>(sy+1, sx+1));
        }
    }
}
```
- shinjuyeop 버전은 회전행렬 `[[cos,−sin],[sin,cos]]` 의 **역행렬**(1/det · adj)을 직접 계산 → 같은 결과 (회전행렬 det = 1).

---

## 4. 선배 코드 주의점 (버그 찾기 문제로 나오기 좋음)

| 코드 | 문제 | 올바른 것 |
|---|---|---|
| versatile `0916_replacement_gray_01.cpp` | `img.at<Vec3b>(y,x)[3]` | 채널 인덱스는 0~2. **`[0]+[1]+[2]`** |
| 〃 | `if (cx < 0 \|\| cx >= cols ...)` 로 경계검사 | **`x`, `y`** 로 검사해야 함 (cx,cy는 고정값) |
| versatile `ImageResize` NN/AVERAGE | 경계검사 없이 `sy+1`, `sx+1` 접근 | 마지막 행/열에서 범위 초과 → 검사 필요 |
| versatile `ImageRotate` AVERAGE | `sx < width` 만 검사하고 `sx+1` 접근 | `sx < width − 1` |

---

## 5. 예상 문제

### A. 코드 빈칸

**A1.** 컬러 영상을 읽고 크기를 구하라.
```cpp
Mat img = imread("test.jpg", ①________);
int height = img.②____;
int width  = img.③____;
Mat result(④______, ⑤______, CV_8UC3);
```

**A2.** (y, x) 픽셀의 R 값을 읽어라.
```cpp
int r = img.at<①_____>(②__, ③__)[④__];
```

**A3.** Resize backward mapping
```cpp
re_width = ①__________;
for (y = 0; y < re_height; y++)
  for (x = 0; x < re_width; x++) {
      pos_x = ②__________;
      pos_y = (1.0 / scale) * y;
      sx = (int)pos_x;  sy = (int)pos_y;
      if (sx >= 0 && sx < ③__________ && sy >= 0 && sy < height - 1) { ... }
  }
```

**A4.** Rotation (중심 기준 backward mapping)
```cpp
rad = degree * ①________;
pos_x =  cos(rad) * (x - width/2.0f) + ②________ * (y - height/2.0f);
pos_y = ③________ * (x - width/2.0f) + cos(rad) * (y - height/2.0f);
pos_x += ④________;
pos_y += height / 2.0f;
```

**A5.** Bilinear 보간 (p, q 는 소수부)
```cpp
v = ①__________ * I(sy, sx)   + p * (1-q) * I(sy, sx+1)
  + (1-p) * q   * I(sy+1, sx) + ②______  * I(sy+1, sx+1);
```

**A6.** NN 보간에서 반올림
```cpp
sx = (int)(pos_x + ①____);
```

### B. 개념 / 알고리즘

**B1.** Forward mapping 대신 backward mapping을 쓰는 이유는?

**B2.** NN, Average, Bilinear 보간의 장단점을 비교하라.

**B3.** 회전 시 원본 범위 밖으로 나가는 픽셀은 어떻게 처리하는가? 내 코드에서 그 픽셀의 색은?

**B4.** `Mat a = b;` 와 `Mat a = b.clone();` 의 차이는?

**B5.** 다음 코드의 버그 2개를 찾아라.
```cpp
for (int y = cy - BLK; y <= cy + BLK; y++)
  for (int x = cx - BLK; x <= cx + BLK; x++) {
    if (cx < 0 || cx >= temp.cols || cy < 0 || cy >= temp.rows) continue;
    int gray = (img.at<Vec3b>(y,x)[1] + img.at<Vec3b>(y,x)[2] + img.at<Vec3b>(y,x)[3]) / 3;
    ...
  }
```

### C. 결과 적기 / 계산

**C1.** 512×512 영상을 scale = 0.7 로 resize 하면 결과 크기는? (int 변환)

**C2.** 원본 픽셀이 다음과 같을 때 (x, y) 좌표 기준
`I(0,0)=10, I(1,0)=20, I(0,1)=30, I(1,1)=40` (I(x,y))
원본 좌표 (pos_x, pos_y) = (0.25, 0.5) 에서
(a) NN (b) Average (c) Bilinear 결과는?

**C3.** `result.at<Vec3b>(y,x) = Vec3b(255, 0, 255);` 를 하면 무슨 색인가?

**C4.** 영상 크기 W=200, H=100, degree=90 으로 내 회전 코드를 실행할 때, 결과 픽셀 (x, y) = (110, 50) 은 원본의 어느 좌표 값을 가져오는가?

**C5.** BLK = 30 일 때 colortopink 코드가 바꾸는 픽셀 수는? (경계 안쪽이라고 가정)

---

<details>
<summary><b>정답 보기</b></summary>

**A1.** ① `CV_LOAD_IMAGE_COLOR` (또는 `IMREAD_COLOR`, `1`) ② `rows` ③ `cols` ④ `height` ⑤ `width`

**A2.** ① `Vec3b` ② `y` ③ `x` ④ `2`

**A3.** ① `(int)(scale * width)` ② `(1.0 / scale) * x` ③ `width - 1`

**A4.** ① `3.141592f / 180.0f` (π/180) ② `sin(rad)` ③ `-sin(rad)` ④ `width / 2.0f`

**A5.** ① `(1-p) * (1-q)` ② `p * q`

**A6.** ① `0.5`

**B1.** Forward는 원본→결과로 보내므로 확대/회전 시 결과에 값이 안 들어간 구멍(hole)이 생기고, 한 결과 픽셀에 여러 값이 겹칠 수 있다. Backward는 결과 픽셀마다 원본 위치를 역계산하므로 모든 결과 픽셀이 정확히 한 번 채워진다.

**B2.** NN: 가장 빠르지만 계단(aliasing) 현상. Average: 4픽셀 평균이라 계단은 줄지만 위치와 무관한 같은 가중치라 흐려지고 부정확. Bilinear: 거리 기반 가중치로 가장 자연스럽지만 연산량이 가장 많음.

**B3.** `if (sx >= 0 && sx < width-1 && sy >= 0 && sy < height-1)` 조건을 만족하지 않으면 값을 쓰지 않는다. 내 코드는 `Mat::zeros`로 초기화했으므로 **검정(0)**. (versatile 코드는 255=흰색으로 채움)

**B4.** `a = b` 는 헤더만 복사(같은 픽셀 메모리 공유) → a를 바꾸면 b도 바뀜. `clone()` 은 픽셀 데이터까지 새로 복사 → 독립적.

**B5.** ① 경계검사를 `cx, cy`(고정값)로 하고 있음 → `x, y` 로 해야 함. ② 채널 인덱스 `[3]`은 범위 밖 → `[0] + [1] + [2]`.

**C1.** 0.7 × 512 = 358.4 → **358 × 358**

**C2.**
(a) NN: `(int)(0.25+0.5)=0`, `(int)(0.5+0.5)=1` → I(0,1) = **30**
(b) Average: (10+20+30+40)/4 = **25**
(c) Bilinear: p=0.25, q=0.5
위쪽: 10×0.75 + 20×0.25 = 12.5 / 아래쪽: 30×0.75 + 40×0.25 = 32.5
→ 12.5×0.5 + 32.5×0.5 = **22.5**

**C3.** B=255, G=0, R=255 → **분홍(마젠타)**

**C4.** dx = 110−100 = 10, dy = 50−50 = 0, θ=90° (cos=0, sin=1)
pos_x = 0·10 + 1·0 + 100 = 100, pos_y = −1·10 + 0·0 + 50 = 40 → 원본 **(100, 40)** 근처 (average라면 (100,40),(101,40),(100,41),(101,41) 평균)

**C5.** (2·30+1)² = 61² = **3721**

</details>
