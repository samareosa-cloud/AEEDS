# AEEDS 시험 대비 스터디 가이드 (W2 ~ W6)

전기전자심화설계 및 소프트웨어실습 (Computer Vision, 김원준 교수님).
강의자료 LectureNote 3~6, 내 코드(이 저장소), 선배 코드 두 개
([versatile0010/Image-Processing](https://github.com/versatile0010/Image-Processing), [shinjuyeop/AEEDS](https://github.com/shinjuyeop/AEEDS))를 읽고 정리했다.

시험 유형: **코드 빈칸 채우기**, **알고리즘·개념 서술**, **결과값 적기/계산**

## 목차

| 주차 | 파일 | 주제 | 강의자료 |
|---|---|---|---|
| W2 | [W2_ImageBasics.md](W2_ImageBasics.md) | Mat, 픽셀 접근, Resize, Rotation, NN/Average/Bilinear | (LN3 review) |
| W3 | [W3_Edge_HOG.md](W3_Edge_HOG.md) | 이산 미분, Prewitt, magnitude/phase, 9-bin, HOG | LN3 |
| W4 | [W4_Harris_FAST.md](W4_Harris_FAST.md) | Harris (SSD→Taylor→M→R), Gaussian, NMS, FAST, corner+HOG 매칭 | LN4 |
| W5 | [W5_Video_STIP_ORB.md](W5_Video_STIP_ORB.md) | STIP, LoG blob, VideoCapture, ORB(FAST+BRIEF), NNDR, RANSAC | LN5 |
| W6 | [W6_OpticalFlow.md](W6_OpticalFlow.md) | OFE, Lucas-Kanade, pyramid, Farneback, Dense trajectories | LN6 |
| 종합 | [MockExam.md](MockExam.md) | 전 범위 모의고사 (빈칸 + 서술 + 계산) | 전체 |

각 주차 파일 구성: **① 한눈에 보기 → ② 개념 설명 → ③ 핵심 코드 → ④ 선배 코드 비교/주의점 → ⑤ 예상 문제 → 정답(접기)**

---

## 공식 · 숫자 치트시트 (시험 직전용)

### 기본
```
Mat(rows, cols, type)   vs   Size(cols, rows)        ← 순서 반대
at<uchar>(y, x)  /  at<Vec3b>(y, x)[0:B, 1:G, 2:R]   ← y 먼저, BGR
rad = deg × π / 180
```

### 보간 (p, q = 소수부)
```
NN       : (int)(pos + 0.5)
Average  : 0.25 × (4픽셀 합)
Bilinear : (1−p)(1−q)·I00 + p(1−q)·I10 + (1−p)q·I01 + pq·I11
회전(backward): pos_x =  cosθ·dx + sinθ·dy + W/2
                pos_y = −sinθ·dx + cosθ·dy + H/2
```

### Edge / HOG
```
Center difference : f(x+1) − f(x−1)
Prewitt fx : [-1 0 1] × 3행      fy : [-1 -1 -1 / 0 0 0 / 1 1 1]
|∇f| = √(fx²+fy²) ≈ |fx|+|fy|      ∠∇f = tan⁻¹(fy/fx)   (법선 방향, 구조 방향은 +90°)
mask 인덱스: mask[(yy−(y−1))*3 + (xx−(x−1))]
표시 정규화: 255·(v−min)/(max−min)   (교수님 코드는 255 − 이것 → edge 검정)
dir = atan2(fy, fx)·180/π;  if(dir<0) dir += 180;  bin = (int)(dir/20);  if(bin>=9) bin=8;
HOG: block 16×16 = cell 8×8 × 4, 9bin × 4 = 36,  L2: h_k = N_k / √(ΣN² + δ)
64×128, stride 8 → 7 × 15 = 105 blocks → 945 (cell X) / 3780 (cell O)
유사도: Euclidean distance √Σ(a−b)²  (작을수록 유사)
```

### Harris
```
E(Δx,Δy) = Σ[I(x+Δx,y+Δy) − I(x,y)]²  ≈ [Δx Δy] M [Δx Δy]ᵀ
M = [[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]
R = det(M) − k·tr²(M),  det = λ1λ2, tr = λ1+λ2,  k = 0.04~0.06
R>0 corner / R<0 edge / |R| 작음 flat
4단계: Gaussian LPF → Ix, Iy → R → NMS
G(x) = 1/(√(2π)σ)·exp(−‖x−c‖²/2σ²);  σ=1, ‖x−c‖²=2 → 0.1468
불변: shift, rotation (조명에도 강함)  /  취약: scale
FAST: 반지름 3 원 위 16픽셀, 연속 N(9 또는 12)개가 밝거나 어두우면 corner
시간: (stop.QuadPart − start.QuadPart) / freq.QuadPart  [초]
```

### Video / ORB
```
STIP: 3×3 S (Lx, Ly, Lt),  H = det(S) − k·tr³(S) = λ1λ2λ3 − k(λ1+λ2+λ3)³
LoG: maxima = 밝은 배경의 어두운 blob, minima = 어두운 배경의 밝은 blob, 크기는 σ
VideoCapture capture(0);  capture >> frame;  if (waitKey(30) >= 0) break;
ORB = FAST(+Harris 상위 N) + orientation(intensity centroid) + rotated BRIEF
BRIEF: τ = 1 if p(x) < p(y) → 비트열, Hamming distance ("BruteForce-Hamming")
knnMatch(k=2) → NNDR: dist1 < 0.8·dist2 → RANSAC(findHomography, 2.5px)
nmatches = matches.size()/2,  ratio = 100·inliers/matches,  ms = 1000·(t2−t1)/getTickFrequency()
```

### Optical flow
```
가정: brightness constancy, small motion, spatial coherence
OFE: Ix·u + Iy·v + It = 0     (It = 현재 − 이전)
LK: [u v]ᵀ = [[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]⁻¹ · [−ΣIxIt, −ΣIyIt]ᵀ   (= Harris M!)
풀림 조건: M 가역, 고유값이 너무 작지 않음
큰 움직임(>1px) → pyramid coarse-to-fine
Farneback: d = −½ A1⁻¹ (b2 − b1)
calcOpticalFlowFarneback(prev, frame, flow, 0.4, 1, 12, 2, 8, 1.2, 0)
  = (pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
flow.at<Point2f>(y, x) → (dx, dy);  320×240, 간격 5 → 64×48 = 3072 점
약점: 조명 변화, occlusion, aperture problem
```

---

## 내 코드에서 발견한 것 (시험 전 확인)

1. **W3 HOG 유사도 지표**: 슬라이드는 *Euclidean distance* 인데 내 코드는 평균 절대 차이. 내 CSV 로 다시 계산하면
   평균 절대 차이 → compare1 이 더 유사 / **Euclidean → compare2 가 더 유사** (7.1316 vs 7.0535). 결론이 뒤집힘.
2. **W4 Harris**: 내 코드는 Step 2·3(gradient, R)만 있고 **Step 1 Gaussian, Step 4 NMS 가 없음** → W4 파일의 3-2, 3-3 코드 꼭 보기.
3. **W4 NMS (versatile0010 선배 코드)**: `r_map[x][y]` 로 항상 왼쪽 위 3×3 만 보는 버그. 올바르게는 `r_map[i+y−1][j+x−1]`.
4. **Sobel vs Prewitt**: versatile0010 코드는 Sobel(가중치 2)을 쓰고 x/y 이름이 뒤바뀜. 시험은 수업 기준 **Prewitt**.
