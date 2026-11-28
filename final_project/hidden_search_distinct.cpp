// macos version
// 숨은그림찾기 : HOG 템플릿 매칭 + 크기 대응 + 회전 대응 (coarse-to-fine)
//   + 자동 크기 대응 : 템플릿은 정사각형 패딩 후 48x48, 장면 배율 범위는 장면 가로 크기에 비례
//   1단계 (거칠게) : 큰 간격 각도(COARSE_STEP) 템플릿으로 모든 배율 x 모든 위치 탐색 -> 점수 높은 후보 NCAND 곳
//   2단계 (자세히) : 후보 주변 (+-REFINE_R px, 배율 앞뒤 1단계) 에서만 각도를 REFINE_STEP 간격으로 촘촘히 비교
//   좌우 반전 : 1단계에서 원래 템플릿 + 좌우 반전 템플릿 둘 다 비교 (USE_FLIP)
//   구별력 보정 : 최종 점수 = 내 점수 - LAMBDA x (그 자리에서 다른 물체 템플릿 최고 점수) -> 흔한 무늬 자리 감점
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

#define PI 3.14159265f
#define WIN 48
#define BLOCK 16
#define STRIDE 8
#define NBIN 9
#define DIM (((WIN - BLOCK) / STRIDE + 1) * ((WIN - BLOCK) / STRIDE + 1) * NBIN)   // 225
#define STEP 3                // 1단계 창 이동 간격

// ---- 실험 설정 ----
#define PUZZLE_DIR   "./data"
#define TEMPLATE_DIR "./data/templates"
#define NPUZZLE      1
#define BASE_WIDTH   800      // 아래 배율 범위를 정한 기준 장면 가로 크기 (px)
#define SCALE_MIN    0.48f    // 가로 800px 장면 기준 배율 범위 (물체가 WIN 근처가 되게)
#define SCALE_MAX    1.20f    // 실제 장면 크기에 맞게 main 에서 자동 조절
#define COARSE_STEP  360      // 1단계 각도 간격 : 360 이면 0도 하나만
#define REFINE_RANGE 30
#define REFINE_STEP  15       // 2단계 각도 간격 (-30, -15, 0, 15, 30)
#define USE_FLIP     1        // 1 이면 좌우 반전 템플릿도 비교 (거북이, 상자처럼 뒤집혀 그려진 물체)
#define NCAND        5        // 1단계에서 남길 후보 수
#define ROT_PENALTY  0.01f    // 회전 30도당 빼는 점수 : 0도보다 확실히 나을 때만 회전 채택
#define REFINE_R     4        // 2단계 : 후보 위치 +- 몇 px
#define LAMBDA       0.3f     // 구별력 보정 세기 : 0 이면 끔 (0.5 이상은 오히려 떨어짐)

int nmsR = 16;

// ===== HOG (본인 W9 코드를 그림 전체 gradient 한 번 계산하도록 나눈 것) =====
void computeGradient(const Mat& img, vector<float>& mag, vector<int>& bin)
{
    int w = img.cols, h = img.rows;
    int maskX[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int maskY[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    mag.assign(w * h, 0);
    bin.assign(w * h, 0);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float gx = 0, gy = 0;
            for (int yy = y - 1; yy <= y + 1; yy++)
                for (int xx = x - 1; xx <= x + 1; xx++) {
                    if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                    float p = img.at<uchar>(yy, xx) / 255.0f;
                    int k = (yy - y + 1) * 3 + (xx - x + 1);
                    gx += maskX[k] * p;
                    gy += maskY[k] * p;
                }
            float theta = atan2(gy, gx) * 180 / PI;
            if (theta < 0) theta += 180;
            if (theta >= 180) theta -= 180;
            int b = (int)(theta / 20);
            if (b >= NBIN) b = NBIN - 1;
            mag[y * w + x] = sqrt(sqrt(gx * gx + gy * gy));
            bin[y * w + x] = b;
        }
    }
}

void hogFromGradient(const vector<float>& mag, const vector<int>& bin, int w, int x0, int y0, float* desc)
{
    int k = 0;
    for (int by = 0; by <= WIN - BLOCK; by += STRIDE) {
        for (int bx = 0; bx <= WIN - BLOCK; bx += STRIDE) {
            float* hh = &desc[k * NBIN];
            for (int i = 0; i < NBIN; i++) hh[i] = 0;
            for (int y = by; y < by + BLOCK; y++)
                for (int x = bx; x < bx + BLOCK; x++) {
                    int id = (y0 + y) * w + (x0 + x);
                    hh[bin[id]] += mag[id];
                }
            float sum = 1e-6f;
            for (int i = 0; i < NBIN; i++) sum += hh[i] * hh[i];
            sum = sqrt(sum);
            for (int i = 0; i < NBIN; i++) hh[i] /= sum;
            k++;
        }
    }
}

void computeHOG(const Mat& img, float* desc)
{
    vector<float> mag;
    vector<int> bin;
    computeGradient(img, mag, bin);
    hogFromGradient(mag, bin, img.cols, 0, 0, desc);
}

float computeSimilarity(const float* ref, const float* tar)
{
    float nomi = 0, refMag = 0, tarMag = 0;
    for (int k = 0; k < DIM; k++) {
        nomi += ref[k] * tar[k];
        refMag += ref[k] * ref[k];
        tarMag += tar[k] * tar[k];
    }
    return nomi / (sqrt(refMag) * sqrt(tarMag) + 1e-6f);
}

// ===== 템플릿 회전 : W2 imageRotation 과 같은 backward mapping (크기 그대로, 빈 곳은 흰색) =====
Mat rotateTemplate(const Mat& src, float degree)
{
    Mat dst(src.rows, src.cols, CV_8UC1, Scalar(255));
    float rad = degree * PI / 180;
    float c = cos(rad), s = sin(rad);
    float cx = (src.cols - 1) / 2.0f, cy = (src.rows - 1) / 2.0f;
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            // 결과 픽셀 -> 원본 좌표 (반시계 degree 회전의 역변환)
            float dx = x - cx, dy = y - cy;
            float sx = c * dx - s * dy + cx;
            float sy = s * dx + c * dy + cy;
            int x0 = (int)floor(sx), y0 = (int)floor(sy);
            if (x0 < 0 || y0 < 0 || x0 + 1 >= src.cols || y0 + 1 >= src.rows) continue;
            float a = sx - x0, b = sy - y0;       // 쌍선형 보간
            float v = (1 - a) * (1 - b) * src.at<uchar>(y0, x0) + a * (1 - b) * src.at<uchar>(y0, x0 + 1)
                    + (1 - a) * b * src.at<uchar>(y0 + 1, x0) + a * b * src.at<uchar>(y0 + 1, x0 + 1);
            dst.at<uchar>(y, x) = (uchar)(v + 0.5f);
        }
    }
    return dst;
}

// 템플릿을 흰 여백으로 정사각형 만들기 (물체는 가운데) : 48x48 로 줄일 때 찌그러지지 않게
Mat padSquare(const Mat& src)
{
    int L = max(src.cols, src.rows);
    Mat dst;
    copyMakeBorder(src, dst, (L - src.rows) / 2, (L - src.rows + 1) / 2,
                   (L - src.cols) / 2, (L - src.cols + 1) / 2, BORDER_CONSTANT, Scalar(255));
    return dst;
}

// 각도별 템플릿 HOG (큰 템플릿을 돌린 뒤 WIN x WIN)
void templateHOG(const Mat& big, float degree, float* desc)
{
    Mat rot = rotateTemplate(big, degree), ref;
    resize(rot, ref, Size(WIN, WIN), 0, 0, INTER_AREA);
    computeHOG(ref, desc);
}

// ===== 후보 목록 : 점수 높은 순 NCAND 개, 원래 그림에서 가까운 후보끼리는 높은 것만 =====
struct Cand { float score; int x, y; float s; float angle; int flip; };   // flip : 1 이면 좌우 반전 템플릿   // x, y : 줄인 그림에서 창 왼쪽 위

void addCand(vector<Cand>& list, Cand c)
{
    float ox = (c.x + WIN / 2) / c.s, oy = (c.y + WIN / 2) / c.s;     // 원래 그림 좌표 (중심)
    for (size_t i = 0; i < list.size(); i++) {
        float px = (list[i].x + WIN / 2) / list[i].s, py = (list[i].y + WIN / 2) / list[i].s;
        if (hypot(ox - px, oy - py) < 20) {          // 같은 자리 (다른 배율/각도) -> 높은 것만
            if (c.score > list[i].score) list[i] = c;
            return;
        }
    }
    list.push_back(c);
    for (int i = (int)list.size() - 1; i > 0 && list[i].score > list[i - 1].score; i--) swap(list[i], list[i - 1]);
    if ((int)list.size() > NCAND) list.pop_back();
}

// 1단계 : 한 배율에서 전체 탐색 -> NMS 통과한 점을 후보 목록에
void coarseScan(const vector<float>& mag, const vector<int>& bin, int w, int h,
                const float* ref, float s, float angle, int flip, vector<Cand>& list)
{
    int nx = w - WIN + 1, ny = h - WIN + 1;
    if (nx <= 0 || ny <= 0) return;
    vector<float> score(nx * ny, -1.0f);
    float tarHOG[DIM];
    for (int y = 0; y < ny; y += STEP)
        for (int x = 0; x < nx; x += STEP) {
            hogFromGradient(mag, bin, w, x, y, tarHOG);
            score[y * nx + x] = computeSimilarity(ref, tarHOG);
        }
    for (int y = 0; y < ny; y += STEP)
        for (int x = 0; x < nx; x += STEP) {
            float o = score[y * nx + x];
            bool isMax = true;
            for (int yy = max(0, y - nmsR); yy <= min(ny - 1, y + nmsR) && isMax; yy++)
                for (int xx = max(0, x - nmsR); xx <= min(nx - 1, x + nmsR); xx++)
                    if (score[yy * nx + xx] > o) { isMax = false; break; }
            if (isMax) addCand(list, { o, x, y, s, angle, flip });
        }
}

// 2단계 : 후보 하나 주변에서 배율 3개 x 각도 x 위치 +-REFINE_R 를 촘촘히
Cand refine(const Mat& tar, const Mat& big0, Cand c)
{
    Mat big = big0;
    if (c.flip) flip(big0, big, 1);                 // 후보가 반전 템플릿에서 나왔으면 반전해서 비교
    Cand best = c;
    float ss[3] = { c.s / 1.1f, c.s, c.s * 1.1f };
    for (int si = 0; si < 3; si++) {
        Mat small;
        resize(tar, small, Size(), ss[si], ss[si], INTER_AREA);
        if (small.cols < WIN || small.rows < WIN) continue;
        // 후보 주변만 잘라서 gradient (전체를 다시 계산할 필요 없음)
        int cx = (int)((c.x + WIN / 2) / c.s * ss[si]), cy = (int)((c.y + WIN / 2) / c.s * ss[si]);
        int x0 = max(0, cx - WIN / 2 - REFINE_R - 2), y0 = max(0, cy - WIN / 2 - REFINE_R - 2);
        int x1 = min(small.cols, cx + WIN / 2 + REFINE_R + 2), y1 = min(small.rows, cy + WIN / 2 + REFINE_R + 2);
        if (x1 - x0 < WIN || y1 - y0 < WIN) continue;
        Mat roi = small(Rect(x0, y0, x1 - x0, y1 - y0)).clone();
        vector<float> mag;
        vector<int> bin;
        computeGradient(roi, mag, bin);

        for (float a = c.angle - REFINE_RANGE; a <= c.angle + REFINE_RANGE + 0.1f; a += REFINE_STEP) {
            float ref[DIM], tarHOG[DIM];
            templateHOG(big, a, ref);
            for (int y = 0; y + WIN <= roi.rows; y++)
                for (int x = 0; x + WIN <= roi.cols; x++) {
                    hogFromGradient(mag, bin, roi.cols, x, y, tarHOG);
                    float sc = computeSimilarity(ref, tarHOG) - ROT_PENALTY * fabsf(a) / 30.0f;
                    if (sc > best.score) best = { sc, x0 + x, y0 + y, ss[si], a, c.flip };
                }
        }
    }
    return best;
}

// ===== 구별력 보정 : 다른 물체 템플릿과도 비슷한 "흔한 무늬" 자리는 감점 (5주차 비율 검사 발상) =====
//   최종 점수 = 내 템플릿 점수 - LAMBDA x (그 자리에서 다른 템플릿들의 점수 중 가장 높은 값)
#define MAXOBJ 40
float otherRef[MAXOBJ][2][DIM];     // 물체별 0도 템플릿 HOG (원래, 반전)
int nOther = 0;

float otherScore(const Mat& tar, const Cand& c, int self)
{
    Mat small;
    resize(tar, small, Size(), c.s, c.s, INTER_AREA);
    int x0 = max(0, c.x - 1), y0 = max(0, c.y - 1);
    int x1 = min(small.cols, c.x + WIN + 1), y1 = min(small.rows, c.y + WIN + 1);
    Mat roi = small(Rect(x0, y0, x1 - x0, y1 - y0)).clone();
    vector<float> m; vector<int> b;
    computeGradient(roi, m, b);
    float h[DIM];
    hogFromGradient(m, b, roi.cols, c.x - x0, c.y - y0, h);
    float best = 0;
    for (int o = 0; o < nOther; o++) {
        if (o == self) continue;
        for (int f = 0; f < 2; f++) best = max(best, computeSimilarity(otherRef[o][f], h));
    }
    return best;
}

int main()
{
    int ok = 0, total = 0;
    float gapSum = 0;
    char path[512];
    int64 t0 = getTickCount();

    for (int n = 1; n <= NPUZZLE; n++) {
        snprintf(path, sizeof(path), "%s/puzzle_%d.png", PUZZLE_DIR, n);
        Mat tar = imread(path, 0);
        if (tar.empty()) { printf("못 읽음: %s\n", path); return -1; }
        Mat show;
        cvtColor(tar, show, COLOR_GRAY2BGR);

        // 배율마다 gradient 는 물체가 바뀌어도 같음 -> 미리 한 번만 계산해서 저장
        vector<float> scales;
        vector<vector<float>> mags;
        vector<vector<int>> bins;
        vector<Size> sizes;
        // 장면 크기 자동 대응 : 800px 기준 배율 범위를 실제 가로 크기에 비례하게 조절
        //   (1600px 장면 -> base 0.5 -> 0.24 ~ 0.6배,  400px 장면 -> base 2.0 -> 0.96 ~ 2.4배)
        //   좌표는 x / s, 박스는 WIN / s 로 되돌리므로 나머지 코드는 그대로 동작
        float base = (float)BASE_WIDTH / tar.cols;
        printf("장면 %d x %d  ->  배율 범위 %.2f ~ %.2f\n", tar.cols, tar.rows, SCALE_MIN * base, SCALE_MAX * base);
        for (float s = SCALE_MIN * base; s <= SCALE_MAX * base + 0.001f; s *= 1.1f) {
            Mat small;
            resize(tar, small, Size(), s, s, INTER_AREA);
            mags.push_back(vector<float>());
            bins.push_back(vector<int>());
            computeGradient(small, mags.back(), bins.back());
            scales.push_back(s);
            sizes.push_back(small.size());
        }

        snprintf(path, sizeof(path), "%s/answer_%d.txt", PUZZLE_DIR, n);
        FILE* fp = fopen(path, "r");
        if (!fp) { printf("못 읽음: %s\n", path); return -1; }
        char header[256];
        if (!fgets(header, sizeof(header), fp)) { fclose(fp); return -1; }

        // 다른 물체 비교용 : 모든 템플릿의 0도 HOG 를 미리 계산
        {
            long pos = ftell(fp);
            char nm[32];  int t1, t2, t3;  float f1, f2;
            nOther = 0;
            while (nOther < MAXOBJ && fscanf(fp, "%31s %d %d %f %f %d", nm, &t1, &t2, &f1, &f2, &t3) == 6) {
                snprintf(path, sizeof(path), "%s/big_%s.png", TEMPLATE_DIR, nm);
                Mat b = imread(path, 0);
                if (b.empty()) continue;
                b = padSquare(b);
                Mat bf;
                flip(b, bf, 1);
                templateHOG(b, 0, otherRef[nOther][0]);
                templateHOG(bf, 0, otherRef[nOther][1]);
                nOther++;
            }
            fseek(fp, pos, SEEK_SET);
        }
        int objIdx = 0;

        char name[32];  int ax, ay, size;  float angle, scale;
        while (fscanf(fp, "%31s %d %d %f %f %d", name, &ax, &ay, &angle, &scale, &size) == 6) {
            snprintf(path, sizeof(path), "%s/big_%s.png", TEMPLATE_DIR, name);
            Mat big = imread(path, 0);
            if (big.empty()) { printf("못 읽음: %s\n", path); continue; }
            big = padSquare(big);                               // 템플릿 크기 · 비율 자동 대응

            // ---- 1단계 : 거친 각도 x 모든 배율 x 전체 위치 -> 후보 NCAND 곳
            vector<Cand> list;
            for (int f = 0; f <= USE_FLIP; f++) {               // 0 : 원래 템플릿, 1 : 좌우 반전 템플릿
                Mat tmpl = big;
                if (f) flip(big, tmpl, 1);
                for (int a = 0; a < 360; a += COARSE_STEP) {
                    float ref[DIM];
                    templateHOG(tmpl, (float)a, ref);
                    for (size_t i = 0; i < scales.size(); i++)
                        coarseScan(mags[i], bins[i], sizes[i].width, sizes[i].height, ref, scales[i], (float)a, f, list);
                }
            }

            // ---- 2단계 : 후보마다 촘촘한 각도로 다시 -> 가장 높은 후보가 답
            vector<Cand> fine;
            for (size_t i = 0; i < list.size(); i++) fine.push_back(refine(tar, big, list[i]));
            // ---- 구별력 보정 : 다른 물체와도 비슷한 자리는 감점
            for (size_t i = 0; i < fine.size(); i++)
                fine[i].score -= LAMBDA * otherScore(tar, fine[i], objIdx);
            objIdx++;
            Cand best = fine[0];
            float second = -1;
            for (size_t i = 1; i < fine.size(); i++) {
                if (fine[i].score > best.score) { second = best.score; best = fine[i]; }
                else if (fine[i].score > second) second = fine[i].score;
            }

            int fx = (int)((best.x + WIN / 2) / best.s), fy = (int)((best.y + WIN / 2) / best.s);
            float d = sqrt((float)((fx - ax) * (fx - ax) + (fy - ay) * (fy - ay)));
            int hit = (d <= size * 0.3f);
            if (hit) ok++;
            gapSum += best.score - second;
            float ang = best.angle > 180 ? best.angle - 360 : best.angle;
            printf("%s", best.flip ? "[반전] " : "       ");
            printf("%-10s 정답(%3d,%3d)  찾음(%3d,%3d)  배율 %.2f  각도 %4.0f  거리 %5.1f  %s  1st %.3f gap %.3f\n",
                   name, ax, ay, fx, fy, best.s, ang, d, hit ? "O" : "X", best.score, best.score - second);

            Scalar color = hit ? Scalar(0, 200, 0) : Scalar(0, 0, 255);
            RotatedRect rr(Point2f((float)fx, (float)fy), Size2f(WIN / best.s, WIN / best.s), -best.angle);
            Point2f pt[4];
            rr.points(pt);
            for (int i = 0; i < 4; i++) line(show, pt[i], pt[(i + 1) % 4], color, 2);   // 돌아간 박스
            putText(show, name, Point(fx - 20, fy - (int)(WIN / best.s / 2) - 4), FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            if (!hit) circle(show, Point(ax, ay), 20, Scalar(255, 0, 0), 2);
            total++;
        }
        fclose(fp);
        snprintf(path, sizeof(path), "%s/result_%d.png", PUZZLE_DIR, n);
        imwrite(path, show);
        imshow("result", show);
        waitKey(0);
    }
    printf("%s : %d / %d   평균 gap %.3f\n", PUZZLE_DIR, ok, total, gapSum / total);
    printf("걸린 시간 %.1f 초\n", (getTickCount() - t0) / getTickFrequency());
    return 0;
}
