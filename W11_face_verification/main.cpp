// macos version (OpenCV 4 / 5)
// LectureNote 09 - Face-based Applications (3)
// Task : Face verification with landmark points
//   1) landmark 66점 전체 사용 (SDM 68점 중 입 안쪽 양끝 60, 64 제외 = 66점 규격)
//   2) 각 landmark 중심 17x17 윈도우
//   3) 윈도우마다 uniform LBP 히스토그램 (59 bin)
//   4) 이어 붙이기: 66 x 59 = 3894 차원
//   5) enroll vs test 유사도 -> 같은 사람 / 다른 사람
//
// 사용법
//   ./landmark_verify                          -> 웹캠 (e: 등록, ESC: 종료)
//   ./landmark_verify enroll.jpg test1.jpg ...  -> 이미지 (첫 장 등록, 나머지 검증)

#include <opencv2/opencv.hpp>
#include "sdm_compat.h"          // SDM(OpenCV 2.4 코드) -> OpenCV 4/5 호환
#include "ldmarkmodel.h"

#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

// ------------------- 상수 정의 -------------------
#define COSINE                   // 유사도: COSINE 또는 INTERSECTION
#define EPS        1e-6f

#define MODEL_PATH "roboman-landmark-model.bin"
#define HAAR_PATH  "haar_roboman_ff_alt2.xml"
#define ENROLL_BIN "enroll_feature.bin"   // 등록 특징 저장 파일

#define NUM_LM     68            // SDM 모델 landmark 수
#define NUM_PT     66            // 사용하는 점 수
#define ULBP_BIN   59            // uniform LBP: 58 uniform + 1 non-uniform
#define DIM        (NUM_PT * ULBP_BIN)   // 66 x 59 = 3894
#define WIN        17            // 윈도우 크기 (17x17)
#define HALF       (WIN / 2)     // 8 -> [cx-8, cx+8]

// 정렬된 얼굴 크기 (턱선까지 들어가게)
#define FACE_W     200
#define FACE_H     220
#define EYE_DIST   64.0f         // 정렬 후 두 눈 중심 사이 거리
#define EYE_Y      85.0f         // 정렬 후 두 눈 중심 y

// 프레임마다 흔들리는 값 안정화
#define BLUR_K         5         // LBP 전 Gaussian blur 크기
#define ENROLL_FRAMES  10        // 등록: 10프레임 특징 평균
#define SMOOTH_N       10        // 판정: 최근 10프레임 유사도 평균

#ifdef COSINE
float threshold_ = 0.85f;       // 이 값 이상이면 같은 사람 (직접 찍어보고 조정)
#else
float threshold_ = 0.70f;
#endif
// -------------------------------------------------

int ptIdx[NUM_PT];               // 사용할 landmark 번호
int lookup[256];                 // LBP 코드(0~255) -> bin(0~58)

// 1) 사용할 점: 0~67 중 60, 64 제외
void makePointIndex()
{
    int k = 0;
    for (int i = 0; i < NUM_LM; i++) {
        if (i == 60 || i == 64) continue;
        ptIdx[k++] = i;
    }
}

// uniform LBP lookup: 원형으로 0<->1 전이가 2번 이하면 uniform(0~57), 나머지 58
void makeLookup()
{
    int idx = 0;
    for (int code = 0; code < 256; code++) {
        int trans = 0;
        for (int b = 0; b < 8; b++) {
            int b1 = (code >> b) & 1;
            int b2 = (code >> ((b + 1) % 8)) & 1;
            if (b1 != b2) trans++;
        }
        lookup[code] = (trans <= 2) ? idx++ : ULBP_BIN - 1;
    }
}

// 영상 전체 LBP 코드 (3x3, 이웃 >= 중심이면 1)
Mat computeLBP(const Mat& gray)
{
    Mat lbp(gray.size(), CV_8UC1, Scalar(0));
    int dy[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };   // 12시부터 시계방향
    int dx[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

    for (int y = 1; y < gray.rows - 1; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {
            uchar c = gray.at<uchar>(y, x);
            int v = 0;
            for (int n = 0; n < 8; n++)
                if (gray.at<uchar>(y + dy[n], x + dx[n]) >= c) v |= (1 << n);
            lbp.at<uchar>(y, x) = (uchar)v;
        }
    }
    return lbp;
}

// 두 눈 기준 정렬 (회전 + 크기 + 위치) -> 거리/기울기 달라도 같은 부위를 같은 크기로 봄
void alignFace(const Mat& gray, const Mat& shape, Mat& face, vector<Point2f>& pts)
{
    Point2f L(0, 0), R(0, 0);
    for (int i = 36; i <= 41; i++) L += Point2f(shape.at<float>(i), shape.at<float>(i + NUM_LM));
    for (int i = 42; i <= 47; i++) R += Point2f(shape.at<float>(i), shape.at<float>(i + NUM_LM));
    L *= (1.0f / 6); R *= (1.0f / 6);

    double angle = atan2(R.y - L.y, R.x - L.x);                 // rad
    double dist  = sqrt((R.x - L.x) * (R.x - L.x) + (R.y - L.y) * (R.y - L.y));
    double scale = EYE_DIST / (dist + EPS);
    Point2f mid  = (L + R) * 0.5f;

    // getRotationMatrix2D 직접 계산 (OpenCV 5에서 geometry 모듈로 이동)
    double al = scale * cos(angle), be = scale * sin(angle);
    Mat M = (Mat_<double>(2, 3) << al, be, (1 - al) * mid.x - be * mid.y,
                                   -be, al, be * mid.x + (1 - al) * mid.y);
    M.at<double>(0, 2) += FACE_W * 0.5 - mid.x;                 // 눈 중심 -> (FACE_W/2, EYE_Y)
    M.at<double>(1, 2) += EYE_Y - mid.y;

    warpAffine(gray, face, M, Size(FACE_W, FACE_H), INTER_LINEAR, BORDER_REPLICATE);

    pts.clear();
    for (int k = 0; k < NUM_PT; k++) {
        double x = shape.at<float>(ptIdx[k]), y = shape.at<float>(ptIdx[k] + NUM_LM);
        pts.push_back(Point2f((float)(M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2)),
                              (float)(M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2))));
    }
}

// 2)~4) 66점마다 17x17 윈도우 uniform LBP 히스토그램 -> 이어 붙이기 (3894차원)
void LBPdescriptor(const Mat& face, const vector<Point2f>& pts, float* desc)
{
    // 카메라 노이즈 / 점 떨림에 LBP가 민감 -> 살짝 블러 후 계산 (같은 얼굴 유사도 안정)
    Mat smooth;
    GaussianBlur(face, smooth, Size(BLUR_K, BLUR_K), 0);
    Mat lbp = computeLBP(smooth);

    for (int k = 0; k < NUM_PT; k++) {
        float* hist = &desc[k * ULBP_BIN];
        for (int b = 0; b < ULBP_BIN; b++) hist[b] = 0;

        int cx = cvRound(pts[k].x), cy = cvRound(pts[k].y);
        for (int y = cy - HALF; y <= cy + HALF; y++) {
            for (int x = cx - HALF; x <= cx + HALF; x++) {
                // 얼굴 밖은 건너뜀 (히스토그램 개수는 항상 66개 유지)
                if (y < 1 || y >= face.rows - 1 || x < 1 || x >= face.cols - 1) continue;
                hist[lookup[lbp.at<uchar>(y, x)]] += 1.0f;
            }
        }

        // 윈도우별 정규화
#ifdef COSINE
        float norm = EPS;                                       // L2
        for (int b = 0; b < ULBP_BIN; b++) norm += hist[b] * hist[b];
        norm = sqrt(norm);
#else
        float norm = EPS;                                       // L1 (합 = 1)
        for (int b = 0; b < ULBP_BIN; b++) norm += hist[b];
#endif
        for (int b = 0; b < ULBP_BIN; b++) hist[b] /= norm;
    }
}

// 5) 유사도 (클수록 같은 사람)
float computeSimilarity(const float* ref, const float* tar)
{
#ifdef COSINE
    float nomi = 0, refMag = 0, tarMag = 0;
    for (int k = 0; k < DIM; k++) {
        nomi   += ref[k] * tar[k];
        refMag += ref[k] * ref[k];
        tarMag += tar[k] * tar[k];
    }
    return nomi / (sqrt(refMag) * sqrt(tarMag) + EPS);
#else
    float s = 0;                                                // histogram intersection
    for (int k = 0; k < DIM; k++) s += min(ref[k], tar[k]);
    return s / NUM_PT;
#endif
}

// landmark 검출 -> 정렬 -> 특징
bool getFeature(ldmarkmodel& model, const Mat& img, Mat& shape, bool detect,
                float* desc, Mat& face, vector<Point2f>& pts)
{
    if (detect) shape = Mat();
    int err = model.track(img, shape, detect);
    if (err == SDM_ERROR_FACENO || shape.empty() || shape.cols != NUM_LM * 2) return false;

    Mat gray;
    if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
    else gray = img;

    alignFace(gray, shape, face, pts);
    LBPdescriptor(face, pts, desc);
    return true;
}

// 등록 특징 저장 / 불러오기
bool saveEnroll(const float* desc)
{
    FILE* fp = fopen(ENROLL_BIN, "wb");
    if (!fp) return false;
    fwrite(desc, sizeof(float), DIM, fp);
    fclose(fp);
    return true;
}

bool loadEnroll(float* desc)
{
    FILE* fp = fopen(ENROLL_BIN, "rb");
    if (!fp) return false;
    size_t n = fread(desc, sizeof(float), DIM, fp);
    fclose(fp);
    return n == DIM;
}

// 점 색깔 (BGR)
#define COLOR_SAME    Scalar(0, 255, 0)     // 초록: 등록된 사람 (ACCEPT)
#define COLOR_OTHER   Scalar(0, 0, 255)     // 빨강: 다른 사람 (REJECT)
#define COLOR_NONE    Scalar(0, 255, 255)   // 노랑: 아직 등록 전

// 그리기: 사용한 66점을 color로 (제외한 60, 64번은 안 그림)
void drawLandmarks(Mat& img, const Mat& shape, Scalar color)
{
    for (int k = 0; k < NUM_PT; k++) {
        int i = ptIdx[k];
        Point p(cvRound(shape.at<float>(i)), cvRound(shape.at<float>(i + NUM_LM)));
        circle(img, p, 2, color, -1);
    }
}

// 정렬된 얼굴 위에 17x17 윈도우 표시
Mat drawWindows(const Mat& face, const vector<Point2f>& pts)
{
    Mat vis;
    cvtColor(face, vis, COLOR_GRAY2BGR);
    for (size_t k = 0; k < pts.size(); k++) {
        Point p(cvRound(pts[k].x), cvRound(pts[k].y));
        rectangle(vis, Rect(p.x - HALF, p.y - HALF, WIN, WIN), Scalar(0, 255, 0), 1);
    }
    resize(vis, vis, Size(), 2.0, 2.0, INTER_NEAREST);
    return vis;
}

// 이미지 모드: 첫 장 등록, 나머지 검증
int runImages(ldmarkmodel& model, int argc, char** argv)
{
    static float enrollDesc[DIM], testDesc[DIM];
    Mat shape, face;
    vector<Point2f> pts;

    Mat enroll = imread(argv[1]);
    if (enroll.empty()) { printf("cannot read %s\n", argv[1]); return -1; }
    if (!getFeature(model, enroll, shape, true, enrollDesc, face, pts)) {
        printf("no face in %s\n", argv[1]); return -1;
    }
    saveEnroll(enrollDesc);
    drawLandmarks(enroll, shape, COLOR_SAME);
    imwrite("enroll_landmarks.bmp", enroll);
    imwrite("enroll_windows.bmp", drawWindows(face, pts));
    printf("enrolled: %s  (feature dim = %d = %d points x %d bins)\n\n", argv[1], DIM, NUM_PT, ULBP_BIN);

    for (int i = 2; i < argc; i++) {
        Mat test = imread(argv[i]);
        if (test.empty()) { printf("cannot read %s\n", argv[i]); continue; }
        if (!getFeature(model, test, shape, true, testDesc, face, pts)) {
            printf("%-25s : no face\n", argv[i]); continue;
        }
        float sim = computeSimilarity(enrollDesc, testDesc);
        printf("%-25s : similarity = %.4f -> %s\n", argv[i], sim,
               sim >= threshold_ ? "ACCEPT (same)" : "REJECT (different)");

        drawLandmarks(test, shape, sim >= threshold_ ? COLOR_SAME : COLOR_OTHER);
        char name[64];
        snprintf(name, sizeof(name), "test%d_landmarks.bmp", i - 1);
        imwrite(name, test);
    }
    return 0;
}

// 웹캠 모드: e 로 등록(파일 저장), 이후 실시간 검증
int runCamera(ldmarkmodel& model)
{
    VideoCapture cap(0);
    if (!cap.isOpened()) { printf("Camera opening failed...\n"); return -1; }

    static float enrollDesc[DIM], curDesc[DIM], enrollSum[DIM];
    bool enrolled = loadEnroll(enrollDesc);       // 전에 등록한 게 있으면 불러옴
    int  enrollCount = -1;                        // -1: 등록 안 하는 중, 0~: 모은 프레임 수
    float simHist[SMOOTH_N];                      // 최근 유사도 기록
    int  simCount = 0;                            // 기록된 개수 (얼굴 놓치면 0으로)
    Mat frame, shape, face;
    vector<Point2f> pts;

    printf("e: enroll (%d frames)   ESC: quit\n", ENROLL_FRAMES);
    if (enrolled) printf("loaded %s\n", ENROLL_BIN);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        bool ok = getFeature(model, frame, shape, true, curDesc, face, pts);   // 매 프레임 새로 검출 (사람이 바뀌어도 바로 찾음)
        if (!ok) simCount = 0;                     // 얼굴 놓치면 평균 다시 시작

        // ---- 등록 중: ENROLL_FRAMES 프레임 특징을 더해서 평균 ----
        if (enrollCount >= 0) {
            if (ok) {
                for (int i = 0; i < DIM; i++) enrollSum[i] += curDesc[i];
                enrollCount++;
                drawLandmarks(frame, shape, COLOR_NONE);
            }
            char txt[64];
            snprintf(txt, sizeof(txt), "enrolling... %d/%d", enrollCount, ENROLL_FRAMES);
            putText(frame, txt, Point(20, frame.rows - 20), FONT_HERSHEY_SIMPLEX, 1.0, COLOR_NONE, 2);

            if (enrollCount == ENROLL_FRAMES) {
                for (int i = 0; i < DIM; i++) enrollDesc[i] = enrollSum[i] / ENROLL_FRAMES;
                enrolled = true;
                enrollCount = -1;
                simCount = 0;
                saveEnroll(enrollDesc);
                imwrite("enroll_windows.bmp", drawWindows(face, pts));
                printf("enrolled! (%d frames averaged, saved %s)\n", ENROLL_FRAMES, ENROLL_BIN);
            }
        }
        // ---- 검증: 최근 SMOOTH_N 프레임 유사도 평균으로 판정 ----
        else if (ok) {
            imshow("Aligned face + 17x17 windows", drawWindows(face, pts));

            if (enrolled) {
                float sim = computeSimilarity(enrollDesc, curDesc);
                simHist[simCount % SMOOTH_N] = sim;
                simCount++;
                int n = min(simCount, SMOOTH_N);
                float avg = 0;
                for (int i = 0; i < n; i++) avg += simHist[i];
                avg /= n;

                bool same = avg >= threshold_;
                Scalar color = same ? COLOR_SAME : COLOR_OTHER;   // 같은 사람 초록, 다른 사람 빨강
                drawLandmarks(frame, shape, color);

                char txt[80];
                snprintf(txt, sizeof(txt), "sim %.3f (now %.3f)  %s", avg, sim, same ? "ACCEPT" : "REJECT");
                putText(frame, txt, Point(20, frame.rows - 20), FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
            } else {
                drawLandmarks(frame, shape, COLOR_NONE);
            }
        }
        if (!enrolled && enrollCount < 0)
            putText(frame, "press 'e' to enroll", Point(20, frame.rows - 20),
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0), 2);

        imshow("Camera", frame);
        int key = waitKey(5);
        if (key == 27) break;
        if (key == 'e') {                          // 등록 시작: 정면 보고 잠깐 가만히
            for (int i = 0; i < DIM; i++) enrollSum[i] = 0;
            enrollCount = 0;
            printf("enrolling... look at the camera\n");
        }
    }
    cap.release();
    destroyAllWindows();
    return 0;
}

int main(int argc, char** argv)
{
    makePointIndex();
    makeLookup();

    ldmarkmodel model;
    if (!load_ldmarkmodel(MODEL_PATH, model)) {
        printf("cannot open %s (실행 폴더에 두세요)\n", MODEL_PATH);
        return -1;
    }
    model.loadFaceDetModelFile(HAAR_PATH);

    if (argc >= 3) return runImages(model, argc, argv);
    return runCamera(model);
}
