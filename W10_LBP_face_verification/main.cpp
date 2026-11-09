// macos version
#include <opencv2/opencv.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/xobjdetect.hpp>   // OpenCV 5부터 Haar cascade가 여기로 이동
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <thread>
#include <chrono>
#include "LBP.h"

using namespace cv;
using namespace std;

#define HAAR_FILE "haarcascade_frontalface_alt.xml"
#define HAAR_URL  "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_alt.xml"

float th = 0.50f;

bool openCamera(VideoCapture& cap)
{
    for (int idx = 0; idx < 4; idx++) {
        cap.open(idx);
        if (!cap.isOpened()) continue;
        Mat f, g;
        for (int i = 0; i < 90; i++) {
            cap >> f;
            if (f.empty()) { this_thread::sleep_for(chrono::milliseconds(30)); continue; }
            cvtColor(f, g, COLOR_BGR2GRAY);
            if (mean(g)[0] > 15) return true;
        }
        cap.release();
    }
    printf("camera open fail\n");
    return false;
}

// 현재 폴더 -> Homebrew 폴더 -> 인터넷 순서로 찾기
bool loadCascade(CascadeClassifier& cascade)
{
    FILE* fp = fopen(HAAR_FILE, "rb");
    if (fp) { fclose(fp); return cascade.load(HAAR_FILE); }

    char path[512] = "";
    FILE* p = popen("find /opt/homebrew /usr/local -name " HAAR_FILE " 2>/dev/null | head -1", "r");
    if (p) {
        if (fgets(path, sizeof(path), p)) path[strcspn(path, "\n")] = 0;
        pclose(p);
    }
    if (path[0] && cascade.load(path)) return true;

    printf("%s 다운로드\n", HAAR_FILE);
    if (system("curl -fsSL -o " HAAR_FILE " " HAAR_URL) != 0) return false;
    return cascade.load(HAAR_FILE);
}

int main()
{
    CascadeClassifier cascade;
    if (!loadCascade(cascade)) {
        printf("cascade load fail\n");
        return -1;
    }

    VideoCapture capture;
    if (!openCamera(capture)) return -1;

    Mat frame, raw;
    vector<Rect> faces;
    vector<float> refLBPhist, tarLBPhist;
    int flag = 0, cnt = 0;

    while (1) {
        capture >> frame;
        if (frame.empty()) { waitKey(10); continue; }
        resize(frame, frame, Size(640, 640 * frame.rows / frame.cols));
        raw = frame.clone();   // 박스 그리기 전 원본 (특징 추출용)
        cnt++;

        cascade.detectMultiScale(frame, faces, 1.1, 4, 0, Size(60, 60));

        // enrollment (ref) : 얼굴 1개만 보일 때 자동 등록
        if (faces.size() == 1 && flag == 0 && cnt > 30) {
            Point lb(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
            Point tr(faces[0].x, faces[0].y);
            LBPdescriptor(raw, refLBPhist, faces[0].x, faces[0].y, faces[0].width, faces[0].height);
            imwrite("enroll.bmp", raw(faces[0]));
            rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
            flag = 1;

            printf("enrolled\n");
        }

        // verification (tar)
        else if (faces.size() > 0 && flag != 0 && cnt > 30) {
            for (size_t i = 0; i < faces.size(); i++) {
                Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                Point tr(faces[i].x, faces[i].y);

                LBPdescriptor(raw, tarLBPhist, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
                float score = computeSimilarity(refLBPhist, tarLBPhist);

                Scalar color = (score < th) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
                if (cnt % 15 == 0)
                    printf("face %zu  score %.3f  %s\n", i, score, score < th ? "MATCH" : "NO");

                rectangle(frame, lb, tr, color, 3, 8, 0);
                char text[16];
                snprintf(text, sizeof(text), "%.3f", score);
                putText(frame, text, Point(tr.x, max(15, tr.y - 8)), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }
        }

        // 얼굴 LBP 이미지 따로 보기
        if (!faces.empty()) {
            Mat gray, face, lbp;
            cvtColor(raw(faces[0]), gray, COLOR_BGR2GRAY);
            resize(gray, face, Size(FACE_SIZE * 2, FACE_SIZE * 2));
            LBPimage(face, lbp);
            imshow("LBP", lbp);
        }

        imshow("Face", frame);

        int key = waitKey(1);
        if (key == 27 || key == 'q') break;
        if (key == '1') flag = 0;                    // 다시 등록
        if (key == '+' || key == '=') th += 0.01f;
        if (key == '-') th -= 0.01f;
    }
    return 0;
}