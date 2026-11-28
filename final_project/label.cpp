// 라벨링 도구 : 장면에서 물체를 하나씩 드래그로 표시 -> answer_1.txt 저장
//   (HiddenTest 정답 파일과 같은 형식 : 이름 중심x 중심y 각도 크기 한변px)
//
//   사용법 : ./label  (아래 DATA 경로만 바꾸기, 물체 이름은 DATA/names.txt)
//   - 왼쪽 위에 지금 찾을 물체 템플릿이 보임
//   - 장면에서 그 물체를 마우스로 드래그해서 박스 치기 (다시 드래그하면 고쳐짐)
//   - SPACE : 확정하고 다음 물체      s : 못 찾음 (건너뛰기)
//   - u     : 바로 전 물체 취소       ESC : 저장하고 끝
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

#define DATA    "./data"
#define MAXOBJ  40

char NAMES[MAXOBJ][32];     // DATA/names.txt 에서 읽음 (한 줄에 이름 하나)
int  NOBJ = 0;

int sx = -1, sy = -1, ex = -1, ey = -1, dragging = 0;
int lx[MAXOBJ], ly[MAXOBJ], lsize[MAXOBJ], done[MAXOBJ];     // 라벨 결과 (done : 1 표시함, 0 아직, -1 못 찾음)

void onMouse(int event, int x, int y, int, void*)
{
    if (event == EVENT_LBUTTONDOWN) { sx = ex = x; sy = ey = y; dragging = 1; }
    else if (event == EVENT_MOUSEMOVE && dragging) { ex = x; ey = y; }
    else if (event == EVENT_LBUTTONUP) { ex = x; ey = y; dragging = 0; }
}

int main()
{
    char path[512];
    snprintf(path, sizeof(path), "%s/names.txt", DATA);
    FILE* fn = fopen(path, "r");
    if (!fn) { printf("못 읽음: %s\n", path); return -1; }
    while (NOBJ < MAXOBJ && fscanf(fn, "%31s", NAMES[NOBJ]) == 1) NOBJ++;
    fclose(fn);
    printf("%d objects\n", NOBJ);

    snprintf(path, sizeof(path), "%s/scene.png", DATA);
    Mat scene = imread(path);
    if (scene.empty()) { printf("못 읽음: %s\n", path); return -1; }

    namedWindow("label");
    setMouseCallback("label", onMouse);

    int k = 0;
    while (k < NOBJ) {
        snprintf(path, sizeof(path), "%s/templates/big_%s.png", DATA, NAMES[k]);
        Mat tmpl = imread(path);

        Mat view = scene.clone();
        // 이미 표시한 것 : 초록 박스
        for (int i = 0; i < k; i++) {
            if (done[i] != 1) continue;
            rectangle(view, Rect(lx[i] - lsize[i] / 2, ly[i] - lsize[i] / 2, lsize[i], lsize[i]), Scalar(0, 200, 0), 1);
            putText(view, NAMES[i], Point(lx[i] - lsize[i] / 2, ly[i] - lsize[i] / 2 - 3), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 150, 0), 1);
        }
        // 지금 드래그 중인 박스 : 빨강
        if (sx >= 0) rectangle(view, Point(sx, sy), Point(ex, ey), Scalar(0, 0, 255), 2);
        // 왼쪽 위 : 찾을 물체
        if (!tmpl.empty()) {
            Mat small;
            resize(tmpl, small, Size(80, 80));
            small.copyTo(view(Rect(5, 5, 80, 80)));
            rectangle(view, Rect(5, 5, 80, 80), Scalar(0, 0, 255), 2);
        }
        char text[96];
        snprintf(text, sizeof(text), "%d/%d %s  (drag, SPACE ok, s skip, u undo)", k + 1, NOBJ, NAMES[k]);
        putText(view, text, Point(92, 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        imshow("label", view);

        int key = waitKey(20);
        if (key == 27) break;
        if (key == ' ' && sx >= 0 && abs(ex - sx) > 3 && abs(ey - sy) > 3) {
            lx[k] = (sx + ex) / 2;
            ly[k] = (sy + ey) / 2;
            lsize[k] = std::max(abs(ex - sx), abs(ey - sy));     // 긴 변 = 물체 크기
            done[k] = 1;
            printf("%-10s (%d, %d) size %d\n", NAMES[k], lx[k], ly[k], lsize[k]);
            k++; sx = -1;
        }
        if (key == 's') { done[k] = -1; printf("%-10s 건너뜀\n", NAMES[k]); k++; sx = -1; }
        if (key == 'u' && k > 0) { k--; done[k] = 0; sx = -1; printf("취소 -> %s 다시\n", NAMES[k]); }
    }

    // 저장 : HiddenTest answer 형식 (각도 0, 크기배율 = 한변 / 36)
    snprintf(path, sizeof(path), "%s/answer_1.txt", DATA);
    FILE* fp = fopen(path, "w");
    fprintf(fp, "# name cx cy angle scale size(px)\n");
    for (int i = 0; i < NOBJ; i++)
        if (done[i] == 1) fprintf(fp, "%s %d %d 0.0 %.2f %d\n", NAMES[i], lx[i], ly[i], lsize[i] / 36.0f, lsize[i]);
    fclose(fp);
    printf("saved %s\n", path);
    return 0;
}
