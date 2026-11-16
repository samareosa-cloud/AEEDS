#include "main.h"

void SLICsegmentation(Mat image)
{
    SLIC slic;
    int x, y;
    int height, width;
    int numlabels;               // Generated number of superpixels
    int m_spcount = 1000;        // Desired number of superpixels (K)
    double m_compactness = 20;   // compactness factor m (1-40)

    height = image.rows;
    width  = image.cols;

    unsigned int* ubuff = (unsigned int*)calloc(height * width, sizeof(unsigned int));
    int* labels = NULL;          // SLIC 함수 안에서 new int[]로 할당됨 (calloc 불필요)

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            ubuff[y * width + x] =  (int)image.at<Vec3b>(y, x)[0]
                                 + ((int)image.at<Vec3b>(y, x)[1] << 8)
                                 + ((int)image.at<Vec3b>(y, x)[2] << 16);
        }
    }

    // 출력 : labels = 픽셀마다 superpixel 번호, numlabels = 실제 생성된 superpixel 개수
    slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(ubuff, width, height, labels,
                                                              numlabels, m_spcount, m_compactness);
    std::cout << "K = " << m_spcount << ", m = " << m_compactness
              << " -> numlabels = " << numlabels << std::endl;

    // for drawing SLIC superpixels
    Mat result(height, width, CV_8UC3);

    slic.DrawContoursAroundSegments(ubuff, labels, width, height, 0);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            result.at<Vec3b>(y, x)[0] = ubuff[y * width + x]       & 0xff;
            result.at<Vec3b>(y, x)[1] = ubuff[y * width + x] >> 8  & 0xff;
            result.at<Vec3b>(y, x)[2] = ubuff[y * width + x] >> 16 & 0xff;
        }
    }

    imwrite("./images/SLIC_segmentation.bmp", result);
    imshow("SLIC segmentation", result);
    waitKey(0);

    free(ubuff);
    delete[] labels;             // new로 할당됐으므로 free가 아니라 delete[]
}
