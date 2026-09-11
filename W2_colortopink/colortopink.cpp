#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	// 이미지 읽기
	Mat imgColor = imread("C:/Users/sinbc/test.jpg", CV_LOAD_IMAGE_COLOR);

	Mat result = imgColor.clone();

	int cx, cy;
	int BLK = 30;

	cout << "Input center x : ";
	cin >> cx;

	cout << "Input center y : ";
	cin >> cy;

	for (int y = cy - BLK; y <= cy + BLK; y++)
	{
		for (int x = cx - BLK; x <= cx + BLK; x++)
		{
			if (x >= 0 && x < imgColor.cols &&
				y >= 0 && y < imgColor.rows)
			{
				int b = imgColor.at<Vec3b>(y, x)[0];
				int g = imgColor.at<Vec3b>(y, x)[1];
				int r = imgColor.at<Vec3b>(y, x)[2];

				//gray 값 계산
				int gray = (r + g + b) / 3;

				//B, G, R을 모두 같은 값으로 바꾸면 회색이 됨.
				result.at<Vec3b>(y, x)[0] = 255;
				result.at<Vec3b>(y, x)[1] = 0;
				result.at<Vec3b>(y, x)[2] = 255;
			}
		}
	}

	imshow("Original", imgColor);
	imshow("Result", result);

	waitKey(0);

	return 0;

}
