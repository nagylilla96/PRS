// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <random>
#include <stdio.h>

const int nrclasses = 6;
char classes[nrclasses][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };

struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

struct clustPoint {
	Point point;
	int cluster;
	int prevCluster;
};

struct meanCalc {
	int sumX, sumY, n;
};

struct classDist {
	float dist;
	int ind;
	bool operator < (const classDist &other) const {
		return dist < other.dist;
	}
};

struct trainingPoint {
	int clasa;
	int x;
	int y;
};

struct adaPoint {
	int X[2];
	int y;
	double w;
};

struct weaklearner {
	int feature;
	int threshold;
	int class_label;
	double error;
	int classify(int X[2])
	{
		if (X[feature] < threshold)
		{
			return class_label;
		}
		else
		{
			return -class_label;
		}
	}
};

struct classifier
{
	int T;
	std::vector<double> alphas;
	std::vector<weaklearner> hs;
	int classify(int x, int y)
	{
		double sum = 0;
		int X[2] = { x, y };
		for (int i = 0; i < T; i++)
		{
			sum += alphas.at(i) * hs.at(i).classify(X);
		}
		if (sum > 0) return 1;
		return -1;
	}
};

int findBin(int value)
{
	if (value >= 0 && value < 32)
	{
		return 0;
	}
	if (value >= 32 && value < 64)
	{
		return 1;
	}
	if (value >= 64 && value < 96)
	{
		return 2;
	}
	if (value >= 96 && value < 128)
	{
		return 3;
	}
	if (value >= 128 && value < 160)
	{
		return 4;
	}
	if (value >= 160 && value < 192)
	{
		return 5;
	}
	if (value >= 192 && value < 224)
	{
		return 6;
	}
	if (value >= 224 && value <= 256)
	{
		return 7;
	}
	return -1;
}

double distance(float x, float y)
{
	return (x - y) * (x - y);
}

// BGR
int* calcHist(Mat img, int nr_bins)
{
	int *hist = (int*)malloc(sizeof(int) * nr_bins * 3);
	for (int i = 0; i < nr_bins * 3; i++)
	{
		hist[i] = 0;
	}

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b color = img.at<Vec3b>(i, j);
			int blue = findBin(color[0]);
			int green = findBin(color[1]);
			int red = findBin(color[2]);
			hist[blue] ++;
			hist[green + nr_bins] ++;
			hist[red + 2 * nr_bins] ++;
		}
	}

	return hist;
}

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the �diblook style�
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}
		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

// Display the points on the image
void displayPoints(Mat img, std::vector<Point2f> points) {
	for (Point2f p : points) {
		img.at<Vec3b>(p.y, p.x)[0] = 0;
		img.at<Vec3b>(p.y, p.x)[1] = 0;
		img.at<Vec3b>(p.y, p.x)[2] = 0;
	}
}

/*
	Calculate theta 1
	n - number of points
*/
float calculateO1(int n, float sumax, float sumay, float sumaxx, float sumaxy) {
	float O1 = 0;
	O1 = (float)(n * sumaxy - sumax * sumay);
	O1 /= (float)(n * sumaxx - sumax * sumax);
	return O1;
}

/*
Calculate theta 0
n - number of points
*/
float calculateO0(int n, float O1, float sumax, float sumay) {
	float O0 = 0;
	O0 = (float)(sumay - O1 * sumax) / (float)n;
	return O0;
}

/*
Calculate beta
n - number of points
*/
float calculateBeta(int n, float sumax, float sumay, float sumaxx, float sumaxy, float sumayy_xx) {
	float beta = 0, par1 = 0, par2 = 0;
	par1 = (float)(2 * sumaxy - 2.0 * sumax * sumay / n);
	par2 = sumayy_xx + sumax * sumax / (float)n - sumay * sumay / (float)n;
	beta = -0.5 * atan2(par1, par2);
	return beta;
}

/*
Calculate ro
n - number of points
*/
float calculateRo(int n, float beta, float sumax, float sumay) {
	float ro = 0;
	ro = (cos(beta) * sumax + sin(beta) * sumay) / (float)n;
	return ro;
}

/*
	Display the image
*/
void drawImage(Mat img, std::vector<Point2f> points, float width, float O0, float O1, float beta, float ro) {
	// Define the points
	Point p1 = Point(0, O0);
	Point p2 = Point(width, O0 + width * O1);
	Point p3 = Point(0, ro / sin(beta));
	Point p4 = Point(width, (ro - width * cos(beta)) / sin(beta));

	for (int i = 0; i < 500; i++)
	{
		for (int j = 0; j < 500; j++)
		{
			// Display an 500x500 white image
			img.at<Vec3b>(i, j)[0] = 255; //blue
			img.at<Vec3b>(i, j)[1] = 255; //green
			img.at<Vec3b>(i, j)[2] = 255; //red
		}
	}

	// Display the points over the image
	displayPoints(img, points);

	// Draw lines
	line(img, p1, p2, Scalar(255, 255, 0), 5);
	line(img, p3, p4, Scalar(0, 0, 0));

	imshow("LeastMeanSquares", img);
	waitKey();
}

void leastMeanSquares()
{
	Mat img(500, 500, CV_8UC3);
	std::vector<Point2f> points;
	float xmin = 9999, ymin = 9999, xmax = 0, ymax = 0, ro = 0, O0 = 0, O1 = 0, beta = 0;
	float sumax = 0, sumay = 0, sumaxx = 0, sumayy_xx = 0, sumaxy = 0;
	char fname[MAX_PATH];
	int nr;

	// Open file
	openFileDlg(fname);
	FILE* f = fopen(fname, "r");

	// Scan the number of (x, y) pairs
	fscanf(f, "%i", &nr);

	for (int i = 0; i < nr; i++)
	{
		// Read each (x, y) pair, add them to the vector and calculate all mins and maxs
		Point2f point;
		float x, y;
		fscanf(f, "%f%f", &x, &y);
		if (x < xmin) xmin = x;
		if (y < ymin) ymin = y;
		if (x > xmax) xmax = x;
		if (y > ymax) ymax = y;
		point.x = (float)x;
		point.y = (float)y;
		points.push_back(point);
	}

	for (int i = 0; i < nr; i++)
	{
		// Normalize points and calculate sums
		points[i].x -= xmin;
		points[i].y -= ymin;
		sumax += points[i].x;
		sumay += points[i].y;
		sumaxy += points[i].x * points[i].y;
		sumaxx += points[i].x * points[i].x;
		sumayy_xx += points[i].y * points[i].y - points[i].x * points[i].x;
	}

	// Close file
	fclose(f);

	// Calculate necessary parameters
	O1 = calculateO1(nr, sumax, sumay, sumaxx, sumaxy);
	O0 = calculateO0(nr, O1, sumax, sumay);
	beta = calculateBeta(nr, sumax, sumay, sumaxx, sumaxy, sumayy_xx);
	ro = calculateRo(nr, beta, sumax, sumay);

	drawImage(img, points, xmax - xmin, O0, O1, beta, ro);
}

void ransac(int t, float p, float q, int s)
{
	std::vector<Point> points;
	Point P1, P2;
	int n = 0, rows, cols, A = 0, B = 0, C = 0, Skmax = 0;
	float N = 0, T = 0;
	char fname[MAX_PATH];

	srand(time(NULL));

	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

	rows = img.rows;
	cols = img.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				points.push_back(Point(j, i));
				n++;
			}
		}
	}

	N = (float)log(1 - p) / log(1 - pow(q, s));
	T = q * n;

	for (int i = 0; i < N; i++)
	{
		int i1, i2, Sk = 0, a, b, c;

		i1 = rand() % n;
		i2 = rand() % n;
		while (i1 == i2)
		{
			i1 = rand() % n;
			i2 = rand() % n;
		}

		Point p1 = Point(points[i1].x, points[i1].y);
		Point p2 = Point(points[i2].x, points[i2].y);
		a = p1.y - p2.y;
		b = p2.x - p1.x;
		c = p1.x * p2.y - p2.x * p1.y;

		for (int j = 0; j < n; j++)
		{
			float dist = 0;
			dist = (float)abs(a * points[j].x + b * points[j].y + c) / sqrt(a * a + b * b);
			if (dist <= t) Sk++;
		}

		if (Sk >= Skmax)
		{
			A = a;
			B = b;
			C = c;
			P1 = p1;
			P2 = p2;
			Skmax = Sk;
		}

		if (Sk >= T) break;
	}

	line(img, P1, P2, Scalar(0, 0, 0));

	imshow("Ransac", img);
	waitKey();
}

bool desc(peak i, peak j) { return i.hval > j.hval; }

void hough()
{
	std::vector<Point> points;
	std::vector<peak> peaks;
	int rows, cols, n = 0, maxHough = 0;
	double diag;
	char fname[MAX_PATH];
	Mat houghImg;

	srand(time(NULL));

	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat colorImg = imread(fname);


	rows = img.rows;
	cols = img.cols;
	diag = sqrt(rows * rows + cols * cols);
	Mat Hough(360, diag + 1, CV_32SC1, Scalar(0));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (img.at<uchar>(i, j) == 255)
			{
				points.push_back(Point(j, i));
				n++;
			}
		}
	}

	for (int i = 0; i < n; i++)
	{
		for (int theta = 0; theta < 360; theta++)
		{
			int ro = points[i].x * cos((theta * CV_PI) / 180) + points[i].y * sin((theta * CV_PI) / 180);
			if (ro >= 0 && ro <= diag)
			{
				Hough.at<int>(theta, ro)++;
				if (Hough.at<int>(theta, ro) > maxHough)
					maxHough = Hough.at<int>(theta, ro);
			}
		}
	}

	for (int ro = 0; ro <= diag; ro++)
	{
		for (int theta = 0; theta < 360; theta++)
		{
			int localmax = 0;
			for (int i = ro - 1; i <= ro + 1; i++)
			{
				if (i > 0 && i <= diag)
				{
					for (int j = theta - 1; j <= theta + 1; j++)
					{
						if (j > 0 && j < 360)
						{
							if (Hough.at<int>(j, i) > localmax)
							{
								localmax = Hough.at<int>(j, i);
							}
						}
					}
				}
			}

			if (Hough.at<int>(theta, ro) == localmax)
			{
				peak p;
				p.ro = ro;
				p.theta = theta;
				p.hval = localmax;
				peaks.push_back(p);
			}
		}
	}

	sort(peaks.begin(), peaks.end(), desc);

	for (int i = 0; i < 15; i++)
	{
		int x, y;
		x = 0;
		y = peaks[i].ro / sin(peaks[i].theta * CV_PI / 180);
		Point P1(x, y);
		x = cols;
		y = (peaks[i].ro - x * cos(peaks[i].theta  * CV_PI / 180)) / sin(peaks[i].theta  * CV_PI / 180);
		Point P2(x, y);
		line(colorImg, P1, P2, Scalar(0, 0, 255));
	}

	Hough.convertTo(houghImg, CV_8UC1, 255.f / maxHough);
	imshow("HoughCurves", houghImg);
	imshow("Hough", colorImg);
	waitKey();
}

void distanceT(int wHV, int wD)
{
	std::vector<Point> points;
	char fname[MAX_PATH], fname0[MAX_PATH];
	int rows, cols, n = 0, sum = 0;

	openFileDlg(fname0);
	openFileDlg(fname);

	Mat src = imread(fname0, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat dt = img.clone();

	rows = img.rows;
	cols = img.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (j - 1 >= 0 && i - 1 >= 0) {
				if (dt.at<uchar>(i - 1, j - 1) + wD < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i - 1, j - 1) + wD;
			}
			if (j - 1 >= 0) {
				if (dt.at<uchar>(i, j - 1) + wHV < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i, j - 1) + wHV;
			}
			if (i - 1 >= 0 && j + 1 < cols) {
				if (dt.at<uchar>(i - 1, j + 1) + wD < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i - 1, j + 1) + wD;
			}
			if (i - 1 >= 0) {
				if (dt.at<uchar>(i - 1, j) + wHV < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i - 1, j) + wHV;
			}
		}
	}

	for (int i = rows - 1; i >= 0; i--)
	{
		for (int j = cols - 1; j >= 0; j--)
		{
			if (i + 1 < rows && j + 1 < cols) {
				if (dt.at<uchar>(i + 1, j + 1) + wD < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i + 1, j + 1) + wD;
			}
			if (j + 1 < cols) {
				if (dt.at<uchar>(i, j + 1) + wHV < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i, j + 1) + wHV;
			}
			if (i + 1 < rows && j - 1 >= 0) {
				if (dt.at<uchar>(i + 1, j - 1) + wD < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i + 1, j - 1) + wD;
			}
			if (i + 1 < rows) {
				if (dt.at<uchar>(i + 1, j) + wHV < dt.at<uchar>(i, j))
					dt.at<uchar>(i, j) = dt.at<uchar>(i + 1, j) + wHV;
			}
		}
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				sum += dt.at<uchar>(i, j);
				n++;
			}
		}
	}

	printf("Mean = %f\n", (float)sum / n);

	imshow("DT", dt);
	waitKey();
}

void dataAnalysis(int fi1, int fi2, int fj1, int fj2)
{
	char folder[256] = "images_faces";
	char fname[256];
	FILE *f, *g;
	int const N = 361;
	int const p = 400;
	double means[361] = { 0 };
	double deviations[361] = { 0 };
	int features[p][N] = { 0 };
	Mat cov = Mat::zeros(N, N, CV_64FC1);
	Mat corr = Mat::zeros(N, N, CV_64FC1);
	Mat img = Mat::zeros(256, 256, CV_8UC1);

	for (int i = 1; i <= p; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i);
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		printf("Reading %s\n", fname);
		for (int j = 0; j < 19; j++)
		{
			for (int k = 0; k < 19; k++)
			{
				features[i - 1][j * 19 + k] = img.at<uchar>(j, k);
				means[j * 19 + k] += img.at<uchar>(j, k);
			}
		}
	}

	for (int i = 0; i < 361; i++)
	{
		means[i] *= 1.0 / p;
	}

	for (int i = 0; i < 361; i++)
	{
		double sum = 0.0;
		for (int k = 0; k < p; k++)
		{
			sum += (features[k][i] - means[i]) * (features[k][i] - means[i]);
		}
		deviations[i] = sqrt(sum * 1.0 / p);
	}

	f = fopen("cov.csv", "w");
	if (f == NULL)
	{
		printf("File not found\n");
		return;
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			double sum = 0.0;
			double devsum = 0.0;
			for (int k = 0; k < p; k++)
			{
				sum += (features[k][i] - means[i]) * (features[k][j] - means[j]);
			}
			cov.at<double>(i, j) = sum * 1.0 / p;
			fprintf(f, "%f, ", cov.at<double>(i, j));
		}
		fprintf(f, "\n");
	}
	fclose(f);

	g = fopen("corr.csv", "w");
	if (g == NULL)
	{
		printf("File not found\n");
		return;
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			corr.at<double>(i, j) = cov.at<double>(i, j) / (deviations[i] * deviations[j]);
			fprintf(g, "%f, ", corr.at<double>(i, j));
		}
		fprintf(g, "\n");
	}
	fclose(g);

	int fi = fi1 * 19 + fj1;
	int fj = fi2 * 19 + fj2;

	for (int k = 0; k < p; k++)
	{
		img.at<uchar>(features[k][fi], features[k][fj]) = 255;
	}

	imshow("Img", img);
	printf("Correlation: %f\n", corr.at<double>(fj, fi));

	waitKey();
}

void kmeans(int K)
{
	char fname[MAX_PATH];
	Point* means = new Point[K];
	std::vector<clustPoint> points;
	int rows, cols, n;

	srand(time(NULL));

	openFileDlg(fname);

	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	rows = src.rows;
	cols = src.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (src.at<uchar>(i, j) == 0) {
				clustPoint cp;
				cp.point = Point(j, i);
				cp.cluster = -1;
				cp.prevCluster = -1;
				points.push_back(cp);
			}
		}
	}

	n = points.size();

	std::cout << "n = " << n << std::endl;

	// Initialize clusters

	for (int i = 0; i < K; i++)
	{
		means[i] = points.at(rand() % n).point;
	}

	bool change = true;


	while (change) {

		// Assignment

		for (int i = 0; i < n; i++)
		{
			double dist = 10000000;
			for (int j = 0; j < K; j++)
			{
				double hdist = (means[j].x - points.at(i).point.x) * (means[j].x - points.at(i).point.x) +
					(means[j].y - points.at(i).point.y) * (means[j].y - points.at(i).point.y);
				if (hdist < dist)
				{
					points.at(i).prevCluster = points.at(i).cluster;
					points.at(i).cluster = j;
					dist = hdist;
					if (points.at(i).prevCluster == points.at(i).cluster) change = false;
				}
			}
		}

		// Update centers

		meanCalc* calcs = new meanCalc[K];
		for (int i = 0; i < K; i++) {
			calcs[i].n = 0;
			calcs[i].sumX = 0;
			calcs[i].sumY = 0;
		}

		for (int j = 0; j < n; j++)
		{
			calcs[points.at(j).cluster].sumX += points.at(j).point.x;
			calcs[points.at(j).cluster].sumY += points.at(j).point.y;
			calcs[points.at(j).cluster].n++;
		}

		for (int i = 0; i < K; i++) {
			means[i].x = calcs[i].sumX / n;
			means[i].y = calcs[i].sumY / n;
		}
		delete[] calcs;
	}

	Vec3b* colors = new Vec3b[K];

	for (int i = 0; i < K; i++) {
		colors[i] = { (uchar)rand(), (uchar)rand(), (uchar)rand() };
	}

	Mat img = Mat::zeros(rows, cols, CV_8UC3);
	Mat imgV = Mat::zeros(rows, cols, CV_8UC3);

	for (int i = 0; i < n; i++)
	{
		img.at<Vec3b>(points.at(i).point.x, points.at(i).point.y) = colors[points.at(i).cluster];
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double dist = 100000000;
			int ind = 0;
			for (int k = 0; k < K; k++) {
				double hdist = (means[k].x - i) * (means[k].x - i) +
					(means[k].y - j) * (means[k].y - j);
				if (hdist < dist)
				{
					dist = hdist;
					ind = k;
				}
			}
			imgV.at<Vec3b>(i, j) = colors[ind];
		}
	}

	imshow("Img", img);
	imshow("ImgV", imgV);
	waitKey();
}

void pca(int K) {
	char fname[MAX_PATH];
	int n, d;
	Mat Lambda, Q;
	double mad = 0;
	int rows, cols;

	openFileDlg(fname);
	FILE* f = fopen(fname, "r");

	fscanf(f, "%d %d", &n, &d);

	printf("n = %d\nd = %d\n", n, d);

	Mat X(n, d, CV_64FC1);
	Mat Xn(n, d, CV_64FC1);
	Mat Qk(d, K, CV_64FC1);
	Mat Xpca(n, K, CV_64FC1);
	Mat Xkt(n, d, CV_64FC1);

	double *means = new double[d];
	for (int i = 0; i < d; i++)
	{
		means[i] = 0;
	}

	double *mins = new double[K];
	double *maxs = new double[K];

	for (int i = 0; i < K; i++)
	{
		mins[i] = 99999999;
		maxs[i] = -99999999;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			fscanf(f, "%lf", &X.at<double>(i, j));
		}
	}

	for (int i = 0; i < d; i++) {
		double sum = 0;
		for (int j = 0; j < n; j++) {
			sum += X.at<double>(j, i);
		}
		means[i] = (double)sum / n;
		printf("means[%d] = %lf\n", i, means[i]);
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			Xn.at<double>(i, j) = X.at<double>(i, j) - means[j];
		}
	}

	Mat C = (Xn.t() * Xn) / (n - 1);
	eigen(C, Lambda, Q);
	Q = Q.t();

	printf("Eigenvalues\n");
	for (int i = 0; i < d; i++)
	{
		printf("%lf\n", Lambda.at<double>(i));
	}

	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < K; j++)
		{
			Qk.at<double>(i, j) = Q.at<double>(i, j);
		}
	}

	Xpca = Xn * Qk;
	Xkt = Xpca * Qk.t();

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			mad += abs(Xn.at<double>(i, j) - Xkt.at<double>(i, j));
		}
	}

	mad /= (n * d);

	printf("mad = %lf\n", mad);

	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (Xpca.at<double>(j, i) < mins[i])
			{
				mins[i] = Xpca.at<double>(j, i);
			}
			if (Xpca.at<double>(j, i) > maxs[i])
			{
				maxs[i] = Xpca.at<double>(j, i);
			}
		}
	}

	for (int i = 0; i < K; i++)
	{
		printf("min[%d] = %lf, max[%d] = %lf\n", i, mins[i], i, maxs[i]);
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < K; j++)
		{
			Xpca.at<double>(i, j) -= mins[j];
		}
	}

	rows = maxs[0] - mins[0] + 1;
	cols = maxs[1] - mins[1] + 1;

	int intmin = maxs[2] - mins[2] + 1;
	int multiplier = 255 / intmin;

	printf("rows: %d, cols: %d\n", rows, cols);

	Mat img = Mat::zeros(rows, cols, CV_8UC1);

	for (int i = 0; i < n; i++)
	{
		int intensity = Xpca.at<double>(i, 2) * intmin;
		img.at<uchar>(Xpca.at<double>(i, 0), Xpca.at<double>(i, 1)) = intensity;
	}

	imshow("PCA", img);

	waitKey();
}

void knn(int M, int K) {
	Mat X(1000, 3 * M, CV_32FC1);
	Mat y(1000, 1, CV_8UC1);
	Mat C(6, 6, CV_32SC1, Scalar(0));

	int c = 0, fileNr = 0, rowX = 0;
	char fname[MAX_PATH];

	for (int c = 0; c < 6; c++) {
		fileNr = 0;
		while (1)
		{
			sprintf(fname, "images_KNN/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

			int *hist = calcHist(img, M);

			for (int d = 0; d < M * 3; d++)
			{
				X.at<float>(rowX, d) = hist[d];
			}

			y.at<uchar>(rowX) = c;
			rowX++;
		}
	}



	// CONFUSED!!!
	for (int c = 0; c < 6; c++) {
		fileNr = 0;
		while (1)
		{
			std::vector<classDist> distances;
			sprintf(fname, "images_KNN/test/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

			int *hist = calcHist(img, M);
			float sum = 0;

			for (int i = 0; i < rowX; i++)
			{
				classDist clD;
				sum = 0;
				for (int d = 0; d < M * 3; d++)
				{
					float dist = distance(X.at<float>(i, d), hist[d]);
					sum += dist;
				}
				sum = sqrt(sum);
				clD.dist = sum;
				clD.ind = y.at<uchar>(i);
				distances.push_back(clD);
			}

			std::sort(distances.begin(), distances.end());

			int thiscls[6] = { 0 };

			for (int i = 0; i < K; i++)
			{
				thiscls[distances.at(i).ind] ++;
			}

			int max = 0;
			int ind = 0;

			for (int i = 0; i < 6; i++)
			{
				if (thiscls[i] > max)
				{
					max = thiscls[i];
					ind = i;
				}
			}
			C.at<int>(c, ind)++;
		}
		printf("c = %d\n", c);
	}

	int diag = 0, alll = 0;

	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if (i == j) diag += C.at<int>(i, j);
			alll += C.at<int>(i, j);
			printf("%d ", C.at<int>(i, j));
		}
		printf("\n");
	}

	printf("Accuracy: %f", (float)((float)diag / (float)alll));

	getchar();
	getchar();
	//imshow("KNN", img);
	waitKey();
}

void bayes()
{
	const int T = 128;
	const int d = 28 * 28;
	const int C = 10;
	int totals = 0;

	char fimgname[MAX_PATH];

	Mat priors(C, 1, CV_64FC1);
	Mat likelihood(C, d, CV_64FC1, Scalar(0));

	std::vector<double> probs;

	for (int c = 0; c < C; c++)
	{
		int index = 0;
		char fname[256];
		while (1)
		{
			sprintf(fname, "d:/Work/PRS/OpenCVApplication-VS2017_OCV340_basic/images_Bayes/train/%d/%06d.png", c, index);
			Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			if (img.cols == 0) break;
			for (int i = 0; i < img.cols; i++)
			{
				for (int j = 0; j < img.cols; j++)
				{
					if (img.at<uchar>(i, j) >= 128)
					{
						likelihood.at<double>(c, i * img.cols + j)++;
					}
				}
			}
			index++;
		}
		totals += index;
		for (int i = 0; i < d; i++)
		{
			likelihood.at<double>(c, i) /= index;
			if (likelihood.at<double>(c, i) == 0)
			{
				likelihood.at<double>(c, i) = 0.00001;
			}
		}
		probs.push_back(index);
		printf("Class %d trained\n", c);
	}

	do {

		std::vector<double> results;

		openFileDlg(fimgname);
		Mat img = imread(fimgname, CV_LOAD_IMAGE_GRAYSCALE);

		for (int c = 0; c < C; c++)
		{
			double sum = 0;
			for (int i = 0; i < img.cols; i++)
			{
				for (int j = 0; j < img.cols; j++)
				{
					if (img.at<uchar>(i, j) >= 128)
					{
						sum += log(likelihood.at<double>(c, i * img.cols + j));
					}
					else
					{
						sum += log(1 - likelihood.at<double>(c, i * img.cols + j));
					}
				}
			}
			sum += log(probs.at(c) / totals);
			printf("sum[%d] = %f\n", c, sum);
			results.push_back(sum);
		}

		for (int c = 0; c < 10; c++)
		{
			printf("For class %d the post-priori is %f\n", c, results.at(c));
		}

		int max = INT_MIN;
		int index = 0;

		for (int c = 0; c < C; c++)
		{
			if (results.at(c) > max) {
				max = results.at(c);
				index = c;
			}
		}

		printf("Your number is: %d\n", index);

		getchar();
	} while (1);
}

void onlinePercetron(float nu, std::vector<float> w0, float elimit, int max_iter)
{
	for (int i = 0; i < max_iter; i++)
	{
		float E = 0;
	}
}

std::vector<trainingPoint> read(int *rows, int *cols, Mat *image)
{
	char fname[MAX_PATH];
	std::vector<trainingPoint> points;


	openFileDlg(fname);
	FILE* f = fopen(fname, "r");

	Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
	*image = img;
	if (img.cols == 0) return points;
	*rows = img.rows;
	*cols = img.cols;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b color = img.at<Vec3b>(i, j);
			if (color[0] == 0 && color[1] == 0 && color[2] == 255) // Red
			{
				trainingPoint tp;
				tp.clasa = -1;
				tp.x = j;
				tp.y = i;
				points.push_back(tp);
			}
			else
			{
				if (color[0] == 255 && color[1] == 0 && color[2] == 0) // Blue
				{
					trainingPoint tp;
					tp.clasa = 1;
					tp.x = j;
					tp.y = i;
					points.push_back(tp);
				}
			}
		}
	}
	printf("Assigned blue and red points\n");

	return points;
}

void drawLine(float w[], Mat img)
{
	Point p1;
	p1.x = 0;
	p1.y = -w[0] / w[2];

	Point p2;
	p2.x = img.cols;
	p2.y = -(w[1] * img.cols + w[0]) / w[2];

	line(img, p1, p2, Vec3b(0, 255, 0));
	imshow("Image", img);
}

void perceptron(float nu, float Elimit, int max_iter)
{
	int rows, cols;
	Mat img;
	std::vector<trainingPoint> points = read(&rows, &cols, &img);
	printf("Rows: %d, cols: %d\n", rows, cols);
	int n = points.size();
	float w[3] = { 0.1, 0.1, 0.1 };

	for (int iter = 0; iter < max_iter; iter++)
	{
		float E = 0;
		for (int i = 0; i < n; i++)
		{
			float z = w[0] + w[1] * points.at(i).x + w[2] * points.at(i).y;
			if (z * points.at(i).clasa <= 0)
			{
				w[0] += nu * points.at(i).clasa;
				w[1] += nu * points.at(i).x * points.at(i).clasa;
				w[2] += nu * points.at(i).y * points.at(i).clasa;
				E++;
			}
		}
		E /= n;
		if (E < Elimit)
			break;
	}

	printf("w = [ %f, %f, %f ]\n", w[0], w[1], w[2]);

	drawLine(w, img);
	waitKey();
}

weaklearner findBestWeakLearner(std::vector<adaPoint> points, int rows, int cols, int n, int t)
{
	weaklearner best;
	double best_error = INT_MAX;
	int img_size;
	int z;
	for (int j = 0; j < 2; j++)
	{
		if (j == 0) img_size = cols;
		else img_size = rows;
		for (int threshold = 0; threshold < img_size; threshold++)
		{
			for (int label = 0; label < 2; label++)
			{
				double error = 0;
				for (int i = 0; i < n; i++)
				{
					if (points.at(i).X[j] < threshold)
					{
						if (label == 0) z = -1;
						else z = 1;
					}
					else
					{
						if (label == 0) z = 1;
						else z = -1;
					}
					if (z * points.at(i).y < 0)
					{
						error += points.at(i).w;
					}
				}
				if (error == 0 && t == 7)
				{
					printf("SUGI PULA");
				}
				if (error < best_error)
				{
					best_error = error;
					best.error = error;
					best.feature = j;
					best.threshold = threshold;
					best.class_label = z;
				}
				if (best_error == 0 && t == 7)
				{
					printf("SUGI PULA");
				}
			}
		}
	}
	if (best_error < 0)
		printf("FUCK\n");
	return best;
}

classifier algAdaBoost(std::vector<adaPoint> points, int n, int T, int rows, int cols)
{
	std::vector<double> alphas;
	std::vector<weaklearner> hs;
	classifier cls;
	for (int i = 0; i < n; i++)
	{
		points.at(i).w = 1.0 / n;
	}

	for (int t = 0; t < T; t++)
	{
		weaklearner best_weak = findBestWeakLearner(points, rows, cols, n, t);
		hs.push_back(best_weak);
		double alpha = 0.5 * log((1 - best_weak.error) / best_weak.error);
		alphas.push_back(alpha);
		double s = 0;
		for (int i = 0; i < n; i++)
		{
			points.at(i).w *= exp(-alpha * points.at(i).y * best_weak.classify(points.at(i).X));
			s += points.at(i).w;
		}
		for (int i = 0; i < n; i++)
		{
			points.at(i).w /= s;
		}
	}
	cls.T = T;
	cls.alphas = alphas;
	cls.hs = hs;
	return cls;
}

void adaboost(int T)
{
	char fname[MAX_PATH];
	std::vector<adaPoint> points;
	int n = 0;

	openFileDlg(fname);
	FILE* f = fopen(fname, "r");

	Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
	if (img.cols == 0) return;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b color = img.at<Vec3b>(i, j);
			if (color[0] == 0 && color[1] == 0 && color[2] == 255) // Red
			{
				adaPoint ap;
				ap.X[0] = i;
				ap.X[1] = j;
				ap.y = 1;
				n++;
				points.push_back(ap);
			}
			else
			{
				if (color[0] == 255 && color[1] == 0 && color[2] == 0) // Blue
				{
					adaPoint ap;
					ap.X[0] = i;
					ap.X[1] = j;
					ap.y = -1;
					n++;
					points.push_back(ap);
				}
			}
		}
	}
	printf("Assigned blue and red points\n");

	classifier cls = algAdaBoost(points, n, T, img.rows, img.cols);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b color = img.at<Vec3b>(i, j);
			if (!(color[0] == 0 && color[1] == 0 && color[2] == 255) // Red
				&& !(color[0] == 255 && color[1] == 0 && color[2] == 0)) // Blue
			{
				if (cls.classify(i, j) == 1) {
					Vec3b color = Vec3b(255, 255, 0);
					img.at<Vec3b>(i, j) = color;
				}
				else {
					Vec3b color = Vec3b(0, 255, 255);
					img.at<Vec3b>(i, j) = color;
				}
			}
		}
	}

	imshow("Result", img);
	waitKey();

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Least Mean Squares\n"); // Lab1
		printf(" 11 - RANSAC line\n"); // Lab2
		printf(" 12 - Hough Transform\n"); // Lab3
		printf(" 13 - Distance Transform and Pattern Matching\n"); // Lab4
		printf(" 14 - Statistical Data Analysis\n"); // Lab5
		printf(" 15 - K-means clustering\n"); // Lab6
		printf(" 16 - PCA\n"); // Lab7
		printf(" 17 - KNN classifier\n"); // Lab8
		printf(" 18 - Naive Bayes Classifier\n"); // Lab9
		printf(" 19 - Perceptron Algorithm\n"); // Lab10
		printf(" 20 - AdaBoost Algotirhm\n"); // Lab11
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			leastMeanSquares();
			break;
		case 11:
			ransac(10, 0.99, 0.8, 2);
			break;
		case 12:
			hough();
			break;
		case 13:
			distanceT(5, 7);
			break;
		case 14:
			dataAnalysis(5, 5, 4, 14);
			break;
		case 15:
			kmeans(3);
			break;
		case 16:
			pca(3);
			break;
		case 17:
			knn(8, 7);
			break;
		case 18:
			bayes();
			break;
		case 19:
			perceptron(0.0001, 0.00001, 100000);
			break;
		case 20:
			adaboost(6);
			break;
		default:
			break;
		}
	} while (op != 0);
	return 0;
}