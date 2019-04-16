///*
//文件名：opencv_test.cpp
//作者：王虎强
//日期：2019/2/25
//功能：调用电脑摄像头读取视频流，对图像进行运行目标提取，将目标中背景中提取出。
//*/
#include "pch.h"
//#define PROGRAMMER_2
#define PROGRAMMER_1
//#define PROGRAMMER_3

#ifdef PROGRAMMER_1
	
	//OpenCV Headers
	#include<opencv/cv.h>
	#include <opencv2/highgui/highgui.hpp>
	#include <opencv2/opencv.hpp>
	//Input-Output
	#include <stdio.h>
	#include <iostream>
	#include <windows.h>
	//Definitions
	#define h 240
	#define w 320
	//NameSpaces
	using namespace cv;
	using namespace std;
	
	//Global variables
	int fcount = 0;//Counts the number of frames		//帧数
	IplImage* FC_FindBiggestContours(IplImage* src);
	CvRect R;
	//Main Function
	int main()
	{
		VideoCapture cap(0);			//调用摄像头
		
		Mat frame_cap;						
		IplImage image;			
		//Structure to get feed from CAM
		
		//Windows
		cvNamedWindow("Live", CV_WINDOW_AUTOSIZE);
		cvNamedWindow("Threshy", CV_WINDOW_AUTOSIZE);
		cvNamedWindow("cnt", CV_WINDOW_AUTOSIZE);
		
		//Image Variables		//建立图像 大小的320*240
		IplImage *frame = cvCreateImage(cvSize(w, h), 8, 3); //Original Image
		IplImage *hsvframe = cvCreateImage(cvSize(w, h), 8, 3);//Image in HSV color space	//图像的HSV颜色空间
		IplImage *threshy = cvCreateImage(cvSize(w, h), 8, 1); //Threshold image of yellow color
		
		//Variables for trackbars
		int h1 = 0; int s1 = 30; int v1 = 80;
		int h2 = 20; int s2 = 150; int v2 = 255;
		//H : 色调   S:饱和度  V:亮度
		//Creating the trackbars
		cvCreateTrackbar("H1", "cnt", &h1, 255, 0);
		cvCreateTrackbar("H2", "cnt", &h2, 255, 0);
		cvCreateTrackbar("S1", "cnt", &s1, 255, 0);
		cvCreateTrackbar("S2", "cnt", &s2, 255, 0);
		cvCreateTrackbar("V1", "cnt", &v1, 255, 0);
		cvCreateTrackbar("V2", "cnt", &v2, 255, 0);
		
		//Infinate Loop
		while (1)
		{
			cap >> frame_cap;
			image = IplImage(frame_cap);
			//Getting the current frame		
			IplImage *fram = &image;
			//If failed to get break the loop
			if (!fram)
				break;
	
			//1.PREPROCESSING OF FRAME
			//Resizing the capture
			cvResize(fram, frame, CV_INTER_LINEAR);
			//Flipping the frame  //旋转图像，绕x轴，这样可以让坐标原点从左上角移动到左下角
			cvFlip(frame, frame, 0);
			//Changing the color space
			cvCvtColor(frame, hsvframe, CV_BGR2HSV);
			//Thresholding the frame for yellow	//
			cvInRangeS(hsvframe, cvScalar(h1, s1, v1), cvScalar(h2, s2, v2), threshy);
			//Filtering the frame  //中值滤波  7*7滤波器
			cvSmooth(threshy, threshy, CV_MEDIAN, 7, 7);
			//Finding largest contour	
			threshy = FC_FindBiggestContours(threshy);
			//Getting the screen information
			int screenx = GetSystemMetrics(SM_CXSCREEN);
			int screeny = GetSystemMetrics(SM_CYSCREEN);
			//Calculating the moments
			CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
			cvMoments(threshy, moments, 1);
			// The actual moment values
			double moment10 = cvGetSpatialMoment(moments, 1, 0);
			double moment01 = cvGetSpatialMoment(moments, 0, 1);
			double area = cvGetCentralMoment(moments, 0, 0);
	
	
			//Getting the current frame
			IplImage *fram2 = &image;
			//If failed to get break the loop
			if (!fram)
				break;
	
			//1.PREPROCESSING OF FRAME
			//Resizing the capture
			cvResize(fram2, frame, CV_INTER_LINEAR);
			//Flipping the frame
			cvFlip(frame, frame, 1);
			//Changing the color space
			cvCvtColor(frame, hsvframe, CV_BGR2HSV);
			//Thresholding the frame for yellow
			cvInRangeS(hsvframe, cvScalar(h1, s1, v1), cvScalar(h2, s2, v2), threshy);
			//Filtering the frame
			cvSmooth(threshy, threshy, CV_MEDIAN, 7, 7);
			//Biggest Contour
			threshy = FC_FindBiggestContours(threshy);
			//Getting the screen information
			int screenx2 = GetSystemMetrics(SM_CXSCREEN);
			int screeny2 = GetSystemMetrics(SM_CYSCREEN);
			//Calculating the moments
			CvMoments *moments2 = (CvMoments*)malloc(sizeof(CvMoments));
			cvMoments(threshy, moments2, 1);
			// The actual moment values
			double moment20 = cvGetSpatialMoment(moments2, 1, 0);
			double moment02 = cvGetSpatialMoment(moments2, 0, 1);
			double area2 = cvGetCentralMoment(moments2, 0, 0);
			//Position Variables
			int x1;
			int y1;
			int x2;
			int y2;
			//Calculating the current position
			x1 = moment10 / area;
			y1 = moment01 / area;
			x2 = moment20 / area2;
			y2 = moment20 / area2;
			//Fitting to the screen
			int x = (int)(x1*screenx / w);
			int y = (int)(y1*screeny / h);
	
			/*if(x>>0 && y>>0 )
			{
			cvLine(frame, cvPoint(x1,y1), cvPoint(x1,y1), cvScalar(0,25,255),5);
			cout<<"X:"<<x<<"\tY:"<<y<<endl;
			}*/
	
			//Moving the mouse pointer
			//SetCursorPos(x,y);
			if (x2 - x1 > 80)
			{
	
				keybd_event(VK_RIGHT, 0, KEYEVENTF_EXTENDEDKEY | 0, 0);
				keybd_event(VK_RIGHT, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
				keybd_event(VK_PRIOR, 0, KEYEVENTF_EXTENDEDKEY | 0, 0);
				keybd_event(VK_PRIOR, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
				cout << "RIGHT" << endl;
			}
	
			if (x1 - x2 > 80)
			{
				keybd_event(VK_LEFT, 0, KEYEVENTF_EXTENDEDKEY | 0, 0);
				keybd_event(VK_LEFT, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
				keybd_event(VK_NEXT, 0, KEYEVENTF_EXTENDEDKEY | 0, 0);
				keybd_event(VK_NEXT, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0);
				cout << "LEFT" << endl;
			}
	
	
	
			//Showing the images
			cvShowImage("Live", frame);
			cvShowImage("Threshy", threshy);
			//Escape Sequence
			char c = cvWaitKey(50);
			if (c == 27)
				break;
	
		}
		//Cleanup
		//cvReleaseCapture(&capture);
		cvDestroyAllWindows();
	
	}
	
	IplImage* FC_FindBiggestContours(IplImage* src)
	{
		IplImage temp = *src;
		IplImage *src_img = cvCreateImage(cvSize(temp.width, temp.height), IPL_DEPTH_32S, 1);
		IplImage *dest = cvCreateImage(cvSize(temp.width, temp.height), IPL_DEPTH_8U, 1);
		CvArr* _mask = &temp;
		int poly1Hull0 = 1;
		CvPoint offset;
		offset.x = 0;
		offset.y = 0;
		CvMat mstub, *mask = cvGetMat(_mask, &mstub);
		CvMemStorage* tempStorage = cvCreateMemStorage();
		CvSeq *contours, *c;
		int nContours = 0;
		double largest_length = 0, len = 0;
		CvContourScanner scanner;
		// clean up raw mask
		cvMorphologyEx(mask, mask, 0, 0, CV_MOP_OPEN, 1);
		cvMorphologyEx(mask, mask, 0, 0, CV_MOP_CLOSE, 1);
		// find contours around only bigger regions
		scanner = cvStartFindContours(mask, tempStorage,
			sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, offset);
		while ((c = cvFindNextContour(scanner)) != 0)
		{
			len = cvContourPerimeter(c);
			if (len > largest_length)
			{
				largest_length = len;
			}
		}
		contours = cvEndFindContours(&scanner);
		scanner = cvStartFindContours(mask, tempStorage,
			sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, offset);
		while ((c = cvFindNextContour(scanner)) != 0)
		{
			len = cvContourPerimeter(c);
			double q = largest_length;
			if (len < q) //Get rid of blob if it's perimeter is too small
				cvSubstituteContour(scanner, 0);
			else  //Smooth it's edges if it's large enough
			{
				CvSeq* newC;
				if (poly1Hull0) //Polygonal approximation of the segmentation
					newC = cvApproxPoly(c, sizeof(CvContour), tempStorage, CV_POLY_APPROX_DP, 2, 0);
				else //Convex Hull of the segmentation
					newC = cvConvexHull2(c, tempStorage, CV_CLOCKWISE, 1);
				cvSubstituteContour(scanner, newC);
				nContours++;
				R = cvBoundingRect(c, 0);
			}
		}
		contours = cvEndFindContours(&scanner);
		// paint the found regions back into the image
		cvZero(src_img);
		cvZero(_mask);
		for (c = contours; c != 0; c = c->h_next)
		{
			cvDrawContours(src_img, c, cvScalarAll(1), cvScalarAll(1), -1, -1, 8,
				cvPoint(-offset.x, -offset.y));
		}
		cvReleaseMemStorage(&tempStorage);
		// convert to 8 bit IplImage
		for (int i = 0; i < src_img->height; i++)
			for (int j = 0; j < src_img->width; j++)
			{
				int idx = CV_IMAGE_ELEM(src_img, int, i, j);  //get reference to pixel at (col,row),
				uchar* dst = &CV_IMAGE_ELEM(dest, uchar, i, j);                          //for multi-channel images (col) should be multiplied by number of channels
				if (idx == -1 || idx == 1)
					*dst = (uchar)255;
				else if (idx <= 0 || idx > 1)
					*dst = (uchar)0; // should not get here
				else {
					*dst = (uchar)0;
				}
			}
		cvReleaseImage(&src_img);
	
		return dest;
	}
#endif

#ifdef PROGRAMMER_2

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


using namespace cv;
using namespace std;

void filter_image(Mat& frame);	//平滑滤波函数

int main()	//main主函数
{
	VideoCapture cap(0);	//打开摄像头
	Mat frame;				//视频流中读取的每一帧图像放入这个变量

	Ptr<BackgroundSubtractorMOG2>p_mog2;	//opencv中的至真摸板类Ptr指向混合高斯背景消除法类

	namedWindow("1");	//设置一个窗口
	namedWindow("2");	//设置一个窗口

	cap >> frame;		//将摄像头中读取的图像放入frame变量中
	Mat Imask;
	Mat Imask1;
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	p_mog2 = createBackgroundSubtractorMOG2(100, 23, true);//创建一个自适应高斯背景消除 100是训练背景的帧数，23是阈值，true为是检测背影

	while (true)	//循环读取视频流
	{
		cap >> frame;	//从摄像头读取的视频流每一帧图像放入

		p_mog2->apply(frame, Imask);	//计算前景掩码
		Canny(Imask, Imask, 2, 7);		//边缘检测
		dilate(Imask, Imask1, element);	//先腐蚀
		erode(Imask1, Imask, element);	//再膨胀

		//adaptiveThreshold(Imask,Imask1,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,3,0);
		imshow("2", frame);		//显示原图像
		imshow("1", Imask);	//显示处理过后的图像

		if (waitKey(30) == 27)	//延时30ms，检测是否按下esc键，按下终止循环，退出程序
		{
			break;
		}
	}
	return 0;
}

void filter_image(Mat& frame)
{
	blur(frame, frame, Size(3, 3));		//这里进行的是均值滤波	blur是opencv中自带的函数，滤波器为3*3的矩形
	equalizeHist(frame, frame);
}
#endif



#ifdef PROGRAMMER_3


#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main()
{
	VideoCapture video(0);
	if (!video.isOpened()) {
		cout << "could not load video file ..." << endl;
		return -1;
	}

	Mat frame, bgMask_MOG2, bgMask_KNN, background;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

	//namedWindow("highway_test", WINDOW_AUTOSIZE);
	//namedWindow("background_mask_by_MOG2", WINDOW_AUTOSIZE);
	namedWindow("background_by_KNN", WINDOW_AUTOSIZE);
	namedWindow("background_mask_by_KNN", WINDOW_AUTOSIZE);

	Ptr<BackgroundSubtractor> ptrMOG2 = createBackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractor> ptrKNN = createBackgroundSubtractorKNN(50, 50, true);

	while (video.read(frame))
	{
		imshow("highway_test", frame);

		//ptrMOG2->apply(frame, bgMask_MOG2);
		//morphologyEx(bgMask_MOG2, bgMask_MOG2, MORPH_OPEN, kernel);
		//threshold(bgMask_MOG2, bgMask_MOG2, 244, 255, THRESH_BINARY);
		//Canny(bgMask_MOG2, bgMask_MOG2,0,0);
		//imshow("background_mask_by_MOG2", bgMask_MOG2);

		ptrKNN->apply(frame, bgMask_KNN);
		blur(bgMask_KNN, bgMask_KNN, Size(3, 3));
		threshold(bgMask_KNN, bgMask_KNN, 250, 255, THRESH_BINARY);
		morphologyEx(bgMask_KNN, bgMask_KNN, MORPH_OPEN, kernel);
		imshow("background_mask_by_KNN", bgMask_KNN);
		//ptrKNN->getBackgroundImage(background);
		//imshow("background_by_KNN", background);

		char c = waitKey(50);
		if (c == 27)
		{
			break;
		}
	}
	waitKey(0);
	video.release();
	return 0;
}

#endif // PROGRAMMER_3

