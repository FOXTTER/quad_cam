#include <cv.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global Variables
const int hue_slider_max = 255;
const int sat_slider_max = 255;
const int val_slider_max = 255;
const int dilation_max = 100;
const int erosion_max = 100;
int hue_max_slider = 180;
int hue_min_slider = 0;
int sat_max_slider = 255;
int sat_min_slider = 0;
int val_max_slider = 255;
int val_min_slider = 0;
int ero_slider = 0;
int dil_slider = 0;

/// Matrices to store images
Mat src1;
Mat src2;
Mat dst;

IplImage* hsvFilter( IplImage* img) {

    //HSV image
    IplImage* imgHSV = cvCreateImage( cvGetSize(img), 8, 3);
    cvCvtColor( img, imgHSV, CV_BGR2HSV);



    //Create binary image with min/max value
    IplImage* imgThresh = cvCreateImage( cvGetSize( img ), 8, 1 );

    cvInRangeS (imgHSV, cvScalar( hue_min_slider , sat_min_slider, val_min_slider), cvScalar(hue_max_slider, sat_max_slider, val_max_slider), imgThresh );
    if(hue_max_slider > 180){
    	IplImage* temp = cvCreateImage( cvGetSize( img ), 8, 1 );
    	cvInRangeS (imgHSV, cvScalar( 0 , sat_min_slider, val_min_slider), cvScalar(hue_max_slider-180, sat_max_slider, val_max_slider), temp );
    	cvAdd(imgThresh,temp,imgThresh);
    	cvReleaseImage( &temp );
    }
    //Clean up
    cvReleaseImage( &imgHSV );  
    return imgThresh;
}

void erosion(Mat src,int erosion_size){
    int erosion_type = MORPH_RECT;
    Mat element = getStructuringElement( erosion_type,
                                   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                   Point( erosion_size, erosion_size ) );
    erode( src, src, element );
}

void dilation(Mat src, int dilation_size){
    int dilation_type = MORPH_RECT;
    Mat dil_element = getStructuringElement( dilation_type,
                                    Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                    Point( dilation_size, dilation_size ) );
    dilate( src, src, dil_element );

}

/**
 * @function on_trackbar
 * @brief Callback for trackbar
 */
void on_trackbar1( int, void* )
{
	
}
void on_trackbar2( int, void* )
{
	
}
void on_trackbar3( int, void* )
{
	
}
void on_trackbar4( int, void* )
{
	
}
void on_trackbar5( int, void* )
{
	
}
void on_trackbar6( int, void* )
{
	
}
void on_trackbar7( int, void* )
{
	
}
void on_trackbar8( int, void* )
{
	
}

int main( int argc, char** argv )
{
 /// Read image ( same size, same type )
 CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
 src1 = cvarrToMat(cvQueryFrame(capture));

 /// Create Windows
 namedWindow("Linear Blend", 1);

 createTrackbar( "Hue min: ", "Linear Blend", &hue_min_slider, hue_slider_max, on_trackbar1 );
 createTrackbar( "Hue max: ", "Linear Blend", &hue_max_slider, hue_slider_max, on_trackbar2 );
 createTrackbar( "Sat min: ", "Linear Blend", &sat_min_slider, sat_slider_max, on_trackbar3 );
 createTrackbar( "Sat max: ", "Linear Blend", &sat_max_slider, sat_slider_max, on_trackbar4 );
 createTrackbar( "Val min: ", "Linear Blend", &val_min_slider, val_slider_max, on_trackbar5 );
 createTrackbar( "Val max: ", "Linear Blend", &val_max_slider, val_slider_max, on_trackbar6 );
 createTrackbar( "Erosion: ", "Linear Blend", &ero_slider, erosion_max, on_trackbar7 );
 createTrackbar( "Dilation ", "Linear Blend", &dil_slider, dilation_max, on_trackbar8 );

 /// Show some stuff
 while(1){
 	src1 = cvarrToMat(cvQueryFrame(capture));
 	IplImage* foo = new IplImage(src1);
 	imshowc("org",src1);
 	foo = hsvFilter(foo);
 	dst = cvarrToMat(foo);
 	dilation(dst,dil_slider);
 	erosion(dst,ero_slider);
 	
 	imshow( "Linear Blend", dst );
 	waitKey(10);
 }
 /// Wait until user press some key
 waitKey(0);
 return 0;
}