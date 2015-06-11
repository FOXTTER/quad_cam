#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
String window_name = "Edge Map";

using namespace std;
char key;

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );
    
    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold, kernel_size );
    
    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
    
    src.copyTo( dst, detected_edges);
    imshow( window_name, dst );
}


/** @function main */
int main()
{
    cvNamedWindow("Camera_Output", 1);    //Create window
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
    while(1){ //Create infinte loop for live streaming
        
        IplImage* frame = cvQueryFrame(capture); //Create image frames from capture
        
        /// Read the image
        src = cv::cvarrToMat(frame);
        
        if( !src.data )
        { return -1; }
        
    
        /// Create a matrix of the same type and size as src (for dst)
        dst.create( src.size(), src.type() );
    
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
    
        /// Create a window
        namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    
        /// Create a Trackbar for user to enter threshold
        createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
    
        /// Show the image
        CannyThreshold(0, 0);
    
        key = cvWaitKey(10);     //Capture Keyboard stroke
        
        if (char(key) == 27){
                break;      //If you hit ESC key loop will break.
            }
        }
    
    cvReleaseCapture(&capture); //Release capture.
    cvDestroyWindow("Camera_Output"); //Destroy Window
    
    return 0;
}