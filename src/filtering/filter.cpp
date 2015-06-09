#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
using namespace cv;
using namespace std;
char key;

IplImage* HSVfilter( IplImage* img) {

    //HSV image
    IplImage* imgHSV = cvCreateImage( cvGetSize(img), 8, 3);
    cvCvtColor( img, imgHSV, CV_BGR2HSV);

    //Create binary image with min/max value
    IplImage* imgThresh = cvCreateImage( cvGetSize( img ), 8, 1 );

    cvInRangeS (imgHSV, cvScalar( 104, 178, 70), cvScalar(130, 240, 124), imgThresh );
}

int main()
{
    cvNamedWindow("Video", 1);    //Create window
    cvNamedWindow("Filtered")
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
    IplImage* frame = 0;
    while(1){ //Create infinte loop for live streaming
        
        if(!(frame = cvQueryFrame(capture))) //Create image frames from capture
            break;

        IplImage* imgThresh = GetThresholdedImageHSV(frame);
            
        }
        
        /// Show your results
        namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
        imshow( "Hough Circle Transform Demo", src );
        
        key = cvWaitKey(10);     //Capture Keyboard stroke
        if (char(key) == 27){
            break;      //If you hit ESC key loop will break.
        }
    }
    cvReleaseCapture(&capture); //Release capture.
    cvDestroyWindow("Camera_Output"); //Destroy Window
    return 0;
}