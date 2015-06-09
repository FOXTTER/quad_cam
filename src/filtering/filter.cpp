#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
using namespace cv;
using namespace std;
char key;

IplImage* hsvFilter( IplImage* img) {

    //HSV image
    IplImage* imgHSV = cvCreateImage( cvGetSize(img), 8, 3);
    cvCvtColor( img, imgHSV, CV_BGR2HSV);

    //Create binary image with min/max value
    IplImage* imgThresh = cvCreateImage( cvGetSize( img ), 8, 1 );

    cvInRangeS (imgHSV, cvScalar( 104, 178, 70), cvScalar(130, 240, 124), imgThresh );

    //Clean up
    cvReleaseImage( &imgHSV );  
    return imgThresh;
}

int main()
{
        //Create window
    
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
    IplImage* frame = 0;
    while(1){ //Create infinte loop for live streaming
        
        frame = cvQueryFrame(capture);

        IplImage* imgThresh = hsvFilter(frame);
            
        
        /// Show your results
        namedWindow("Video", CV_WINDOW_AUTOSIZE);
        imshow( "Video", cvarrToMat(frame) );
        namedWindow("Filtered", CV_WINDOW_AUTOSIZE);
        imshow( "Filtered", cvarrToMat(imgThresh));
        
        key = cvWaitKey(10);     //Capture Keyboard stroke
        if (char(key) == 27){
            break;      //If you hit ESC key loop will break.
        }
    }
    cvReleaseCapture(&capture); //Release capture.
    cvDestroyWindow("Camera_Output"); //Destroy Window
    return 0;
}