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

    cvInRangeS (imgHSV, cvScalar( 50 , 80, 0), cvScalar(150, 220, 255), imgThresh );

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
        int erosion_type = MORPH_ELLIPSE;

        int erosion_size = 20;
        IplImage* imgThresh = hsvFilter(frame);
        Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
        Mat test = cvarrToMat(imgThresh);
        Mat prev = test.clone();
        erode( test, test, element );
        int dilation_type = MORPH_ELLIPSE;
        int dilation_size = 20;
        Mat dil_element = getStructuringElement( dilation_type,
                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        Point( dilation_size, dilation_size ) );
        dilate( test, test, dil_element );
        
        /// Show your results
        namedWindow("original", CV_WINDOW_AUTOSIZE);
        imshow( "original", cvarrToMat(frame) );
        namedWindow("Video", CV_WINDOW_AUTOSIZE);
        imshow( "Video", prev );
        namedWindow("Filtered", CV_WINDOW_AUTOSIZE);
        imshow( "Filtered", test);
        
        key = cvWaitKey(10);     //Capture Keyboard stroke
        if (char(key) == 27){
            break;      //If you hit ESC key loop will break.
        }
    }
    cvReleaseCapture(&capture); //Release capture.
    cvDestroyWindow("Camera_Output"); //Destroy Window
    return 0;
}