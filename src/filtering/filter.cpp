#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
using namespace cv;
using namespace std;
char key;
enum color_type {
    red,
    blue
};

IplImage* hsvFilter( IplImage* img, color_type color) {

    //HSV image
    IplImage* imgHSV = cvCreateImage( cvGetSize(img), 8, 3);
    cvCvtColor( img, imgHSV, CV_BGR2HSV);



    //Create binary image with min/max value
    IplImage* imgThresh = cvCreateImage( cvGetSize( img ), 8, 1 );

    switch(color) {
        case red:
            cvInRangeS (imgHSV, cvScalar( 0 , 130, 0), cvScalar(30, 190, 255), imgThresh );
            break;
        case blue:
            cvInRangeS (imgHSV, cvScalar( 50 , 80, 0), cvScalar(150, 220, 255), imgThresh );
            break;
        default:
            cvInRangeS (imgHSV, cvScalar( 50 , 80, 0), cvScalar(150, 220, 255), imgThresh );
    }

    //Clean up
    cvReleaseImage( &imgHSV );  
    return imgThresh;
}


void erosion(Mat src){
    int erosion_type = MORPH_ELLIPSE;
    int erosion_size = 20;
    Mat element = getStructuringElement( erosion_type,
                                   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                   Point( erosion_size, erosion_size ) );
    erode( src, src, element );
}

void dilation(Mat src){
    int dilation_type = MORPH_ELLIPSE;
    int dilation_size = 20;
    Mat dil_element = getStructuringElement( dilation_type,
                                    Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                    Point( dilation_size, dilation_size ) );
    dilate( src, src, dil_element );

}


int main()
{
        //Create window
    
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
    IplImage* frame = 0;
    while(1){ //Create infinte loop for live streaming
        
        frame = cvQueryFrame(capture);
        IplImage* imgThresh = hsvFilter(frame, blue);
        Mat test = cvarrToMat(imgThresh);
        erosion(test);
        dilation(test);
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