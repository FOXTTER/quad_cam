#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
using namespace cv;
using namespace std;
char key;
int main()
{
    cvNamedWindow("Camera_Output", 1);    //Create window
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
    while(1){ //Create infinte loop for live streaming
        
        IplImage* frame = cvQueryFrame(capture); //Create image frames from capture
        
        Mat src, src_gray;
        
        
        /// Read the image
        src = cv::cvarrToMat(frame);
        
        if( !src.data )
        { return -1; }
        
        
        /// Convert it to gray
        cvtColor( src, src_gray, CV_BGR2GRAY );
        
        /// Reduce the noise so we avoid false circle detection
        GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
        
        std::vector<Vec3f> circles;
        
        /// Apply the Hough Transform to find the circles
        HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/100, 50, 140, 0, 0 );
        
        // std
        //HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
        
        /// Draw the circles detected
        int sum_x = 0;
        int sum_y = 0;
        int numbOfcircles = 0;
        
        for( size_t i = 0; i < circles.size(); i++ )
        {
            //Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            
            for(int j = 0; j < circles.size();j++) {
                if (j==i){
                    continue;
                }
                int x = circles[i][0];
                int dx = circles[j][0];
                int y = circles[i][1];
                int dy = circles[j][1];
                
                if (sqrt(abs(x-dx)*abs(x-dx) + abs(y-dy)*abs(y-dy)) < 15) {
                    if(abs(cvRound(circles[i][2]-circles[j][2])) > 5 ){
                        numbOfcircles++;
                        sum_x = circles[i][0];
                        sum_y = circles[i][1];
                    }
                }
            }
            // draw circles (average)
            if (numbOfcircles > 0) {
                // circle center
                Point center(sum_x/numbOfcircles, sum_y/numbOfcircles);
                circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
                // circle outline
                // circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
            }
            
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