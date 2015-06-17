#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Empty.h>
#include <ardrone_autonomy/Navdata.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <geometry_msgs/Twist.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <string>
#include <math.h>

using namespace cv;
using namespace std;
#define IMAGE_PATH "/ardrone/image_raw" //Quadcopter
//#define IMAGE_PATH "/image_raw" //Webcam
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
int button_val = 0;
bool freeze = false;

/// Matrices to store images
Mat src1;
Mat dst;
image_transport::Subscriber image_sub_;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  
public:

  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe(IMAGE_PATH, 1, 
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
  
    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);w
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Draw an example circle on the video stream
    //if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    //  cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
    
    if (!freeze)
    {
        src1 = cv_ptr->image;
    }

    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};
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
static void on_mouse( int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
        return;
    freeze = !freeze;
    ROS_INFO("Button: %d",freeze);
    
}

int main( int argc, char** argv )
{
 ros::init(argc, argv, "test");
 ImageConverter ic;
 ros::Rate r(10); // 10 hz
 ros::spinOnce();
 /// Read image ( same size, same type )
 //it_(nh_);
 //CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
 //src1 = cvarrToMat(cvQueryFrame(capture));
 //src1 = cvarrToMat(cvQueryFrame(capture));
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
 setMouseCallback("Linear Blend", on_mouse,0);
 double time_start=(double)ros::Time::now().toSec();
 while (ros::ok() && ((double)ros::Time::now().toSec()< time_start+5)){

 }
 /// Show some stuff
 while(ros::ok()){
 	//src1 = cvarrToMat(cvQueryFrame(capture));
 	ros::spinOnce();
 	IplImage* foo = new IplImage(src1);
 	imshow("org",src1);
 	foo = hsvFilter(foo);
 	dst = cvarrToMat(foo);
 	dilation(dst,dil_slider);
 	erosion(dst,ero_slider);
    imshow("Linear Blend",dst);
    r.sleep();
 }
 return 0;
}