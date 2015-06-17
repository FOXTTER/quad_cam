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
//#define IMAGE_PATH "/ardrone/image_raw" //Quadcopter
#define IMAGE_PATH "/image_raw" //Webcam
/// Global Variables
const int dilation_max = 50;
const int erosion_max = 50;
int ero_slider = 0;
int dil_slider = 0;
int button_val = 0;
int can_low = 90;
int can_high = 100;
int can_max = 100;
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

void erosion(Mat src,int erosion_size){
    int erosion_type = MORPH_ELLIPSE;
    Mat element = getStructuringElement( erosion_type,
                                   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                   Point( erosion_size, erosion_size ) );
    erode( src, src, element );
}

void dilation(Mat src, int dilation_size){
    int dilation_type = MORPH_ELLIPSE;
    Mat dil_element = getStructuringElement( dilation_type,
                                    Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                    Point( dilation_size, dilation_size ) );
    dilate( src, src, dil_element );

}
void on_trackbar1( int, void* )
{
  
}
void on_trackbar2( int, void* )
{
  
}
void on_trackbar3( int, void* )
{
  
}

static void on_mouse( int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
        return;
    freeze = !freeze;
    ROS_INFO("Button = (%d,%d)",x,y);
    
}


void drawCircles(Mat src){
  Mat src_gray = src;
  cvtColor( src, src_gray, CV_BGR2GRAY );
  
  /// Reduce the noise so we avoid false circle detection
  GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
  
  std::vector<Vec3f> circles;
  
  /// Apply the Hough Transform to find the circles
  //HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/100, 20, 140, 0, 0 );
  
  // std
  HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 10, 100, 0, 0 );
  
  /// Draw the circles detected
  int sum_x = 0;
  int sum_y = 0;
  int numbOfcircles = 0;
  cvtColor(src_gray,src_gray,COLOR_GRAY2RGB);
  
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );      
  }
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
 //createTrackbar( "Erosion: ", "Linear Blend", &ero_slider, erosion_max, on_trackbar1 );
 createTrackbar( "Dilation ", "Linear Blend", &dil_slider, dilation_max, on_trackbar2 );
 createTrackbar( "Canny low: ", "Linear Blend", &can_low, can_max, on_trackbar3 );
 setMouseCallback("Linear Blend", on_mouse,0);
 double time_start=(double)ros::Time::now().toSec();
 while (ros::ok() && ((double)ros::Time::now().toSec()< time_start+2)){

 }
 /// Show some stuff
 while(ros::ok()){
 	//src1 = cvarrToMat(cvQueryFrame(capture));
 	ros::spinOnce();
 	//IplImage* foo = new IplImage(src1);
 	imshow("org",src1);
  dst = src1;
  Canny(src1, dst, can_low, can_low*3);
  //dilation(dst,dil_slider);
  //bitwise_not( dst, dst);
  //cvtColor(dst,dst,COLOR_GRAY2BGR);
  //drawCircles(dst);
 	//foo = hsvFilter(foo);
 	//dst = cvarrToMat(foo);
 	//dilation(dst,dil_slider);
 	//erosion(dst,ero_slider);
  imshow("Linear Blend",dst);
  r.sleep();
 }
 return 0;
}