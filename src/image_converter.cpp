#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Empty.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <geometry_msgs/Twist.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <string>
using namespace cv;
using namespace std;
//#define IMAGE_PATH "/ardrone/image_raw" //Quadcopter
#define IMAGE_PATH "/image_raw" //Webcam
//Comment to test github
static const std::string OPENCV_WINDOW = "Image window";
Point2f measured(0,0);
geometry_msgs::Twist twist_msg;
geometry_msgs::Twist twist_msg_hover;
geometry_msgs::Twist twist_msg_pshover;
std_msgs::Empty emp_msg;


class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  
public:

  void drawText(string text, Mat* img)
  {
  int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 2;
  int thickness = 3;

  //Mat img(600, 800, CV_8UC3, Scalar::all(0));

  int baseline=0;
  Size textSize = getTextSize(text, fontFace,
                              fontScale, thickness, &baseline);
  baseline += thickness;

  // center the text
  Point textOrg((img->cols - textSize.width)/2,
                (img->rows + textSize.height)/2);

  // draw the box
  rectangle(*img, textOrg + Point(0, baseline),
            textOrg + Point(textSize.width, -textSize.height),
            Scalar(0,0,255));
  // ... and the baseline first
  line(*img, textOrg + Point(0, thickness),
       textOrg + Point(textSize.width, thickness),
       Scalar(0, 0, 255));

  // then put the text itself
  putText(*img, text, textOrg, fontFace, fontScale,
          Scalar::all(255), thickness, 8);
  }

  //Function stub for the computer vision
  cv::Point2f getCoordinates(cv::Mat img){
  	return Point2f(0,0);
  }

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

  Point circleDetection(Mat src){
  	Mat src_gray;
 		// Convert it to gray
  	cvtColor( src, src_gray, CV_BGR2GRAY );
  	
  	/// Reduce the noise so we avoid false circle detection
  	GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
  	
  	std::vector<Vec3f> circles;
  	
  	/// Apply the Hough Transform to find the circles
  	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/100, 20, 140, 0, 0 );
  	
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
  	                sum_x += circles[i][0];
  	                sum_y += circles[i][1];
  	            }
  	        }
  	    }
  	    // draw circles (average)
  	    if (numbOfcircles > 0) {
  	        // circle center
  	        Point center(sum_x/numbOfcircles, sum_y/numbOfcircles);
  	        //circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
  	        // circle outline
  	        // circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
  	        return center;
  	    }
  	    
  	}		
  }

  Point blobDetection(Mat src){
  	Mat src_gray;
  	//TEST AF BLOB DETECTION
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

   // Set up the detector with default parameters.
    cv::SimpleBlobDetector::Params params;
    params.filterByColor = true;
    params.blobColor = 255;
    SimpleBlobDetector detector(params);
         
    // Detect blobs.
    std::vector<KeyPoint> keypoints;
    detector.detect( src, keypoints);
         
    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
    Mat im_with_keypoints;
    drawKeypoints( src, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
         
    // Show blobs
     

    if (keypoints.size() > 0){
      std::stringstream ss;
      ss << "(" <<  (keypoints[0].pt.x-320) << ", " << (keypoints[0].pt.y-240) << ")";
      string tekst = ss.str();
      cv::Size siz = cv::getTextSize(tekst, fontface, scale, thickness, &baseline);
      cv::Point2f pt(keypoints[0].pt.x - (siz.width / 2), keypoints[0].pt.y - siz.height*2);
      cv::putText(im_with_keypoints, tekst, pt, fontface, scale, CV_RGB(255,0,0), thickness, 8);
      return keypoints[0].pt;
    }
  }

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
	    dilate(src, src, dil_element );
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
    IplImage* img = new IplImage(cv_ptr->image);
    Mat org = cvarrToMat(img).clone();
    img = hsvFilter(img);
    Mat src = cvarrToMat(img);
    erosion(src);
    //dilation(src);
    //cv::flip(src,src,1);
    measured = blobDetection(src);
    circle( src, measured, 3, Scalar(0,255,0), -1, 8, 0 );
    
    imshow("Original", org );
    imshow("keypoints", src ); 

    //SLUT TEST




    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

//Husk at dobbelttjek disse værdier
double lambdaX = 17.5; //Grader
double lambdaY = 22.5; //Grader
double pixelDistX = 72; //Pixels
double pixelDistY = 88; //Pixels

double getErrorX(int pixErrorX){
	double alphaX = 0; // SKAL HENTES FRA QUADCOPTEREN
	double betaX = atan(tan(lambdaX/2)*(pixErrorX)/pixelDistX);
	double height = 0; //HØJDEMÅLING FRA ULTRALYD
	return height * tan(alphaX+betaX);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  double dt = 0.1;
  ros::NodeHandle node;
  ros::Rate r(1/dt); // 10 hz
  ros::Publisher pub_empty_land;
	ros::Publisher pub_twist;
	ros::Publisher pub_empty_takeoff;
	ros::Publisher pub_empty_reset;
	//pub_twist = node.advertise<geometry_msgs::Twist>("/cmd_vel", 1); /* Message queue length is just 1 */
	//pub_empty_takeoff = node.advertise<std_msgs::Empty>("/ardrone/takeoff", 1); /* Message queue length is just 1 */
	//pub_empty_land = node.advertise<std_msgs::Empty>("/ardrone/land", 1); /* Message queue length is just 1 */
	//pub_empty_reset = node.advertise<std_msgs::Empty>("/ardrone/reset", 1); /* Message queue length is just 1 */

	twist_msg_hover.linear.x=0.0; 
	twist_msg_hover.linear.y=0.0;
	twist_msg_hover.linear.z=0.0;
	twist_msg_hover.angular.x=0.0; 
	twist_msg_hover.angular.y=0.0;
	twist_msg_hover.angular.z=0.0;  
	double time = (double)ros::Time::now().toSec(); 
	double previous_errorX = 0;
	double outputX = 0;
	double integralX = 0;
	double derivativeX = 0;
	double errorX = 0;
	double previous_errorY = 0;
	double outputY = 0;
	double integralY = 0;
	double derivativeY = 0;
	double errorY = 0;
	Point2f target(0,0);
	double Kp = 1;
	double Ki = 0;
	double Kd = 0;
	while (ros::ok())
	{
  	ros::spinOnce();
  	errorX = target.x - measured.x;
  	integralX = integralX + errorX * dt;
  	derivativeX = (errorX-previous_errorX)/dt;
  	outputX = Kp*errorX + Ki * integralX + Kd * derivativeX;
  	errorY = target.y - measured.y;
  	integralY = integralY + errorY * dt;
  	derivativeY = (errorY-previous_errorY)/dt;
  	outputY = Kp*errorY + Ki * integralY + Kd * derivativeY;
  	ROS_INFO("Blob at (%g, %g)", measured.x-320, measured.y-240);
  	r.sleep();
	}
  return 0;
}