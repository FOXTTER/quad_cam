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
#define X 0
#define Y 1
#define ROT 2
#define Z 3

//Values for HSV filter
#define BLUE_HUE_LOW 83 
#define BLUE_HUE_HIGH 126
#define BLUE_SAT_LOW 60
#define BLUE_SAT_HIGH 255
#define BLUE_VAL_LOW 57
#define BLUE_VAL_HIGH 117
#define RED_HUE_LOW 144
#define RED_HUE_HIGH 187 
#define RED_SAT_LOW 71
#define RED_SAT_HIGH 243
#define RED_VAL_LOW 64
#define RED_VAL_HIGH 208

static const std::string OPENCV_WINDOW = "Image window";
double  measured[4] = {};
geometry_msgs::Twist twist_msg;
geometry_msgs::Twist twist_msg_hover;
geometry_msgs::Twist twist_msg_pshover;
std_msgs::Empty emp_msg;
ardrone_autonomy::Navdata msg_in_global;
	//Husk at dobbelttjek disse værdier
	double gammaX = 40; //Grader
	double gammaY = 64; //Grader
	double pixelDistX = 180; //Pixels
	double pixelDistY = 320; //Pixels


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

  Point blobDetection(Mat src){
  	Mat src_gray;
  	//TEST AF BLOB DETECTION
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

   // Set up the detector with default parameters.
    cv::SimpleBlobDetector::Params params;
    // foelgende fra DroneLander
    params.thresholdStep = 10;
    params.minThreshold = 60;
    params.maxThreshold = 300;
    params.minRepeatability = 1;
    params.minDistBetweenBlobs = 100;
    params.filterByColor = true;
    params.blobColor = 255;
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 10000;
    params.filterByCircularity = 0;
    params.minCircularity = 8.0000001192092896e-001;
    params.maxCircularity = 3.4028234663852886e+038;
    params.filterByInertia = true;
    params.minInertiaRatio = 1.0000000149011612e-001;
    params.maxInertiaRatio = 3.4028234663852886e+038;
    params.filterByConvexity = true;
    params.minConvexity = 0.5;
    params.maxConvexity = 1.5;
    // vores oårindelige parameter 
    /*
    params.filterByArea =true;
    params.maxArea = 50000;
    params.minArea = 1000;
    params.filterByColor = true;
    params.filterByCircularity = false;
    params.blobColor = 255;
    params.filterByConvexity = true;
    */
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
    return Point(-1,-1);
  }
  enum color_type {
    red,
    blue
	};

	double getAngle(Point2f blue, Point2f red) {
	  int kateteA = red.y - blue.y;
	  int kateteB = red.x - blue.x;

	  return atan2(kateteA, kateteB);
	}


  IplImage* hsvFilter( IplImage* img, color_type color) {

    //HSV image
    IplImage* imgHSV = cvCreateImage( cvGetSize(img), 8, 3);
    cvCvtColor( img, imgHSV, CV_BGR2HSV);



    //Create binary image with min/max value
    IplImage* imgThresh = cvCreateImage( cvGetSize( img ), 8, 1 );

    switch(color) {
        case red:
            cvInRangeS (imgHSV, cvScalar( RED_HUE_LOW , RED_SAT_LOW, RED_VAL_LOW), cvScalar(RED_HUE_HIGH, RED_SAT_HIGH, RED_VAL_HIGH), imgThresh );
            //Hvis den er i det røde skal vi wrappe den
            if(RED_HUE_HIGH > 180){
    					IplImage* temp = cvCreateImage( cvGetSize( img ), 8, 1 );
    					cvInRangeS (imgHSV, cvScalar( 0 , RED_SAT_LOW, RED_VAL_LOW), cvScalar(RED_HUE_HIGH-180, RED_SAT_HIGH, RED_VAL_HIGH), temp );
    					cvAdd(imgThresh,temp,imgThresh);
    					cvReleaseImage( &temp );
    				}
            break;
        case blue:
            cvInRangeS (imgHSV, cvScalar( BLUE_HUE_LOW , BLUE_SAT_LOW, BLUE_VAL_LOW), cvScalar(BLUE_HUE_HIGH, BLUE_SAT_HIGH, BLUE_VAL_HIGH), imgThresh );
            break;
        default:
            cvInRangeS (imgHSV, cvScalar( 50 , 80, 0), cvScalar(150, 220, 255), imgThresh );
    }

    //Clean up
    cvReleaseImage( &imgHSV );  
    return imgThresh;
}

	void erosion(Mat src, int erosion_size){
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
	    dilate(src, src, dil_element );
	}
	
	double getPosX(int pixErrorX){
		double alphaX = ((msg_in_global.rotY*3.14)/180); // SKAL HENTES FRA QUADCOPTEREN
		double betaX = -atan(tan(gammaX/2)*(pixErrorX)/pixelDistX);
		double height = 1;//;msg_in_global.altd/1000; //HØJDEMÅLING FRA ULTRALYD
		return height * tan(alphaX+betaX);
	}
	double getPosY(int pixErrorY){
		double alphaY = ((msg_in_global.rotX*3.14)/180); // SKAL HENTES FRA QUADCOPTEREN
		double betaY = -atan(tan(gammaY/2)*(pixErrorY)/pixelDistY);
		double height = 1;//msg_in_global.altd/1000; //HØJDEMÅLING FRA ULTRALYD
		return height * tan(alphaY+betaY);
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
    Mat blueMat = cvarrToMat(hsvFilter(img,blue));
    Mat redMat  = cvarrToMat(hsvFilter(img,red));
    Mat src = cvarrToMat(img);
    erosion(blueMat,5);
    dilation(blueMat,0);
    erosion(redMat,7);
    dilation(redMat,0);
    Point2f blaa = blobDetection(blueMat);
    Point2f roed = blobDetection(redMat);
    Mat test = redMat + blueMat;
    cvtColor(test,test,COLOR_GRAY2RGB);
    circle( test, Point2f(320,180), 10, Scalar(255,0,0), -1, 8, 0 );
    if (blaa.x == -1 || roed.x == -1)
    {
    	//measured[X] = 0;
    	//measured[Y] = 0;
    	
    }else{
    	measured[X] = getPosX(((blaa.y+roed.y)/2)-180);
    	measured[Y] = getPosY(((blaa.x+roed.x)/2)-320);
    	circle( test, Point2f((blaa.x+roed.x)/2,(blaa.y+roed.y)/2), 10, Scalar(0,255,0), -1, 8, 0 );
    }
    measured[ROT] = getAngle(blaa, roed);

    
    
    
    //circle( test, Point2f(measured[X]+320,measured[Y]+240), 10, Scalar(0,255,0), -1, 8, 0 );



    //imshow("Red", redMat );
    //imshow("Blue", blueMat ); 
    imshow("test", test);
    imshow("Original",org);

    //SLUT TEST




    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};


void nav_callback(const ardrone_autonomy::Navdata& msg_in)
{
		//Take in state of ardrone	
		msg_in_global = msg_in;
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
	ros::Subscriber nav_sub;
	nav_sub = node.subscribe("/ardrone/navdata", 1, nav_callback);
	pub_twist = node.advertise<geometry_msgs::Twist>("/cmd_vel", 1); /* Message queue length is just 1 */
	pub_empty_takeoff = node.advertise<std_msgs::Empty>("/ardrone/takeoff", 1); /* Message queue length is just 1 */
	pub_empty_land = node.advertise<std_msgs::Empty>("/ardrone/land", 1); /* Message queue length is just 1 */
	pub_empty_reset = node.advertise<std_msgs::Empty>("/ardrone/reset", 1); /* Message queue length is just 1 */
	twist_msg_hover.linear.x=0.0; 
	twist_msg_hover.linear.y=0.0;
	twist_msg_hover.linear.z=0.0;
	twist_msg_hover.angular.x=0.0; 
	twist_msg_hover.angular.y=0.0;
	twist_msg_hover.angular.z=0.0;
	double time = (double)ros::Time::now().toSec(); 
	
  double previous_error[4] = {};
  double error[4] = {};
  double output[4] = {};
  double derivative[4] = {};
  double integral[4] = {};
  double target[4] = {};

	double Kp[4] = {0.1,0.1,1/6.28,0};
	double Ki[4] = {};
	double Kd[4] = {};
	double time_start=(double)ros::Time::now().toSec();
	while (ros::ok() && ((double)ros::Time::now().toSec()< time_start+1.0));
	if (msg_in_global.state == 0) {
		ros::spinOnce();
  	//pub_empty_reset.publish(emp_msg);
	}
	//pub_empty_takeoff.publish(emp_msg);
	//pub_twist.publish(twist_msg_hover);
	time_start=(double)ros::Time::now().toSec();
	while (ros::ok() && ((double)ros::Time::now().toSec()< time_start+5.0));
	while (ros::ok() || ((double)ros::Time::now().toSec()< time_start+20) && msg_in_global.altd < 1500)
	{
  	ros::spinOnce();
  	ROS_INFO("Angle y: %g",msg_in_global.rotY);
    for(int i = 0; i < 4; i++) {
      error[i] = target[i] - measured[i];
      integral[i] = integral[i] + error[i] * dt;
      derivative[i] = (error[i]-previous_error[i])/dt;
      output[i] = Kp[i]*error[i] + Ki[i] * integral[i] + Kd[i] * derivative[i];
      previous_error[i] = error[i];
    }
    twist_msg.angular.z= output[ROT];
    twist_msg.linear.x = output[X];
    twist_msg.linear.y = output[Y];

    //pub_twist.publish(twist_msg);
  	//ROS_INFO("Blob at (%g, %g)", measured[0]-320, measured[1]-240);
  	//ROS_INFO("Angle: %g", measured[ROT]);
  	ROS_INFO("Measured pos = (%g,%g)",measured[X],measured[Y]);
  	ROS_INFO("Measured rot = %g", measured[ROT]);
  	ROS_INFO("Output x: %g", output[X]);
  	ROS_INFO("Output y: %g", output[Y]);
  	ROS_INFO("Output rot: %g", output[ROT]);
  	char key = cvWaitKey(10);     //Capture Keyboard stroke
    if (char(key) == 27){
        break;      //If you hit ESC key loop will break.
    }
  	r.sleep();
	}
	pub_empty_land.publish(emp_msg);
  return 0;
}