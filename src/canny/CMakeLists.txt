cmake_minimum_required(VERSION 2.8)
project( canny )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( canny canny.cpp )
target_link_libraries( canny ${OpenCV_LIBS} )



































