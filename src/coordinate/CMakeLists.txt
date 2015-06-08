cmake_minimum_required(VERSION 2.8)
project( Squares )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Squares Squares
.cpp )
target_link_libraries( Squares ${OpenCV_LIBS} )