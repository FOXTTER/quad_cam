cmake_minimum_required(VERSION 2.8)
project( erosion_dilatation )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( erosion_dilatation erosion_dilatation.cpp )
target_link_libraries( erosion_dilatation ${OpenCV_LIBS} )



































