#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <list>

#include <iostream>

using namespace std;
using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

/** Global variables */
String frontalface_cascade_name = "/home/eduardo/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml";
String profileface_cascade_name = "/home/eduardo/opencv-3.1.0/data/haarcascades/haarcascade_profileface.xml";
String eyes_cascade_name = "/home/eduardo/opencv-3.1.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier frontalface_cascade;
CascadeClassifier profileface_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

int main(int argc, char** argv) {
    //string filename = "test.ts";
    //string filename = "/home/eduardo/Vídeos/obras/hbo.ts";
    string filename = "/home/eduardo/Vídeos/obras/i_am_legend.ts";
    //string filename = "/home/eduardo/OpenTLD/build/bin/mig2.mp4";
    //string filename = "/home/eduardo/Vídeos/band.ts";
    
    VideoCapture capture(filename);
    Mat frame;

    if(!capture.isOpened())
        throw "Error when reading steam_avi";

    namedWindow(window_name, 1);
    
    frontalface_cascade.load(frontalface_cascade_name);
    profileface_cascade.load(profileface_cascade_name);
    eyes_cascade.load(eyes_cascade_name);
    
    while(true) {
        capture >> frame;
        
        if(frame.empty())
            break;
        
        detectAndDisplay(frame);
        
        //imshow("w", frame);
        //waitKey(20); // waits to display frame
    }
    
    waitKey(0); // key press to close window
    // releases and window destroy are automatic in C++ interface
}

void detectAndDisplay(Mat frame) {
    std::vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    frontalface_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for(size_t i = 0; i < faces.size(); i++ ) {
        Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        ellipse(frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0);

        Mat faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

        for(size_t j = 0; j < eyes.size(); j++) {
            Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            circle(frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0);
        }
    }
    
    profileface_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for(size_t i = 0; i < faces.size(); i++ ) {
        Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        ellipse(frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0);
    }

    //-- Show what you got
    imshow( window_name, frame );
    waitKey(20);
 }