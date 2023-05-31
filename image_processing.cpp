#include "image_processing.h"
#include "opencv2/opencv.hpp"
#include <iostream>

//ZaK
//background model not implemented correctly, only simply threshold applied

CImageProcessor::CImageProcessor() {
    for(uint32 i=0; i<3; i++) {
        /* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
        m_proc_image[i] = new cv::Mat();
    }
}

CImageProcessor::~CImageProcessor() {
    for(uint32 i=0; i<3; i++) {
        delete m_proc_image[i];
    }
}

cv::Mat* CImageProcessor::GetProcImage(uint32 i) {
    if(2 < i) {
        i = 2;
    }
    return m_proc_image[i];
}

int CImageProcessor::DoProcess(cv::Mat* image) {
    // Start time measurement
    int64 startTic = cv::getTickCount();

    // Parameter
    double threshold = 80.0;

    if(!image) return(EINVALID_PARAMETER);  

    cv::Mat grayImage;
    cv::Mat colorImage;
    cv::Mat dirImage;
    uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};
    
    // Convert the image to grayscale
    if(image->channels() > 1) {
        cv::cvtColor( *image, grayImage, cv::COLOR_BGR2GRAY );
        colorImage = image->clone();
    } else {
        grayImage = *image;
        cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
    }   

    // Simple low-pass filter
    cv::Mat graySmooth;
    cv::blur(grayImage, graySmooth, cv::Size(5,5));

    // Compute the partial derivatives
    cv::Mat imgDx;
    cv::Mat imgDy;
    cv::Sobel(grayImage, imgDx, CV_16S, 1, 0);
    cv::Sobel(grayImage, imgDy, CV_16S, 0, 1);

    // Binning of the edge direction into four angle intervals
    dirImage = cv::Mat(grayImage.size(), CV_8U);

    // Initialize the 3D matrix to hold the counters for each pixel
    int dims[] = {grayImage.rows, grayImage.cols, 5};
    cv::Mat bkgrModel(3, dims, CV_32S, cv::Scalar(0));

    for(int rows = 0; rows < imgDx.rows; rows++) {
        for(int cols = 0; cols < imgDx.cols; cols++) {
            double dx = imgDx.at<int16_t>(rows, cols);
            double dy = imgDy.at<int16_t>(rows, cols);
            double dr2 = dx*dx + dy*dy;
            int index = 0;
            if(dr2 > threshold*threshold) {
                double alpha = atan2(dy, dx);
                index = 1 + (int) ((alpha + M_PI)/(M_PI/4));
                colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);
            }
            dirImage.at<uint8>(rows, cols) = index;

            // Increment the corresponding counter in the bkgrModel matrix
            bkgrModel.at<int>(rows, cols, index)++;
        }
    }

    // Define the foreground
    cv::Mat foreground = grayImage.clone();
    cv::threshold(foreground, foreground, threshold, 255, cv::THRESH_BINARY);

    // Morphologische Operationen
    cv::Mat improvedForeground = foreground.clone();

    // Erstelle ein Strukturelement für die morphologischen Operationen
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // Durchführen der Erosion und der Dilatation
    cv::erode(improvedForeground, improvedForeground, kernel);
    cv::dilate(improvedForeground, improvedForeground, kernel);

    // Segmentierung und Merkmalsextraktion durch Konturenerkennung
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(improvedForeground, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Konturen grafisch darstellen
    cv::Mat contouredImage = cv::Mat::zeros(improvedForeground.size(), CV_8UC3);
    for(int i = 0; i < contours.size(); i++) {
        cv::drawContours(contouredImage, contours, i, cv::Scalar(255, 0, 0), 2, cv::LINE_8, hierarchy, 0);
    }

    *m_proc_image[0] = colorImage;
    *m_proc_image[1] = foreground;
    *m_proc_image[2] = contouredImage;

    // End time measurement
    int64 endTic = cv::getTickCount();

    // Compute the elapsed time
    double deltaTime = (double)(endTic - startTic) / cv::getTickFrequency();

    // Print the result
    std::cout << "Time: " << (int)(1000 * deltaTime) << " ms" << std::endl;

    return 0; // If the process has successfully completed
}
