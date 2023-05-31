#include "image_processing.h"


CImageProcessor::CImageProcessor() {
	for(uint32 i=0; i<3; i++) {
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
    for(int i0 = 0; i0 < 5; i0++) {
        bkgrModel[i0] = new cv::Mat();
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
	if(!image) return(EINVALID_PARAMETER);
    
    int64 startTic = cv::getTickCount();

    cv::Mat colorImage;
    cv::Mat grayImage;

    if(image->channels() > 1) {
        cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
        colorImage = image->clone();
    } else {
        grayImage = *image;
        cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
    }

    cv::Mat graySmooth;
    cv::blur(grayImage, graySmooth, cv::Size(15,15));

    cv::Mat imgDx;
    cv::Mat imgDy;

    cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0, 3, 1, 0);
    cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1, 3, 1, 0);

    uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

    if ((*bkgrModel[0]).size() != grayImage.size()) {
        for(int i0 = 0; i0 < 5; i0++) {
            bkgrModel[i0] = new cv::Mat(grayImage.size(), CV_8U);
        }
    }
    cv::Mat dirImage(grayImage.size(), CV_8U);
    cv::Mat foreground = cv::Mat(grayImage.size(), CV_8U);
    for(int rows = 0; rows < imgDx.rows; rows++) {
        for(int cols = 0; cols < imgDx.cols; cols++) {
            double dx = imgDx.at<int16_t>(rows, cols);
            double dy = imgDy.at<int16_t>(rows, cols);
            double dr2 = dx*dx + dy*dy;
            int index = 0;

            for(int i0 = 0; i0 < 5; i0++) {
                if ((*bkgrModel[i0]).at<uint8_t>(rows, cols) > 0) {
                    (*bkgrModel[i0]).at<uint8_t>(rows, cols) -= 1;
                }
            }

            if(dr2 > threshold*threshold) {
                double alpha = atan2(dy, dx);
                index = round((alpha+3.14159)/(2*3.14159)*4);
                colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);

                if ((*bkgrModel[index]).at<uint8_t>(rows, cols) < 250) {
                    (*bkgrModel[index]).at<uint8_t>(rows, cols) += 2;
                }

                int maxId = -1;
                uint8_t maxVal = 0;
                for(int i0 = 0; i0 < 5; i0++) {
                    if ((*bkgrModel[i0]).at<uint8_t>(rows, cols) > maxVal) {
                        maxVal = (*bkgrModel[i0]).at<uint8_t>(rows, cols);
                        maxId = i0;
                    }
                }
                if ((maxVal > countThresh) && (maxId != index)) {
                    foreground.at<uint8>(rows, cols) = 255;
                }
                else {
                    foreground.at<uint8>(rows, cols) = 0;
                }
            }
            else {
                if ((*bkgrModel[0]).at<uint8_t>(rows, cols) < 250) {
                    (*bkgrModel[0]).at<uint8_t>(rows, cols) += 2;
                }
                foreground.at<uint8>(rows, cols) = 0;
            }

            dirImage.at<uint8>(rows, cols) = index;
        }
    }

    *m_proc_image[0] = colorImage;
    *m_proc_image[1] = foreground;

    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
    cv::morphologyEx(foreground, foreground, cv::MORPH_CLOSE, kernel);

    cv::Mat stats, centroids, labelImage;
    connectedComponentsWithStats(foreground, labelImage, stats, centroids);

    cv::Mat resultImage = image->clone();
    for (int i = 1; i < stats.rows; i++) {
        int topLeftx = stats.at<int>(i, 0);
        int topLefty = stats.at<int>(i, 1);
        int width = stats.at<int>(i, 2);
        int height = stats.at<int>(i, 3);
        int area = stats.at<int>(i, 4);
        double cx = centroids.at<double>(i, 0);
        double cy = centroids.at<double>(i, 1);

        if (area > areaThresh) {
            cv::Point2d cent(cx, cy);
            cv::circle(resultImage, cent, 5, cv::Scalar(0, 0, 255), -1);

            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(foreground, contours, hierarchy, cv::RETR_EXTERNAL ,
                cv::CHAIN_APPROX_SIMPLE);
                for(unsigned int idx = 0 ; idx < contours.size(); idx++ ) {
                //area
                double area = cv::contourArea(contours[idx]);
                //bounding rectangle
                cv::Rect rect = cv::boundingRect(contours[idx]);
                //center of gravity
                // center of mass
                cv::Moments moment = cv::moments(contours[idx]);
                double cx = moment.m10 / moment.m00;
                double cy = moment.m01 / moment.m00;
                //to draw counter to index idx in image
                cv::drawContours(resultImage, contours, idx, cv::Scalar(0, 255, 0), 1, 8 );
            }
        }
    }

    int64 endTic = cv::getTickCount();
    double deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
    std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;

    *m_proc_image[2] = resultImage;

	return(SUCCESS);
}
