// 28.05.2023, Niklas Rösch

#include "image_processing.h"

CImageProcessor::CImageProcessor():m_init(false) {
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

	// measuring ticks
	int64 startTic = cv::getTickCount();
	
	if(!image) return(EINVALID_PARAMETER);	

		cv::Mat grayImage;
		cv::Mat diffImage;
		cv::Mat binaryImage;
		cv::Mat graySmooth;
		cv::Mat imgDx;
		cv::Mat imgDy;
		cv::Mat dirImage(grayImage.size(), CV_8U);
		cv::Mat morphImage;
		cv::Mat contImage;

		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};	// defining output colors
	
		double binaryThreshold = 50;
		double partialThreshold = 50;
		double areaThreshold = 3000;

		int binThreshold = 50;
		int numBins = 4;
		int binIndex = 0;

		if(image->channels() > 1 ) {
			cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
			colorImage = image->clone();
		} else {
			grayImage = *image;
			cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
		}

		if(!m_init) {
			fgrModel = cv::Mat(480, 640, CV_8U);

			for (int i0 = 0; i0 < 5; i0++)
			{
				bkgrModel[i0] = new cv::Mat(grayImage.size(), CV_8U);
			}
			m_init = true;
		}
	
	if (mPrevImage.size() != cv::Size()) {

	// diffImage 
	cv::absdiff(*image, mPrevImage, diffImage);

	// binaryImage
	cv::threshold(diffImage, binaryImage, binaryThreshold, 255, cv::THRESH_BINARY);

	// lowpass filtering
	cv::blur(grayImage, graySmooth, cv::Size(5,5));

	// partial derivatives	
	cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0, 3, 1, 0);	// x-direction
	cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1, 3, 1, 0);	// y-direction

	// binning
	for (int rows = 0; rows < imgDx.rows; rows++) {
		for (int cols = 0; cols < imgDy.cols; cols++) {
			
	 		double dx = imgDx.at<int16_t>(rows, cols);
	 		double dy = imgDy.at<int16_t>(rows, cols);

	 		double dr2 = dx*dx + dy*dy;						// squared norm of derivative

	 		binIndex = 0;

	 		if (dr2 > partialThreshold*partialThreshold) {	// squared norm of derivative greater than squared threshold?
		
				double alpha = atan2(dy, dx);				// angle of derivative (-pi to pi)
	 			
				//index = fmod(floor((alpha+M_PI/numBins)/(2*M_PI/numBins)), numBins)+1;	// own implementation
				//ZaK: wrong binning
				//binIndex = 1+(int)((alpha+(3/4*M_PI))/(M_PI/2)); // binning from lesson
				binIndex = 1 + ((int)((alpha + M_PI + M_PI / 4) / (2 * M_PI) * 4)) % 4;
				
	 			colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[binIndex - 1]);	// coloring pixels
			}

			for (int i = 0; i <= numBins; i++) {
             	if (i == binIndex) {
            	 	bkgrModel[i]->at<uint8>(rows, cols) += 1;
	 				if (bkgrModel[i]->at<uint8>(rows, cols) > binThreshold) {
	 					bkgrModel[i]->at<uint8>(rows, cols) = binThreshold;
	 				}
                } else if (bkgrModel[i]->at<uint8>(rows, cols) > 0) {
                    bkgrModel[i]->at<uint8>(rows, cols) -= 1;
                }
	 		}	

			if (bkgrModel[binIndex]->at<uint8>(rows, cols) > (binThreshold/2)) {
                fgrModel.at<uint8>(rows, cols) = 0;
            } else {
                fgrModel.at<uint8_t>(rows, cols) = 255;
            }
		}
	}

	// morphology
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(9,9));
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9,9));
	cv::morphologyEx(fgrModel, morphImage, cv::MORPH_CLOSE, kernel);

	cv::Mat contImage = image->clone();

	cv::findContours(morphImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (unsigned int idx = 0; idx < contours.size(); idx++) {
		double area = cv::contourArea(contours[idx]);
		cv::Moments moment = cv::moments(contours[idx]);
		double cx = moment.m10 / moment.m00;
		double cy = moment.m01 / moment.m00;

		cv::Point2d cent(cx, cy);

		if (area > areaThreshold) {
		cv::drawContours(contImage, contours, idx, cv::Scalar(255), 2);
		cv::circle(contImage, cent, 5, cv::Scalar(128, 0, 0), -1);
		}
	}

	*m_proc_image[0] = colorImage;
	*m_proc_image[1] = fgrModel;
	*m_proc_image[2] = contImage;

	}

	mPrevImage = grayImage.clone();
	
	// measuring ticks
	int64 endTic = cv::getTickCount();
	double deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
	std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;

	return(SUCCESS);
}



