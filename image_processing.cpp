#include "image_processing.h"

#define Testat 1

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

#if Testat
int CImageProcessor::DoProcess(cv::Mat* image) {
	
	if(!image) return(EINVALID_PARAMETER);	

	double threshold = 40;
	double minArea = 100;
	cv::Mat colorImage;	//for edge detection

	//convert image to gray-scale image
	cv::Mat grayImage;
	if(image->channels() > 1) {
		cv::cvtColor( *image, grayImage, cv::COLOR_RGB2GRAY );
		colorImage = image->clone();
	}
	else{
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}
	
	//image processing:
	
	//low pass filter
	cv::Mat graySmooth;
	cv::blur(grayImage, graySmooth, cv::Size(5,5));
	
	//partial derivatives with sobel filter (result 16bit signed)
	cv::Mat imgDx;
	cv::Mat imgDy;
	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0); //I_x
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0); //I_y
	
	//norm-derivative
	cv::Mat dirImage(grayImage.size(), CV_8U);
	cv::Mat foreground(grayImage.size(), CV_8U);

	for (int i = 0; i < 5; i++) {
		background[i] = new cv::Mat(grayImage.size(), CV_8U);
	}
	
	uint8 colorMap[4][3] = {{255,0,0}, {0,255,0}, {0,0,255}, {255,255,0}};
	
	//Binning
	for(int rows = 0; rows < imgDx.rows; rows++){
		for(int cols = 0; cols < imgDx.cols; cols++){
			//if(rows == 130 && cols == 221) {
			//	int stop = 0;
			//}
			double dx = imgDx.at<int16_t>(rows,cols);
			double dy = imgDy.at<int16_t>(rows,cols);
			double dr2 = dx*dx + dy*dy; //squared norm
			int index = 0;

			// Bei folgendem Abschnitt Hilfe von Felix Reding erhalten
			// Alle channel welche nicht channel 0 sind subtrahieren
			for(int i = 0; i < 5; i++) {
				if ((*background[i]).at<uint8_t>(rows, cols) > 0) {
					(*background[i]).at<uint8_t>(rows, cols) -= 1;
				}
			}

			if(dr2 > threshold*threshold){	//compare norm with threshold
				double alpha = atan2(dy,dx);
				// Assign values for the different bins
				index = round((alpha+3.14159)/(2*3.14159)*4);
				colorImage.at<cv::Vec3b>(rows,cols) = cv::Vec3b(colorMap[index-1]);
				
				// Add the wrongly subtracted one back
				if ((*background[index]).at<uint8_t>(rows, cols) < 250) {
					(*background[index]).at<uint8_t>(rows, cols) += 2;
				}

				int maxId = -1;
				uint8_t maxVal = 0;

				// Check for the max value across the channels
				for(int i = 0; i < 5; i++) {
					if ((*background[i]).at<uint8_t>(rows, cols) > maxVal) {
						maxVal = (*background[i]).at<uint8_t>(rows, cols);
						maxId = i;
					}
				}
				if ((maxVal > 25) && (maxId != index)) {
					foreground.at<uint8>(rows, cols) = 255;
				}
				else {
					foreground.at<uint8>(rows, cols) = 0;
				}
			}
			else {
				if (((*background[0])).at<uint8_t>(rows, cols) < 250) {
					((*background[0])).at<uint8_t>(rows, cols) += 2;
				}
				foreground.at<uint8>(rows, cols) = 0;
			} 
			dirImage.at<uint8_t>(rows,cols) = index;
		}	
	}

	//morphology
	cv::Mat kernel = cv::Mat::ones(7, 7, CV_8UC1); //structurelement
	cv::Mat temp = foreground.clone();
	cv::morphologyEx(temp, foreground, cv::MORPH_OPEN, kernel);
	temp = foreground.clone();
	cv::morphologyEx(temp, foreground, cv::MORPH_CLOSE, kernel); //morphology -> closing (with structurelement 'kernel')

	//region labeling
	cv::Mat stats, centroids, labelImage;
	connectedComponentsWithStats(foreground, labelImage,stats, centroids); // region labeling on foreground (result=labelImage)
	cv::Mat resultImage = (*image).clone();
	for (int i = 1; i < stats.rows; i++) {
		//Bounding Box
		int topLeftx = stats.at<int>(i, 0);
		int topLefty = stats.at<int>(i, 1);
		int width = stats.at<int>(i, 2);
		int height = stats.at<int>(i, 3);
		//area
		int area = stats.at<int>(i, 4);
		//center of mass (centroid)
		double cx = centroids.at<double>(i, 0);
		double cy = centroids.at<double>(i, 1);
		//plot region labeling
		if(minArea < area){
			cv::Rect rect(topLeftx, topLefty, width, height);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));
			cv::Point2d cent(cx, cy);
			cv::circle(resultImage, cent, 5, cv::Scalar(128, 0, 0), -1);
		}
	}

	//save images to web view 'Ansicht'
	*m_proc_image[0] = colorImage;
	*m_proc_image[1] = foreground;
	*m_proc_image[2] = resultImage;
	//*m_proc_image[3] = resultImage;
	//}


	for (int i = 0; i < 5; i++) {
		delete background[i];
	}


	m_PrevImage = grayImage.clone();
	return(SUCCESS);
}
#else
int CImageProcessor::DoProcess(cv::Mat* image) {
	
	if(!image) return(EINVALID_PARAMETER);	
        
		cv::Mat grayImage;

		if (image->channels() > 1) {
			cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		} else {
			grayImage = *image;
		}

		cv::Mat diffImage;

		if (m_PrevImage.size() != cv::Size()) {
			cv::absdiff(m_PrevImage, grayImage, diffImage);

			cv::Mat binaryImage;
			cv::threshold(diffImage, binaryImage, 20, 255, cv::THRESH_BINARY);

			cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
			cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

			cv::Mat stats, centroids, labelImage;
			connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

			cv::Mat resultImage = image->clone();
			for (int i = 1; i < stats.rows; i++) {
				int topLeftx = stats.at<int>(i, 0);
				int topLefty = stats.at<int>(i, 1);
				int width = stats.at<int>(i, 2);
				int height = stats.at<int>(i, 3);

				int area = stats.at<int>(i, 4);

				double cx = centroids.at<double>(i, 0);
				double cy = centroids.at<double>(i, 1);

				cv::Rect rect(topLeftx, topLefty, width, height);
				cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));
				cv::Point2d cent(cx, cy);
				cv::circle(resultImage, cent, 5, cv::Scalar(128, 0, 0), -1);
			}

			*m_proc_image[0] = labelImage;
			*m_proc_image[1] = binaryImage;
			*m_proc_image[2] = resultImage;
		}

		m_PrevImage = grayImage.clone();
	return(SUCCESS);
}
#endif