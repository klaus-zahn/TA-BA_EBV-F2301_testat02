

#include "image_processing.h"


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
	
	if(!image) return(EINVALID_PARAMETER);

	int64 startTic = cv::getTickCount();

	const double threshold = 65*65;
	const uint8_t smoothSize = 5;
	const uint8_t cntThreshold = 25;
	const uint8_t cntMax = 25*2;
	const int32_t minArea = 400;

	cv::Mat grayImage;
	cv::Mat colorImage;

	if (image->channels() > 1)
	{
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		colorImage = image->clone();
	}
	else
	{
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}

	cv::Mat graySmooth;
	cv::blur(grayImage, graySmooth, cv::Size(smoothSize, smoothSize));

	cv::Mat imgDx;
	cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0);

	cv::Mat imgDy;
	cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1);

	uint8_t colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

	cv::Mat dirImage(grayImage.size(), CV_8U);

	static int dims[] = {grayImage.rows, grayImage.cols, 5};
	static cv::Mat bkgrModel(3, dims, CV_8S, cntThreshold);

	cv::Mat bkgrImage(grayImage.size(), CV_8U);
	cv::Mat foreImage(grayImage.size(), CV_8U);
	cv::Mat openImage(grayImage.size(), CV_8U);
	cv::Mat closeImage(grayImage.size(), CV_8U);
	cv::Mat coloredImage = colorImage.clone();

	for (int rows = 0; rows < imgDx.rows; rows++)
	{
		for (int cols = 0; cols < imgDx.cols; cols++)
		{
			double dx = imgDx.at<int16_t>(rows, cols);
			double dy = imgDy.at<int16_t>(rows, cols);
			double dr2 = dx * dx + dy * dy;
			uint8_t index = 0;
			uint8_t max = 0;
			uint8_t maxIndex = 0;
			if (dr2 > threshold)
			{
				double alpha = atan2(dy, dx);// + 5 * M_PI_4;
				index = 1 + (int)(((alpha + M_PI + M_PI_4) / M_PI_2));
				coloredImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);
			}
			for (int i = 0; i < 5; i++)
			{
				uint8_t cnt = bkgrModel.at<uint8>(rows, cols, i);
				if (i==index)
				{
					if (cnt < cntMax)
					{
						cnt++;
					}
				}
				else
				{
					if (cnt > 0)
					{
						cnt--;
					}
				}
				if (cnt > max && cnt > cntThreshold)
				{
					max = cnt;
					maxIndex = i; 
				}
				bkgrModel.at<uint8>(rows, cols, i) = cnt;
			}
			dirImage.at<uint8>(rows, cols) = index;
			bkgrImage.at<uint8>(rows, cols) = maxIndex;
			foreImage.at<uint8>(rows, cols) = index!=maxIndex;
		}
	}
	cv::Mat openFilter = cv::Mat::ones(4, 4, CV_8UC1);
	cv::morphologyEx(foreImage, openImage, cv::MORPH_OPEN, openFilter);

	cv::Mat closeFilter = cv::Mat::ones(25, 25, CV_8UC1);
	cv::morphologyEx(openImage, closeImage, cv::MORPH_CLOSE, closeFilter);
	

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(closeImage, contours, hierarchy, cv::RETR_EXTERNAL , cv::CHAIN_APPROX_SIMPLE);

	cv::Mat resultImage = colorImage.clone();

	for(unsigned int i = 0 ; i < contours.size(); i++ ) {
		//area
		double area = cv::contourArea(contours[i]);

		if ( area >= minArea ) {
			// center of mass
			cv::Moments moment = cv::moments(contours[i]);
			double cx = moment.m10 / moment.m00;
			double cy = moment.m01 / moment.m00;

			cv::circle(resultImage, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);
			
			//to draw counter to index idx in image
			cv::drawContours(resultImage, contours, i, cv::Scalar(255,255,0), 1, 8 );
		}
	}

	*m_proc_image[0] = resultImage;
	*m_proc_image[1] = coloredImage;
	*m_proc_image[2] = closeImage;

	int64 endTic = cv::getTickCount();
	double deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
	std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;

	/*
		const double threshold = 30;
		const double alpha = 0.95;

		//static cv::Mat mPrevImage;
		static cv::Mat mBkgrImage;

		cv::Mat grayImage;

		if (image->channels() > 1)
		{
			cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
		}
		else
		{
			grayImage = *image;
		}

		//if (mPrevImage.size() != cv::Size())
		if (mBkgrImage.size() != cv::Size())
		{
			cv::addWeighted(mBkgrImage, alpha, grayImage, 1 - alpha, 0, mBkgrImage);

			cv::Mat diffImage;
			//cv::absdiff(mPrevImage, grayImage, diffImage);
			cv::absdiff(mBkgrImage, grayImage, diffImage);
			cv::Mat binaryImage;
			cv::threshold(diffImage, binaryImage, threshold, 255, cv::THRESH_BINARY);

			cv::Mat kernel = cv::Mat::ones(50, 50, CV_8UC1);
			cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

			cv::Mat stats, centroids, labelImage;
			connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);
			cv::Mat resultImage = grayImage.clone();

			for (int i = 1; i < stats.rows; i++)
			{
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

			*m_proc_image[0] = resultImage;
			*m_proc_image[1] = mBkgrImage;
			*m_proc_image[2] = labelImage;
		}
		else
		{
			mBkgrImage = grayImage.clone();
		}
		*/
		//mPrevImage = grayImage.clone();
		/*

		cv::subtract(cv::Scalar::all(255), *image,*m_proc_image[0]);
		cv::subtract(cv::Scalar::all(255), grayImage,*m_proc_image[1]);
		cv::subtract(cv::Scalar::all(255), grayImage,*m_proc_image[2]);
		*/
        
      //  cv::imwrite("dx.png", *m_proc_image[0]);
      //  cv::imwrite("dy.png", *m_proc_image[1]);

	return(SUCCESS);
}









