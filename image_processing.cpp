

#include "image_processing.h"

CImageProcessor::CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
}

CImageProcessor::~CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		delete m_proc_image[i];
	}
}

cv::Mat *CImageProcessor::GetProcImage(uint32 i)
{
	if (2 < i)
	{
		i = 2;
	}
	return m_proc_image[i];
}

int CImageProcessor::DoProcess(cv::Mat *image)
{

	if (!image)
		return (EINVALID_PARAMETER);

	int64 startTic = cv::getTickCount();

	// Gray Image
	cv::Mat grayImage;
	// Color Image
	cv::Mat colorImage;
	// Sobel Filter in dx
	cv::Mat imgDx;
	// Sobel Filter in dy
	cv::Mat imgDy;
	// Gray Smooth Image
	cv::Mat graySmooth;
	// size Tiefpassfilter
	uint8 sizeTPFilter = 5;
	// define Value for threshold
	uint8 threshold = 30;

	uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

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

	// init bkgrModel
	int dims[] = {grayImage.rows,grayImage.cols,5};
	static cv::Mat bkgrModel(3, dims, CV_8S, 25);

	// dir Image
	cv::Mat dirImage(grayImage.size(), CV_8U);

	// Background Image
	cv::Mat bkgrImage(grayImage.size(), CV_8U);

	// Foreground Image
	cv::Mat frgrImage(grayImage.size(), CV_8U);

	// Result Image
	cv::Mat resultImage(grayImage.size(), CV_8U);

	// Int for position of max value
	uint8_t pos = 0;

	// int for max background value
	uint8_t bkgr = 0;
	int a = 0;

	// if (mPrevImage.size() != cv::Size()) {
	// Tiefpassfilter
	cv::blur(grayImage, graySmooth, cv::Size(sizeTPFilter, sizeTPFilter));

	cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0, 3, 1, 0);
	cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1, 3, 1, 0);

	for (int rows = 0; rows < imgDx.rows; rows++)
	{
		for (int cols = 0; cols < imgDx.cols; cols++)
		{
			double dx = imgDx.at<int16_t>(rows, cols);
			double dy = imgDy.at<int16_t>(rows, cols);
			double dr2 = dx * dx + dy * dy;
			int index = 0;
			if (dr2 > (threshold * threshold))
			{
				double alpha = atan2(dy, dx);
				//ZaK: wrong binning
				//index = 1 + (int)((alpha + M_PI) / (M_PI / 2));
				index = 1 + ((int)((alpha + M_PI + M_PI / 4) / (2 * M_PI) * 4)) % 4;
				//ZaK: c.f. below
				colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);
			}
			dirImage.at<uint8>(rows, cols) = index;
			//ZaK: set only if dr2 > thr²
			//colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);
			
			// Int for position of max value
			pos = 0;

			// int for max background value
			bkgr = 0;

			// Binning
			for (int i = 0; i < 5; i++){
				a = bkgrModel.at<uint8_t>(rows,cols,i);
				a = a - 1;

				if (dirImage.at<uint8>(rows, cols) == i){
					a = a + 2;
				}

				if (a < 0)
				{
					a = 0;
				}
				else if (a > 50)
				{
					a = 50;
				}

				if( (a > 25) && (a > bkgr)){
					pos = i;
					bkgr = a;
				}
				bkgrModel.at<uint8_t>(rows,cols,i) = a;
			}

			bkgrImage.at<uint8_t>(rows,cols) = pos;

			// frgrImage.at<uint8_t>(rows,cols) = ((bkgrImage.at<uint8_t>(rows,cols)) != (dirImage.at<uint8>(rows, cols)));

			if ((bkgrImage.at<uint8_t>(rows,cols)) == (dirImage.at<uint8>(rows, cols))){
				frgrImage.at<uint8_t>(rows,cols) = 0;
			}else{
				frgrImage.at<uint8_t>(rows,cols) = 255;
			}
		}
	}

	// Values for Opening and Closing
	int valueOpen = 5;
	int valueClose = 10;

	// Opening
	cv::Mat kernel = cv::Mat::ones(valueOpen, valueOpen, CV_8UC1);
	cv::morphologyEx(frgrImage, frgrImage, cv::MORPH_OPEN, kernel);
	

	// Closing
	kernel = cv::Mat::ones(valueClose, valueClose, CV_8UC1);
	cv::morphologyEx(frgrImage, frgrImage, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	double area_min = 100;

	cv::findContours(frgrImage, contours, hierarchy, cv::RETR_EXTERNAL , cv::CHAIN_APPROX_SIMPLE);

	resultImage = image->clone();
	
	for(unsigned int idx = 0 ; idx < contours.size(); idx++ ) {
		//area
		double area = cv::contourArea(contours[idx]);

		if ( area >= area_min ) {

			// //bounding rectangle
			// cv::Rect rect = cv::boundingRect(contours[idx]);

			// center of mass
			cv::Moments moment = cv::moments(contours[idx]);
			double cx = moment.m10 / moment.m00;
			double cy = moment.m01 / moment.m00;

			//to draw counter to index idx in image
			cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8 );

			//draw center of mass #todo exclude drawing, if it is on a border
			cv::drawMarker(resultImage, cv::Point(cx,cy), cv::Scalar(255), 1, 10, 3, 8);
		}
	}
	int64 endTic = cv::getTickCount();
	double deltaTime = (double) (endTic - startTic)/cv::getTickFrequency();
	std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;

	*m_proc_image[0] = colorImage;
	*m_proc_image[1] = frgrImage;
	*m_proc_image[2] = resultImage;
	
	

	return (SUCCESS);
}