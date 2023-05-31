

#include "image_processing.h"

CImageProcessor::CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
	for (int i0 = 0; i0 < 5; i0++)
	{
		bkgrModel[i0] = NULL;
	}
}

CImageProcessor::~CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		delete m_proc_image[i];
	}
	for (int i0 = 0; i0 < 5; i0++)
	{
		delete bkgrModel[i0];
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
	int64 startTic = cv::getTickCount();
	cv::Mat grayImage;
	cv::Mat colorImage;
	cv::Mat graySmooth;
	cv::Mat imgDx;
	cv::Mat imgDy;

	double gradThreshold = 50;
	uint8_t bkgrThreshold = 50;

	if (!image)
		return (EINVALID_PARAMETER);

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

	cv::Mat directionsImg = colorImage.clone();
	cv::Mat fgrModel = cv::Mat(grayImage.size(), CV_8U);
	if (bkgrModel[0] == NULL)
	{
		for (int i0 = 0; i0 < 5; i0++)
		{
			bkgrModel[i0] = new cv::Mat(grayImage.size(), CV_8U);
		}
	}

	// Tiefpassfiltern
	cv::blur(grayImage, graySmooth, cv::Size(5, 5));

	// Kantenfilter
	// cv::Sobel(graySmooth, imgDx, -1, 1, 0, 3, 1, 128);
	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);
	// cv::Sobel(graySmooth, imgDy, -1, 0, 1, 3, 1, 128);
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);

	// Normableitung
	cv::Mat dirImage(grayImage.size(), CV_8U);
	uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

	// Iteration über alle Pixel (über das ganze Bild)
	for (int rows = 0; rows < imgDx.rows; rows++)
	{
		for (int cols = 0; cols < imgDx.cols; cols++)
		{
			double dx = imgDx.at<int16_t>(rows, cols);
			double dy = imgDy.at<int16_t>(rows, cols);
			double dr2 = dx * dx + dy * dy;
			// default Index = 0
			int index = 0;
			if (dr2 > gradThreshold * gradThreshold)
			{
				double alpha = atan2(dy, dx);
				index = 1 + (int)((alpha + M_PI) / (M_PI / 2) + (M_PI / 4));
				directionsImg.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);
			}
			dirImage.at<uint8_t>(rows, cols) = index;

			uint8_t maxValue = 0;
			uint8_t maxIndex = 0;

			for (int i0 = 0; i0 < 5; i0++)
			{
				if (bkgrModel[i0]->at<uint8_t>(rows, cols) > maxValue)
				{
					maxValue = bkgrModel[i0]->at<uint8_t>(rows, cols);
					maxIndex = i0;
				}

				if (i0 == index)
				{
					bkgrModel[i0]->at<uint8_t>(rows, cols) += 1;
					if (bkgrModel[i0]->at<uint8_t>(rows, cols) == bkgrThreshold + 1)
					{
						bkgrModel[i0]->at<uint8_t>(rows, cols) -= 1;
					}
				}
				else
				{
					bkgrModel[i0]->at<uint8_t>(rows, cols) -= 1;
					if (bkgrModel[i0]->at<uint8_t>(rows, cols) == 255)
					{
						bkgrModel[i0]->at<uint8_t>(rows, cols) += 1;
					}
				}
			}
			fgrModel.at<uint8_t>(rows, cols) = (maxValue > bkgrThreshold / 2) && (index != maxIndex);
		}
	}
	cv::Mat morphImg(grayImage.size(), CV_8U);
	cv::Mat morphImg2(grayImage.size(), CV_8U);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	cv::morphologyEx(fgrModel, morphImg, cv::MORPH_OPEN, element);
	cv::morphologyEx(morphImg, morphImg2, cv::MORPH_CLOSE, element);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(morphImg2, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat contourImg = colorImage.clone();
	for (unsigned int idx = 0; idx < contours.size(); idx++)
	{
		// area
		double area = cv::contourArea(contours[idx]);
		double minArea = 100;
		if (area > minArea)
		{

			//  center of mass
			cv::Moments moment = cv::moments(contours[idx]);
			double cx = moment.m10 / moment.m00;
			double cy = moment.m01 / moment.m00;
			// Zeichnen Sie den Schwerpunkt in das Bild
			cv::circle(contourImg, cv::Point2d(cx, cy), 3, cv::Scalar(255, 0, 0), 3);
			// to draw counter to index idx in image
			cv::drawContours(contourImg, contours, idx, cv::Scalar(0, 0, 255), 3);
		}
	}

	// Fgr = and((maxValue > maxCounter/2),(ImageBin+1 ~= maxIndex));

	*m_proc_image[0] = directionsImg;
	*m_proc_image[1] = fgrModel;
	*m_proc_image[2] = contourImg;

	// make deploy
	// make run
	int64 endTic = cv::getTickCount();
	double deltaTime = (double)(endTic - startTic) / cv::getTickFrequency();
	std::cout << "time:" << (int)(1000 * deltaTime) << " ms" << std::endl;
	return (SUCCESS);
}
