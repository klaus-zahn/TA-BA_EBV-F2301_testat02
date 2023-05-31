// 28.05.2023, Niklas Rösch

/*! @file image_processing.h
 * @brief Image Manipulation class
 */

#ifndef IMAGE_PROCESSING_H_
#define IMAGE_PROCESSING_H_

#include "opencv.hpp"

#include "includes.h"
#include "camera.h"



class CImageProcessor {
public:
	CImageProcessor();
	~CImageProcessor();
	
	int DoProcess(cv::Mat* image);

		
	cv::Mat mPrevImage;  // cv::Mat is an OpenCV object
	cv::Mat mBkgrImage;
	cv::Mat grayImage;
	cv::Mat colorImage;
	cv::Mat resultImage;
	


	cv::Mat* GetProcImage(uint32 i);

private:
	cv::Mat* m_proc_image[3];/* we have three processing images for visualization available */
	cv::Mat fgrModel;
	cv::Mat *bkgrModel[5];				// creating counters
	bool m_init;
};


#endif /* IMAGE_PROCESSING_H_ */
