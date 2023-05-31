

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












/*********************************************************************************************************
 * 	
 *  	TESTAT - EBV - FS23
 *  	BY ODERMATT MARCO
 * 	
 * 		DOES DO THINGS WITH IMAGES
 * 		BINNING AND STUFF OFC.
 * 		
 * 
*********************************************************************************************************/


int CImageProcessor::DoProcess(cv::Mat* image) {
	//check for error
	if(!image) return(EINVALID_PARAMETER);	


// ---------------------------
// ---- Prepare Data ---------
// ---------------------------
	cv::Mat grayImage;
	cv::Mat colorImage;

	// make a grayscale image regardless of the image
	if(image->channels() > 1) {
		cv::cvtColor(*image, grayImage, cv::COLOR_BGR2GRAY);	//if color image, convert to grayscale img
		colorImage = image->clone();
	}
	else {
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2BGR);
	}


// ---------------------------
// ---- Init -----------------
// ---------------------------
	cv::Mat graySmooth;	 //for TP-Filtering
	cv::Mat imgDx, imgDy;  //for partial derivation
	int threshold = 30;  //threshhold value for edge detection
	cv::Mat dirImage(grayImage.size(), CV_8U);	//for Binning
	uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};  //colors for showing Binning in color
	uint8 MaxIndex = 50;  //max value of background counter
	cv::Mat fgrImage = grayImage.clone();  //foreground image
	cv::Mat detectImage = colorImage.clone();  //foreground image
	int MinArea = 10*10;  //min size for drawing bounding box

	//measure time of Processing
	int64 startTic = cv::getTickCount();
// ---------------------------
// ---- Do the Processing ----
// ---------------------------
	if(mPrevImage.size() != cv::Size())	//is data fine? has it allready passed once?$
	{
		// -- do all the processing in here --

	// ---- TP-Filtering ----
		cv::blur(grayImage, graySmooth, cv::Size(5,5));	//create smooth image with filter of 5x5

	// ---- partial derivation ----
		//cv::Sobel(graySmooth, imgDx, -1, 1, 0, 3, 1, 128);
		//cv::Sobel(graySmooth, imgDy, -1, 0, 1, 3, 1, 128);
		cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0, 3, 1, 0);  //we rather take it in 16bit to avoid saturation
		cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1);  //simplified through default values
		
	// ---- binning ----
		for(int rows = 0; rows < imgDx.rows; rows++) {
			for(int cols = 0; cols < imgDx.cols; cols++) {
				double dx = imgDx.at<int16_t>(rows, cols);
				double dy = imgDy.at<int16_t>(rows, cols);
				double dr2 = dx*dx + dy*dy;
				int index = 0;
				//find the fitting bin
				if(dr2 > threshold*threshold)  //is value significant in magnitude?
				{	
					double alpha = atan2(dy, dx);

					//rearrange alpha to shift 45deg + arrange bin in increasing order
					alpha += (M_PI * 1/4);
					if(alpha < 0)
					{
						alpha += (2 * M_PI);
					}
					index = 1 + (int) ((alpha) / (M_PI/2 ));  //sort out bins of the size Pi/2

					//set corresponding color pixel to color value of bin
					colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);  //if inside if - background is colored
				}
				//set corresponding pixel to value of bin
				dirImage.at<uint8>(rows, cols) = index;  //if outside if - background is Black


			//--- update background values ---
				//decrement all counters
				if( bkgrModel[0]->at<uint8>(rows, cols) >= 1 )
					bkgrModel[0]->at<uint8>(rows, cols) --;
				if( bkgrModel[1]->at<uint8>(rows, cols) >= 1 )
					bkgrModel[1]->at<uint8>(rows, cols) --;
				if( bkgrModel[2]->at<uint8>(rows, cols) >= 1 )
					bkgrModel[2]->at<uint8>(rows, cols) --;
				if( bkgrModel[3]->at<uint8>(rows, cols) >= 1 )
					bkgrModel[3]->at<uint8>(rows, cols) --;
				if( bkgrModel[4]->at<uint8>(rows, cols) >= 1 )
					bkgrModel[4]->at<uint8>(rows, cols) --;

				//increment specific counter
				if( bkgrModel[index]->at<uint8>(rows, cols) < MaxIndex )
					bkgrModel[index]->at<uint8>(rows, cols) += 2;
				
				//find max bin value  //#todo make this part more efficient
				uint8 maxBin;
				if ( bkgrModel[0]->at<uint8>(rows, cols) < bkgrModel[1]->at<uint8>(rows, cols) )
				{
					if ( bkgrModel[1]->at<uint8>(rows, cols) < bkgrModel[2]->at<uint8>(rows, cols) )
					{
						if ( bkgrModel[2]->at<uint8>(rows, cols) < bkgrModel[3]->at<uint8>(rows, cols) )
						{
							if ( bkgrModel[3]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 3;
							}
						}
						else
						{
							if ( bkgrModel[2]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 2;
							}
						}
					}
					else
					{
						if ( bkgrModel[1]->at<uint8>(rows, cols) < bkgrModel[3]->at<uint8>(rows, cols) )
						{
							if ( bkgrModel[3]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 3;
							}
						}
						else
						{
							if ( bkgrModel[1]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 1;
							}
						}
					}
				}else
				{
					if ( bkgrModel[0]->at<uint8>(rows, cols) < bkgrModel[2]->at<uint8>(rows, cols) )
					{
						if ( bkgrModel[2]->at<uint8>(rows, cols) < bkgrModel[3]->at<uint8>(rows, cols) )
						{
							if ( bkgrModel[3]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 3;
							}
						}
						else
						{
							if ( bkgrModel[2]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 2;
							}
						}
					}
					else
					{
						if ( bkgrModel[0]->at<uint8>(rows, cols) < bkgrModel[3]->at<uint8>(rows, cols) )
						{
							if ( bkgrModel[3]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 3;
							}
						}
						else
						{
							if ( bkgrModel[0]->at<uint8>(rows, cols) < bkgrModel[4]->at<uint8>(rows, cols) )
							{
								maxBin = 4;
							}
							else
							{
								maxBin = 0;
							}
						}
					}
				}
				
				//draw foreground pixels
				uint8 currentBin = index;
				fgrImage.at<uint8>(rows, cols) = ( ( bkgrModel[maxBin]->at<uint8>(rows, cols) > (MaxIndex/2) ) && (currentBin != maxBin) );
			}		 
		} //foreach bit end
		
	// ---- morphologic operations ----
		cv::Mat kernel;
		kernel = cv::Mat::ones(5, 5, CV_8UC1);
		cv::morphologyEx(fgrImage, fgrImage, cv::MORPH_OPEN, kernel);
		kernel = cv::Mat::ones(20, 20, CV_8UC1);
		cv::morphologyEx(fgrImage, fgrImage, cv::MORPH_CLOSE, kernel);
		
	// ---- contour ----
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours( fgrImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
		for(unsigned int idx = 0 ; idx < contours.size(); idx++ ) {
			//area
			double area = cv::contourArea(contours[idx]);
			if(area > MinArea)
			{
				//bounding rectangle
				cv::Rect rect = cv::boundingRect(contours[idx]);
				//center of gravity
				// center of mass
				cv::Moments moment = cv::moments(contours[idx]);
				double cx = moment.m10 / moment.m00;
				double cy = moment.m01 / moment.m00;
				//to draw counter to index idx in image
				cv::drawContours(detectImage, contours, idx, cv::Scalar(255), 1, 8 );
				//draw center of mass #todo exclude drawing, if it is on a border
				cv::drawMarker(detectImage, cv::Point(cx,cy), cv::Scalar(255), 1, 10, 3, 8);
				//detectImage.at<cv::Vec3b>(cx,cy) = {255, 0, 0};
				/*
				detectImage.at<cv::Vec3b>(cx-1,cy-1) = {0, 255, 255};
				detectImage.at<cv::Vec3b>(cx-1,cy+1) = {0, 255, 255};
				detectImage.at<cv::Vec3b>(cx+1,cy-1) = {0, 255, 255};
				detectImage.at<cv::Vec3b>(cx+1,cy+1) = {0, 255, 255};
				*/
			}
		}


	// ---- processXY ----
		


	}	
	else	//if the first time
	{
		for(int i0 = 0; i0 < 5; i0++)
		{
			bkgrModel[i0] = new cv::Mat(grayImage.size(), CV_8U);
			bkgrModel[i0]->setTo((MaxIndex/2));
/*
			for(int rows = 0; rows < imgDx.rows; rows++) {
				for(int cols = 0; cols < imgDx.cols; cols++) {
					bkgrModel[i0]->at<uint8>(rows, cols) = 24;
				}
			}
*/			
		}

		//#todo output a black image for the output if it was the first time
		//diffImage = grayImage * 0;		//output at least a black image with correct pixel dimensions
	}

// ---------------------------
// ---- Age and save ---------
// ---------------------------
	mPrevImage = grayImage.clone();		//generate image for next round

// ---------------------------
// ---- Output ---------------
// ---------------------------
	*m_proc_image[0] = colorImage;
	*m_proc_image[1] = fgrImage;
	*m_proc_image[2] = detectImage;

// ---------------------------
// ---- Time Measurement -----
// ---------------------------
	int64 endTic = cv::getTickCount();

	double deltaTime = (double) ( endTic - startTic )/cv::getTickFrequency();
	std::cout << "time:" << (int) (1000*deltaTime) << " ms" << std::endl;



	
















		//#todo finishing touches
		//modify diff image so noise is not present
		//diffImage.at<uint8>(0,0)=255;		//set one pixel to max value
		//diffImage.at<uint8>(1,0)=0;			//set one pixel to min value

	// ---- 3) Visualisierung übungen ---
		//*m_proc_image[0] = labelImage;
		//*m_proc_image[1] = binaryImage;
		//*m_proc_image[2] = resultImage;

















		/*		UNUSED CODE

		#define USE_SIMPLECONTOUR (1)
		// ---- Testat gedöns ----
#if (USE_SIMPLECONTOUR)
			double alpha = 0.9;
			cv::addWeighted(mBkgrImage, alpha, grayImage, 1-alpha, 0, mBkgrImage);
#else
			double alpha = 0.9;
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
				cv::drawContours(resultImage, contours, idx, cv::Scalar(255), 1, 8 );
			}
#endif



// code for Processinc

		// ---- a) Difference Image ----
			cv::Mat diffImage;
			cv::absdiff(grayImage, mPrevImage, diffImage);

		// ---- b) Binarisierung ----
			double threshold = 50;
			cv::threshold(diffImage, binaryImage, threshold, 255, cv::THRESH_BINARY);

		// ---- c) Morphologie ----
			cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
			cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

		// ---- d) Region Labeling ----
			cv::Mat stats, centroids;
			connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

			//access values from Labeled Images data
			for (int i = 1; i < stats.rows; i++) {
			int topLeftx = stats.at<int>(i, 0);
			int topLefty = stats.at<int>(i, 1);
			int width = stats.at<int>(i, 2);
			int height = stats.at<int>(i, 3);
			int area = stats.at<int>(i, 4);
			double cx = centroids.at<double>(i, 0);
			double cy = centroids.at<double>(i, 1);

			//draw bounding box over image copy
			cv::Rect rect(topLeftx, topLefty, width, height);
			cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0));
			//draw centerPoint onto image
			cv::Point2d cent(cx, cy);
			cv::circle(resultImage, cent, 5, cv::Scalar(128, 0, 0), -1);
			}	//end Labeling

*/





	return(SUCCESS);
}









