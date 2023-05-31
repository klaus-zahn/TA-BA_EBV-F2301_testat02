#include "image_processing.h"

CImageProcessor::CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		/* index 0 is 3 channels and indicies 1/2 are 1 channel deep */
		m_proc_image[i] = new cv::Mat();
	}
	//cv::Mat* bkgrModel[6];
	for(uint32 i0 = 0; i0 < 6; i0++)
	{
		bkgrModel[i0] = new cv::Mat();
	}
}

CImageProcessor::~CImageProcessor()
{
	for (uint32 i = 0; i < 3; i++)
	{
		delete m_proc_image[i];
	}
	for(uint32 i0 = 0; i0 < 6; i0++)
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
	//cv::Mat* bkgrModel[6];
	if (!image) //kein Bild geladen? dann verlasse die Funktion
	{
		return (EINVALID_PARAMETER);
	}



	//instanziere die Klassen, bzw. erzeuge die Objekte
	cv::Mat grayImage;  //hier wird ein Objekt der Klasse Mat erstellt
	cv::Mat colorImage;
	cv::Mat resultImage; 
	cv::Mat binaryImage; 
	cv::Mat labelImage; 
	cv::Mat diffImage; 

	cv::Mat graySmooth;

	cv::Mat imgDx;
	cv::Mat imgDy;
	uint8 maxCounter = 50;

	uint8 lowlight = 0;  //um ein Pixel zu highlighten
	
	if((*bkgrModel[0]).size() != grayImage.size())
	{
		for(int i0 = 0; i0<6; i0++)
		{
			bkgrModel[i0] = new cv::Mat(grayImage.size(), CV_8U);
		}
	}

	double threshold = 60;

	uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

	//prüfe ob das Bild farbig oder SW ist
	if(image->channels() > 1) //das Bild ist farbig und hat 3 Kanäle
	{
		cv::cvtColor( *image, grayImage, cv::COLOR_BGR2GRAY); //BGR weil das die Reihenfolge ist, die tatsächlich in OpenCV drin ist irgendwie
		colorImage = image->clone();
	}
	else //das Bild ist SW und hat einen Kanal
	{
		grayImage = *image;
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2BGR);
	}



	//Weichzeichnen
	cv::blur(grayImage, graySmooth, cv::Size(5,5));

	//Ableitungen berechnen
	cv::Sobel(graySmooth, imgDx, CV_16S, 1, 0, 3, 1, 0); 
	cv::Sobel(graySmooth, imgDy, CV_16S, 0, 1, 3, 1, 0); //wir machen mit eine 16bit Konvertierung um genug Informationsbits zu haben, da bis zu 1020 (4*255) Werte verwendet werden können.

	cv::Mat dirImage(grayImage.size(), CV_8U);

	for(int rows = 0; rows < imgDx.rows; rows++)
	{
		for(int cols = 0; cols < imgDx.cols; cols++)
		{
			double dx = imgDx.at<int16_t>(rows, cols);
			double dy = imgDy.at<int16_t>(rows, cols);

			double dr2 = dx*dx + dy*dy;

			int index = 0;
			if(dr2 > threshold*threshold)
			{
				double alpha = atan2(dy, dx);
				index = 1 + (int) ((alpha + (0.75*M_PI))/(M_PI/2)); //Binning
				//TODO: noch 45 Grad verschieben

				colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index -1]);
			}

			dirImage.at<uint8>(rows,cols) = index;
			
			//da der Index hier bestimmt ist, wird jetzt damit die Statistik gemacht
			int low = 0;
			
			for(int i = 0; i<5; i++)
			{
				uint8 testnumber = bkgrModel[i]->at<uint8>(rows,cols); //nur zum gucken was drin ist :)

				if(i == index) //soll hochgezählt werden
				{
					if((bkgrModel[i]->at<uint8>(rows,cols)) < maxCounter)
					{
						bkgrModel[i]->at<uint8>(rows,cols)++;
					}
					else
					{
						bkgrModel[i]->at<uint8>(rows,cols) = maxCounter;
					}
				}
				else //soll runterzählt werden
				{
					if(bkgrModel[i]->at<uint8>(rows,cols) > 0)
					{
						bkgrModel[i]->at<uint8>(rows,cols)--;
					}
					else
					{
						bkgrModel[i]->at<uint8>(rows,cols) = 0;
					}
				}


				//falls der Wert hier über maxCounter/2 ist, kann er im bkgrModel[5] markieren und einen counter setzen um nicht mehr aufgerufen zu werden

				
			}
			//
			if ((*bkgrModel[index]).at<uint8>(rows, cols) > (maxCounter / 2))
			{

				(*bkgrModel[5]).at<uint8>(rows, cols) = 255;

			}
			else
			{
				(*bkgrModel[5]).at<uint8_t>(rows, cols) = 0;

			}
			// ich muss diesen Wert der über dem halben maxCounter ist mit dem aktuellen vergleichen, falls er dann nicht übereinstimmt, highlighten
 
		}

	}

	// Binearisierung

		//cv::threshold(*bkgrModel[5], binaryImage, 60, 255, cv::THRESH_BINARY);

	// Bildausgabe

	*m_proc_image[0] = colorImage; // Bildverarbeitung 1

	*m_proc_image[1] = dirImage; // Bildverarbeitung 2 //dirImage

	*m_proc_image[2] = *bkgrModel[5]; // Bildverarbeitung 3


	return (SUCCESS);
}



	/*
	if (!image)
		return (EINVALID_PARAMETER);

	cv::Mat grayImage;
	cv::Mat resultImage;
	cv::Mat binaryImage;
	cv::Mat labelImage;
	cv::Mat diffImage;
	int minArea = 100;

	if (image->channels() > 1) // wieviele Kanäle? farbig 3, SW 1
	{
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY);
	}
	else
	{
		grayImage = *image;
	}

	// Differenzbild

	if (mPrevImage.size() != cv::Size()) // überpringen wenn leer
	{
		cv::absdiff(mPrevImage, grayImage, diffImage);

		diffImage = cv::abs(mPrevImage - grayImage);

		// das hier sind Adressen, weil es Arrays sind

		// Binearisierung
		double threshold = 60;

		cv::threshold(diffImage, binaryImage, threshold, 255, cv::THRESH_BINARY);

		// Morphologie

		cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
		cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);

		// Region labeling

		cv::Mat stats, centroids;

		connectedComponentsWithStats(binaryImage, labelImage, stats, centroids);

		resultImage = image->clone();
		

		for (int i = 1; i < stats.rows; i++) // stats sind die erkannten objekte
		{
			int topLeftx = stats.at<int>(i, 0); // obere linke ecke des rechtecks
			int topLefty = stats.at<int>(i, 1);
			int width = stats.at<int>(i, 2);  // so breit ist das rechteck
			int height = stats.at<int>(i, 3); // so hoch ist das Rechteck

			int area = stats.at<int>(i, 4);

			double cx = centroids.at<double>(i, 0); // center of the Object
			double cy = centroids.at<double>(i, 1); // center of the Object

			if (minArea < area)
			{

				cv::Rect rect(topLeftx, topLefty, width, height);
				cv::rectangle(resultImage, rect, cv::Scalar(255, 0, 0)); // mark Rechteck

				cv::Point2d cent(cx, cy);
				cv::circle(resultImage, cent, 5, cv::Scalar(128, 0, 0), -1); // mark center of the Object
			}
		}
	}

	mPrevImage = grayImage.clone();

	// Bildausgabe

	*m_proc_image[0] = labelImage; // Bildverarbeitung 1

	*m_proc_image[1] = binaryImage; // Bildverarbeitung 2

	*m_proc_image[2] = resultImage; // Bildverarbeitung 3

	return (SUCCESS);

	// unnötige Codeschnipsel

	// cv::subtract(cv::Scalar::all(255), *image,*m_proc_image[0]);
	// cv::subtract(cv::Scalar::all(255), grayImage, *m_proc_image[1]);

	//  cv::imwrite("dx.png", *m_proc_image[0]);
	//  cv::imwrite("dy.png", *m_proc_image[1]);
	*/