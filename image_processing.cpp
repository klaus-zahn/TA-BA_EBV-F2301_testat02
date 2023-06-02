//ZaK
//code crashed, some parts are missing

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
	// Pictures
	static cv::Mat grayImage;  // Graubild
	static cv::Mat colorImage; // Farbbild
	static cv::Mat graySmooth; // Geglättetesbild
	static cv::Mat imgDx;	   // Ableitungs-Bild
	static cv::Mat imgDy;	   // Ableitungs-Bild
	static cv::Mat dirImage;   //
	static cv::Mat bgImage;	   // Hintergrundbild
	static cv::Mat fgImage;	   // Fordergrundbild
	uint8 bkgrModel[5][400][600];
	uint8 colorMap[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};

	// Variablen
	int threshold = 20;
	uint8 BKGmax = 50;
	// static cv::Mat mPrevImage;
	// static cv::Mat diffImage;
	// static cv::Mat binaryImage;
	// static cv::Mat labelImage;
	// static cv::Mat resultImage;
	// static cv::Mat morphImage;
	// double threshold = 20.0;
	// int minArea = 100;

	if (!image)
		return (EINVALID_PARAMETER);

	if (image->channels() > 1) // Kontrolle ob es ein Farbbild ist
	{
		cv::cvtColor(*image, grayImage, cv::COLOR_RGB2GRAY); // Farbbild in Graubild umwandeln
		colorImage = image->clone();
	}
	else
	{
		grayImage = *image; // Graubild behalten
		cv::cvtColor(*image, colorImage, cv::COLOR_GRAY2RGB);
	}
	cv::blur(grayImage, graySmooth, cv::Size(5, 5)); // Tiefpassfilter
	cv::Sobel(grayImage, imgDx, CV_16S, 1, 0, 3, 1, 0);
	cv::Sobel(grayImage, imgDy, CV_16S, 0, 1, 3, 1, 0);
	// cv::Sobel(graySmooth, imgDx, -1, 1, 0, 3, 1, 128); //Ableitung Berechnen und in imgDx Ablegen
	// Eingabebild
	// Ausgabebild
	// Format des Ausgabebildes: -1 / CV_16S 16Bit damit das bild nicht statturiert
	// Ordnung der Ableitung in x: 1
	// Ordnung der Ableitung in y: 1
	// Grösse der Filtermaske: 3
	// Skalierung des Ausgabenbildes: 1
	// Additiver Term für Ausgabebild: 128
	dirImage = cv::Mat(grayImage.size(), CV_8U);
	for (int rows = 0; rows < imgDx.rows; rows++)
	{
		for (int cols = 0; cols < imgDx.cols; cols++)
		{
			double dx = imgDx.at<int16_t>(rows, cols);
			double dy = imgDy.at<int16_t>(rows, cols);
			double dr2 = dx * dx + dy * dy;
			uint8 index = 0;
			if (dr2 > threshold * threshold)
			{
				double alpha = atan2(dy, dx);
				// index = 1 + (uint8) (((alpha+CV_PI) / (CV_PI/2)));
				alpha = alpha + CV_PI;
				if ((alpha <= (CV_PI / 4)) | (alpha >= (7 * CV_PI / 4)))
				{
					index = 1;
				}
				else if (alpha <= (3 * CV_PI / 4))
				{
					index = 2;
				}
				else if (alpha <= (5 * CV_PI / 4))
				{
					index = 3;
				}
				else if (alpha <= (7 * CV_PI / 4))
				{
					index = 4;
				}
				colorImage.at<cv::Vec3b>(rows, cols) = cv::Vec3b(colorMap[index - 1]);
			}
			dirImage.at<uint8>(rows, cols) = index;
			for (int i0 = 0; i0 < 5; i0++)
			{
				if (index == i0)
				{
					if (bkgrModel[i0][rows][cols] <= BKGmax)
					{
						bkgrModel[i0][rows][cols]=bkgrModel[i0][rows][cols] + 1; // Counter erhöhen
					}
					else
					{
						bkgrModel[i0][rows][cols] = BKGmax; // Max Wert
					}
				}
				else
				{
					if (bkgrModel[i0][rows][cols] >= 1)
					{
						bkgrModel[i0][rows][cols]= bkgrModel[i0][rows][cols] - 1; // Counter reduzieren
					}
					else
					{
						bkgrModel[i0][rows][cols] = 0; // min Wert ist 0
					}
				}
			}
			if (bkgrModel[0][rows][cols] > bkgrModel[1][rows][cols])
			{
				if (bkgrModel[0][rows][cols] > bkgrModel[2][rows][cols])
				{
					if (bkgrModel[0][rows][cols] > bkgrModel[3][rows][cols])
					{
						if (bkgrModel[0][rows][cols] > bkgrModel[4][rows][cols])
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 0;
						}
						else
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 4;
						}
					}
					else
					{
						if (bkgrModel[3][rows][cols] > bkgrModel[4][rows][cols])
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 3;
						}
						else
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 4;
						}
					}
				}
				else
				{
					if (bkgrModel[2][rows][cols] > bkgrModel[3][rows][cols])
					{
						if (bkgrModel[2][rows][cols] > bkgrModel[4][rows][cols])
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 2;
						}
						else
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 4;
						}
					}
					else
					{
						if (bkgrModel[3][rows][cols] > bkgrModel[4][rows][cols])
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 3;
						}
						else
						{
							bgImage.at<uint8>(rows, cols) = (uint8) 4;
						}
					}
				}
			}
			else if (bkgrModel[1][rows][cols] > bkgrModel[2][rows][cols])
			{
				if (bkgrModel[1][rows][cols] > bkgrModel[3][rows][cols])
				{
					if (bkgrModel[1][rows][cols] > bkgrModel[4][rows][cols])
					{
						bgImage.at<uint8>(rows, cols) = (uint8) 1;
					}
					else
					{
						bgImage.at<uint8>(rows, cols) = (uint8) 4;
					}
				}
				else
				{
					if (bkgrModel[3][rows][cols] > bkgrModel[4][rows][cols])
					{
						bgImage.at<uint8>(rows, cols) = (uint8) 3;
					}
					else
					{
						bgImage.at<uint8>(rows, cols) = (uint8) 4;
					}
				}
			}
			else if (bkgrModel[2][rows][cols] > bkgrModel[3][rows][cols])
			{
				if (bkgrModel[2][rows][cols] > bkgrModel[4][rows][cols])
				{
					bgImage.at<uint8>(rows, cols) = (uint8) 2;
				}
				else
				{
					bgImage.at<uint8>(rows, cols) = (uint8) 4;
				}
			}
			else if (bkgrModel[3][rows][cols] > bkgrModel[4][rows][cols])
			{
				bgImage.at<uint8>(rows, cols) = (uint8) 3;
			}
			else
			{
				bgImage.at<uint8>(rows, cols) = (uint8) 4;
			}

			fgImage = dirImage-bgImage; // Vordergrundbild
			
		}
	}
	
	*m_proc_image[0] = bgImage;
	*m_proc_image[1] = colorImage;
	*m_proc_image[2] = dirImage;
	return (SUCCESS);
}