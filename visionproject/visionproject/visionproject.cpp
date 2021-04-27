#include "stdafx.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include <vector> 
#include<cmath>
#include <iostream>
#include <list> 
#include <iterator>
#include <random> 
#include <ctime>
#include <math.h>
#include <cmath>

using namespace cv;
using namespace std;



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////Filters////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// filtre médian
Mat filtreMedianNVG(Mat src, int voisinage) {
	Mat dst = src.clone();
	if (src.channels() != 1 || dst.channels() != 1) return src;
	if (src.rows != dst.rows || src.cols != dst.cols) return src;
	if (voisinage % 2 != 1) return src;
	Rect roi = Rect(0, 0, voisinage, voisinage);
	int *voisins = new int[voisinage*voisinage]; // un tableau pour le calcul de la médiane
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			if (x< (voisinage - 1) / 2 || x>(src.rows - 1 - (voisinage - 1) / 2) || y< (voisinage - 1) / 2 
				|| y>(src.cols - 1 - (voisinage - 1) / 2)) // pour le bord copier les mêmes valeurs des pixels
			{
				dst.at<uchar>(x, y) = src.at<uchar>(x, y);
			}
			else
			{
				// on centre le voisinage sur le pixel en cours
				roi.y = x - (voisinage - 1) / 2;
				roi.x = y - (voisinage - 1) / 2;
				Mat img_roi = src(roi);
				for (int i = 0; i < voisinage; i++) {
					for (int j = 0; j < voisinage; j++) {
						voisins[i*voisinage + j] = img_roi.at<uchar>(i, j);
					}
				}
				sort(voisins, voisins + (voisinage*voisinage)); // on classe les valeurs
				dst.at<uchar>(x, y) = voisins[(voisinage - 1) / 2 + 1]; // on choisit la valeur médiane
			}
		}
	}
	return dst;
}

//filtre moyen
Mat filtreMoyenNVG(Mat src, int voisinage) {
	Mat dst = src.clone();
	if (src.channels() != 1 || dst.channels() != 1) return src;// vérifier que c’est en niveau de gris
	if (src.rows != dst.rows || src.cols != dst.cols) return src;// même dimensions
																 //on vérifie que le voisinage est impair sinon on le corrige
	if (voisinage % 2 != 1) return src;
	Rect rec_roi = Rect(0, 0, voisinage, voisinage);// on initialise un carré pour la ROI
	int moyenne = 0;
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			if (x< (voisinage - 1) / 2 || x>(src.rows - 1 - (voisinage - 1) / 2) || y< (voisinage - 1) / 2 
				|| y>(src.cols - 1 - (voisinage - 1) / 2)) // pour le bord copier les mêmes valeurs des pixels
			{
				dst.at<uchar>(x, y) = src.at<uchar>(x, y); // pour le bord copier les mêmes valeurs des pixels
			}
			else
			{
				moyenne = 0;
				rec_roi.y = x - (voisinage - 1) / 2; // on centre le voisinage sur le pixel en cours
				rec_roi.x = y - (voisinage - 1) / 2;
				Mat img_roi = src(rec_roi);// on initialise la région d’intérêt
				for (int i = 0; i < voisinage; i++) { // on calcule la somme du voisinage
					for (int j = 0; j < voisinage; j++) {
						moyenne += img_roi.at<uchar>(i, j);
					}
				}
				moyenne /= voisinage*voisinage;// la moyenne
				dst.at<uchar>(x, y) = moyenne;
			}
		}
	}
	return dst;
}

//erodation
Mat Erodation(Mat src) {
	Mat dist;
	int erode_type = 2;
	int erode_size = 12;
	Mat element1 = getStructuringElement(erode_type, Size(2 * erode_size + 1, 2 * erode_size + 1));
	erode(src, dist, element1);
	return dist;
}

//delatation
Mat delatation(Mat src) {
	Mat dist;
	dilate(src, dist, Mat(), Point(-1, -1), 2, 1, 1);
	return dist;
}

//filtre gaussian
Mat gaussian(Mat src) {
	Mat dist;
	Mat gaussien;
	Point anchor;
	double delta;
	int ddepth;
	anchor = Point(-1, -1);
	delta = 0; //1 2 1 // 0 -1 0
	ddepth = -1; //2 4 2 //-1 4 -1
				 //1 2 1 // 0 -1 0
	gaussien = Mat::zeros(3, 3, CV_32FC1);
	gaussien.at<float>(0, 1) = (float)-1;
	gaussien.at<float>(1, 0) = (float)-1;
	gaussien.at<float>(1, 2) = (float)-1;
	gaussien.at<float>(2, 1) = (float)-1;
	gaussien.at<float>(1, 1) = (float)4;
	filter2D(src, dist, ddepth, gaussien, anchor, delta, BORDER_DEFAULT);
	return dist;
	//dist qui contient le résultat

}

//filtre laplacian
Mat Laplacian(Mat src) {
	Mat abs_dst;
	Mat src_gray, dst;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	// Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(src, src_gray, COLOR_BGR2GRAY); // Convert the image to grayscale
	Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(dst, abs_dst);
	return abs_dst;
}

//filtre gradient
Mat GradientX(Mat src) {
	Mat dstx;
	Sobel(src, dstx, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	return dstx;
}

Mat GradientY(Mat src) {
	Mat dsty;
	Sobel(src, dsty, CV_8U, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	return dsty;
}


Mat filtre(Mat frame) {
	for (int x = 0; x < frame.rows; x++) {
		for (int y = 0; y < frame.cols; y++) {
			if ((int)frame.at<uchar>(x, y) >= 200) {
				frame.at<uchar>(x, y) = (uchar)255;
			}
			if ((int)frame.at<uchar>(x, y) < 200) {
				frame.at<uchar>(x, y) = (uchar)0;
			}
		}
	}
	return frame;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////Generation de video////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int randInt() {
	unsigned long j;
	srand((unsigned)time(NULL));
	return rand();
}
int collisionSystem(int pos[8][6], int objType[8], int index, int nbrObj) {
	// if there is a object in a collision path with this object return its id
	int centerA[2] = { 0, 0 };
	int maxRadiusA = 0;
	if (objType[index] == 0)
	{
		centerA[0] = pos[index][0];centerA[1] = pos[index][1];maxRadiusA = sqrt(2 * pow(pos[index][4], 2)) + 25;
	}
	else {
		centerA[0] = (pos[index][0] + pos[index][4] / 2);centerA[1] = (pos[index][1] + pos[index][5] / 2);
		if ((pos[index][4]) >= (pos[index][5])) maxRadiusA = pos[index][4] / 2 + 25;
		else maxRadiusA = pos[index][5] / 2 + 25;
	}
	for (int i = 0; i < nbrObj; i++) {
		if (i == index) continue;
		int centerB[2] = { 0, 0 };
		int maxRadiusB = 0;
		if (objType[i] == 0)
		{
			centerB[0] = pos[i][0];centerB[1] = pos[i][1];maxRadiusB = sqrt(2 * pow(pos[i][4], 2)) + 25;
		}
		else {
			centerB[0] = (pos[i][0] + pos[i][4] / 2);centerB[1] = (pos[i][1] + pos[i][5] / 2);
			if ((pos[i][4]) >= (pos[i][5])) maxRadiusB = pos[i][4] / 2 + 25;
			else maxRadiusB = pos[i][5] / 2 + 25;
		}

		int dist = sqrt(pow(centerA[0] - centerB[0], 2) + pow(centerA[1] - centerB[1], 2));
		if ((maxRadiusA + maxRadiusB) >= dist)
			return i;
	}
	return -1;
}

void triangle(Mat inputImage, int x, int y, int width, int length, Scalar bg) {
	vector<Point> vPolygonPoint;
	vPolygonPoint.push_back(Point(x, y));
	vPolygonPoint.push_back(Point(x + width, y));
	vPolygonPoint.push_back(Point(x + (int)(width / 2), y + length));
	vector<vector<Point>> contours;
	contours.push_back(vPolygonPoint);
	fillPoly(inputImage, contours, bg, 8);
}

void generateVideo(double fps,double length,int frameRes,double nbrNoise,int noiseRefreshRate,int meanX,
	int meanY,int nbrOfObjects,int borderSize,bool pointsNoise) {
	int bg[3] = { 255, 255, 255 };
	int objType[9] = { 0, 0, 1, 1, 1, 1, 2, 2, 2 };
	int pos[9][6] = {
		{ 030, 070, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 30, 0 },
		{ 020, frameRes - 50, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 30, 0 },
		{ frameRes - 10, 15, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 50, 50 },
		{ frameRes - 55, frameRes - 25, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 50, 50 },
		{ 025, frameRes / 2, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 50, 30 },
		{ frameRes - 20, frameRes / 2, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 50, 30 },
		{ frameRes / 2, 025, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 50, 50 },
		{ 220, frameRes - 250, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 50, 50 },
		{ frameRes / 2, frameRes - 25, (meanX - (int)(randInt() % (meanX / 2))), (meanY - (int)(randInt() % (meanY / 2))), 50, 30 }
	};
	int objBg[9][3] = {
		{ 200, 200, 200 },{ 205, 205, 0 },{ 205, 155, 100 },{ 205, 0, 205 },{ 0, 205, 205 },{ 205, 0, 0 },
		{ 0, 205, 0 },{ 0, 0, 205 },{ 100, 100, 100 },
	};
	namedWindow("camera", WINDOW_AUTOSIZE);
	VideoWriter video("C://Users//ASUS//Desktop//visionvideo//sample25.avi", VideoWriter::fourcc('R', 'G', 'B', 'A'), (double)fps, 
		Size(frameRes, frameRes));
	Mat noise = Mat(frameRes, frameRes, CV_8UC3, Scalar(bg[0], bg[1], bg[2]));
	int c = 0; // counter of generated and printed frames
	for (;;) {
		Mat frame = Mat(frameRes, frameRes, CV_8UC3, Scalar(bg[0], bg[1], bg[2]));
		if (frame.empty() || c == length * fps) break;
		if (c % noiseRefreshRate == 0) {
			noise = Mat(frameRes, frameRes, CV_8UC3, Scalar(bg[0], bg[1], bg[2]));
			for (int y = 0; y < nbrNoise; y++) {// for each row
				if (pointsNoise) {
					int pix = rand() % noise.cols;int piy = rand() % noise.rows;
					Vec3b color = noise.at<Vec3b>(Point(pix, piy));
					color[0] = rand() % 255;color[1] = rand() % 255;color[2] = rand() % 255;noise.at<Vec3b>(Point(pix, piy)) = color;
				}
				else {
					int type = (rand() + y) % 3;
					int xN = (int)((rand()*y * 2) % (frame.cols - 2 * borderSize - 10)) + (borderSize);
					int yN = (int)((rand()) % (frame.rows - 2 * borderSize - 10)) + (borderSize);
					int c1 = (int)((rand()*y) % 150);
					int c2 = (int)((rand() * 2 * y) % 150);
					int c3 = (int)((rand() * 3 * y) % 150);
					if (type == 0)circle(noise, Point(xN, yN), 3, Scalar(c1, c2, c3), FILLED);
					else if (type == 1) rectangle(noise, Rect(xN, yN, 10, 10), Scalar(c1, c2, c3), FILLED);
					else triangle(noise, xN, yN, 10, 10, Scalar(c1, c2, c3));
				}
			}
		}
		for (int y = 0; y < noise.rows; y++) {// for each row
			for (int x = 0; x < noise.cols; x++) { // turn some pixals into noise
				frame.at<Vec3b>(Point(x, y)) = noise.at<Vec3b>(Point(x, y));
			}
		}
		bool objRedirected[sizeof(objType)];
		for (int i = 0; i < nbrOfObjects; i++) objRedirected[i] = false;
		for (int i = 0; i < nbrOfObjects; i++) {
			pos[i][0] += pos[i][2];
			pos[i][1] += pos[i][3];
			if (objType[i] == 0) { // circle
				if (pos[i][0] <= pos[i][4] + borderSize) {
					pos[i][0] = pos[i][4] + borderSize + 1;
					pos[i][2] *= -1;
				}
				else if ((pos[i][0] + pos[i][4]) >= frame.cols - borderSize) {
					pos[i][0] = frame.cols - pos[i][4] - borderSize - 1;
					pos[i][2] *= -1;
				}
				if (pos[i][1] <= pos[i][4] + borderSize) {
					pos[i][1] = pos[i][4] + borderSize + 1;
					pos[i][3] *= -1;
				}
				else if ((pos[i][1] + pos[i][4]) >= frame.rows - borderSize) {
					pos[i][1] = frame.rows - pos[i][4] - borderSize - 1;
					pos[i][3] *= -1;
				}
			}
			else { // rectangle
				if (pos[i][0] <= borderSize) {
					pos[i][0] = borderSize + 1;
					pos[i][2] *= -1;
				}
				else if ((pos[i][0] + pos[i][4]) >= frame.cols - borderSize) {
					pos[i][0] = frame.cols - pos[i][4] - borderSize - 1;
					pos[i][2] *= -1;
				}
				if (pos[i][1] <= borderSize) {
					pos[i][1] = borderSize + 1;
					pos[i][3] *= -1;
				}
				else if ((pos[i][1] + pos[i][5]) >= frame.rows - borderSize) {
					pos[i][1] = frame.rows - pos[i][5] - borderSize - 1;
					pos[i][3] *= -1;
				}
			}
			if (!objRedirected[i])
			{
				int inCollisionPath = collisionSystem(pos, objType, i, nbrOfObjects);
				if (inCollisionPath != -1)
				{
					pos[i][2] *= -1;
					pos[i][3] *= -1;
					pos[i][0] += 2 * pos[i][2];
					pos[i][1] += 2 * pos[i][3];

				}
				objRedirected[i] = true;
			}
			if (objType[i] == 0)
				circle(frame, Point(pos[i][0], pos[i][1]), pos[i][4], Scalar(objBg[i][0], objBg[i][1], objBg[i][2]), FILLED);
			else if (objType[i] == 1)
				rectangle(frame, Rect(pos[i][0], pos[i][1], pos[i][4], pos[i][5]), Scalar(objBg[i][0], objBg[i][1], objBg[i][2]), FILLED);
			else
				triangle(frame, pos[i][0], pos[i][1], pos[i][4], pos[i][5], Scalar(objBg[i][0], objBg[i][1], objBg[i][2]));
		}
		video.write(frame);
		if (waitKey((1 / fps) * 1000) == 27) break;
		imshow("camera", frame);
		c++;
	}
	video.release();
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////Shape detection////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int SquareDetection(int x, int y, Mat frame, Mat src) {
	int line = x; int col = y;
	//avancer dans les lignes
	int longueurligne = -1;
	while ((col - 2) > 0 && (int)frame.at<uchar>(x, col) == 255 && (int)frame.at<uchar>(x, col - 2) == 0) {
		x++;
		longueurligne++;
	}
	//retourner au dernier point blanc
	x = x - 2;
	//avancer dans les colonnes
	int longueurCol = -1;
	while ((line - 2) > 0 && (int)frame.at<uchar>(line, y) == 255 && (int)frame.at<uchar>(line - 2, y) == 0) {
		y++;
		longueurCol++;
	}
	//retourner au dernier point blanc
	y = y - 2;
	//verifier les cotés du carrés si sont égales
	if (longueurligne == longueurCol && longueurligne > 5) {
		String display = "Square ((" + to_string(line) + "," + to_string(col)+ "),"
			+"(" + to_string(line)+ "," + to_string(y) + "),"
			+ "(" + to_string(x) + "," + to_string(col) + "),"
			+ "(" + to_string(x) + "," + to_string(y) + ")),"
			+ "(" + to_string(src.at<Vec3b>(line + 5, col + 5)[2])
			+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[1])
			+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[0]) + ")"
			;
		putText(src, display, Point(col, line), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0));
		return 1;
	}
	return 0;
}

int RectDetection(int x, int y, Mat frame, Mat src) {
	int line = x; int col = y;
	//avancer dans les lignes
	int longueurligne = -1;
	while ((col - 2) > 0 && (int)frame.at<uchar>(x, col) == 255 && (int)frame.at<uchar>(x, col - 2) == 0) {
		x++;longueurligne++;
	}
	//retourner au dernier point blanc
	x = x - 2;
	//avancer dans les colonnes
	int longueurCol = -1;
	while ((line - 2) > 0 && (int)frame.at<uchar>(line, y) == 255 && (int)frame.at<uchar>(line - 2, y) == 0) {
		y++;longueurCol++;
	}
	//retourner au dernier point blanc
	y = y - 2;
	//verifier que le point d n'est pas blanc
	if (longueurligne != longueurCol && longueurligne > 5 && longueurCol>5) {

		String display = "Rectangle ((" + to_string(line) + "," + to_string(col) + "),"
			+ "(" + to_string(line) + "," + to_string(y) + "),"
			+ "(" + to_string(x) + "," + to_string(col) + "),"
			+ "(" + to_string(x) + "," + to_string(y) + ")),"
			+ "(" + to_string(src.at<Vec3b>(line+5, col+5)[2])
			+ "," + to_string(src.at<Vec3b>(line+5, col+5)[1])
			+ "," + to_string(src.at<Vec3b>(line+5, col+5)[0])+")"
			;
		putText(src, display, Point(col, line), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0));
		return 1;
	}
	return 0;
}

int CircleDetection(int x, int y, Mat frame, Mat src) {
	int line = x;
	int col = y;
	int startline = x+1;
	// avancer avec 1 pixels en bas pour eviter le problème de bruit dans le circle
	if (startline > 0  && (col+5)>0 && (int)frame.at<uchar>(x, col+5) == 0) {
		//avancer dans les lignes
		int longueurligne = -1;
		while ((col - 1) > 0 && (int)frame.at<uchar>(startline, col) == 255 &&  (int)frame.at<uchar>(startline, col - 1) == 255 
			&& (col + 1) > 0 && (int)frame.at<uchar>(startline, col + 1) == 255) {
			startline++;
			longueurligne++;
		}

		startline = startline - 1;

		//le circle est entouré par un carré verifier que les 4 points du carré sont des points noires.
		if ((int)frame.at<uchar>(startline, col - longueurligne / 2) == 0
			&& (int)frame.at<uchar>(startline, col + longueurligne / 2) == 0
			&& (int)frame.at<uchar>(line, col - longueurligne / 2) == 0
			&& (int)frame.at<uchar>(line, col + longueurligne / 2) == 0
			) {
			String display = "Circle ((" + to_string(line+longueurligne / 2) + "," + to_string(col+longueurligne/2) + "),"
				+ "(" + to_string(longueurligne/2) +  ")),"
				+ "(" + to_string(src.at<Vec3b>(line + 5, col + 5)[2])
				+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[1])
				+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[0]) + ")"
				;
			putText(src, display, Point(col, line), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0));
			return 1;
		}
	}
	return 0;
}

int TriangleDetection(int x, int y, Mat frame, Mat src) {
	int line = x; int col = y;
	//avancer dans les colonne
	int longueurCol = -1;
	while ((line - 2) > 0 && (int)frame.at<uchar>(line, y) == 255 && (int)frame.at<uchar>(line - 2, y) == 0) {
		y++; longueurCol++;
	}
	//retourner au dernier point blanc
	y = y - 1;	
	//trouver la longueur des lignes
	int centre = longueurCol / 2;
	int longueurligne = -1;
	int sommet = centre;
	while (sommet < frame.cols && (int)frame.at<uchar>(sommet, col) == 255) {
		longueurligne++; sommet++;
	}
	//retourner au dernier point non blanc
	sommet = sommet - 1;//le sommet du triangle
	//verifier que le point d n'est pas blanc
	cout << longueurCol << endl;
	if (longueurCol>5 
		&& sommet>7
		&& (int)frame.at<uchar>(sommet, centre-longueurCol/2) == 0
		&& (int)frame.at<uchar>(sommet, centre + longueurCol / 2) == 0
		&& (line-1)>0
		&& (int)frame.at<uchar>(line -1 , centre) == 0 ) {
		String display = "Triangle ((" + to_string(line) + "," + to_string(col) + "),"
			+ "(" + to_string(line) + "," + to_string(y) + "),"
			+ "(" + to_string(sommet) + "," + to_string(col+longueurCol/2) + ")),"
			+ "(" + to_string(src.at<Vec3b>(line + 5, col + 5)[2])
			+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[1])
			+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[0]) + ")"
			;
		putText(src, display, Point(col, line), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0));
		return 1;
	}
	return 0;
}


int TriangleDetection2(int x, int y, Mat frame, Mat src) {
	int line = x;
	int col = y;
	int startline = x + 1;
	// avancer avec 1 pixels en bas pour eviter le problème de bruit dans le circle
	if (startline > 0 && (col + 5)>0 && (int)frame.at<uchar>(x, col + 5) == 0) {
		//avancer dans les lignes
		int longueurligne = -1;
		while ((col - 1) > 0 && (int)frame.at<uchar>(startline, col) == 255 && (int)frame.at<uchar>(startline, col - 1) == 255
			&& (col + 1) > 0 && (int)frame.at<uchar>(startline, col + 1) == 255) {
			startline++;
			longueurligne++;
		}

		startline = startline - 1;

		//le circle est entouré par un carré verifier que les 4 points du carré sont des points noires.
		if ((int)frame.at<uchar>(startline, col - longueurligne / 2) == 0
			&& (int)frame.at<uchar>(startline, col + longueurligne / 2) == 0
			&& (int)frame.at<uchar>(line, col - longueurligne / 2) == 255
			&& (int)frame.at<uchar>(line, col + longueurligne / 2) == 255
			&& (int)frame.at<uchar>(line+1, col) == 255
			&& longueurligne>5
			) {
			String display = "Triangle ((" + to_string(line + longueurligne / 2) + "," + to_string(col + longueurligne / 2) + "),"
				+ "(" + to_string(longueurligne / 2) + ")),"
				+ "(" + to_string(src.at<Vec3b>(line + 5, col + 5)[2])
				+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[1])
				+ "," + to_string(src.at<Vec3b>(line + 5, col + 5)[0]) + ")"
				;
			putText(src, display, Point(col, line), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0));
			//putText(src, display, Point(col, line), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 255, 255));
			return 1;
		}
	}
	return 0;
}

void DetectionDeGeometrie(Mat src, Mat frame) {
	int line, col;
	int i, j;
	int maxrow = frame.rows;
	int maxcol = frame.cols;
	for (i = 0; i < maxrow; i++) {
		for (j = 0; j < maxcol; j++) {
			if (j - 1 > 0 && (int)frame.at<uchar>(i, j) > 200 && j - 1 > 0 && (int)frame.at<uchar>(i, j - 1) == 0 && i - 1 > 0 
				&& (int)frame.at<uchar>(i - 1, j) == 0) {
				if (SquareDetection(i, j, frame,src) == 1) {
					continue;
				}
				else {
					if (RectDetection(i, j, frame, src) == 1) {
						continue;
					}
					else {
						if (CircleDetection(i, j, frame, src) == 1) {
							continue;
						}
						else {
							if (TriangleDetection(i, j, frame, src) == 1) {
								continue;
							}
							else {
								if (TriangleDetection2(i, j, frame, src) == 1) {
									continue;
								}
							}
						}
					}
				}	
			}
		}
	}
}

int lire_frame() {
	// la capture video ici un fichier
	VideoCapture cap("C://Users//ASUS//Desktop//visionvideo//sample25.avi");
	if (!cap.isOpened())
		return 0;


	int iLowH = 0;
	int iHighH = 360;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 50;
	int iHighV = 250;

	int i = 0;
	for (;;)
	{
		Mat imgOriginal;

		bool bSuccess = cap.read(imgOriginal);

		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}


		Mat imgHSV;
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);



		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		imgThresholded = filtre(imgThresholded);
		DetectionDeGeometrie(imgOriginal, imgThresholded);

		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		imshow("Original", imgOriginal); //show the original image

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "appuyer sur etc" << endl;
			break;
		}
	}

	return 0;
}


int _tmain(int argc, _TCHAR* argv[])
{
	lire_frame();
	
	/*generateVideo(
	30,
	10,
	720,
	10,
	15,
	3,
	3,
	8,
	50,
	false);*/

	/*Mat src = imread("C://Users//ASUS//Desktop//lenna.png", IMREAD_GRAYSCALE);
	Mat dist = filtreMedianNVG(src, 5);
	imshow("original", src);
	imshow("median", dist);*/

	waitKey();
	return 0;
}

