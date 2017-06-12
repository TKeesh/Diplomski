#include <cstdio>
#include <queue>
#include <cstdlib>
#include <random>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "common.h"

#include "munkres.h"

using namespace cv;
using namespace std;

#define CV_LOAD_IMAGE_ANYDEPTH 2
#define CV_LOAD_IMAGE_ANYCOLOR 4
#define COLOR_BGR2GRAY 6
#define CV_BRG2GRAY COLOR_BGR2COLOR

struct Position{
	int row;
	int col;
	int frame;
	int startRow;
	int startCol;
	int width;
	int height;
	int id;
	Position(int row=0, int col=0, int frame=0, int startRow=0, int startCol=0, int width=0, int height=0, int id=0):row(row), col(col), startRow(startRow), startCol(startCol), width(width), height(height), frame(frame), id(id){}
	friend bool operator < (const Position &p1, const Position &p2){
		if (p1.row<p2.row){
			return true;
		}
		if (p1.row>p2.row){
			return false;
		}
		if (p1.col<p2.col){
			return true;
		}
		if (p1.col>p2.col){
			return false;
		}
		if (p1.startRow<p2.startRow){
			return true;
		}
		if (p1.startRow>p2.startRow){
			return false;
		}
		if (p1.startCol<p2.startCol){
			return true;
		}
		if (p1.startCol>p2.startCol){
			return false;
		}
		if (p1.width<p2.width){
			return true;
		}
		if (p1.width>p2.width){
			return false;
		}
		if (p1.height<p2.height){
			return true;
		}
		if (p1.height>p2.height){
			return false;
		}
		if (p1.frame<p2.frame){
			return true;
		}
		if (p1.frame>p2.frame){
			return false;
		}
		return p1.id<p2.id;
	}
};

struct Item{
	int position, value;
	Item(int position=0, int value=0):position(position), value(value){}
	friend bool operator < (const Item &i1, const Item &i2){
		if (i1.value<i2.value){
			return true;
		}
		if (i1.value>i2.value){
			return false;
		}
		return i1.position<i2.position;
	}
};

struct Group{
	int size;
	int id;
	Group(int size=0, int id=0):size(size), id(id){}
	friend bool operator < (const struct Group &g1, const struct Group &g2){
		if (g1.size<g2.size){
			return true;
		}
		if (g1.size>g2.size){
			return false;
		}
		return g1.id<g2.id;
	}
};

struct UnionFind{
	int n;
	int *parent;
	int *size;
	int *sum;
	
	UnionFind(int n=0):n(n){
		parent=new int[n];
		size=new int[n];
		sum=new int[n];
		Clear();
	}
	
	~UnionFind(){
		delete[] parent;
		delete[] size;
	}
	
	void Add(int x, int value=0){
		parent[x]=x;
		size[x]=1;
		sum[x]=value;
	}

	int Find(int x){
		if (parent[x]!=x){
			parent[x]=Find(parent[x]);
		}
		return parent[x];
	}
	
	void Union(int x, int y, bool sizePriority=false){
		int px=Find(x);
		int py=Find(y);
		if (px!=py){
			if ((sizePriority && size[px]<size[py]) || px<py){
				parent[px]=py;
				size[py]+=size[px];
				sum[py]+=sum[px];
			} else{
				parent[py]=px;
				size[px]+=size[py];
				sum[px]+=sum[py];
			}
		}
	}

	void Clear(){
		for (int i=0;i<n;++i){
			parent[i]=i;
			size[i]=0;
			sum[i]=0;
		}
	}
};

int** GetIntMatrix(int rows, int cols, bool setToZero=false){
	
	int **matrix=new int*[rows];
	for (int i=0;i<rows;++i){
		matrix[i]=new int[cols];
		if (setToZero==true){
			for (int j=0;j<cols;++j){
				matrix[i][j]=0;
			}
		}
	}

	return matrix;
}

void FreeIntMatrix(int **matrix, int rows){
	
	for (int i=0;i<rows;++i){
		delete[] matrix[i];
	}

	delete[] matrix;
}

bool IsForegroundPixel2(Vec3b point, double redLower=0.3450, double redUpper=0.3661, double greenLower=0.4600, double greenUpper=0.5075, double greenThreshold=35){
	double s=point[0]+point[1]+point[2];
	if (s==0){
		return false;
	}
	double r=point[2]/s;
	double g=point[1]/s;
	return ((redLower<=r && r<=redUpper && greenLower<=g && g<=greenUpper)==false || point[1]<=greenThreshold);
}

void GetBackgroundMask(Mat img, int ***flagPtr, UnionFind &uf, double greenFactor=1.0, double redFactor=1.0, double greenFactor2=1.3, double previousSizeThreshold=2.0, bool yAligned=false){

	int rows=img.rows;
	int cols=img.cols;

	int **flag=*flagPtr;
	uf.Clear();
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			Vec3b &sourcePoint=*(((Vec3b *)(img.data))+i*cols+j);
			//if (point[1]>point[2] && point[2]>point[0]){
			if (sourcePoint[1]>greenFactor*sourcePoint[2] && sourcePoint[2]>redFactor*sourcePoint[0] && sourcePoint[1]>greenFactor2*sourcePoint[0]){
				uf.Add(i*cols+j, i);
				flag[i][j]=1;
				if (i>0){
					if (flag[i-1][j]==1){
						uf.Union(i*cols+j, (i-1)*cols+j);
					}
				}
				if (j>0){
					if (flag[i][j-1]==1){
						uf.Union(i*cols+j, i*cols+j-1);
					}
				}
			} else{
				flag[i][j]=0;
			}

		}
	}
		
	vector<Group> groups;
	int msp=0;
	set<int> distinctGroups;
	for (int i=0;i<rows*cols;++i){
		if (uf.size[msp]<uf.size[i]){
			msp=i;
		}
	}
	int sizeThreshold=uf.size[msp];
	double meanY=uf.sum[msp]/(double)uf.size[msp];
	for (int i=0;i<rows*cols;++i){
		if (i==uf.parent[i] && sizeThreshold<uf.size[i]*previousSizeThreshold){
			if (yAligned==false || fabs(uf.sum[i]/(double)uf.size[i]-meanY)<0.1*rows){
				uf.Union(i, msp, true);
			}
		}
	}
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]==1 && uf.Find(i*cols+j)!=msp){
				flag[i][j]=0;
			}
		}
	}

}

void GetBackgroundMask2(Mat img, int ***flagPtr, UnionFind &uf, double redLower=0.3450, double redUpper=0.3661, double greenLower=0.4600, double greenUpper=0.5075, double previousSizeThreshold=2.0, bool yAligned=false){

	int rows=img.rows;
	int cols=img.cols;

	int **flag=*flagPtr;
	uf.Clear();
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			Vec3b &sourcePoint=*(((Vec3b *)(img.data))+i*cols+j);
			//if (point[1]>point[2] && point[2]>point[0]){
			double s=sourcePoint[0]+sourcePoint[1]+sourcePoint[2];
			if (s>0){
				double r=sourcePoint[2]/s;
				double g=sourcePoint[1]/s;
				if (redLower<=r && r<=redUpper && greenLower<=g && g<=greenUpper){
					uf.Add(i*cols+j, i);
					flag[i][j]=1;
					if (i>0){
						if (flag[i-1][j]==1){
							uf.Union(i*cols+j, (i-1)*cols+j);
						}
					}
					if (j>0){
						if (flag[i][j-1]==1){
							uf.Union(i*cols+j, i*cols+j-1);
						}
					}
				} else{
					flag[i][j]=0;
				}
			} else{
				flag[i][j]=0;
			}
		}
	}
		
	vector<Group> groups;
	int msp=0;
	set<int> distinctGroups;
	for (int i=0;i<rows*cols;++i){
		if (uf.size[msp]<uf.size[i]){
			msp=i;
		}
	}
	int sizeThreshold=uf.size[msp];
	double meanY=uf.sum[msp]/(double)uf.size[msp];
	for (int i=0;i<rows*cols;++i){
		if (i==uf.parent[i] && sizeThreshold<uf.size[i]*previousSizeThreshold){
			if (yAligned==false || fabs(uf.sum[i]/(double)uf.size[i]-meanY)<0.1*rows){
				uf.Union(i, msp, true);
			}
		}
	}
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]==1 && uf.Find(i*cols+j)!=msp){
				flag[i][j]=0;
			}
		}
	}

}

void GetFilledBackgroundMask2(Mat img, int ***flagPtr, UnionFind &uf, double redLower=0.3450, double redUpper=0.3661, double greenLower=0.4600, double greenUpper=0.5075, double previousSizeThreshold=2.0, bool combineWithPrevious=false){

	int rows=img.rows;
	int cols=img.cols;
	int **previousFlag=NULL;

	if (combineWithPrevious==true){
		previousFlag=GetIntMatrix(rows, cols, true);
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				previousFlag[i][j]=(*flagPtr)[i][j];
			}
		}
	}

	GetBackgroundMask2(img, flagPtr, uf, redLower, redUpper, greenLower, greenUpper, previousSizeThreshold);
	
	int **flag=*flagPtr;
	uf.Clear();

	int border=rows*cols;
	uf.Add(border);

	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]!=1){
				uf.Add(i*cols+j);
				if (i==0 || j==0 || i==rows-1 || j==cols-1){
					uf.Union(i*cols+j, border);
				} else{
					if (i>0 && flag[i-1][j]!=1){
						uf.Union(i*cols+j, (i-1)*cols+j);
					}
						
					if(j>0 && flag[i][j-1]!=1){
						uf.Union(i*cols+j, i*cols+j-1);
					}
				}
					
			}
		}
	}

	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]==0 && uf.Find(i*cols+j)!=border || (combineWithPrevious==true && previousFlag[i][j]==1)){
				flag[i][j]=1;
			}
		}
	}

	if (combineWithPrevious==true){
		FreeIntMatrix(previousFlag, img.rows);
	}

}

void GetBackground(VideoCapture video, Mat &background, int skip=0, int step=30, int take=30, double greenFactor=1.0, double redFactor=1.0, double greenFactor2=1.3, double previousSizeThreshold=2.0, bool yAligned=false){

	while(skip>0){
		Mat img;
		video>>img;
		if (img.empty()){
			return;
		}
		--skip;
	}

	int **flag=NULL;
	int **count=NULL;
	UnionFind *uf=NULL;
	
	int rows=0;
	int cols=0;
	background=Mat(0, 0, CV_64FC3);

	int currentStep=1;
	while(take>0){
		Mat img;
		video>>img;
		if (img.empty()){
			break;
		}
		--currentStep;
		if (currentStep!=0){
			continue;
		}

		if (rows==0){
			rows=img.rows;
			cols=img.cols;
			background=Mat(rows, cols, CV_64FC3);
			flag=GetIntMatrix(rows, cols);
			count=GetIntMatrix(rows, cols);
			for (int i=0;i<rows;++i){
				for (int j=0;j<cols;++j){
					count[i][j]=0;
				}
			}
			uf=new UnionFind(rows*cols+1);
		}

		currentStep=step;
		--take;

		GetBackgroundMask(img, &flag, *uf, greenFactor, redFactor, greenFactor2, previousSizeThreshold, yAligned);

		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				if (flag[i][j]==1){
					++count[i][j];
					*(((Vec3d *)(background.data))+i*cols+j)+=*(((Vec3b *)(img.data))+i*cols+j);
				}
			}
		}

	}

	if (rows!=0){
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				if (count[i][j]!=0){
					*(((Vec3d *)(background.data))+i*cols+j)/=count[i][j];
				}
			}
		}
		background.convertTo(background, CV_8U);
	}

	delete uf;

}

void GetBackground2(VideoCapture video, Mat &background, int skip=0, int step=30, int take=30, double redLower=0.3450, double redUpper=0.3661, double greenLower=0.4600, double greenUpper=0.5075, double previousSizeThreshold=2.0, bool yAligned=false){

	while(skip>0){
		Mat img;
		video>>img;
		if (img.empty()){
			return;
		}
		--skip;
	}

	int **flag=NULL;
	int **count=NULL;
	UnionFind *uf=NULL;
	
	int rows=0;
	int cols=0;
	background=Mat(0, 0, CV_64FC3);

	int currentStep=1;
	while(take>0){
		Mat img;
		video>>img;
		if (img.empty()){
			break;
		}
		--currentStep;
		if (currentStep!=0){
			continue;
		}

		if (rows==0){
			rows=img.rows;
			cols=img.cols;
			background=Mat(rows, cols, CV_64FC3);
			flag=GetIntMatrix(rows, cols);
			count=GetIntMatrix(rows, cols);
			for (int i=0;i<rows;++i){
				for (int j=0;j<cols;++j){
					count[i][j]=0;
				}
			}
			uf=new UnionFind(rows*cols+1);
		}

		currentStep=step;
		--take;

		GetBackgroundMask2(img, &flag, *uf, redLower, redUpper, greenLower, greenUpper, previousSizeThreshold, yAligned);

		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				if (flag[i][j]==1){
					++count[i][j];
					*(((Vec3d *)(background.data))+i*cols+j)+=*(((Vec3b *)(img.data))+i*cols+j);
				}
			}
		}

	}

	if (rows!=0){
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				if (count[i][j]!=0){
					*(((Vec3d *)(background.data))+i*cols+j)/=count[i][j];
				}
			}
		}
		background.convertTo(background, CV_8U);
	}

	delete uf;

}

void inline MinMaxRowColWithCount(int &minRow, int &maxRow, int &minCol, int &maxCol, const int i, const int j){
	if (minRow==-1 || i<minRow){
		minRow=i;
	}
	
	if (maxRow==-1 || maxRow<i){
		maxRow=i;
	}

	if (minCol==-1 || j<minCol){
		minCol=j;
	}

	if (maxCol==-1 || maxCol<j){
		maxCol=j;
	}

}

void inline MinMaxRowCol(int &minRow, int &maxRow, int &minCol, int &maxCol, const int i, const int j, const int count=1){
	if (count!=0){
		MinMaxRowColWithCount(minRow, maxRow, minCol, maxCol, i, j);
	}
}

struct BackgroundFetcher5{
	Mat *images;
	int ***flags;
	int **count;
	int **untouchedCount;
	int n;
	int size;
	int minimumSize;
	int start;
	int newPosition;
	int untouchedTTL;
	Mat background;
	int rows;
	UnionFind *uf;
	double redLower;
	double redUpper;
	double greenLower;
	double greenUpper;
	double previousSizeThreshold;
	bool yAligned;
	int minRow;
	int maxRow;
	int minCol;
	int maxCol;
	
	BackgroundFetcher5(int n, double redLower=0.3450, double redUpper=0.3661, double greenLower=0.4600, double greenUpper=0.5075, double previousSizeThreshold=2.0, bool yAligned=true, int minimumSize=3, int untouchedTTL=30):n(n), redLower(redLower), redUpper(redUpper), greenLower(greenLower), greenUpper(greenUpper), previousSizeThreshold(previousSizeThreshold), yAligned(yAligned), minimumSize(minimumSize), untouchedTTL(untouchedTTL){
		images=new Mat[n];
		uf=NULL;
		size=0;
		start=0;
		newPosition=0;
		background=Mat(0, 0, CV_64FC3);
		int rows=0;
		count=NULL;
		untouchedCount=NULL;
		flags=new int**[n];
		for (int i=0;i<n;++i){
			flags[i]=NULL;
		}
		minRow=-1;
		maxRow=-1;
		minCol=-1;
		maxCol=-1;
	}

	~BackgroundFetcher5(){
		for (int i=0;i<n;++i){
			if (flags[i]!=NULL){
				FreeIntMatrix(flags[i], rows);
			}
			images[i].release();
		}
		if (count!=NULL){
			FreeIntMatrix(count, rows);
			FreeIntMatrix(untouchedCount, rows);
		}
		delete[] images;
		delete[] flags;
		if (uf!=NULL){
			delete uf;
		}
	}
	
	void Clear(){
		if (size==0){
			return;
		}
		Mat &img=images[start];
		int rows=img.rows;
		int cols=img.cols;

		size=0;
		start=0;
		newPosition=0;
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				count[i][j]=0;
				untouchedCount[i][j]=0;
				(*(((Vec3d *)(background.data))+i*cols+j))=Vec3d(0.0, 0.0, 0.0);
			}
		}
		
	}

	void Remove(){
		if (size==0){
			return;
		}

		minRow=-1;
		maxRow=-1;
		minCol=-1;
		maxCol=-1;

		Mat &img=images[start];
		int rows=img.rows;
		int cols=img.cols;
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				if (flags[start][i][j]==1){
					bool remove=true;
					if (count[i][j]<=minimumSize && untouchedTTL<untouchedCount[i][j]){
						for (int fi=0;fi<size;++fi){
							int frame=(newPosition+n-fi-1)%n;
							if (frame==start){
								break;
							}
							if (flags[frame][i][j]==0){
								*(((Vec3b *)(images[frame].data))+i*cols+j)=*(((Vec3b *)(img.data))+i*cols+j);
								flags[frame][i][j]==1;
								remove=false;
								break;
							}
						}
					}
					if (remove==true){
						--count[i][j];
						(*(((Vec3d *)(background.data))+i*cols+j))-=*(((Vec3b *)(img.data))+i*cols+j);
					}
				}
				MinMaxRowCol(minRow, maxRow, minCol, maxCol, i, j, count[i][j]);
			}
		}
		img.release();
		start=(start+1)%n;
		--size;
	}

	void Add(Mat img){
		if (size==n){
			Remove();
		}
		
		minRow=-1;
		maxRow=-1;
		minCol=-1;
		maxCol=-1;

		++size;
		img.copyTo(images[newPosition]);
		
		rows=img.rows;
		int cols=img.cols;
		
		if (uf==NULL){
			uf=new UnionFind(rows*cols+1);
		}
		if (flags[newPosition]==NULL){
			flags[newPosition]=GetIntMatrix(img.rows, img.cols);
		}
		if (background.rows==0){
			background=Mat::zeros(img.rows, img.cols, CV_64FC3);
			count=GetIntMatrix(img.rows, img.cols, true);
			untouchedCount=GetIntMatrix(img.rows, img.cols, true);
		}
		
		GetBackgroundMask2(images[newPosition], flags+newPosition, *uf, redLower, redUpper, greenLower, greenUpper, previousSizeThreshold, yAligned);
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				if (flags[newPosition][i][j]==1){
					++count[i][j];
					untouchedCount[i][j]=0;
					(*(((Vec3d *)(background.data))+i*cols+j))+=*(((Vec3b *)(img.data))+i*cols+j);
				} else{
					++untouchedCount[i][j];
				}
				MinMaxRowCol(minRow, maxRow, minCol, maxCol, i, j, count[i][j]);
			}
		}
		newPosition=(newPosition+1)%n;
	}
	
	void GetBackground(Mat &result){
		if (background.rows==0){
			return;
		}
		int rows=background.rows;
		int cols=background.cols;
		result=Mat::zeros(rows, cols, CV_8UC3);
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				if (count[i][j]!=0){
					*(((Vec3b *)(result.data))+i*cols+j)=(*(((Vec3d *)(background.data))+i*cols+j))/count[i][j];
				}
			}
		}
	}

};

void GetBase(const char *path, char *base){
	int l=strlen(path);
	int start=0;
	int end=-1;
	for (int i=l-1;i>-1;--i){
		if (end==-1 && path[i]=='.'){
			end=i;
		}
		if (path[i]=='/' || path[i]=='\\'){
			start=i+1;
			break;
		}
	}

	for (int i=start;i<end;++i){
		base[i-start]=path[i];
	}
	base[end-start]='\0';

}

void GetBackgroundSmartly2(const char *videoPath, Mat &background, int skip=0, int step=30, int take=30, const char *backgroundsPath="D:/backgrounds/", bool write=false, double redLower=0.3450, double redUpper=0.3661, double greenLower=0.4600, double greenUpper=0.5075, double previousSizeThreshold=2.0){

	char base[1025];
	GetBase(videoPath, base);
	char path[1025];
	sprintf(path, "%s/%s_%d_%d_%d.png", backgroundsPath, base, skip, step, take);

	FILE *input=fopen(path, "rb");
	
	if (input==NULL){
		VideoCapture video=VideoCapture(videoPath);
		GetBackground2(video, background, skip, step, take, redLower, redUpper, greenLower, greenUpper);
		video.release();
		if (write==true){
			imwrite(path, background);
		}
	} else{
		fclose(input);
		background=imread(path, 6);
	}

}

void GetForegroundFlag(Mat img, Mat background, int **terrainMask, double threshold, int **flag, int minRow=-1, int maxRow=-1, int minCol=-1, int maxCol=-1){

	int rows=img.rows;
	int cols=img.cols;

	if (minRow==-1){
		minRow=0;
	}
	if (maxRow==-1){
		maxRow=rows-1;
	}
	if (minCol==-1){
		minCol=0;
	}
	if (maxCol==-1){
		maxCol=cols-1;
	}

	for (int i=minRow;i<=maxRow;++i){
		for (int j=minCol;j<=maxCol;++j){
			flag[i][j]=0;
			if (terrainMask[i][j]!=0){
				Vec3b backgroundPoint=*(((Vec3b *)(background.data))+i*cols+j);
				if (backgroundPoint[0]!=0 || backgroundPoint[1]!=0 || backgroundPoint[2]){
					const Vec3b point=*(((Vec3b *)(img.data))+i*cols+j);
				
					//TESTING
					if (false){
						double thresholdAngle=6;
						Vec3d v1=point;
						Vec3d v2=backgroundPoint;
						double currentAngle=0;
						double a=0;
						double b=0;
						for (int i=0;i<3;++i){
							currentAngle+=v1[i]*v2[i];
							a+=v1[i]*v1[i];
							b+=v2[i]*v2[i];
						}
						currentAngle/=sqrt(a*b);
						currentAngle=180*acos(currentAngle)/3.141592654;
						//printf("%lf\n", currentAngle);
						if (thresholdAngle<currentAngle){
							flag[i][j]=1;
						}
						continue;
					}
				

					double d=0.0;
					for (int k=0;k<3;++k){
						double dd=point[k]-backgroundPoint[k];
						dd*=dd;
						d+=dd;
					}
					if (threshold<d){
						flag[i][j]=1;
					}
				}
			}
		}
	}

}

void GetForegroundFlag(Mat img, Mat background, int **terrainMask, double threshold, int **flag, int **suddenlyChanged, int minRow=-1, int maxRow=-1, int minCol=-1, int maxCol=-1){

	int rows=img.rows;
	int cols=img.cols;

	if (minRow==-1){
		minRow=0;
	}
	if (maxRow==-1){
		maxRow=rows-1;
	}
	if (minCol==-1){
		minCol=0;
	}
	if (maxCol==-1){
		maxCol=cols-1;
	}

	for (int i=minRow;i<=maxRow;++i){
		for (int j=minCol;j<=maxCol;++j){
			flag[i][j]=0;
			if (terrainMask==NULL || terrainMask[i][j]!=0){
				Vec3b backgroundPoint=*(((Vec3b *)(background.data))+i*cols+j);
				if (backgroundPoint[0]!=0 || backgroundPoint[1]!=0 || backgroundPoint[2]){
					const Vec3b point=*(((Vec3b *)(img.data))+i*cols+j);
				
					//TESTING
					if (false){
						double thresholdAngle=6;
						Vec3d v1=point;
						Vec3d v2=backgroundPoint;
						double currentAngle=0;
						double a=0;
						double b=0;
						for (int i=0;i<3;++i){
							currentAngle+=v1[i]*v2[i];
							a+=v1[i]*v1[i];
							b+=v2[i]*v2[i];
						}
						currentAngle/=sqrt(a*b);
						currentAngle=180*acos(currentAngle)/3.141592654;
						//printf("%lf\n", currentAngle);
						if (thresholdAngle<currentAngle){
							flag[i][j]=1;
						}
						continue;
					}
				

					double d=0.0;
					for (int k=0;k<3;++k){
						double dd=point[k]-backgroundPoint[k];
						dd*=dd;
						d+=dd;
					}
					if (threshold<d){
						flag[i][j]=1;
					} else{
						suddenlyChanged[i][j]=0;
					}
				} else{
					suddenlyChanged[i][j]=0;
				}
			}
		}
	}

}

void GetForegroundFlag(Mat img, Mat background, int **terrainMask, double threshold, double greenThreshold, int **flag, int **suddenlyChanged, int minRow=-1, int maxRow=-1, int minCol=-1, int maxCol=-1){
	
	int rows=img.rows;
	int cols=img.cols;

	if (minRow==-1){
		minRow=0;
	}
	if (maxRow==-1){
		maxRow=rows-1;
	}
	if (minCol==-1){
		minCol=0;
	}
	if (maxCol==-1){
		maxCol=cols-1;
	}

	for (int i=minRow;i<=maxRow;++i){
		for (int j=minCol;j<=maxCol;++j){
			flag[i][j]=0;
			if (terrainMask==NULL || terrainMask[i][j]!=0){
				Vec3b backgroundPoint=*(((Vec3b *)(background.data))+i*cols+j);
				if (backgroundPoint[0]!=0 || backgroundPoint[1]!=0 || backgroundPoint[2]){
					const Vec3b point=*(((Vec3b *)(img.data))+i*cols+j);
				
					//TESTING
					if (false){
						double thresholdAngle=6;
						Vec3d v1=point;
						Vec3d v2=backgroundPoint;
						double currentAngle=0;
						double a=0;
						double b=0;
						for (int i=0;i<3;++i){
							currentAngle+=v1[i]*v2[i];
							a+=v1[i]*v1[i];
							b+=v2[i]*v2[i];
						}
						currentAngle/=sqrt(a*b);
						currentAngle=180*acos(currentAngle)/3.141592654;
						//printf("%lf\n", currentAngle);
						if (thresholdAngle<currentAngle){
							flag[i][j]=1;
						}
						continue;
					}
				

					double d=0.0;
					for (int k=0;k<3;++k){
						double dd=point[k]-backgroundPoint[k];
						dd*=dd;
						d+=dd;
					}
					if (point[1]<greenThreshold){
						double dd=greenThreshold-point[1];
						dd*=dd;
						d+=dd;
					}
					if (threshold<d){
						flag[i][j]=1;
					} else{
						suddenlyChanged[i][j]=0;
					}
				} else{
					suddenlyChanged[i][j]=0;
				}
			}
		}
	}

}

void GetForegroundFlagWithRespectToPreviousFrameAndBackground2(Mat img, Mat previous, Mat background, int **terrainMask, double threshold, double thresholdForPrevious, double greenThreshold, int **flag, int **suddenlyChanged, double redLower=0.3450, double redUpper=0.3661, double greenLower=0.4600, double greenUpper=0.5075, int minRow=-1, int maxRow=-1, int minCol=-1, int maxCol=-1){
	
	if (suddenlyChanged==NULL){
		GetForegroundFlag(img, background, terrainMask, threshold, flag, minRow, maxRow, minCol, maxCol);
	} else{
		GetForegroundFlag(img, background, terrainMask, threshold, flag, suddenlyChanged, minRow, maxRow, minCol, maxCol);
	}

	int rows=img.rows;
	int cols=img.cols;

	if (minRow==-1){
		minRow=0;
	}
	if (maxRow==-1){
		maxRow=rows-1;
	}
	if (minCol==-1){
		minCol=0;
	}
	if (maxCol==-1){
		maxCol=cols-1;
	}

	for (int i=minRow;i<=maxRow;++i){
		for (int j=minCol;j<=maxCol;++j){
			if (flag[i][j]!=0){
				Vec3b backgroundPoint=*(((Vec3b *)(background.data))+i*cols+j);
				if (backgroundPoint[0]!=0 || backgroundPoint[1]!=0 || backgroundPoint[2]){
					const Vec3b point=*(((Vec3b *)(img.data))+i*cols+j);
					const Vec3b previousPoint=*(((Vec3b *)(previous.data))+i*cols+j);
				
					double d=0.0;
					for (int k=0;k<3;++k){
						double dd=point[k]-previousPoint[k];
						dd*=dd;
						d+=dd;
					}
					if (point[1]<greenThreshold){
						double dd=greenThreshold-point[1];
						dd*=dd;
						d+=dd;
					}
					
					if (thresholdForPrevious<d){
						suddenlyChanged[i][j]=1;
					}
					//if (suddenlyChanged[i][j]==0){
					if (suddenlyChanged[i][j]==0 && IsForegroundPixel2(point, redLower, redUpper, greenLower, greenUpper)==false){
						flag[i][j]=0;
					}
				}
			}
		}
	}

}

void GetGroups(int **flag, int rows, int cols, vector<vector<Position>> &groups, UnionFind &uf, bool ufIsInitialized=false, int minRow=-1, int maxRow=-1, int minCol=-1, int maxCol=-1){

	if (minRow==-1){
		minRow=0;
	}
	if (maxRow==-1){
		maxRow=rows-1;
	}
	if (minCol==-1){
		minCol=0;
	}
	if (maxCol==-1){
		maxCol=cols-1;
	}
	
	int border=rows*cols;
	
	if (ufIsInitialized==false){
		uf.Clear();
		uf.Add(border);
		
		for (int i=minRow;i<=maxRow;++i){
			for (int j=minCol;j<=maxCol;++j){
				if (flag[i][j]==1){
					uf.Add(i*cols+j);
					if (i==0 || j==0 || i==rows-1 || j==cols-1){
						uf.Union(i*cols+j, border);
					} else{
						if (i>0 && flag[i-1][j]==1){
							uf.Union(i*cols+j, (i-1)*cols+j);
						}
						
						if(j>0 && flag[i][j-1]==1){
							uf.Union(i*cols+j, i*cols+j-1);
						}
					}
					
				}
			}
		}
	}

	map<int, vector<Position>> positions;

	for (int i=minRow;i<=maxRow;++i){
		for (int j=minCol;j<=maxCol;++j){
			if (flag[i][j]==1){
				int parent=uf.Find(i*cols+j);
				if (parent!=border){
					positions[parent].push_back(Position(i, j));
				}
			}
		}
	}

	groups.clear();
	
	for (auto mi=positions.begin();mi!=positions.end();++mi){
		groups.push_back(mi->second);
	}
	
}

enum BoundingBoxType{
	NORMAL=0,
	PUSHED_OUT=1,
	PUSHER=2,
	FILLED=4
};

struct BoundingBox{
	int minRow;
	int maxRow;
	int minCol;
	int maxCol;
	int frame;
	int type;
	Vec3b meanColor;
	BoundingBox(int minRow=0, int maxRow=0, int minCol=0, int maxCol=0, int frame=0, int type=0, Vec3b meanColor=Vec3b(0, 0, 0)):minRow(minRow), maxRow(maxRow), minCol(minCol), maxCol(maxCol), frame(frame), type(type), meanColor(meanColor){}
};

void GetBoundingBox(const vector<Position> &points, int &minRow, int &maxRow, int &minCol, int &maxCol){
	int n=points.size();
	if (n==0){
		return;
	}
	minRow=points[0].row;
	maxRow=minRow;
	minCol=points[0].col;
	maxCol=minCol;
	for (int i=0;i<n;++i){
		int cr=points[i].row;
		int cc=points[i].col;
		if (cr<minRow){
			minRow=cr;
		}
		if (maxRow<cr){
			maxRow=cr;
		}
		if (cc<minCol){
			minCol=cc;
		}
		if (maxCol<cc){
			maxCol=cc;
		}
				
	}
	
}

BoundingBox GetBoundingBox(const vector<Position> &points){
	int n=points.size();
	if (n==0){
		return BoundingBox(-1, -1, -1, -1);
	}
	int minRow=points[0].row;
	int maxRow=minRow;
	int minCol=points[0].col;
	int maxCol=minCol;
	for (int i=0;i<n;++i){
		int cr=points[i].row;
		int cc=points[i].col;
		if (cr<minRow){
			minRow=cr;
		}
		if (maxRow<cr){
			maxRow=cr;
		}
		if (cc<minCol){
			minCol=cc;
		}
		if (maxCol<cc){
			maxCol=cc;
		}
				
	}
	
	return BoundingBox(minRow, maxRow, minCol, maxCol);
}

void DrawRectangle(Mat &img, int minRow, int maxRow, int minCol, int maxCol, int size=-1){
	double height=maxRow-minRow;
	double width=maxCol-minCol;

	Scalar drawColor=Scalar(0, 0, 255);
	if (height*1.5<width || width*10<height || (size!=-1 && size*5<height*width)){
		drawColor=Scalar(255, 0, 0);
	}

	rectangle(img, Point(minCol, minRow), Point(maxCol, maxRow), drawColor);
}

void DrawRectangles(Mat &img, const vector<BoundingBox> &boundingBoxes){

	for (int i=0;i<boundingBoxes.size();++i){
		int minRow=boundingBoxes[i].minRow;
		int maxRow=boundingBoxes[i].maxRow;
		int minCol=boundingBoxes[i].minCol;
		int maxCol=boundingBoxes[i].maxCol;
		
		DrawRectangle(img, minRow, maxRow, minCol, maxCol, -1);
	}

}

Mat terrainSelectionImg;
vector<Position> terrainSelectionPositions;
Position terrainSelectionPointerPoint;
const char *terrainSelectionWindowName = "terrain_selection";
bool terrainSelectionActionsPerformed = false;
bool terrainSelectionMouseDown = false;
int terrainSelectionX = 0;
int terrainSelectionY = 0;
bool terrainSelectionMovePerformed = false;

bool LineIntersectionExists(double p0_x, double p0_y, double p1_x, double p1_y, double p2_x, double p2_y, double p3_x, double p3_y){
    
	double s1_x=p1_x-p0_x;
	double s1_y=p1_y-p0_y;
    double s2_x=p3_x-p2_x;
	double s2_y=p3_y-p2_y;

    double s=(-s1_y*(p0_x-p2_x)+s1_x*(p0_y-p2_y))/(-s2_x*s1_y+s1_x*s2_y);
    double t=(s2_x*(p0_y-p2_y)-s2_y*(p0_x-p2_x))/(-s2_x*s1_y+s1_x*s2_y);

    if (s>=0 && s<=1 && t>=0 && t<=1){
        return true;
    }

    return false;
}

void TerrainSelectionMouseCallback(int event, int x, int y, int flags, void* userdata){
	
	int radius=10;
	int radiusIdx=-1;

	for (int i=0;i<terrainSelectionPositions.size();++i){
		int x2=terrainSelectionPositions[i].col;
		int y2=terrainSelectionPositions[i].row;

		int d=(x-x2)*(x-x2)+(y-y2)*(y-y2);
		if (d<=radius*radius){
			radiusIdx=i;
			break;
		}
	}

	if (event==EVENT_LBUTTONDOWN || event==EVENT_MBUTTONDOWN || event==EVENT_RBUTTONDOWN){
		terrainSelectionMouseDown=true;
		terrainSelectionMovePerformed=false;
	}

	bool disregardUp=false;
	if (event==EVENT_LBUTTONUP || event==EVENT_MBUTTONUP || event==EVENT_RBUTTONUP){
		terrainSelectionMouseDown=false;
		disregardUp=terrainSelectionMovePerformed;
		terrainSelectionMovePerformed=false;
	}

	if (event==EVENT_LBUTTONUP && disregardUp==false){
		if (radiusIdx!=-1){
			
			terrainSelectionActionsPerformed=true;
			int n=terrainSelectionPositions.size();
			bool collision=false;
			for (int i=radiusIdx;i<n-3;++i){
				for (int j=i+2;j<n-1;++j){
					if (LineIntersectionExists(terrainSelectionPositions[i].col, terrainSelectionPositions[i].row, terrainSelectionPositions[i+1].col, terrainSelectionPositions[i+1].row, terrainSelectionPositions[j].col, terrainSelectionPositions[j].row, terrainSelectionPositions[j+1].col, terrainSelectionPositions[j+1].row)==true){
						collision=true;
						break;
					}
				}
			}
			
			if (collision==false){
				Position add=terrainSelectionPositions[radiusIdx];
				if (radiusIdx>0){
					for (int i=0;i<terrainSelectionPositions.size()-radiusIdx;++i){
						terrainSelectionPositions[i]=terrainSelectionPositions[i+radiusIdx];
					}
					terrainSelectionPositions.resize(terrainSelectionPositions.size()-radiusIdx);
					radiusIdx=0;
				}
				terrainSelectionPositions.push_back(add);
			}
		} else{
			terrainSelectionPositions.push_back(Position(y, x));
		}

	} else if (event==EVENT_RBUTTONUP && disregardUp==false){

		terrainSelectionActionsPerformed=true;
		terrainSelectionPositions.pop_back();
		
	} else if (event==EVENT_MBUTTONUP && disregardUp==false){
		
		terrainSelectionActionsPerformed=true;
		terrainSelectionPositions.clear();

	} else if (event==EVENT_MOUSEMOVE){
		
		if (terrainSelectionMouseDown==true){
			int dx=x-terrainSelectionX;
			int dy=y-terrainSelectionY;
			for (int i=0;i<terrainSelectionPositions.size();++i){
				terrainSelectionPositions[i].col+=dx;
				terrainSelectionPositions[i].row+=dy;
			}
			terrainSelectionMovePerformed=true;
			terrainSelectionActionsPerformed=true;
		}
		terrainSelectionPointerPoint=Position(y, x);
		terrainSelectionX=x;
		terrainSelectionY=y;
	}

	Mat img;
	terrainSelectionImg.copyTo(img);

	for (int i=1;i<terrainSelectionPositions.size();++i){
		Point previous=Point(terrainSelectionPositions[i-1].col, terrainSelectionPositions[i-1].row);
		Point current=Point(terrainSelectionPositions[i].col, terrainSelectionPositions[i].row);
		line(img, previous, current, Scalar(0, 0, 255));
	}

	if (terrainSelectionPositions.size()>0 && terrainSelectionMouseDown==false){
		Point previous=Point(terrainSelectionPositions[terrainSelectionPositions.size()-1].col, terrainSelectionPositions[terrainSelectionPositions.size()-1].row);
		Point current=Point(terrainSelectionPointerPoint.col, terrainSelectionPointerPoint.row);
		line(img, previous, current, Scalar(0, 0, 255));
	}

	for (int i=0;i<terrainSelectionPositions.size();++i){
		Point center=Point(terrainSelectionPositions[i].col, terrainSelectionPositions[i].row);
		Scalar color=Scalar(0, 0, 255);
		if (i==radiusIdx && i!=terrainSelectionPositions.size()-1 && terrainSelectionPositions.size()-i>2){
			color=Scalar(255, 0, 0);
		}
		circle(img, center, radius, color);
	}

	imshow(terrainSelectionWindowName, img);

}

void CleanUpSelectedTerrain(int ***flagPtr, UnionFind *uf, int rows, int cols){

	int **flag=*flagPtr;
	uf->Clear();
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]==1){
				uf->Add(i*cols+j, i);
				if (i>0){
					if (flag[i-1][j]==1){
						uf->Union(i*cols+j, (i-1)*cols+j);
					}
				}
				if (j>0){
					if (flag[i][j-1]==1){
						uf->Union(i*cols+j, i*cols+j-1);
					}
				}
			} else{
				flag[i][j]=0;
			}

		}
	}
		
	vector<Group> groups;
	int msp=0;
	set<int> distinctGroups;
	for (int i=0;i<rows*cols;++i){
		if (uf->size[msp]<uf->size[i]){
			msp=i;
		}
	}
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]==1 && uf->Find(i*cols+j)!=msp){
				flag[i][j]=0;
			}
		}
	}

	int border=rows*cols;
	uf->Clear();
	uf->Add(border);
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]==0){
					
				uf->Add(i*cols+j);
				if (i==0 || j==0 || i==rows-1 || j==cols-1){
					uf->Union(i*cols+j, border);
				} else{
					if (i>0 && flag[i-1][j]==0){
						uf->Union(i*cols+j, (i-1)*cols+j);
					}
						
					if(j>0 && flag[i][j-1]==0){
						uf->Union(i*cols+j, i*cols+j-1);
					}
				}
					
			}
		}
	}

	int f=uf->Find(border);
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (flag[i][j]==0 && uf->Find(i*cols+j)!=border){
				flag[i][j]=1;
			}
		}
	}

}

int **SelectTerrain(double f=1.0){
	int rows=terrainSelectionImg.rows;
	int cols=terrainSelectionImg.cols;
	
	resize(terrainSelectionImg, terrainSelectionImg, Size(terrainSelectionImg.cols/f, terrainSelectionImg.rows/f));

	terrainSelectionActionsPerformed=false;
	terrainSelectionMouseDown=false;
	terrainSelectionMovePerformed=false;

	bool selected=false;

	while(selected==false){
		imshow(terrainSelectionWindowName, terrainSelectionImg);
		setMouseCallback(terrainSelectionWindowName, TerrainSelectionMouseCallback, NULL);
		waitKey();
		if (terrainSelectionPositions.size()<3 || terrainSelectionPositions[0].row!=terrainSelectionPositions[terrainSelectionPositions.size()-1].row || terrainSelectionPositions[0].col!=terrainSelectionPositions[terrainSelectionPositions.size()-1].col){
			terrainSelectionPositions.clear();
		} else{
			selected=true;
		}
	}

	for (int i=0;i<terrainSelectionPositions.size();++i){
		terrainSelectionPositions[i].row/=f;
		terrainSelectionPositions[i].col/=f;
	}

	int **terrainMask=GetIntMatrix(rows, cols, true);

	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			
			int count=0;

			int n=terrainSelectionPositions.size();
			for (int ii=0;ii<n-1;++ii){
				if (LineIntersectionExists(terrainSelectionPositions[ii].col, terrainSelectionPositions[ii].row, terrainSelectionPositions[ii+1].col, terrainSelectionPositions[ii+1].row, 0, 0, j, i)==true){
					++count;
				}
			}

			/*
			for (int ii=0;ii<n;++ii){
				if (terrainSelectionPositions[ii].row<=j && terrainSelectionPositions[ii].col<=i){
					int dx1=terrainSelectionPositions[ii].col;
					int dx2=i-terrainSelectionPositions[ii].col;
					int dy1=terrainSelectionPositions[ii].row;
					int dy2=j-terrainSelectionPositions[ii].row;

					if (dx1*dy2==dy1*dx2){
						--count;
					}

				}
			}
			*/

			if (count%2!=0){
				terrainMask[i][j]=1;
			}

		}
	}

	UnionFind *uf=new UnionFind(rows*cols+1);
	CleanUpSelectedTerrain(&terrainMask, uf, rows, cols);
	delete uf;

	/*
	int move[4][2]={
				{-1, 0},
				{0, 1},
				{1, 0},
				{0, -1}
			};
	queue<Position> q1;
	q1.push(Position(0, 0));
	terrainMask[0][0]=2;
	while(q1.empty()==false){
		int row=q1.front().row;
		int col=q1.front().col;
		q1.pop();

		for (int i=0;i<4;++i){
			int row2=row+move[i][0];
			int col2=col+move[i][1];
			if (row2>=0 && row2<rows && col2>0 && col2<cols && terrainMask[row2][col2]==0){
				terrainMask[row2][col2]=2;
				q1.push(Position(row2, col2));
			}
		}
	}

	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (terrainMask[i][j]==0){
				terrainMask[i][j]=1;
			} else if (terrainMask[i][j]==2){
				terrainMask[i][j]=0;
			}
		}
	}
	*/

	destroyWindow(terrainSelectionWindowName);

	return terrainMask;
}

int **SelectTerrainSmartly(const char *videoPath, int skip=0, int step=30, int take=30, const char *backgroundsPath="D:/terrains/", bool write=false, double f=1.0){
	char base[1025];
	GetBase(videoPath, base);
	char path[1025];
	sprintf(path, "%s/%s_%d_%d_%d.png", backgroundsPath, base, skip, step, take);

	int rows=terrainSelectionImg.rows;
	int cols=terrainSelectionImg.cols;

	FILE *input=fopen(path, "rb");
	
	int **terrainMask=NULL;
	if (input==NULL){
		VideoCapture video=VideoCapture(videoPath);
		terrainMask=SelectTerrain(f);
		if (write==true){
			Mat img=Mat::zeros(rows, cols, CV_8UC1);
			for (int i=0;i<rows;++i){
				for (int j=0;j<cols;++j){
					*(((uchar *)(img.data))+i*cols+j)=255*terrainMask[i][j];
				}
			}
			imwrite(path, img);
		}
	} else{
		fclose(input);
		Mat img=imread(path, 6);
		terrainMask=GetIntMatrix(rows, cols, true);
		for (int i=0;i<rows;++i){
			for (int j=0;j<cols;++j){
				uchar value=*(((uchar *)(img.data))+i*cols+j);
				if (value==0){
					terrainMask[i][j]=0;
				} else{
					terrainMask[i][j]=1;
				}
			}
		}
	}

	return terrainMask;
}

int **SelectTerrainSmartly(const char *videoPath, Mat &background, int skip=0, int step=30, int take=30, const char *backgroundsPath="D:/terrains/", bool write=false, double f=1.0){
	return SelectTerrainSmartly(videoPath, skip, step, take, backgroundsPath, write, f);
}

struct TrackingData{
	struct DisposedComparison{
		bool operator ()(const TrackingData *t1, const TrackingData *t2) const {
			if (t1->lastFrame<t2->lastFrame){
				return true;
			}
			if (t2->lastFrame<t1->lastFrame){
				return false;
			}
			if (t1->id<t2->id){
				return true;
			}
			if (t2->id<t1->id){
				return false;
			}
			return t1<t2;
		}
	};
	int id;
	vector<Position> positions;
	vector<BoundingBox> previous;
	int lastFrame;
	bool pushedOut;
	TrackingData *pushedOutBy;
	bool isTracked;
	Vec3b meanColor;
	Position meanPosition;
	set<TrackingData*, DisposedComparison> pushedOutGroups;
	int framesOutsideOfTerrain;
	TrackingData(vector<Position> positions=vector<Position>(), vector<BoundingBox> previous=vector<BoundingBox>(), int lastFrame=-1, bool pushedOut=false, bool isTracked=true, int framesOutsideOfTerrain=0):positions(positions), previous(previous), lastFrame(lastFrame), pushedOut(pushedOut), isTracked(isTracked), id(createdGroupsCount++), framesOutsideOfTerrain(framesOutsideOfTerrain){
		pushedOutBy=nullptr;
		meanColor=Vec3b(0, 0, 0);
		meanPosition=Position(-1, -1);
	}
	void RemovePushedOutBy(){
		if (pushedOutBy!=nullptr){
			auto found=pushedOutBy->pushedOutGroups.find(this);
			if (found!=pushedOutBy->pushedOutGroups.end()){
				pushedOutBy->pushedOutGroups.erase(found);
			}
		}
		pushedOutBy=nullptr;
		pushedOut=false;
	}
	void SetPushedByOutBySmartly(TrackingData *pusher, bool copyPushedOutGroups=false){
		RemovePushedOutBy();
		pushedOutBy=pusher;
		if (pusher!=nullptr){
			pusher->pushedOutGroups.insert(this);
			pushedOut=true;
		} else{
			pushedOut=false;
		}
		if (copyPushedOutGroups==true && pusher!=nullptr){
			for (auto si=pushedOutGroups.cbegin();si!=pushedOutGroups.cend();++si){
				(*si)->pushedOutBy=pusher;
				pusher->pushedOutGroups.insert(*si);
			}
			pushedOutGroups.clear();
		}
	}
	void ClearPushedOutGropusSmartly(){
		for (auto si=pushedOutGroups.cbegin();si!=pushedOutGroups.cend();++si){
			(*si)->pushedOut=false;
			(*si)->pushedOutBy=nullptr;
		}
		pushedOutGroups.clear();
	}
	void CalculateMeanColor(const Mat &img){
		int rows=img.rows;
		int cols=img.cols;

		int count=0;
		Vec3d mean(0.0, 0.0, 0.0);
		for (int i=0;i<positions.size();++i){
			int r=positions[i].row;
			int c=positions[i].col;
			if (r>=0 && r<rows && c>=0 && c<cols){
				mean+=*(((Vec3b *)(img.data))+r*cols+c);
				++count;
			}
		}
		if (count==0){
			meanColor=Vec3b(0, 0, 0);
		} else{
			meanColor=mean/count;
		}
	}
	void CalculateMeanPosition(){
		
		int n=positions.size();
		if (n==0){
			meanPosition=Position(-1, -1);
			return;
		}
		int row=0;
		int col=0;
		for (int i=0;i<n;++i){
			row+=positions[i].row;
			col+=positions[i].col;
		}
		meanPosition=Position(row/n, col/n);
	}
private:
	static int createdGroupsCount;
};

int TrackingData::createdGroupsCount=0;

TrackingData*** GetTrackingDataPointerMatrix(int rows, int cols, bool setToNull=false){
	
	TrackingData ***matrix=new TrackingData**[rows];
	for (int i=0;i<rows;++i){
		matrix[i]=new TrackingData*[cols];
		if (setToNull==true){
			for (int j=0;j<cols;++j){
				matrix[i][j]=nullptr;
			}
		}
	}

	return matrix;
}

void FreeTrackingDataPointerMatrix(TrackingData ***matrix, int rows){
	
	for (int i=0;i<rows;++i){
		delete[] matrix[i];
	}

	delete[] matrix;
}

int move8[8][2]={
				{-1, 0},
				{-1, 1},
				{0, 1},
				{1, 1},
				{1, 0},
				{1, -1},
				{0, -1},
				{-1, -1}
			};

void Spread(const vector<Position> &group, const Mat &img, const Mat &background, int **terrainMask, double greenThreshold, int **flag, int **spreadData, int **visited, int spreadCount, int visitCount, Position &seedPosition, bool insideTerrain=true){
	
	int rows=img.rows;
	int cols=img.cols;

	for (int i=0;i<group.size();++i){
		Vec3b backgroundPoint=*(((Vec3b *)(background.data))+group[i].row*cols+group[i].col);
		int row=group[i].row;
		int col=group[i].col;
		if ((insideTerrain==false || terrainMask==NULL || terrainMask[row][col]!=0) && backgroundPoint[0]+backgroundPoint[1]+backgroundPoint[2]>0 && visited[row][col]!=visitCount){
			const Vec3b point=*(((Vec3b *)(img.data))+group[i].row*cols+group[i].col);
				
			double d=0.0;
			for (int k=0;k<3;++k){
				double dd=point[k]-backgroundPoint[k];
				dd*=dd;
				d+=dd;
			}
			if (point[1]<greenThreshold){
				double dd=greenThreshold-point[1];
				dd*=dd;
				d+=dd;
			}
			flag[group[i].row][group[i].col]=spreadCount;
			spreadData[group[i].row][group[i].col]=d;
		}
	}

	double md=-1;
	int mdIdx=0;
	for (int i=0;i<group.size();++i){
		if ((insideTerrain==false || terrainMask==NULL || terrainMask[group[i].row][group[i].col]!=0) && flag[group[i].row][group[i].col]==spreadCount){
			Vec3b point=*(((Vec3b *)(img.data))+group[i].row*cols+group[i].col);
			double d=spreadData[group[i].row][group[i].col];
			for (int k=0;k<8;++k){
				int nr=group[i].row+move8[k][0];
				int nc=group[i].col+move8[k][1];
				if (nr>=0 && nr<rows && nc>=0 && nc<cols && flag[nr][nc]==spreadCount){
					d+=spreadData[nr][nc];
				}
			}
			if (md<d || md==-1){
				md=d;
				mdIdx=i;
			}
		}
	}
	
	seedPosition=group[mdIdx];

}

bool Visit(TrackingData *trackedGroup, const Mat &img, const Mat &background, int **terrainMask, double greenThreshold, int **visited, int visitCount, const Position &seedPosition, int scanningAttempts, double threshold, int minimumGroupSize, TrackingData ***owners=nullptr, bool insideTerrain=true, int maximumWidth=-1, int maximumHeight=-1, double remainingFactor=1.2){
	
	bool takeIt=false;
	
	int rows=img.rows;
	int cols=img.cols;
	
	int currentScanningAttempts=scanningAttempts;
	double currentThreshold=threshold;

	while(currentScanningAttempts-->0){
				
		int remaining=remainingFactor*trackedGroup->positions.size();

		vector<Position> newPositions;

		int minRow=seedPosition.row;
		int maxRow=seedPosition.row;
		int minCol=seedPosition.col;
		int maxCol=seedPosition.col;

		queue<Position> q1;
		TrackingData *touchedOwner=nullptr;
		bool touchedOther=false;
		if (visited[seedPosition.row][seedPosition.col]==visitCount){
			touchedOther=true;
			if (owners!=nullptr){
				touchedOwner=owners[seedPosition.row][seedPosition.col];
			}
		} else{
			visited[seedPosition.row][seedPosition.col]=visitCount;
			newPositions.push_back(seedPosition);
			q1.push(seedPosition);
		}
		while(q1.empty()==false && remaining>0){
			Position p=q1.front();
			q1.pop();

			for (int i=0;i<8;++i){
				int row=p.row+move8[i][0];
				int col=p.col+move8[i][1];
				if (row<0 || col<0 || row>rows-1 || col>cols-1){
					continue;
				}
				if (visited[row][col]==visitCount){
					if (row!=seedPosition.row && col!=seedPosition.col){
						touchedOther=true;
						if (owners!=nullptr){
							touchedOwner=owners[row][col];
						}
					}
					continue;
				}
					
				Vec3b point=*(((Vec3b *)(img.data))+row*cols+col);
				Vec3b backgroundPoint=*(((Vec3b *)(background.data))+row*cols+col);
					
				double d=0.0;
				for (int k=0;k<3;++k){
					double dd=point[k]-backgroundPoint[k];
					dd*=dd;
					d+=dd;
				}
				if (point[1]<greenThreshold){
					double dd=greenThreshold-point[1];
					dd*=dd;
					dd+=5;
					d+=dd;
				}
				if ((insideTerrain==false || terrainMask==NULL || terrainMask[row][col]!=0) && currentThreshold<d && backgroundPoint[0]+backgroundPoint[1]+backgroundPoint[2]>0){
					int currentMinRow=minRow;
					int currentMaxRow=maxRow;
					int currentMinCol=minCol;
					int currentMaxCol=maxCol;

					MinMaxRowCol(currentMinRow, currentMaxRow, currentMinCol, currentMaxCol, row, col);
					if ((maximumHeight==-1 || currentMaxRow-currentMinRow+1<=maximumHeight) && (maximumWidth==-1 || currentMaxCol-currentMinCol+1<=maximumWidth)){
						minRow=currentMinRow;
						maxRow=currentMaxRow;
						minCol=currentMinCol;
						maxCol=currentMaxCol;
						--remaining;
						visited[row][col]=visitCount;
						newPositions.push_back(Position(row, col));
						q1.push(Position(row, col));
					}
				}
				
			}

		}

		if (newPositions.size()>=minimumGroupSize){
			vector<Position> &positions=trackedGroup->positions;
			//positions.clear();
			positions=newPositions;
			takeIt=true;
			
			trackedGroup->pushedOut=false;
			trackedGroup->SetPushedByOutBySmartly(nullptr);

			if (owners!=nullptr){
				for (int i=0;i<positions.size();++i){
					owners[positions[i].row][positions[i].col]=trackedGroup;
				}
			}
			if (currentScanningAttempts<scanningAttempts-1){
				//rescanned=true;
				//printf("%d %d Success after rescanning (%d)!\n", framesCount, gi, scanningAttempts-1-currentScanningAttempts);
			} else{
				//rescanned=false;
			}

			break;
		} else{
			for (int i=0;i<newPositions.size();++i){
				visited[newPositions[i].row][newPositions[i].col]=visitCount-1;
			}
			trackedGroup->pushedOut=touchedOther;
			
			if (touchedOther==false){
				currentThreshold*=0.8;
			} else{

				trackedGroup->SetPushedByOutBySmartly(touchedOwner);
				break;
			}
		}
	}

	return takeIt;
}

void GetWiderAreaPositions(TrackingData *trackedGroup, vector<Position> &positions, int rows, int cols, int previousLookSize=15, double enlargementFactor=3.0){
	BoundingBox boundingBox=GetBoundingBox(trackedGroup->positions);
	int dx=boundingBox.maxCol-boundingBox.minCol+1;
	int dy=boundingBox.maxRow-boundingBox.minRow+1;

	int back=trackedGroup->previous.size()-previousLookSize;
	if (back<0){
		back=0;
	}
	
	//printf("%d %d      ", dx, dy);
	for (int i=back;i<trackedGroup->previous.size();++i){
		BoundingBox previous=trackedGroup->previous[i];
		int cdx=previous.maxCol-previous.minCol+1;
		int cdy=previous.maxRow-previous.minRow+1;
		if (dx<cdx){
			dx=cdx;
		}
		if (dy<cdy){
			dy=cdy;
		}
	}
	
	//printf("%d %d\n", dx, dy);

	int startRow=boundingBox.minRow-dx*enlargementFactor;
	if (startRow<0){
		startRow=0;
	}
	int endRow=boundingBox.maxRow+dx*enlargementFactor;
	if (endRow>=rows){
		endRow=rows-1;
	}
	int startCol=boundingBox.minCol-dy*enlargementFactor;
	if (startCol<0){
		startCol=0;
	}
	int endCol=boundingBox.maxCol+dy*enlargementFactor;
	if (endCol>=cols){
		endCol=cols-1;
	}
	for (int row=startRow;row<=endRow;++row){
		for (int col=startCol;col<=endCol;++col){
			positions.push_back(Position(row, col));
		}
	}
				
}

void AddMissingPrevious(TrackingData *tracked, BoundingBox start, BoundingBox end, int n){
	
	for (int i=0;i<n;++i){
		BoundingBox boundingBox=BoundingBox(((n-i-1)*start.minRow+(i+1)*end.minRow)/n, ((n-i-1)*start.maxRow+(i+1)*end.maxRow)/n, ((n-i-1)*start.minCol+(i+1)*end.minCol)/n, ((n-i-1)*start.maxCol+(i+1)*end.maxCol)/n);
		boundingBox.type=BoundingBoxType::FILLED;
		tracked->previous.push_back(boundingBox);
	}

}

void DrawTrajectories(Mat &img2, const vector<TrackingData*> &trackedGroups, int trajectoryDrawingLength, Scalar color=Scalar(0, 0, 255)){

	for (int gi=0;gi<trackedGroups.size();++gi){
		TrackingData *trackedGroup=trackedGroups[gi];
		int stop=trackedGroup->previous.size()-trajectoryDrawingLength;
		if (stop<0){
			stop=0;
		}
		
		for (int i=trackedGroup->previous.size()-1;i>stop;--i){
			line(img2, Point((trackedGroup->previous[i].maxCol+trackedGroup->previous[i].minCol)/2, trackedGroup->previous[i].maxRow), Point((trackedGroup->previous[i-1].maxCol+trackedGroup->previous[i-1].minCol)/2, trackedGroup->previous[i-1].maxRow), color);
		}
	}

}

void ReconnectGroups(TrackingData *add, TrackingData *newlyFoundGroup, const Mat &img, int framesCount, TrackingData *takePrevious=nullptr){
	
	//AddMissingPrevious(add, add->previous[add->previous.size()-1], GetBoundingBox(newlyFoundGroup->positions), framesCount-add->lastFrame);
	if (takePrevious!=nullptr){
		for (int j=takePrevious->previous.size()-framesCount+add->lastFrame+1;j<takePrevious->previous.size();++j){
			BoundingBox boundingBox=takePrevious->previous[j];
			boundingBox.type=BoundingBoxType::FILLED;
			add->previous.push_back(boundingBox);
		}
	} else{
		if (add->previous.size()>0){
			AddMissingPrevious(add, add->previous[add->previous.size()-1], GetBoundingBox(newlyFoundGroup->positions), framesCount-add->lastFrame-1);
		}
	}

	add->isTracked=true;
	add->pushedOut=false;
	add->SetPushedByOutBySmartly(nullptr);
	add->lastFrame=framesCount;
	add->positions=newlyFoundGroup->positions;
	add->CalculateMeanColor(img);
	add->CalculateMeanPosition();

	add->ClearPushedOutGropusSmartly();
	
}

int CountPositionsInsideTerrain(const vector<Position> &positions, const Mat &terrain){
	int count=0;

	int rows=terrain.rows;
	int cols=terrain.cols;

	for (int i=0;i<positions.size();++i){
		int row=positions[i].row;
		int col=positions[i].col;
		if (*(((uchar *)(terrain.data))+row*cols+col)!=0){
			++count;
		}
	}

	return count;
}

template<typename T>
T CalculateMean(const vector<T> &data){
	T m=0;

	int n=data.size();
	for (int i=0;i<n;++i){
		m+=data[i];
	}
	m/=n;
	
	return m;
}

template<typename T>
T CalculateStandardDeviation(const vector<T> &data, T m){
	
	T std=0;

	int n=data.size();
	for (int i=0;i<n;++i){
		T d=data[i]-m;
		std+=d*d;
	}
	std=sqrt(std/(n-1));

	return std;
}

template<typename T>
T CalculateStandardDeviation(const vector<T> &data){
	return CalculateStandardDeviation(data, CalculateMean(data));
}

void CreateMaskFromFlags(int **flag, Mat &mask, int rows, int cols){
	
	mask=Mat::zeros(rows, cols, CV_8UC1);

	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			int bf=flag[i][j];
			*(((uchar *)(mask.data))+i*cols+j)=255*bf;
		}
	}
	
}

void CalculateColorChromaticityBounds(const Mat &source, const Mat &mask, double &redLower, double &redUpper, double &greenLower, double &greenUpper, double spreadFactor=2.0){
	
	Vec3d meanColor=Vec3d(0.0, 0.0, 0.0);
	vector<double> channels[3];

	Mat img;
	source.convertTo(img, CV_64F);

	int rows=img.rows;
	int cols=img.cols;

	vector<double> red;
	vector<double> green;
	
	int n=0;
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (*(((const uchar *)(mask.data))+i*cols+j)!=0){
				Vec3d &point=*(((Vec3d *)(img.data))+i*cols+j);
				double s=point[0]+point[1]+point[2];
				if (s>0){
					red.push_back(point[2]/s);
					green.push_back(point[1]/s);
				}
			}
		}
	}

	double redMean=CalculateMean(red);
	double greenMean=CalculateMean(green);

	double redStd=CalculateStandardDeviation(red, redMean);
	double greenStd=CalculateStandardDeviation(green, greenMean);
	
	redLower=redMean-spreadFactor*redStd;
	redUpper=redMean+spreadFactor*redStd;
	greenLower=greenMean-spreadFactor*greenStd;
	greenUpper=greenMean+spreadFactor*greenStd;
}

double CalculateApproximateDifference2(const Mat &img1, const Mat &img2, int step=10, int **terrainMask=NULL, double threshold=5.0){
	int rows=img1.rows;
	int cols=img1.cols;

	if (rows!=img2.rows || cols!=img2.cols){
		return -1;
	}

	int n=0;
	int count=0;
	for (int i=0;i<rows;i+=step){
		for (int j=0;j<cols;j+=step){
			if (terrainMask==NULL || terrainMask[i][j]!=0){
				for (int k=0;k<3;++k){
					double dd=(*(((Vec3b *)(img1.data))+i*cols+j))[k]-(*(((Vec3b *)(img2.data))+i*cols+j))[k];
					if (dd<0){
						dd=-dd;
					}
					if (threshold<dd){
						++count;
						break;
					}
				}
				++n;
			}
		}
	}

	return count/(float)n;
}

void CalculateCenter(const Mat &img, int &row, int &col){
	
	int rows=img.rows;
	int cols=img.cols;

	int n=0;
	row=0;
	col=0;
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			if (*(((uchar *)(img.data))+i*cols+j)!=0){
				row+=i;
				col+=j;
				++n;
			}
		}
	}
	row/=n;
	col/=n;

}

void usleep(__int64 usec){ 
    HANDLE timer; 
    LARGE_INTEGER ft; 

    ft.QuadPart = -(10*usec); // Convert to 100 nanosecond interval, negative value indicates relative time

    timer = CreateWaitableTimer(NULL, TRUE, NULL); 
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0); 
    WaitForSingleObject(timer, INFINITE); 
    CloseHandle(timer); 
}

void Beep3(){
	printf("\a");
	usleep(750000);
	printf("\a");
	usleep(750000);
	printf("\a");
}

vector<Point> teamSelection;
const char *teamSelectionWindowName = "i";

void TeamSelectionMouseCallback(int event, int x, int y, int flags, void* userdata) {
	Point center;
	Scalar color = Scalar(0, 0, 255);
	int radius = 3;

	if (event == EVENT_LBUTTONDOWN || event == EVENT_MBUTTONDOWN || event == EVENT_RBUTTONDOWN) {
		teamSelection.push_back(Point(x, y));

		center = Point(x, y);
		circle(terrainSelectionImg, center, radius, color, -1);
		imshow(teamSelectionWindowName, terrainSelectionImg);
	}

}

void SelectTeam() {
	int selected = 0;	
	Mat tmpImg = terrainSelectionImg.clone();

	while (1) {
		setMouseCallback(teamSelectionWindowName, TeamSelectionMouseCallback, NULL);
		waitKey();
		
		if (teamSelection.size() != 4) {
			teamSelection.clear();
			tmpImg.copyTo(terrainSelectionImg);
			imshow(teamSelectionWindowName, tmpImg);
		}
		else break;		
	}

	tmpImg.release();
}

struct MyTracking {
	vector<Point> contourPoints;
	Mat rect;
	Point center;
	int team;
	int area;
	int visited;
	MatND hist;
};

vector<MyTracking> myTracking, newMyTracking;
int avgContourArea;

struct HistData {
	MatND histData;
	int id;
	Vec3b color;
};
vector<HistData> histData;

bool analyse(Mat img, vector<vector<Point>> contours) {
	avgContourArea = 0;
	int channels[] = { 0, 1, 2 };
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { s_ranges, s_ranges, s_ranges };
	int h_bins = 50; int s_bins = 50;
	int histSize[] = { h_bins, s_bins, s_bins };
	Mat hsvImg, rangeImg1, rangeImg2;
	MatND histImg;

	int found = 0;
	double inside;
	for (int i = 0; i < contours.size(); ++i) {
		avgContourArea += contourArea(contours[i]);
		for (int j = 1; j < 3; ++j) {
			inside = pointPolygonTest(contours[i], teamSelection[j], false);
			if (inside > 0) {
				++found;
				cvtColor(img(boundingRect(contours[i])), hsvImg, COLOR_BGR2RGB);
				calcHist(&hsvImg, 1, channels, Mat(), histImg, 2, histSize, ranges, true, false);
				normalize(histImg, histImg, 0, 1, NORM_MINMAX, -1, Mat());
				HistData tmpHistData;
				tmpHistData.histData = histImg;
				tmpHistData.id = j;
				tmpHistData.color = img.at<Vec3b>(teamSelection[j]);
				histData.push_back(tmpHistData);
			}
		}		
	}
	avgContourArea /= contours.size();

	Moments m;
	for (int i = 0; i < contours.size(); ++i) {		
		if (contourArea(contours[i]) < avgContourArea*0.5) continue;			
		MyTracking tmpData;
		tmpData.area = contourArea(contours[i]);
		tmpData.contourPoints = contours[i];
		m = moments(contours[i], true);
		tmpData.center = Point(m.m10 / m.m00, m.m01 / m.m00);
		tmpData.visited = 0;
		tmpData.rect = img(boundingRect(contours[i]));
		
		
		cvtColor(tmpData.rect, hsvImg, COLOR_BGR2RGB);
		calcHist(&hsvImg, 1, channels, Mat(), histImg, 2, histSize, ranges, true, false);
		normalize(histImg, histImg, 0, 1, NORM_MINMAX, -1, Mat());
		tmpData.hist = histImg;
		
		if (pointPolygonTest(contours[i], teamSelection[0], false) > 0) {
			tmpData.team = 0; 
			++found;
		}		
		else if (pointPolygonTest(contours[i], teamSelection[3], false) > 0) {
			tmpData.team = 3; 
			++found;
		}
		else {
			/*double team1 = compareHist(histData[0].histData, histImg, 1);
			double team2 = compareHist(histData[1].histData, histImg, 1);
			printf("%d %d\n", team1, team2);
			if (team1 < team2) tmpData.team = histData[0].id;
			else tmpData.team = histData[1].id;*/
			Scalar bgrL = (histData[0].color[0] - 1, histData[0].color[1] - 1, histData[0].color[2] - 1);
			Scalar bgrU = (histData[0].color[0] + 1, histData[0].color[1] + 1, histData[0].color[2] + 1);
			inRange(tmpData.rect, bgrL, bgrU, rangeImg1);
			bgrL = (histData[1].color[0] - 1, histData[1].color[1] - 1, histData[1].color[2] - 1);
			bgrU = (histData[1].color[0] + 1, histData[1].color[1] + 1, histData[1].color[2] + 1);
			inRange(tmpData.rect, bgrL, bgrU, rangeImg2);
			if (countNonZero(rangeImg1) < countNonZero(rangeImg2)) tmpData.team = histData[1].id;
			else tmpData.team = histData[0].id;
		}
		
		myTracking.push_back(tmpData);
	}

	if (found == 4) return true;
	else return false;
}

void Test97() {
	char videoPath[1024] = "C:/Users/etomiki/Desktop/Nogomet/Dinamo_vs._Rijeka_-_panorama_view.mp4";
	//char videoPath[1024] = "C:/Users/etomiki/Desktop/Nogomet/Inter - Dinamo wide angle.mp4";	


	const char *drawnResultsPath = "C:/Users/etomiki/Desktop/Nogomet/labeled_images/";
	char videoBase[1025];
	GetBase(videoPath, videoBase);

	double f = 1.0;

	//double thresholdFactor=0.75;
	double thresholdFactor = 0.8;
	double threshold = thresholdFactor*1000.0;
	double thresholdForPrevious = thresholdFactor*250.0;

	//int maximumGroupsCount=30;
	int maximumGroupsCount = 35;

	int minimumGroupSize = 3;
	int minimumGroupSizeAtFirstDetection = 5;

	bool cameraMoved = false;
	bool cameraWasMoving = false;
	bool forceModelBuilding = false;
	//double cameraMovedThreshold=0.1;
	double cameraMovedThreshold = 0.2;
	double pixelChangedThreshold = 5.0;
	Mat lastGoodImage;
	int cameraMovedStep = 20;

	//int n=30;
	int n = 20;
	int skip = 0;
	//int skip=50;
	int step = 30;
	int take = n;

	VideoCapture prevideo = VideoCapture(videoPath);
	Mat preImg;
	prevideo >> preImg;
	prevideo.release();

	preImg.copyTo(lastGoodImage);

	int rows = preImg.rows;
	int cols = preImg.cols;

	preImg.copyTo(terrainSelectionImg);
	int **terrainMask = SelectTerrainSmartly(videoPath, skip, step, take, "C:/Users/etomiki/Desktop/Nogomet/terrains/", true, f);

	Mat terrainMaskImg = Mat::zeros(rows, cols, CV_8UC1);
	for (int i = 0; i<rows; ++i) {
		for (int j = 0; j<cols; ++j) {
			*(((uchar *)(terrainMaskImg.data)) + i*cols + j) = 255 * terrainMask[i][j];
		}
	}
	//imshow("terrain", terrainMaskImg);
	
	int keyPressed = waitKey(1);

	double redLower = 0.3450;
	double redUpper = 0.3661;
	double greenLower = 0.4600;
	double greenUpper = 0.5075;

	double spreadFactor = 4.0;

	CalculateColorChromaticityBounds(preImg, terrainMaskImg, redLower, redUpper, greenLower, greenUpper, spreadFactor);

	int chromaticityBoundsCalculationStep = 25;
	int chromaticityBoundsCalculationCount = chromaticityBoundsCalculationStep;

	double greenThreshold = 45;

	double previousSizeThreshold = 2.0;

	int backgroundCount = 1;
	int backgroundLimit = 30;

	int redetectStep = 2;
	int redetectCount = redetectStep;

	int scanningAttempts = 3;

	int allowedFramesOutsideOfTerrain = 300;

	int backFramesToCheckForCloseTracked = 50;
	int backFramesToCheckForStrongClosePushedOut = 50;
	int backFramesToCheckForClosePushedOut = 150;

	int trajectoryDrawingLength = 100;

	double remainingFactor = 1.2;

	Mat background;
	GetBackgroundSmartly2(videoPath, background, skip, step, take, "C:/Users/etomiki/Desktop/Nogomet/backgrounds/", true, redLower, redUpper, greenLower, greenUpper);

	//double sameGroupFieldDistancePercentage=0.07;
	//double sameGroupFieldDistancePercentage = backFramesToCheckForStrongClosePushedOut*0.0028;
	//int sameGroupFieldDistance = background.cols*sameGroupFieldDistancePercentage;
	//int sameGroupBackFramesForSpeed = 10;

	int maximumWidth = cols*0.075;
	int maximumHeight = rows*0.05;

	int **backgroundFlag = GetIntMatrix(rows, cols);
	int **foregroundFlag = GetIntMatrix(rows, cols);
	int **currentBackgroundFlag = GetIntMatrix(rows, cols);
	UnionFind uf(rows*cols + 1);

	Mat img;
	BackgroundFetcher5 *bf = new BackgroundFetcher5(n, redLower, redUpper, greenLower, greenUpper, previousSizeThreshold);
	//BackgroundFetcher5 *bf=new BackgroundFetcher5(n, redLower, redUpper, greenLower, greenUpper, previousSizeThreshold, true, 0);

	int minRow = -1;
	int maxRow = -1;
	int minCol = -1;
	int maxCol = -1;

	for (int i = 0; i<rows; ++i) {
		for (int j = 0; j<cols; ++j) {
			const Vec3b &point = *(((Vec3b *)(background.data)) + i*cols + j);
			int s = point[0] + point[1] + point[2];
			MinMaxRowCol(minRow, maxRow, minCol, maxCol, i, j, s);
		}
	}

	int currentStep = 1;
	int framesCount = 0;
	Mat previous = Mat::zeros(0, 0, CV_8UC3);
	int **flag = GetIntMatrix(rows, cols, true);
	int **suddenlyChanged = GetIntMatrix(rows, cols, true);

	int spreadCount = 0;
	int **spreadData = GetIntMatrix(rows, cols, true);

	GetForegroundFlag(preImg, background, terrainMask, threshold, greenThreshold, flag, suddenlyChanged, minRow, maxRow, minCol, maxCol);

	Mat testImgMine = Mat::zeros(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			*(((uchar *)(testImgMine.data)) + i*cols + j) = 255 * flag[i][j];
		}
	}
	Mat element = getStructuringElement(CV_SHAPE_ELLIPSE, Size(3, 3), Point(1, 1));
	//morphologyEx(testImgMine, testImgMine, 3, element);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat imageCopy = testImgMine.clone();
	findContours(imageCopy, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);
	imshow("testImgMine", testImgMine);

	/*vector<vector<Position> > groups;
	GetGroups(flag, img.rows, img.cols, groups, uf, false, minRow, maxRow, minCol, maxCol);

	vector<TrackingData*> trackedGroups;
	set<TrackingData*, TrackingData::DisposedComparison> disposedGroups;

	for (int i = 0; i<groups.size(); ++i) {
		trackedGroups.push_back(new TrackingData(groups[i]));
	}

	int **visited = GetIntMatrix(rows, cols, true);

	TrackingData ***owners = GetTrackingDataPointerMatrix(rows, cols, true);*/

	VideoCapture video = VideoCapture(videoPath);

	Mat currentMask;
	int currentMaskCounter = 1;
	int currentMaskReset = 50;

	int visitCount = 0;

	while (true) {
		++framesCount;

		cameraMoved = false;

		Mat img;
		video >> img;

		if (img.empty()) {
			break;
		}

		if (previous.rows != 0) {
			double difference = CalculateApproximateDifference2(img, previous, cameraMovedStep, terrainMask, pixelChangedThreshold);
			//printf("%lf\n", difference);
			if (cameraMovedThreshold<difference) {
				printf("%lf\n", difference);
				if (cameraWasMoving == false) {
					previous.copyTo(lastGoodImage);
				}
				cameraWasMoving = true;
				//the line below is to disable possible usage of terrain for factors recalculation in a potentially critical moment when the terrain selection may not be valid anymore
				++chromaticityBoundsCalculationCount;
			}
			else {
				if (cameraWasMoving == true) {
					printf("Camera moved.\n");

					double difference = CalculateApproximateDifference2(img, lastGoodImage, cameraMovedStep, terrainMask, pixelChangedThreshold);

					if (difference <= cameraMovedThreshold) {
						printf("However, it returned to the beginning again.");
					}
					else {
						Beep3();

						img.copyTo(terrainSelectionImg);
						terrainMask = SelectTerrain(f);
						CreateMaskFromFlags(terrainMask, terrainMaskImg, rows, cols);
						imshow("terrain", terrainMaskImg);

						if (terrainSelectionActionsPerformed == true) {
							cameraMoved = true;

							forceModelBuilding = true;
							printf("Clearing background.\n");
							bf->Clear();
							//delete bf;
							//bf=new BackgroundFetcher5(n, redLower, redUpper, greenLower, greenUpper, previousSizeThreshold);

							currentMaskCounter = 1;
						}
					}
				}

				img.copyTo(lastGoodImage);
				cameraWasMoving = false;
			}
		}
		--currentMaskCounter;
		if (currentMaskCounter == 0) {
			currentMaskCounter = currentMaskReset;

			bool combineWithPrevious = false;
			GetFilledBackgroundMask2(img, &currentBackgroundFlag, uf, redLower, redUpper, greenLower, greenUpper, previousSizeThreshold, combineWithPrevious);
			int initialRow = 0;
			int initialCol = 0;
			if (cameraMoved == true) {
				CalculateCenter(currentMask, initialRow, initialCol);
			}
			CreateMaskFromFlags(currentBackgroundFlag, currentMask, rows, cols);
			//imshow("cm", currentMask);
		}

		--currentStep;
		if (currentStep == 0 || forceModelBuilding == true) {
			if (forceModelBuilding == false) {
				currentStep = step;
			}

			bf->Add(img);

			if (bf->size == n || forceModelBuilding == true) {
				bf->GetBackground(background);
				minRow = bf->minRow;
				maxRow = bf->maxRow;
				minCol = bf->minCol;
				maxCol = bf->maxCol;
			}
			if (bf->size == n) {
				forceModelBuilding = false;
				currentStep = step;
			}
		}

		/*++spreadCount;
		++visitCount;
		vector<bool> isTaken;
		set<TrackingData*, TrackingData::DisposedComparison> currentlyDisposedGroups;
		for (int gi = 0; gi<trackedGroups.size(); ++gi) {

			Position seedPosition;
			Spread(trackedGroups[gi]->positions, img, background, terrainMask, greenThreshold, flag, spreadData, visited, spreadCount, visitCount, seedPosition, false);

			bool takeIt = Visit(trackedGroups[gi], img, background, terrainMask, greenThreshold, visited, visitCount, seedPosition, scanningAttempts, threshold, minimumGroupSize, owners, false, maximumWidth, maximumHeight, remainingFactor);

			isTaken.push_back(takeIt);

		}

		for (int gi = 0; gi<isTaken.size(); ++gi) {
			if (isTaken[gi] == false && trackedGroups[gi]->pushedOut == false) {

				vector<Position> positions;
				GetWiderAreaPositions(trackedGroups[gi], positions, rows, cols, 25, 3);

				Position seedPosition;
				Spread(positions, img, background, terrainMask, greenThreshold, flag, spreadData, visited, spreadCount, visitCount, seedPosition, false);

				bool takeIt = Visit(trackedGroups[gi], img, background, terrainMask, greenThreshold, visited, visitCount, seedPosition, scanningAttempts, threshold, minimumGroupSize, owners, false, maximumWidth, maximumHeight, remainingFactor);

				if (takeIt == false && trackedGroups[gi]->pushedOut == true) {
					trackedGroups[gi]->pushedOut = false;
				}

				isTaken[gi] = takeIt;

				if (takeIt == true) {
					//printf("%d %d taken after wider area rescanning\n", framesCount, gi);
				}

			}

			if (isTaken[gi] == true) {
				if (trackedGroups[gi]->previous.size() >= 2) {
					BoundingBox before = trackedGroups[gi]->previous[trackedGroups[gi]->previous.size() - 2];
					BoundingBox after = trackedGroups[gi]->previous[trackedGroups[gi]->previous.size() - 1];
					int areaBefore = (before.maxRow - before.minRow + 1)*(before.maxCol - before.minCol + 1);
					int areaAfter = (after.maxRow - after.minRow + 1)*(after.maxCol - after.minCol + 1);

					if (areaAfter * 3<areaBefore) {
						//printf("%d Alert!\n", framesCount);
						redetectCount = 1;
					}
				}
			}
		}

		//checking if a group is too long fully outside of the terrain
		for (int i = 0; i<isTaken.size(); ++i) {
			if (isTaken[i] == true) {
				int inside = CountPositionsInsideTerrain(trackedGroups[i]->positions, terrainMaskImg);
				if (inside == 0) {
					++trackedGroups[i]->framesOutsideOfTerrain;
					if (allowedFramesOutsideOfTerrain<trackedGroups[i]->framesOutsideOfTerrain) {
						trackedGroups[i]->framesOutsideOfTerrain = 0;
						isTaken[i] = false;
					}
				}
			}
		}

		vector<TrackingData*> newTrackedGroups;
		for (int i = 0; i<isTaken.size(); ++i) {
			if (isTaken[i] == true) {
				trackedGroups[i]->lastFrame = framesCount;
				newTrackedGroups.push_back(trackedGroups[i]);
			}
			else {
				currentlyDisposedGroups.insert(trackedGroups[i]);
			}
			trackedGroups[i]->isTracked = isTaken[i];
		}*/

		if (--chromaticityBoundsCalculationCount == 0) {
			chromaticityBoundsCalculationCount = chromaticityBoundsCalculationStep;
			CalculateColorChromaticityBounds(img, terrainMaskImg, redLower, redUpper, greenLower, greenUpper, spreadFactor);
			bf->redLower = redLower;
			bf->redUpper = redUpper;
			bf->greenLower = greenLower;
			bf->greenUpper = greenUpper;
		}
		
		//vector<TrackingData*> newlyFoundGroups;
		if (--redetectCount == 0) {
			redetectCount = redetectStep;

			if (previous.rows == 0) {
				GetForegroundFlag(img, background, terrainMask, threshold, greenThreshold, flag, suddenlyChanged, minRow, maxRow, minCol, maxCol);
			}
			else {
				GetForegroundFlagWithRespectToPreviousFrameAndBackground2(img, previous, background, terrainMask, threshold, thresholdForPrevious, greenThreshold, flag, suddenlyChanged, redLower, redUpper, greenLower, greenUpper, minRow, maxRow, minCol, maxCol);
			}

			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					*(((uchar *)(testImgMine.data)) + i*cols + j) = 255 * flag[i][j];
				}
			}
			Mat imageCopy = testImgMine.clone();
			morphologyEx(testImgMine, testImgMine, 3, element);
			imshow("testImgMine", testImgMine);			
			morphologyEx(imageCopy, imageCopy, 4, element);
			findContours(imageCopy, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);
						
			imshow("testImgMineNew", imageCopy);

			/*
			vector<vector<Position>> groups;
			GetGroups(flag, img.rows, img.cols, groups, uf, false, minRow, maxRow, minCol, maxCol);

			for (int gi = 0; gi<groups.size(); ++gi) {
				const vector<Position> &group = groups[gi];

				bool take = true;
				for (int i = 0; i<group.size(); ++i) {
					if (visited[group[i].row][group[i].col] == visitCount) {
						take = false;
						break;
					}
				}

				//if (take==true && group.size()>=minimumGroupSize){
				if (take == true && group.size() >= minimumGroupSizeAtFirstDetection) {
					TrackingData *newGroup = new TrackingData(group);
					newGroup->isTracked = true;
					newlyFoundGroups.push_back(newGroup);
					//isRescanned.push_back(false);
				}
			}
			*/
		}

		/*for (int i = 0; i<newTrackedGroups.size(); ++i) {
			newTrackedGroups[i]->CalculateMeanColor(img);
			newTrackedGroups[i]->CalculateMeanPosition();
		}*/

		/*
		if (newlyFoundGroups.size()>0) {
			for (int i = 0; i<newlyFoundGroups.size(); ++i) {
				newlyFoundGroups[i]->CalculateMeanColor(img);
				newlyFoundGroups[i]->CalculateMeanPosition();

				TrackingData *add = newlyFoundGroups[i];

				//STATISTICS for reconnecting rules
				//first, determine close groups
				vector<pair<int, TrackingData*> > closeTracked;
				for (int j = 0; j<newTrackedGroups.size(); ++j) {
					TrackingData *tracked = newTrackedGroups[j];
					if (tracked->meanPosition.row != -1) {
						int dr = newlyFoundGroups[i]->meanPosition.row - tracked->meanPosition.row;
						dr *= dr;
						int dc = newlyFoundGroups[i]->meanPosition.col - tracked->meanPosition.col;
						dc *= dc;
						int d = dr + dc;
						if (d <= sameGroupFieldDistance*sameGroupFieldDistance) {
							closeTracked.push_back(make_pair(d, tracked));
						}
					}
				}

				auto closeComparison = [](const pair<int, TrackingData *> &p1, const pair<int, TrackingData *> &p2) {if (p1.first<p2.first) { return true; }if (p2.first<p1.first) { return false; }return p2.second->lastFrame<p1.second->lastFrame; };
				sort(closeTracked.begin(), closeTracked.end(), closeComparison);

				//then check whether the closest group has shrunk (lately)
				bool sizeShrinked = false;
				bool sizeShrinkedLately = false;
				if (closeTracked.size()>0 && closeTracked[0].second->previous.size()>0) {
					BoundingBox before = closeTracked[0].second->previous[closeTracked[0].second->previous.size() - 1];
					BoundingBox after = GetBoundingBox(closeTracked[0].second->positions);
					int areaBefore = (before.maxRow - before.minRow + 1)*(before.maxCol - before.minCol + 1);
					int areaAfter = (after.maxRow - after.minRow + 1)*(after.maxCol - after.minCol + 1);

					if (areaAfter*1.5<areaBefore) {
						sizeShrinked = true;
					}

					if (sizeShrinked == true) {
						sizeShrinkedLately = true;
					}
					else {
						int stop = closeTracked[0].second->previous.size() - backFramesToCheckForCloseTracked;
						if (stop<0) {
							stop = 0;
						}
						for (int j = closeTracked[0].second->previous.size() - 1; j>stop; --j) {
							BoundingBox before = closeTracked[0].second->previous[j - 1];
							BoundingBox after = GetBoundingBox(closeTracked[0].second->positions);

							int areaBefore = (before.maxRow - before.minRow + 1)*(before.maxCol - before.minCol + 1);
							int areaAfter = (after.maxRow - after.minRow + 1)*(after.maxCol - after.minCol + 1);

							if (areaAfter*1.5<areaBefore) {
								sizeShrinkedLately = true;
								break;
							}
						}
					}
				}

				//next, determine the closest disposed groups
				vector<pair<int, TrackingData*> > closeDisposed;
				for (auto si = disposedGroups.rbegin(); si != disposedGroups.rend(); ++si) {
					TrackingData *disposed = *si;
					if (framesCount - disposed->lastFrame>backFramesToCheckForStrongClosePushedOut) {
						break;
					}
					if (disposed->meanPosition.row != -1) {
						int dr = newlyFoundGroups[i]->meanPosition.row - disposed->meanPosition.row;
						int dc = newlyFoundGroups[i]->meanPosition.col - disposed->meanPosition.col;
						dr *= dr;
						dc *= dc;
						int d = dr + dc;
						int backFrames = sameGroupBackFramesForSpeed;
						if (disposed->previous.size()<backFrames) {
							backFrames = disposed->previous.size();
						}
						if (backFrames<2) {
							continue;
						}
						int sdr = 0;
						int sdc = 0;
						int lastRow = disposed->previous[disposed->previous.size() - backFrames].maxRow;
						int lastCol = (disposed->previous[disposed->previous.size() - backFrames].maxCol - disposed->previous[disposed->previous.size() - backFrames].minCol) / 2;
						int estimatedMaximalDistance = 0;
						for (int i = disposed->previous.size() - backFrames + 1; i<disposed->previous.size(); ++i) {
							int currentRow = disposed->previous[i].maxRow;
							int currentCol = (disposed->previous[i].maxCol - disposed->previous[i].minCol) / 2;
							int sdr = lastRow - currentRow;
							int sdc = lastCol - currentCol;
							estimatedMaximalDistance += sqrt(sdr*sdr + sdc*sdc);
							lastRow = currentRow;
							lastCol = currentCol;
						}
						estimatedMaximalDistance /= backFrames - 1;
						estimatedMaximalDistance *= framesCount - disposed->lastFrame;
						d = sqrt(d);
						if (d <= 1.5*estimatedMaximalDistance) {
							closeDisposed.push_back(make_pair(d, disposed));
						}
					}
				}
				sort(closeDisposed.begin(), closeDisposed.end(), closeComparison);

				//RULE
				//check for trajectories connecting when the disposed group closest to the current newly found group was pushed out by the closest tracked group
				if (closeDisposed.size()>0) {
					//if (closeTracked[0].second->pushedOutGroups.find(closeDisposed[0].second)!=closeTracked[0].second->pushedOutGroups.end()){
					if (closeTracked.size()>0 && closeDisposed[0].second->pushedOut == true && closeDisposed[0].second->pushedOutBy == closeTracked[0].second && (sizeShrinked == true || sizeShrinkedLately == true)) {
						//if (closeDisposed[0].second->pushedOut==true && closeDisposed[0].second->pushedOutBy==closeTracked[0].second){
						add = closeDisposed[0].second;
						disposedGroups.erase(disposedGroups.find(add));

						ReconnectGroups(add, newlyFoundGroups[i], img, framesCount, closeTracked[0].second);
					}
				}

				//RULE
				//check for the closest disposed group that was not pushed out and that is significantly closer than other similar groups
				//somewhat unstable and therefore currently disabled
				if (add == newlyFoundGroups[i] && closeDisposed.size()>0) {

					if (closeDisposed[0].second->pushedOut == false && (closeDisposed.size() == 1 || closeDisposed[0].first * 5 <= closeDisposed[1].first)) {
						add = closeDisposed[0].second;
						disposedGroups.erase(disposedGroups.find(add));

						ReconnectGroups(add, newlyFoundGroups[i], img, framesCount, nullptr);
					}
				}

				if (add == newlyFoundGroups[i] && closeTracked.size()>0 && (closeTracked.size() == 1 || closeTracked[0].first * 3 <= closeTracked[1].first) && (closeTracked[0].second->pushedOutGroups.size()>0) && currentlyDisposedGroups.find(*(closeTracked[0].second->pushedOutGroups.begin())) == currentlyDisposedGroups.end() && framesCount - (*(closeTracked[0].second->pushedOutGroups.begin()))->lastFrame <= backFramesToCheckForClosePushedOut) {
					add = *(closeTracked[0].second->pushedOutGroups.begin());
					disposedGroups.erase(disposedGroups.find(add));

					ReconnectGroups(add, newlyFoundGroups[i], img, framesCount, nullptr);
				}
				//

				//AFTER ALL RULES
				//adding the new (or old) group to the list of new tracked groups
				if (add != nullptr) {
					newTrackedGroups.push_back(add);
				}

			}

		}
		*/

		/*trackedGroups = newTrackedGroups;
		
		if (maximumGroupsCount<trackedGroups.size()) {
			sort(trackedGroups.begin(), trackedGroups.end(), [](const TrackingData *d1, const TrackingData *d2) {return d1->positions.size()*d1->previous.size()>d2->positions.size()*d2->previous.size(); });
			trackedGroups.resize(maximumGroupsCount);
		}

		for (auto si = currentlyDisposedGroups.cbegin(); si != currentlyDisposedGroups.cend(); ++si) {
			disposedGroups.insert(*si);
		}

		vector<BoundingBox> boundingBoxes;
		for (int gi = 0; gi<trackedGroups.size(); ++gi) {
			BoundingBox boundingBox = GetBoundingBox(trackedGroups[gi]->positions);
			boundingBox.frame = framesCount;
			boundingBox.type = BoundingBoxType::NORMAL;
			boundingBox.meanColor = trackedGroups[gi]->meanColor;
			trackedGroups[gi]->previous.push_back(boundingBox);
			trackedGroups[gi]->lastFrame = framesCount;
			boundingBoxes.push_back(boundingBox);
		}

		for (auto si = currentlyDisposedGroups.cbegin(); si != currentlyDisposedGroups.cend(); ++si) {
			TrackingData *currentlyDisposedGroup = *si;
			if (currentlyDisposedGroup->pushedOut == true) {
				if (currentlyDisposedGroup->previous.size()>0) {
					currentlyDisposedGroup->previous[currentlyDisposedGroup->previous.size() - 1].type |= BoundingBoxType::PUSHED_OUT;
					if (currentlyDisposedGroup->pushedOutBy != nullptr && currentlyDisposedGroup->pushedOutBy->previous.size()>0) {
						currentlyDisposedGroup->pushedOutBy->previous[currentlyDisposedGroup->pushedOutBy->previous.size() - 1].type |= BoundingBoxType::PUSHER;
					}
				}
			}
		}
		*/

		img.copyTo(previous);
		//Mat img2;
		//img.copyTo(img2);
		img.copyTo(img);

		//DrawTrajectories(img2, trackedGroups, trajectoryDrawingLength);
		//DrawRectangles(img2, boundingBoxes);

		resize(img, img, Size(img.cols / f, img.rows / f));
		//resize(img2, img2, Size(img2.cols / f, img2.rows / f));

		/*char framesCountText[17];
		sprintf(framesCountText, "%d", framesCount);
		putText(img2, framesCountText, Point(125, 50), CV_FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255));*/

		Mat displayBackground;
		resize(background, displayBackground, Size(background.cols / f, background.rows / f));
		
		keyPressed = waitKey(1);
		if (keyPressed == 'd' || keyPressed == 'D') {
			//while (1) {				
				img.copyTo(terrainSelectionImg);
				drawContours(terrainSelectionImg, contours, -1, Scalar(0, 255, 255));
				teamSelection.clear();
				SelectTeam();

				if (!analyse(img, contours)) printf("problem");
				printf("%d\n", histData[0].id);
				printf("%d\n", histData[1].id);
				Mat z = Mat::zeros(50, 50, CV_8UC3);
				for (int i = 0; i < myTracking.size(); ++i) {
					printf("%d\n", myTracking[i].team);
					imshow("testing_r", z);
					imshow("testing_r", myTracking[i].rect);
					waitKey();
				}

				for (int i = 0; i < myTracking.size(); ++i) {
					if (myTracking[i].team == 0 || myTracking[i].team == 1) polylines(img, myTracking[i].contourPoints, true, Scalar(0, 0, 255), 1, 8);
					else polylines(img, myTracking[i].contourPoints, true, Scalar(255, 0, 0), 1, 8);
				}
				imshow("i", img);
				keyPressed = waitKey();
				//if (keyPressed == 13) break;
			//}			
		}
		else drawContours(img, contours, -1, Scalar(0, 255, 255));

		imshow("i", img);
		imshow("b", displayBackground);
		//imshow("i2", img2);

		

		img.release();
		//foreground.release();
		//img2.release();
		if (framesCount == 300) {
			//break;
		}
	}


	delete bf;

	FreeIntMatrix(backgroundFlag, rows);
	FreeIntMatrix(foregroundFlag, rows);
	//FreeIntMatrix(visited, rows);
	FreeIntMatrix(flag, rows);
	FreeIntMatrix(suddenlyChanged, rows);
	FreeIntMatrix(spreadData, rows);
	FreeIntMatrix(terrainMask, rows);
	FreeIntMatrix(currentBackgroundFlag, rows);

	//FreeTrackingDataPointerMatrix(owners, rows);
}

int main(int argc, char **argv){
	
	Test97();
	
	return 0;
}
