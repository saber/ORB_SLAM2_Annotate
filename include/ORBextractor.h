/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;    // 当前节点中包含的关键点个数
    cv::Point2i UL, UR, BL, BR; //  一个节点对应的： UL: 左上角坐标   UR: 右上角坐标    BL: 左下角坐标   BR: 右下角坐标
    std::list<ExtractorNode>::iterator lit; // 实际上指向自己。
    bool bNoMore;   // 表示该节点不可在细分，即该节点就是叶子节点了
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask, // 利用 cv::Mat() 表示 Mask 忽略
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }
    // size= nlevels
    std::vector<cv::Mat> mvImagePyramid;    // 图像金字塔，每层保存的图像(降采样后的)

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
    // 老版本的计算关键点,此时系统没有使用
    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern; // 将模式数组两个点作为 cv::Point

    // 下面针对 TUM 数据集参数
    int nfeatures;  // 1000
    double scaleFactor; // 1.2
    int nlevels;    // 8
    int iniThFAST;  // 20
    int minThFAST;  // 7
    // size = nlevels
    std::vector<int> mnFeaturesPerLevel;    // 每一层的特征数量(根据一副图像总特征数，按照每层/scalFactor，可以依次计算每一层特征数量)

    std::vector<int> umax;  //  保存关键点小块获取的模式，需要看 IC_Angle() 函数以及 ORBextractor构造函数即可明白！

    // 下面 4 个 vector 大小都是 nlevels = 8
    std::vector<float> mvScaleFactor;       //  [i] = [i-1] * scaleFactor   ; [0] = 1，内部值为 1 1.2 1.2^2 1.2^3...
    std::vector<float> mvInvScaleFactor;    //  [i] = 1/mvScaleFactor[i]
    std::vector<float> mvLevelSigma2;       //  [i] = mvScaleFactor[i]*mvScaleFactor[i];    [0] = 1
    std::vector<float> mvInvLevelSigma2;    //  [i] = 1/mvLevelSigma2[i]
};

} //namespace ORB_SLAM

#endif

