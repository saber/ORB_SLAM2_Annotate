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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"


namespace ORB_SLAM2
{
//! \brief 主要内容就是寻找更多匹配，可能是利用地图点投影到给定帧(关键帧/普通阵）之间匹配(基于关键点描述子汉明距离)，
//! 当然帧的 pose(T/Sim3) 已知（可能是一个初始值），或者是利用 BOW 寻找匹配。
//! \details 具体的寻找匹配都有如下(按照如下函数声明顺序)：
//!   1) 在追踪线程中，根据局部地图点(局部关键帧有限制)利用追踪帧的初始 pose 寻找更多匹配，进而为后面位姿优化做准备
//!   2) 在追踪线程中，追踪帧利用上一个跟踪帧 pose,根据运动模型得到初始 pose，然后寻找更多的匹配。之后才会使用 1) 的方法再次优化当前追踪帧 pose
//!   3) 在追踪线程中，当追踪失败时，会自动启动重定位模式，利用 bow 与关键帧进行搜寻更多匹配，然后得到初始 pose。之后在与当前帧再次寻找更多匹配
//!   5) 在重定位，追踪帧和关键帧进行搜索匹配。之前通过关键帧数据库查找与当前追踪帧有共同单词的关键帧和其进行一一匹配。看哪个关键帧与追踪帧匹配数多
//!   7) 在单目地图初始化时，利用两个参考帧和当前帧，进行关键点匹配。
//!   8) 在局部建图线程中创建地图点时做的。将当前关键帧的临近关键帧组和当前关键帧进行搜索匹配，进而三角化更多的地图点
//!   4)6)9) 都是在闭环线程中的同一个函数里面调用的，调用顺序是 6)->9)->4)，过程如下，首先是通过 bow 寻找潜在闭环帧和当前闭环帧匹配点对，
//!     匹配数符合要求后，再次通过 searchBySim3 寻找更多的匹配，匹配数量符合要求，说明这个潜在闭环帧很有可能是闭环帧，然后得到一个初始 scw 估计，
//!     在根据4) 投影匹配，根据 scw 将该潜在闭环帧对应的临近关键帧包含的所有的地图点投影到基准闭环帧，找到更多的匹配，匹配对数符合一定要求，才算是找到了闭环！
//!   10) 通过将地图点投影到关键帧上，寻找更多的匹配点对（内部可能会替换已有点对），用在局部建图线程函数
//!   11) 与 10）功能一样，也是通过投影到 pkF 上，寻找更多的匹配，这里用在闭环线程中，为了让找到的闭环帧的临近关键帧上的所有地图点，
//!      替换当前基准闭环帧不好的匹配关系。之后为本质图优化做准备
class ORBmatcher
{    
public:

    ORBmatcher(float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b); //(0)

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3); //(1)

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono); //(2)

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist); //(3)

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
     int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th); //(4)

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame. 关键帧地图点和当前普通帧 orb 特征点进行搜索匹配
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches); //(5)
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12); // (6)

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10); //(7)

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                               std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo); //(8)

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th); //(9)

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0); //(10)

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint); // (11)

public:

    static const int TH_LOW;    // 50   两个描述子算作匹配点的距离的阈值
    static const int TH_HIGH;   // 100
    static const int HISTO_LENGTH;  // 30 直方图横坐标范围


protected:

    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

    float RadiusByViewingCos(const float &viewCos);

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;    // 构造 ORBmatcher 时给定的参数，其用在描述子匹配汉明距离阈值筛选。
                        // if(bestDist<(float)bestDist2*mfNNratio)，满足此条件时，才有可能是一个匹配点对，当然后面还会有直方图筛选。
    bool mbCheckOrientation;    // true: 表示需要用直方图来筛选一些不好的匹配。一般情况下都需要！可以看 SearchForInitialization() 函数内部有使用过程！
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
