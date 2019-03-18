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


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"



namespace ORB_SLAM2
{
//! \brief 根据 Closed-form solution of absolute orientation using unit quaternions 论文，计算两个关键帧之间的相似变换。
//!   只要给出 3 个点就可以计算一个 Sim3 ，然后用 rnasac 迭代多次，得到最好的 sim3

class Sim3Solver
{
public:

    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);

    cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    cv::Mat GetEstimatedRotation();
    cv::Mat GetEstimatedTranslation();
    float GetEstimatedScale();


protected:

    void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

    void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

    void CheckInliers();

    void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
    void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);


protected:

    // KeyFrames and matches
    KeyFrame* mpKF1; // 闭环线程正在处理的关键帧(基准关键帧)
    KeyFrame* mpKF2; // 经过一致性检验的闭环关键帧

    // 下面存储的值，都是在匹配点对中。经过了检验对应的地图点是有效后，才加入的。
    // 而下面的索引号都是一一对应的 (个别不是的已经标记出)
    std::vector<cv::Mat> mvX3Dc1; // init :reserve = mN1,关键帧 1 对应的地图点，变换到关键帧 1 对应的相机坐标系
    std::vector<cv::Mat> mvX3Dc2; // init 与上同理。关键帧 2 对应的地图点，变换到关键帧 2 对应的相机坐标系
    std::vector<MapPoint*> mvpMapPoints1; // init : reserve = mN1,在匹配点对中，实际有效的地图点。关键帧 1 对应的
    std::vector<MapPoint*> mvpMapPoints2; // init 同上，后面同上
    std::vector<MapPoint*> mvpMatches12; // (原始匹配关系，没有检验是否都是有效的地图点)关键帧 1 与关键帧 2 匹配的点对. 内部元素是：关键帧1 序号为 i 的关键点匹配的关键帧 2 的地图点
    std::vector<size_t> mvnIndices1; // init: reserve = mN1, 实际大小就是 N。 mvnIndices[i] = idx: 记录 当地图点有效时，对应的关键帧 1 原始关键点/地图点的索引编号idx.
                                     // 通过索引 idx 就可以直接在关键帧 1 中找到对应关键点。
                                     // 内部元素个数其实就是实际有效匹配点对数。
    std::vector<size_t> mvSigmaSquare1;
    std::vector<size_t> mvSigmaSquare2;
    std::vector<size_t> mvnMaxError1; // 关键帧 1 图像坐标系上的重投影误差。 [i] 代表第 i 个关键点
    std::vector<size_t> mvnMaxError2; // 关键帧 2 图像坐标系上的重投影误差。 [i] 代表第 i 个关键点

    int N; // 实际有效的匹配点对数
    int mN1; // 以关键帧 1 为基准的,关键帧 1 和关键帧 2 初始匹配的点对数

    // Current Estimation 根据 3 个任选 点对，按照论文 Horn 计算步骤，得到的旋转、平移、尺度因子
    cv::Mat mR12i; // 旋转
    cv::Mat mt12i; // 平移
    float ms12i; // 单目尺度因子 s, 对于双目和 rgbd 为 1
    cv::Mat mT12i; // 上面计算出来的 s旋转 + 平移
    cv::Mat mT21i; // T21i^-1
    std::vector<bool> mvbInliersi; // 标记是否是内点 init = resize(N).
    int mnInliersi; // 当前计算的变换矩阵，对应的内点集个数 CheckInliers()

    // Current Ransac State // 下面值记录的是 RANSAC 迭代过程中最好的值。变量含义可以参考上面介绍的
    int mnIterations; // init = 0 当前迭代次数
    std::vector<bool> mvbBestInliers;
    int mnBestInliers; // init = 0
    cv::Mat mBestT12;
    cv::Mat mBestRotation; // mR12i 带有尺度信息。
    cv::Mat mBestTranslation;
    float mBestScale;

    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale; // 单目为 false,双目和 RGB-D 为 true

    // Indices for random selection
    std::vector<size_t> mvAllIndices; // init: reserve = mN1,最后一个元素就代表实际有效的匹配点对数 maxN。每个元素都是从 0-maxN 依次递增的。

    // Projections
    std::vector<cv::Mat> mvP1im1; // 在关键帧 1 图像坐标系上的点坐标，与 mvX3Dc1 是一一对应的
    std::vector<cv::Mat> mvP2im2; // 与上同理

    // RANSAC probability
    double mRansacProb; // init = 0.99 随机采样 s 个样本中至少有一次没有野值的概率为 p

    // RANSAC min inliers
    int mRansacMinInliers; // 最小内点集个数 init = 6，在 ComputeSim3 时，给了 20

    // RANSAC max iterations
    int mRansacMaxIts; // 最大迭代次数 init = 300,在 ComputeSim3 时，给了 300

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    cv::Mat mK1; // 关键帧 1 对应的内参矩阵
    cv::Mat mK2;

};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
