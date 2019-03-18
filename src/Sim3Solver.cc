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


#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{


Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
    mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();

    mN1 = vpMatched12.size(); // 以关键帧1 为基准的关键点个数

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    cv::Mat Rcw1 = pKF1->GetRotation();
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);

    size_t idx=0; // 记录有效匹配点对数
    for(int i1=0; i1<mN1; i1++)
    {
        if(vpMatched12[i1])
        {
            MapPoint* pMP1 = vpKeyFrameMP1[i1];
            MapPoint* pMP2 = vpMatched12[i1];

            if(!pMP1)
                continue;

            if(pMP1->isBad() || pMP2->isBad())
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(indexKF1<0 || indexKF2<0) // 地图点与关键帧没有关联
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1]; // 获取索引对应的关键点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210*sigmaSquare1); // ???? 为什么选取这个参数？？？:这个对应概率为 99% 时，一个像素误差，参考 A2.2 卡方分布表
            mvnMaxError2.push_back(9.210*sigmaSquare2);

            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);

            cv::Mat X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);

            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);

            mvAllIndices.push_back(idx);
            idx++;
        }
    }

    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();
}
// 默认参数如下：
//   probability = 0.99,
//   minInliers = 6 ,
//   maxIterations = 300
// 在实际用的时候这里 minInliers = 20
void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;    

    N = mvpMapPoints1.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N) // 有效内点集就是最小内点集，那么只能迭代一次
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3))); // 这个公式有问题？？？

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts)); // 更新 RANSAC 最大迭代次数

    mnIterations = 0;
}

//! \see Closed-form solution of absolute orientation using unit quaternions 三对点确定 sim3 相似变换
//! \param nIterations 指定的迭代次数(5)
//! \param bNoMore true: 达到了最大迭代次数。但是仍然没有达到要求。此时说明检测的闭环帧 pKF2 不好
//! \param vbInliers vbInliers[i] = value; 表示的是关键帧 1 对应的关键点 i 是否是内点， value = true; 表示是内点.与原始关键帧1的关键点是一一对应的
//! \param nInliers 对应最好的内点个数
//! \return 相似变换
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false); // 这一步很关键。保证无效的内点也是 false
    nInliers=0;

    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3,3,CV_32F); // 可以装 3 组点(列为主)
    cv::Mat P3Dc2i(3,3,CV_32F);

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations) // 这里实际上限制了最大迭代次数为 mRangsacMaxIts
    {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices; // RANSAC 随机选取的索引

        // Get min set of points 不能选取重复的元素
        for(short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1); // 获得指定元素区间的随机数

            int idx = vAvailableIndices[randi];

            mvX3Dc1[idx].copyTo(P3Dc1i.col(i)); // 点对赋值到指定位置
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

            vAvailableIndices[randi] = vAvailableIndices.back(); // 防止取出重复的元素
            vAvailableIndices.pop_back();
        }

        // 在初始化时，已经得到固定了尺度信息，为什么在还要计算相似变换？？？
        // (随着误差的累计，运动尺度会飘移，导致地图点尺度也会飘移。因此在遇到闭环时，两者之间运动其实相差一个尺度。也就是多了一个自由度，变为了 sim3)
        ComputeSim3(P3Dc1i,P3Dc2i); // 根据匹配的点对计算 Sim3 相似变换矩阵。

        CheckInliers(); // 计算当前相似变换矩阵对应的内点集个数

        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if(mnInliersi>mRansacMinInliers) // 这里判断内点集满足最低条件，就表示当前变换，内点个数足够多。这里这么做的原因是，这个函数仅仅得到一个初始值。后面其他函数还会再次进行优化！
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true; //
                return mBestT12;
            }
        }
    }

    // 达到了最大迭代次数，但是仍然没有满足 『足够内点』这一条件那么此次 Sim3Solver 求解失败。对应的闭环关键帧不好
    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

// P: 3x3 ，每一列是一个 3d 点坐标。在相机坐标系
// Pr: 3x3 ,每一列同样是一个 3d 点坐标。以质心为原点的坐标
// C: 3 个点的质心
//    计算公式参考： Closed-form solution of absolute orientation using unit quaternions 2C 部分 Centroids of the Sets of Measurements
void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    cv::reduce(P,C,1,CV_REDUCE_SUM); // 这里是把 3x3 矩阵 P 的列元素全部对应元素求和，变为一个 3x1 向量 C
    C = C/P.cols; // 求取质心坐标

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;
    }
}

//! \see 就是按照 Horn 论文 4C 部分介绍的计算步骤。
//! \brief 这里是要计算两个相机坐标之间的相似变换。变换是由点集 2 ---> 点集 1
//! \note P1 P2 是两幅图像的 3 对匹配点对。
//!     且 P1 p2 都是 3x3 矩阵，每一列都是一个 3d 点。 以 p1 为例。包含的点是相机坐标系的 3d 点。
//!    计算了下面的内部变量：
//!    Current Estimation 根据 3 个任选 点对，按照论文 Horn 计算步骤，得到的旋转、平移、尺度因子
//!    cv::Mat mR12i; // 旋转
//!    cv::Mat mt12i; // 平移
//!    float ms12i; // 单目尺度因子 s, 对于双目和 rgbd 为 1
//!    cv::Mat mT12i; // 上面计算出来的 s旋转 + 平移
//!    cv::Mat mT21i; // T21i^-1
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1) 3x3 CV_32F
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    ComputeCentroid(P1,Pr1,O1); // 计算质心，以及以质心为原点的点集坐标
    ComputeCentroid(P2,Pr2,O2);

    // Step 2: Compute M matrix

    cv::Mat M = Pr2*Pr1.t(); // 参考 Horn 论文 4A 部分,此时这个就表示由点集 2 ---> 点集 1

    // Step 3: Compute N matrix // 同样参考 Horn 论文 4A 部分 N 用 M 元素表达。

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;

    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation // 这里 evec[0] 是单位向量 4x1,最大特征值对应的特征向量

    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin(theta/2)*axis)
                                             // 根据参考论文中的给出的四元数部分q的公式有如下发现：虚数模长就是
                                            // 也就是这个向量 vec 的模长就是 sin(theta/2)。计算方式也可以参考 14 讲 3.19 式

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    double ang=atan2(norm(vec),evec.at<float>(0,0)); // sin(theta/2),cos(theta/2) 已知后，计算 tan(theta/2) 反求旋转角 theta

    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half // 轴角

    mR12i.create(3,3,P1.type());

    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

    // 利用 2D 部分计算尺度 s
    // Step 5: Rotate set 2

    cv::Mat P3 = mR12i*Pr2;

    // Step 6: Scale // 对照论文 Horn 的 2D Finding the Scale

    if(!mbFixScale)
    {
        double nom = Pr1.dot(P3); // 对于矩阵来说，其实就是把矩阵变为整个列向量。然后按照向量形式计算点乘.就是 sigma( rr,i . R(r'l,i) ) 就是 s 公式的分母
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        cv::pow(P3,2,aux_P3); // 因为旋转一个向量前后模长不变，所以这里直接计算旋转后的向量
        double den = 0;

        // 计算论文中 s 的分子部分
        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);
            }
        }

        ms12i = nom/den;
    }
    else
        ms12i = 1.0f;

    // Step 7: Translation

    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2; // 计算平移

    // 上面计算完 R t s 后即可计算相似变换
    // Step 8: Transformation

    // Step 8.1 T12
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
}

// 分别根据计算出来的变换矩阵 mT12i mT21i 将两个关键帧对应的 3d 点投影到对方的图像坐标系上。
// 根据提前设定的每个关键点对应的误差阈值，判断当前点是否属于内点
void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1; // 像素坐标
    Project(mvX3Dc2,vP2im1,mT12i,mK1); // 3d - 2d 投影变换，2 个 3d 点集分别投影到另一个图像坐标系上，进行内点的检查
    Project(mvX3Dc1,vP1im2,mT21i,mK2);

    mnInliersi=0;

    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i]) // 满足重投影误差
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}

// 获得 Ransac 估计的R
cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}
/// \brief 将 3D 点集根据内参和外参投影到另一个图像坐标系
/// \param vP3Dw 给定的 3d 点集
/// \param vP2D 变换后的 2d 图像坐标系下的点
/// \param Tcw 变换矩阵
/// \param K 内参矩阵
/// \return
void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;
        const float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

// 将给定的所有相机坐标系 3D 点 3x1。变换到图像坐标系上。得到图像坐标系点对 2x1
void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        const float invz = 1/(vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0)*invz;
        const float y = vP3Dc[i].at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy)); // 变换到图像坐标系
    }
}

} //namespace ORB_SLAM
