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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}
// 用在追踪过程中。SearchLocalPoints() 使用的。将局部地图点投影到当前帧。寻找匹配。这里的匹配后。其实3d点不需要计算了。
//    F 内部的 mvpMapPoints 已经改变了。可能会增加一些地图点
//    返回：新的地图点和当前帧匹配的总对数
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;
    // 这里仅处理能够投影到 Frame F 上的点集。(不包含之前已经存在 F 上的了)
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView) // 说明该地图点已经属于当前 Frame 了或者无法投影到当前帧
            continue;

        if(pMP->isBad()) // 坏点
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r*=th;

        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0) // ???为什么被关键帧观测过的点不处理？？？
                    continue;

            if(F.mvuRight[idx]>0) // 双目
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            F.mvpMapPoints[bestIdx]=pMP; // 找到匹配点对
            nmatches++;
        }
    }

    return nmatches;
}
// viewCos 是 cos(角度) 值。角度越小 cos 值越大。越小角度。说明地图点与当前帧越符合要求。那么半径可以小些
float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998) // 接近零度
        return 2.5;
    else
        return 4.0;
}

// 检查点到对极线距离平方是否满足小于 1 个像素误差。true: 满足条件，才算是符合对极约束
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c] 获得对极直线方程
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den; // 点到对极线距离 平方

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave]; // 为什么要乘？？？？
}
// 关键帧和普通新来的帧进行特征匹配，利用了 DBoW2 特征向量加速匹配
// vpMapPointMatches[i] = pMap: i： 表示当前帧关键点索引 pMap : 表示关键帧的地图点。这两个是一个匹配的点对 3d-2d
// 一般来说。这种匹配的点对很少。因为关键帧初始匹配时有地图点的关键点也就是 100 个左右。
// 此时我们实际上是用关键帧的地图点和当前普通帧的 2d 关键点进行匹配。其中还有一些误匹配。这样一来成功匹配上的点很少很少。如何才能让这个匹配更多？
//    实际上在局部建图线程中，对一些关键帧没有对应地图点的关键帧会做一些新的匹配和三角化，所以当前关键帧的地图点个数会增多。所以间接让这个匹配更多！
// 返回有效的匹配个数
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    // 获取 pKF 关键帧所有有效的地图点(和关键帧自己本身关键点对应)
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec; // 关键帧在成功加入时，就已经执行了特征向量和词袋向量的计算了

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();
    // DBoW2::FeatureVector :std::map<NodeId, std::vector<unsigned int> > // 这个 map 是按照升序排列的，
            // 按照平均来说这里每个 NodeId 是 k=10，其中后面对应的描述子如果也按照平均分配的话。那么这里把所有关键点分配到其中一个 NodeId
            // 那么此时就是 1/10。这样算起来。匹配过程平均来说是按照 1/10 的特征数进行暴力匹配。况且下面选取了 NodeId 对应的关键帧中能够恢复出地图点的那些描述子
            // 与普通帧的相对应的 NodeId 描述子进行暴力匹配。这样数量会相应的降低。但是在单目初始化时，有对应地图点的关键点其实也就是 100 多一点。那么按照下面的方式
            // 其实真正和普通帧能够配对上的普通关键点个数很少了。然后新的地图点加入其实没有多少个
    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first) // NodeId 一致,则在对应的 second 包含的描述子中找潜在的匹配
        {
            const vector<unsigned int> vIndicesKF = KFit->second; // 关键帧自己本身关键点对应的描述子序号
            const vector<unsigned int> vIndicesF = Fit->second; // 同上，这里是该普通帧的描述子序号，也是关键点序号
            // 遍历存储的所有关键帧描述子序号
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF]; // 关键帧对应的关键点序号

                MapPoint* pMP = vpMapPointsKF[realIdxKF]; // 取出当前关键点对应的地图点（不一定存在）

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF); // 获取关键帧关键点(有对应地图点)对应的描述子

                int bestDist1=256; // orb 描述子最大距离
                int bestIdxF =-1 ;
                int bestDist2=256;
                // 对于关键帧中有对应地图点的描述子，与普通帧的潜在描述子进行一一比较配准。
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF]) // 这里有点多余？不多余，是因为每次循环一圈，就会产生与关键帧对应的点匹配点。
                                                    // 此时记录普通帧关键点 realIdx 对应的关键帧地图点 pMP.这样做保证了当前普通帧关键点和地图点是一一对应的
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                    const int dist =  DescriptorDistance(dKF,dF); // 计算描述子距离(关键帧---当前帧),256 维向量
                    // 计算与当前关键帧描述子，距离最短和第二短的当前普通帧的描述子序号以及距离
                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2)) // 在初始化完毕时这里的 mfNNratio = 0.7
                    {
                        vpMapPointMatches[bestIdxF]=pMP; // 记录当前普通帧 bestIdxF 关键点序号，有关键帧 pMP 对应的地图点。先到先记录，然后就不变了。
                                                        // 后面没有像初始化时那种筛选机制，这样实际上会造成一定的误匹配！需要改进！当然作者可能考虑到时间问题

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if(mbCheckOrientation)
                        {   // 建立直方图
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor); // 直方图坐标系坐标。以 30° 为一个分段
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        } // 调整以找到对应相同树节点的 id。才能实现匹配
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first); // >= Fit->first
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        // 在直方图中找到角度差最多的角度差索引
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3) // 满足此条件，说明都是比较好的匹配。
                continue;
            // 不好的匹配认为是错误匹配。这里直接 nmatches--
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

//! \brief 找到好的匹配点对，将 vpPoints 内的地图点全部投影到 pKF 上，寻找潜在的匹配点对。 然后存入 vpMatched 中。
//!        (有可能关键帧 pKF 上的关键点没有对应的地图点，但是仍然有机会寻找到匹配的地图点)
//! \param pKF
//! \param Scw  世界到 pKF 的一个相似变换
//! \param vpPoints 所有的地图点。（将要投影到 pKF 上的地图点）
//! \param vpMatched  潜在的匹配点对。 pKF 关键帧上的地图点（基准）----> 检测到的真正闭环帧上的匹配的地图点
//! \param th  10
//! \return 当前 pKF 关键帧的匹配点对数
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0))); // 旋转 R 的列向量是单位向量。因此 row(0).dot(row(0)) 就是 s^2
    cv::Mat Rcw = sRcw/scw; // 不带尺度的旋转
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw; // 将原来的 sR + t 等效写为  R + t/s 的一个刚体变换。等效指的是：利用这个 [R,t/s] 刚体变换投影地图点到相机图像坐标系是一样的结果。
    cv::Mat Ow = -Rcw.t()*tcw; // 求出相机光心在世界坐标系下的坐标

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL)); // 去除空的匹配

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw; // (此时相机坐标系下的坐标与真实值相差一个尺度因子 s，因为上面仅仅等效了投影到图像坐标系下的坐标。对于相机坐标是相差尺度因子 s 的)

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal(); // 这种方式仅仅在 Fuse() 函数和该函数内部用到了。为什么其他地方不会用？？？
                                       // 因为其他 SearchByProjection() 函数都是在追踪线程做的。那个时候的位姿其实不太准确.
                                       // 尽量放宽条件找到更多的匹配。这里是为了严格找到匹配点对。不好的点对会使闭环纠正误差更大

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx]) // 目的是为了找到新的匹配点对
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}
// 下面讨论的点都是去畸变后的
// vnMatches12:   保存的是 F1 图像 与 F2 图像配准好的关键点序号（对应 F2 图像）,
//                 此时 index = vnMatches12[i] 如果值不是 -1 ，那么说明 F1 图像上序号 i 对应的关键点可以在
//                 F2 图像上找到对应的匹配点，并且匹配点序号是 index。可以通过 F2.mvKeysUn[index]
//                 获取图像 mCurrentFrame 上的关键点。
// vbPrevMatched: 保存的是 F1 图像 与 F2 图像配准好的关键点（在 F2 图像上）
//                 可以这样理解： vbPrevMatched[i] = point; 此时 i 表示 F1 图像上的关键点序号。point就表示
//                 匹配的关键点 (在图像 F2 上),如果数组元素为空，即没有点，那么说明没有匹配上
// windowSize: 100 以第一副图像关键点坐标为中心，然后在第二幅图像上的窗口内找备选匹配点
// nmatches: 返回的有效匹配对数
// 综述：对图像1的所有关键点，在图像2上（备选关键点(可以指定所在金字塔层)）进行暴力匹配。找到最好的匹配点（度量是汉明距离小于某个合适的阈值）。
//      对于图像2关键点被图像1多个点匹配上。那么只统计最后一次配对。这是初始寻找匹配。在寻找的过程中统计匹配点对之间的角度差直方图。
//      （实际上这个直方图对于刚刚说的多次匹配情况没有做冗余处理，可能会有点影响，但是不大）。之后根据直方图呈现的信息。
//      对那些潜在的误匹配进行去除（实际上如果正常匹配，一般角度差应该集中在一个值，即直方图只有一个峰值，但是程序处理过程中，统计了三个峰值，其他的小峰值都是误匹配）
//      ，最后对 vbPrevMatched 进行更新，实际上这个变量存储是图像2被匹配的关键点。
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0; // 良好匹配的个数
    // 直接看前面的参数介绍(记录匹配的参数，目前是记录第一幅图像的关键点，对应匹配的第二幅图像的关键点序号，如果内部为 -1 说明这个图像1关键点没有找到匹配的图像2的关键点)
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];  // vector<int> rotHist[30] 表示两个匹配点角度差构成的直方图。直方图内部保存的是匹配好的第一帧图像关键点索引。
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);   // 记录第二幅图像上关键点与第一幅图像的关键点匹配的最短距离
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1); // 记录第二图像关键点对应匹配的 第一幅图像关键点序号

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;    // 当前关键点所在图像金字塔层数
        if(level1>0) // 只处理原始图像的关键点，略过降采样的那些关键点
            continue;
        // 在第二副图像上获取候选描述子(索引方式记录)。与第一幅图像的关键点描述子进行匹配。
        // 里面用了在 Frame 构造函数中将关键点分发到网格内。通过网格来快速寻找范围 windowSize 内的潜在配对点
        // 这里的寻找匹配值在金字塔层数相同的情况下找到。
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX; // 两个关键点描述子之间最短的距离
        int bestDist2 = INT_MAX;    // 第二短的距离
        int bestIdx2 = -1;  // 最短距离对应的第二副图像描述子索引
        // 给定第一幅图像的一个描述子，遍历第二幅图像的候选描述子。找到最好的匹配。和第二好的匹配 （好坏的标准是 两个描述子之间的距离）
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            // 计算描述子之间距离，用来匹配(待细看！)
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            // 更新第一和第二短描述子之间的距离
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
        // 对上面获得的最短距离 小于最低阈值，以及小于 第二小阈值的 mfNNratio 倍数，此时才算是好的匹配。这当然是第一层筛选。后面还会有角度直方图层次筛选！
        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)    // 说明第二幅图像上序号为 bestIdx2 的关键点 已经和之前的第一幅图像的某个关键点匹配好了。
                                                // 而此时第一幅图像又有关键点与该特征点匹配上了。那么很明显出现了歧义。这里简单的做了替换
                {                               // 但是理想做法应该是比较两次 bestDist 的值，哪个值小，应该匹配对应的关键点！
                    vnMatches12[vnMatches21[bestIdx2]]=-1;  // 重置记录 12 是否匹配的值，那么之前的图像1的关键点相当于没有匹配上。
                    nmatches--; // 匹配好的计数器更新
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                // 统计角度直方图，方便后面检验匹配是否正确。如果匹配正确，说明某个角度（匹配点角度差）对应的第一幅图像的关键点特别多
                // 匹配错误就去除那些不好的匹配：清理 vnMatches12[]对应的值为 -1 ，就表示图像1对应的关键点没有匹配
                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);   // bin[0,12]
                    if(bin==HISTO_LENGTH)   // bin == 30?， bin 只可能 rot[1,360] /30 = [1/30,12] 所以这里不可能成立！
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1); // 可能在统计角度直方图，直方图划分为 [0,12]，这个直方图统计的关键点个数会 >= nmatches
                }                               // 因为上面在第一幅图像中多个关键点在第二幅图像上找到了同一个关键点与之对应，此时 nmatches 相当于不增加
                                                // 但是这里仍然会进行统计，所以导致统计的数量 >= nmatches，似乎没有太大副作用
            }
        }

    }
    // 根据角度直方图(匹配点角度差)，检验匹配是否正确。如果匹配正确，说明某个角度（匹配点角度差）对应的第一幅图像的关键点特别多（相当于聚类），这样才算是真正匹配上了
    // 对于那些角度对应的关键点少的情况。肯定就是误匹配的。此时匹配错误就要去除那些不好的匹配：置 vnMatches12[]对应的值为 -1 ，就表示图像1对应的关键点没有匹配
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        // 取出角度直方图中关键点数量最多三个值。当然如果第2、第3多的数量小于第1多数量的 0.1 倍。那说明 第2 第3多角度对应的关键点也是误匹配的
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
        // 这里对误匹配 vnMatches12 置位 -1，更新匹配结果 nmatches
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            //
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];   // 图像1关键点序号！
                if(vnMatches12[idx1]>=0)    // 说明图像1上的关键点有对应的匹配好的图像2上的关键点！
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched   // 这里写成 iend1=F1.mvKeysUn.size() 会更好，虽然下面的结果一样
    // 更新 vpPrevMatched ：记录了上次匹配中，第二副图像有效匹配的特征点
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)  // 与第一幅图像有效匹配的点
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}
//! \brief 当前函数仅仅在闭环线程中调用。因为下面一切概念以闭环线程用到的概念为基准。当然传统上就是找到两个关键帧之间的匹配点对。返回匹配点对数
//! \param pKF1 闭环线程正在处理的关键帧
//! \param pKF2 经过一致性检验的初始闭环帧
//! \param vpMatches12 :[i] = MapPoint2; i 表示关键帧 1 对应的关键点序号。 MapPoint2 是此时该关键点对应的关键帧 2 相应的地图点.匹配点对 3d-2d。需要注意的是
//!    在寻找匹配点对的时候，关键帧 1 对应的关键点原来就有属于自己的地图点。这里仅仅为了找匹配，看看这个关键点与关键帧 2 哪个地图点对应。
//! \return 有效匹配的点对数
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false); // 表示当前关键帧 2 对应的关键点已经找到了匹配

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin(); // std::map<NodeId, std::vector<unsigned int> >
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first) // 找到相同节点.然后进行描述子之间的匹配！
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++) // 遍历关键帧 1 特征向量节点 id 对应的描述子。和关键帧 2 特征向量节点 id 对应的描述子进行匹配
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1]; // 地图点存在，才找匹配点对。
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++) // 遍历关键帧 2 特征向量节点 Id 对应的描述子
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        } // 下面两个步骤都是为了使得两个关键帧对应的节点 id 一致。才能进行描述子之间的匹配
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}
// 对关键帧 1 中关键点没有对应的地图点。找到关键帧 2 中的匹配。通过对极约束进行判断是否是匹配点对。为之后的三角化新的地图点做准备！
// vMatchedPairs: <size_t, size_t>: 第一个是关键帧 1 对应的关键点匹配序号：第二个是对应的关键帧 2 匹配点的序号
// F12: 给定的两个关键帧之间的基础矩阵(可以通过两个关键帧之间的位姿和内参矩阵进行求解)
//    return: 成功匹配的数量
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image，将关键帧 1 --->关键帧 2
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w; // 关键帧 1 光心在关键帧 2 坐标系下的坐标
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx; // 关键帧1光心在关键帧 2 的像素坐标，实际上就是极点 e2 {参照 14 讲书上 141 页下面对极几何}
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints 对没有跟踪的关键点找到对应的匹配！对于关键点有对应的地图点，我们略过不处理！
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0; // 最后匹配的有效个数
    vector<bool> vbMatched2(pKF2->N,false); // 记录关键帧 2 对应的关键点索引是否已经和关键帧 1 进行匹配了
    vector<int> vMatches12(pKF1->N,-1); // vMatches12[idx1]=bestIdx2; 表示在关键帧1的一个没有对应地图点的关键点号 idx1，对应匹配关键帧 2 序号为 bestIdx2，这个变量是匹配点对

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
    // 注意这里是给定两个关键帧的位姿，然后找到匹配。所以下面用了点到对极线的距离误差在 1 个像素范围内，表示潜在的匹配
    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first) // 两个节点 id 相同，则保存的对应描述子之间可能存在匹配
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++) // 遍历特征向量 1 的所有描述子序号，然后在特征向量 2 中遍历查找
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1); // 获得对应的地图点
                
                // If there is already a MapPoint skip
                if(pMP1) // 表示当前的关键点对应的地图点已经被三角化过了。我们需要的是没有三角化过的关键点。然后将其三角化
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0; // 双目

                if(bOnlyStereo) // == false
                    if(!bStereo1)
                        continue;
                // 下面是新的关键点找到对应的匹配！
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1]; // 双目？
                
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1; // 表示符合匹配的点对索引
                // 在关键帧 2 特征向量中找到符合匹配
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2) // 如果该地图点已经匹配了，或者该关键点对应有地图点。那么这里不做匹配查找
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0; // 双目

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2) // 对于单目来说，这里成立
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave]) // 这个表示什么？？？？？？？？？？？？？？？？？？？？？？
                            continue;
                    }

                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2)) // 检查当前假定的匹配对是否满足对极约束(1 个像素误差之内)
                    {
                        bestIdx2 = idx2; // 更新最好索引
                        bestDist = dist; //更新最近距离
                    }
                }
                
                if(bestIdx2>=0) // 表示有了符合条件的匹配
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2; // 记录匹配关系
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first); // 这个从侧面可以知道，特征向量就是按照从小到大的顺序进行排序的
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++) // 对于统计检查不是匹配的点对，需要剔除
            {
                vMatches12[rotHist[i][j]]=-1; // 清除不是匹配的关系
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear(); // 使得 size() == 0
    vMatchedPairs.reserve(nmatches); // 有效匹配个数

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0) // 表示没有匹配上，或者对应的关键点本身就有地图点
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i])); // 匹配关系
    }

    return nmatches;
}
// pKF: 关键帧
// th: 默认 = 3
// vpMapPoints: 将要被投影的地图点
// 将地图点投影到给定的关键帧上。在关键帧 pKF 上找到与该地图点最佳匹配关键点。如果这个关键点在关键帧 pKF 上原来就有对应的地图点。
// 那么需要根据条件选择其中一个地图点（另一个地图点相应的被替换）。如果这个关键点在关键帧 pKF 上原来没有对应的地图点。那么就要
// 让这个地图点与关键点之间建立联系。pMP->AddObservation(pKF,bestIdx); // 记录地图点被 pKF 关键帧观测过。
//  pKF->AddMapPoint(pMP,bestIdx); // 关键帧的关键点和地图点进行关联
//    return: 返回与 pKF 成功匹配的地图点个数
//    注意：这里的地图点 vpMapPoints 有些点可能最后被地图删除
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0; // 记录在当前地图点中有多少个地图点与关键帧 pKF 进行了成功的匹配

    const int nMPs = vpMapPoints.size();
    // 地图点不存在，则不处理
    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP) // 地图点不存在,无法投影，接下来就没法处理
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF)) // 当前地图点是坏点不需要处理。地图点记录当前关键帧，这样就不需要下面寻找与当前地图点之间的匹配了。
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw; // 关键帧坐标系坐标

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz; // 归一化坐标系
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx; // 地图点投影到关键帧图像坐标系
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz; // 双目有效

        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 获得尺度不变距离
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow; // o-->p 的向量
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF); // 预测当前点在关键帧金字塔图像上的第几层

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel]; // 以原始图像尺度为基准给定的 th 阈值，因此金字塔图像上要应用这个阈值时，需要乘上一个尺度因子

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius); // 在关键帧 pKF 上，找到潜在的匹配点索引

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius，循环遍历上面得到的潜在的匹配点，找到与当前地图点描述子更好的匹配

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256; // 最短汉明距离
        int bestIdx = -1; // 对应最短汉明距离时，关键点序号
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) // 必须在预测的金字塔层数上下一个金字塔层范围内
                continue;

            if(pKF->mvuRight[idx]>=0) // 双目
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else    // 单目
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx; // 投影误差 x 坐标
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey; // 计算投影点到潜在匹配点之间的距离

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99) // 这个是换算到原始金字塔图像上{1 个像素误差}，其实原理还未明白？？？？？？？？？？？？
                    continue;
            }
            // 此时说明，刚刚哪个点确实与投影点相差一个像素，则可能是匹配点，下面需要计算汉明距离，再次确认，实际上这里汉明距离如果度量的不准，一切都白费了。所以用什么度量很重要！
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF) // 在 pKF 关键帧上已经有了对应的地图点，那么需要检查看看该关键点对应的两个地图点哪个与其是准确对应关系。谁被关键帧观测的次数多。然后保留就那个
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations()) // 当前地图点被观测的次数少，被替换
                        pMP->Replace(pMPinKF); // pMP 地图点被替换
                    else
                        pMPinKF->Replace(pMP);
                    // 在这里经过打印，可以发现两个地图点的 mnFirstKFid 可能是他那个一个。那么这个地图点在追踪时，作为局部地图点，就是会同时增加 mnVisible mnFound 两个数
//                    std::cout << "打印两个地图点的 id: " << pMPinKF->mnFirstKFid << " " << pMP->mnFirstKFid << std::endl;
                }
            }
            else // 如果在 pKF 上对应的序号，没有地图点，那么需要添加 pKF 与当前地图点之间的联系。并且更新地图点被哪个关键帧观测
            {
                pMP->AddObservation(pKF,bestIdx); // 记录地图点被 pKF 关键帧观测过。
                pKF->AddMapPoint(pMP,bestIdx); // pKF 关键帧需要为关键点新增加地图点

            }
            nFused++;
        }
    }
    // 这里其实地图点更新后没有更新一下关键帧对应的信息 pKF->UpdateConnections()，
    // 其实本函数仅仅在 LocalMapping::SearchInNeighbors() 函数中调用，局部建图正在处理的关键帧对应的地图点投影到临近关键帧中。
    // 此时临近关键帧应该更新一下对应关系才对。因为这些临近关键帧很可能会在跟踪线程中用到！！！！
    return nFused;
}

//! \brief 将 vpPoints 地图点利用相似变换矩阵 Scw 投影到 pKF 关键帧图像坐标系上，在 pKF 对应的地图点中找到匹配关系。用 vpReplacePoint 记录 vpPoints 匹配到 pKF 中的地图点
//!        且 vpReplacePoint 与 vpPoints 是一一对应的
//! \param pKF 投影到目标关键帧
//! \param Scw pKF 对应的相似变换矩阵
//! \param vpPoints 将要被投影的地图点集
//! \param th :4 投影搜索半径阈值（如何设定？？？）
//! \param vpReplacePoint 记录 vpPoints 中的地图点匹配的地图点（pKF 对应的）
//! \return
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0))); // 获得尺度信息 s
    cv::Mat Rcw = sRcw/scw; // 不带尺度的原始旋转 R
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw; // t/s   ，此时变为 R + t/s,投影点到图像坐标系没有影响，但是相机坐标系就差一个尺度因子 s
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0; // 成功匹配的点对数

    const int nPoints = vpPoints.size(); // 将要投影的地图点个数

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal(); // 在使用这个之前不需要更新当前地图点的 Normal 向量吗？？

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF; // 记录将要被替换的 pKF 对应的不好的地图点，之后在退出该函数后，遍历替换！
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx); // 新增关键帧和地图点之间的关联
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}
//! \brief 在关键帧 1 和关键帧 2 上找到新的匹配关系 vpMatches12 具体含义看参数说明。利用计算出来的尺度因子得到的相似变换。
//!        分别将两个关键帧上的所有没有匹配的关键点对应的地图点投影到对方上面.寻找新的匹配关系。最后根据一致性，找到两次投影匹配重复的匹配点对。
//!        认为是真正的匹配点对。然后更新 vpMatches12 参数。
//! \param pKF1 基准关键帧
//! \param pKF2 潜在闭环帧
//! \param vpMatches12(内部会增加新的匹配关系，已经有的匹配关系不会改变)： 匹配关系 vpMatches12[i] = Mapp; 表示关键帧 pKF1 的关键点对应的关键帧 pKF2 的地图点
//! \param s12 尺度因子，对于双目和 RGB-D 来说为 1.对于单目来说是计算出来的。
//! \param R12
//! \param t12
//! \param th 7.5:含义未知？？
//! \return 当前又找到的点对数
//! \question 这里的寻找匹配点对为什么与之前的不同？？感觉简化了不少？？是因为计算原因吗？？？
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx; // 内参
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false); // 标记相应的关键点已经有对应的匹配点
    vector<bool> vbAlreadyMatched2(N2,false);

    // 记录已经存在的匹配关系
    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1); // 记录关键点对应的关键帧 2的关键点序号
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search 对于单目来说，做的工作就是一个相似变换。寻找匹配的过程其实是一样的。
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1]) // 仍然表示关键点有对应的地图点才会寻找匹配
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;
        // 转化为相机图像坐标系，检查投影点是否在对应图像上.检查距离
        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius); // 在关键帧 2 上找到潜在的匹配点

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++) // 找到最好的匹配！
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH) // 这个阈值会不会有点大？？
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF1 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1) // 表示两次查找的结果相同，认为是真正的匹配点对
            {
                vpMatches12[i1] = vpMapPoints2[idx2]; // 增加匹配点对
                nFound++; // 找到的匹配点对数
            }
        }
    }

    return nFound;
}
// 根据上一帧的地图点与当前帧的关键点进行匹配。（一般来说当前的 pose 都是一个初始值）
// 返回：此时有效的匹配点数
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw; // 当前相机光心在世界坐标系下的坐标

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw; // 当前相机光心到上一帧相机坐标系下的坐标

    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono; // false 如果是单目调用这个函数
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono; // false

    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i]; // 获取一个上一帧的地图点，与当前帧的潜在匹配关键点匹配

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                cv::Mat x3Dw = pMP->GetWorldPos(); // 世界坐标系 3d 点坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw; // 当前坐标系下的 3d 点坐标（这个点其实是不准确的，因为当前帧 pose 是根据匀速运动模型来推断的）

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                if(invzc<0) // 说明当前点深度为负值，不在当前相机坐标系下。但是目前不在当前相机坐标系，但是优化当前帧 pose 后，
                            // 可能仍然在当前相机坐标系下。因为当前帧 Pose 是根据运动模型推断的，不够准确
                    continue;
                // 变换到当前相机图像坐标系下{先变为归一化平面，然后在根据内参变为图像坐标系}
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
                // 判断是否在图像内
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                int nLastOctave = LastFrame.mvKeys[i].octave; // 关键点对应的金字塔层数

                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave]; // 可能是金字塔分层后，图像需要缩放。导致关键点半径选择要以原始图像为标准。这里就差一个尺度因子

                vector<size_t> vIndices2; // 保存上一帧关键点在当前图像的潜在匹配点,内部存储的是去除畸变的关键点序号

                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit; // 当前图像去除畸变的关键点序号
                    if(CurrentFrame.mvpMapPoints[i2]) // 对于当前关键点已经被上一帧的某个地图点匹配上了，那么这里直接跳过！
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.mvuRight[i2]>0) // 对双目有效
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP; // 当前帧地图点集加入上一帧匹配的地图点，这里仅仅是粗略加入。调用本函数之后还会进行 pose 优化。因为这不一定是真正的地图点
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}
// 找到普通帧和关键帧匹配点对。调用这个需要当前帧的 pose 要有一个大概的值。不能什么都没有，否则效果肯定不好!
// CurrentFrame: 当前追踪帧(在重定位模式下)
// pKF: 潜在的匹配关键帧
// sAlreadyFound: 当前帧有效地图点！
// th: 找潜在匹配点的阈值
// ORBdist: 匹配计算描述子之间距离。选择好匹配的阈值
//    可能会使 CurrentFrame 的地图点增加一些！
//  返回值： 有效增加的地图点数
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)// 10,100 --- 3,64
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches(); // 获取当前帧所有效地图点

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP) // 表示地图点存在
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP)) // 不能是坏点，以及在当前帧中没有找到该地图点
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0); // 相机坐标系坐标
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx; // 图像坐标
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX) // 边界条件
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;
                // 这里没有调用 pMP->GetNormal() 函数判断角度问题，可能是为了放宽条件找到更多的匹配？
                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame); // 预测该地图点属于当前帧的哪个金字塔图像上

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel]; // 这个值确定？？？？

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2]) // 存在
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP; // 记录匹配好的地图点！后面通过直方图进行再次筛选不好的地图点！
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL; // 剔除不好的地图点！！！
                    nmatches--; // 递减匹配数量！
                }
            }
        }
    }

    return nmatches;
}
// histo: 匹配点角度之差构成的角度直方图（直方图纵坐标是这个角度对应的第一幅图像的关键点索引）
// ind1: 对应角度差直方图横坐标值，此值对应的纵坐标最大(说明该 ind1 角度对应的图像的特征点数量最多)。
// ind2: 第二多的索引
// ind3: 第三多的索引
// 统计角度直方图中那些角度对应的图像1关键点数量最多。取出前三个多的。当然如果第2、第3多的数量小于第1多数量的 0.1 ，那说明 第2 第3多角度对应的关键点也是误匹配的
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    // 这三个数记录了角度直方图中，纵坐标前三大的纵坐标坐标值
    int max1=0;
    int max2=0;
    int max3=0;
    // 匹配特征点角度差直方图中，前三大的特征点数量！
    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();  // 获取当前索引对应特征点的个数
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }
    // 说明这 max2、max3 两个数量很小，此时角度对应的特征点一定是误匹配的。因为太少了说明不是好的匹配。
    // 这个阈值其实可以调节小点，这样的话可以让单目初始化成功的概率更高些！
    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
// 计算两个描述子之间的汉明距离 a ,b 内部存储的是 256 bit 向量，实际存储的格式 cv::Mat，为 1 x (256/8) = 1 x 32 的矩阵，元素为 1 个字节
// 所以这里计算用了位操作！
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();   // int = 4 字节需要 pa 移动 8 次 = 32个字节
    const int *pb = b.ptr<int32_t>();

    int dist=0;
    // 这里没有细看（待研究！）
    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
