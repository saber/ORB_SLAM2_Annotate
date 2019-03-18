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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;
// 创建新的地图点
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}
// 获得该地图点的世界坐标！
cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

// 每个地图点包含了归属的关键帧，以及在哪个关键帧中对应的关键点序号，之后记录该点被关键帧观测的次数
// 单目来说：观测次数每次加 1
// 双目来说：观测次数每次加 2 因为有两个摄像头
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);

    //  只有地图点归属的关键帧之前没有插入过，才会进行插入
    if(mObservations.count(pKF))
        return;
    mObservations[pKF]=idx;
    // 记录当前地图点被多少个关键帧看到
    if(pKF->mvuRight[idx]>=0)
        nObs+=2;
    else
        nObs++; // 针对单目
}

// 如果当前地图点和关键帧 pKF 建立了关联。那么取消这个关联。减少当前地图点被关键帧观测的次数。
// 如果当前地图点对应的参考关键帧是 pKF，那么选择其他关键帧作为参考关键帧。如果最后当前地图点被观测的次数小于等于2
//    那么此时就会将这个点标记为坏点。然后在地图中去除该点
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF); // 擦除所有元素

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2) // 说明当前被关键帧观测的太少，需要去除这个地图点。看做不好的地图点
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag(); // 在地图中清除当前地图点。
}
// 获得该地图点被那些关键帧观测到
map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}
// 当前地图点被多少个关键帧观测到
int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

// 在地图中将这个地图点清除。然后与当前地图点有关联的关键帧，要取消这个关联
// 在局部建图中剔除地图点函数中调用以及局部 BA 优化后有些地图点可能是坏点。对于不满足论文中 VI LOCAL MAPPING B 两个条件时，地图点都会被标记为坏点，然后在去除坏点
// // 但是坏点本身内存没有剔除.不知道什么时候会清理这些不好的地图点？？？
// 目前来说没有发现哪里直接 delete 这些不好的地图点内存。
// 某个地方会对地图点拥有耦合关系。没法简单的在地图擦除函数中直接 delete 。这也是一个改进的点!!!
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second); // 清理关键帧对应的匹配关系
    }
    // 擦除地图中包含的该地图点
    mpMap->EraseMapPoint(this);
    mappoint_culling_ = true; // 自己设定地图点是被删除了！！ 测试变量，
}

// 当前地图点被哪个地图点代替
MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

//! \brief 用 pMP 地图点替换当前地图点。对当前地图点中那些没有与 pMP 关联的关键帧。然后将 pMP 与之进行关联。
//!       最后在地图中将该地图点删除。此时当前地图点变为坏点。
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId) // 两个是一样的地图点，则不需要代替
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true; // 坏点标注
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    // 与当前地图点有联系的关键帧。与 pMP 地图点进行关联
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF)) // 当前地图点没被 pKF 观测到。因为 pMP 地图点替换当前地图点。所以就需要把 当前地图点的那些关键帧与 给定的 pMP 地图点进行相应的关联
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP); // 将关键帧 pKF 对应位置替换为新的地图点
            pMP->AddObservation(pKF,mit->second); // 给定地图点增加观测
        }
        else // 如果 pKF 关键帧与 pMP 地图点进行了关联。那么需要取消 pKF 与当前地图点之间的联系。因为当前地图点已经被 pMP 替换了。
        {
            pKF->EraseMapPointMatch(mit->second); // 取消关联
        }
    }
    pMP->IncreaseFound(nfound); // 这里为什么直接增加这个次数？？？这样不是会有重复增加的吗？？？？？
                                // 这里其实是有问题的，如果两个地图点在追踪时没有同时遍历，其实是可以直接增加的。但是如果两个地图点都在追踪线程中同样被追踪。那么这里增加其实是有重复的
                                // 经过测试，这个函数在 ORBmatcher::Fuse（）函数中调用。然后在那个函数中打印两个替换地图点的 mnFirstId 发现会有相同的。那么在追踪过程中肯定会同时增加
                                // mnFound mnVisible 这两个变量。那么这里直接把数量相加。其实是有问题的！
                                // 但是从另种角度看：这个地图点本身在之前已经被追踪到，此时仅仅是替换，所以应当包含之前的观测变量。
    pMP->IncreaseVisible(nvisible); // 与上同理？？？？？
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}
// 当前地图点被追踪帧在调用 Tracking::SearchLocalPoints() 函数时进行增加可视次数。这里增加的是初始匹配和局部地图点中能够被那个追踪帧可视。
//    但是值得注意的是，最后将局部地图点投影到当前帧后，还有进行一次 3d-2d 的位姿优化，这些可观测的点在优化后会剔除一部分。
// 剩下的部分地图点就是下面函数中增加 mnFound。可以知道这两个变量的比值其实反应的是这些追踪帧初始 pose 匹配时的精度！这两个值越接近，说明跟踪时，初始估计的 pose 越精确！
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}
// 地图点最终经过 TrackLocalMap() 函数中优化后，确实被帧观测到，就会增加这个计数
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}
// 可以知道这两个变量的比值其实反应的是初始 pose 匹配时的精度！这两个值越接近，说明初始估计的 pose 越精确！
//    可以看上面 IncreaseVisible() 函数说明
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}
// 只要增加了新关键帧，那么就会调用了一次。在当前地图点被观测的所有关键帧中。取出对应的所有关键点描述子。然后两两描述子之间计算汉明距离形成 NXN的距离矩阵
// 每个描述子都有 N 个距离。然后选择中位数距离。 这样会有 N 个中位数距离。然后选择其中最小的距离。
// 然后记录这个描述子为最好的描述子拷贝给 mDescriptor成员变量，是为了快速匹配
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors; // 一个 cv::Mat 是一个描述子

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty()) // 当前地图点没有被关键帧观测过！
        return;

    vDescriptors.reserve(observations.size()); // 该地图点被观测的关键帧以及对应的关键点序号

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad()) // 只有当前关键帧不是冗余的情况（即不会被剔除） 取出被观测关键帧对应的关键点描述子
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them // 计算任意两个描述子之间的汉明距离
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0; // 矩阵第 i 行对应的距离中位数最小
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N); // 矩阵 Distances 中每一行存储的距离值
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)]; // 中位数对应的距离：就是每个描述子与其他描述子之间的距离的中位数，下面选择最小的中位数代表最好的描述子

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}
// 获得该地图点在给定关键帧下的索引（可以找到对应的关键点）。如果当前地图点没有与该关键帧相连，则返回 -1
int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

// 判断当前地图点所属关键帧集中是否包含给定的关键帧。 true： 给定关键帧已经与当前地图点构成了联系
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

// 每次该地图点被一个关键帧观测到就需要调用这个函数更新,以及在一次 BA 优化后，三维点更新后，也需要再次更新，因为 mWorldPos 变化了，导致下面三个变量都会发生变化：
// mfMaxDistance mfMinDistance mNormalVector 这几个变量 对应论文 III.SYSTEM OVERVIEW 中 C 部分
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;
        pRefKF=mpRefKF;
        Pos = mWorldPos.clone();
    }

    if(observations.empty()) // 因为一些地图点可能被置位坏点，然后清理了 这个变量的值。所以需要检查一下
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave; // 关键点所属金字塔层
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level]; // 当前层尺度因子
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

    // 这里需要说明一点：图像金字塔是将采样得到的一堆图像。[参考:http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/pyramids/pyramids.html]
    // 金字塔最后一层的图像尺寸是最小的。图像分辨率是最低的。相当于把近东西拿到了远处。也就是说。我们有了金字塔层上面的一个关键点 kp。以及他与相机光心在世界坐标系下的距离 dist
    // 此时的 dist 是以原始图像为基准计算的[因为关键点坐标都转换为原始图像坐标系，然后三角化为地图点]。实际上该点距离就是 dist。??????
    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor; // 因为前面
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mNormalVector = normal/n;
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

//! \brief 给定该地图点到 Frame 的距离后，判断当前地图点在给定的帧下图像金字塔第几层
//! \param currentDist: 当前世界坐标点到当前相机光心距离。
//! \param pF: 追踪帧
//! \return: 当前距离对应的金字塔图像第几层
//! \question 从下面代码中可以推断出如下式子，但是最后推断的结果不知道怎么回事？:::答 参考「问题已解决.md」 第 4 点
//!     mfMaxDistance = dist * 1.2^L; // L : 是该地图点第一次在三角化时，相对于参考的关键帧关键点所在金字塔层数。 dist 是当时三角化后，点到参考关键帧光心距离。
//!     ratio = mfMaxDistance/curDist = dist * 1.2^L/curDist;
//!     nScale = log(ratio)/log(1.2)= lg(ratio);
//!  ==>   1.2^nScale = ratio;
//!  ==>   1.2^nScale = dist *1.2^L/curDist;
//!  ==>   curDist * 1.2^nScale = dist * 1.2^L;
//!  从最后这个公式可以推导出。程序中所说的尺度不变距离。 mfMaxDistance = dist * 1.2^L
//!  实际上是一个定值。（在当前地图点对应参考关键帧不变的情况下）但是目前不理解这个值为什么是定值？？
//!  这个值在 MapPoint::UpdateNormalAndDepth() 中更新的。
int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist; // mfMaxDistance 是在自己
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
