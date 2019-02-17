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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId; // 当前地图点 id
    static long unsigned int nNextId; // init = 0, 在构造时 ++
    long int mnFirstKFid;   // 第一个关联的关键帧 id ,通常是初始化时哪个 current frame
    long int mnFirstFrame;  // 第一个 Frame 的 id (成功初始化后的对应 currentFrame)，在每次 Frame() 构造时，id 都会增加。所以初始化成功时 id 大多数已经不是 1
    int nObs;   // init = 0,该地图点被多少个关键帧观测的次数

    // Variables used by the tracking 在跟踪线程中会不断的变化的！与当前帧有关
    float mTrackProjX; // 投影到追踪帧的图像坐标系的 u 坐标
    float mTrackProjY; // 投影到追踪帧的图像坐标系的 v 坐标
    float mTrackProjXR; // 对应投影到双目相机的右边摄像头图像坐标系的 u 坐标
    // 这个变量 == false
    // a: 局部地图点不满足 isInFrustum()。
    // b: TrackWithMotionModel TrackReferenceKeyFrame 完毕后得到pose。然后在 Tracking::SearchLocalPoints() 中，属于当前追踪帧的地图点了。
    // c: 在函数内部初始匹配后是内点，但在优化过程中认为是外点。比如在TrackReferenceKeyFrame，TrackWithMotionModel 这两个函数中,有一个初始化匹配，
    //      然后内部调用了 PoseOptimization()优化
    // 上面的结果后，最后在 ORBmatcher::SearchByProjection() 函数内部进行判断这个变量。这个是 false 就跳过该点
    bool mbTrackInView;
    int mnTrackScaleLevel; // 当前地图点在当前追踪帧中对应第几层图像金字塔上
    float mTrackViewCos; // 地图点与追踪帧光心得到的向量与跟踪帧(Mean viewing direction)的单位向量的 cos 夹角
    long unsigned int mnTrackReferenceForFrame; // init = 0 // 当前地图点在追踪 TrackLocalMap()时，可能被追踪帧观测到。这里记录那个追踪帧的 id 号
    long unsigned int mnLastFrameSeen; // init = 0 ,记录当前地图点上次是被哪个帧匹配过和观测过。这里记录那个 Frame 的 mnid

    // Variables used by local mapping
    long unsigned int mnBALocalForKF; // init = 0, 在局部建图线程中 BA 优化时，以哪个关键帧 id 为基准。与下面的 id 一样
    long unsigned int mnFuseCandidateForKF; // init = 0，这个值等于正在处理的关键帧的id，表示已经加入了与当前正在处理的关键帧进行融合的地图点集

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF; // init = 0 记录标记  = mpCurrentKF->mnId 闭环线程正在处理的关键帧的 id
    long unsigned int mnCorrectedByKF; // init = 0 记录标记 = mpCurrentKF->mnId ，是否被当前闭环线程正在处理的关键帧标记过
    long unsigned int mnCorrectedReference; // init = 0 当前地图点在遍历哪个关键帧 id 时进行纠正的(该关键帧的),这些关键帧属于闭环线程正在处理的关键帧的临近关键帧集
    cv::Mat mPosGBA; // 闭环线程进行全局 BA 优化时，更新后的地图点坐标,此时没有更新 mWorldPos 的值，是因为闭环线程 BA 优化函数后面需要用到 mWorldPos 值，最后会用其更新 mWorldPos 的值
    long unsigned int mnBAGlobalForKF; // init = 0 表示在闭环线程中进行全局 BA 优化时，正在处理的关键帧的 id


    static std::mutex mGlobalMutex;

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos; // 世界坐标点三维坐标,一个点对应一个 MapPoint 类

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;  // 记录该地图点被哪个关键帧观测过，并且记录了相应关键帧上面对应的关键点序号。对于单目来说，
                                                // 仅仅在单目初始化追踪线程中创建地图点时写入的，剩下的就是在局部建图线程中？

     // Mean viewing direction
     cv::Mat mNormalVector; // 论文中说的是：这个点和关键帧相机中心的连线（可以有多个这样的连线，因为点可以被多个关键帧观测到），
                            // 所有的连线在世界坐标系的方向向量(单位化后)直接求和，然后除以连线的个数就得到了这个平均值。所以这个向量仍然是单位向量
                            // 具体计算在：MapPoint::UpdateNormalAndDepth() 函数中。一般用在检查一个地图点到相机光心向量与当前这个向量的夹角。然后判断该点是不是在当前帧可视范围内。等等。。
     // Best descriptor to fast matching // 不知道保留这个会有什么意义？？
     cv::Mat mDescriptor; // 当前 mObservations 中，所有关键帧及其相应匹配的关键点描述子。计算两两描述子之间的距离。选择中位数距离最小的关键帧对应的描述子
                          // MapPoint::ComputeDistinctiveDescriptors() 函数赋值的。但是有什么意义？？？
     // Reference KeyFrame
     KeyFrame* mpRefKF; // 在哪个关键帧三角化得到的点,单目初始化时，是在当前帧中被三角化的，保留的是当前帧(mnCurrentFrame)。
                        // 剩下的就是在 Optimizer.cc 中 LocalBundleAdjustment() 函数中调用  MapPoint::EraseObservation(KeyFrame* pKF) 来改变

     // Tracking counters 两个变量都是在 TrackingLocalMap() 中进行变化的！也就是需要经过一个追踪局部地图才能最后确定当前地图点的下面这个两个属性！
     int mnVisible; // init = 1 表示这个地图点被追踪帧(累计)观测几次(但是这个仅仅是没有经过 TrackLocalMap() 中的位姿优化时，地图点能够被追踪帧观测到，此时这个值就会增加，
                    // 与下面的 mnFound 不同，mnFound 是在位姿优化后如果这个地图点仍然可以被追踪帧观测到，那么就会增加 mnFound 数量。
                    // 与上面的 nObs 不同。
     int mnFound; // init = 1 表示该地图点经过 TrackLocalMap() 内部位姿优化后。如果该地图点仍然可以被追踪帧观测到，然后才会增加这个值。
                  // 注意这里对于单目初始化帧不记录。因为在初始化时创建了地图点，创建时地图点这个值初始化为 1，其实就表示被观测到了。
                  // 剩下的就是在 Tracking::TrackLocalMap() 函数中被追踪帧观测到然后进行增加。还有就是在 Tracking::Track() 定位模式中会增加。然后在 MapPoint::Replace() 中会增加
                  // 这个函数会在 ORBmatcher.cc Fuse() 和 LoopClosing.cc 中会进行调用

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;    // 初始 = false，如果为 true 表示这个地图点需要去掉,在 SetBadFlag() 中设置
     MapPoint* mpReplaced; // init = NULL //  当前地图点被哪个地图点代替：什么时候会被代替？？？ 在 ORBmatcher.cc  Fuse() 函数中，
                           // 是因为两个地图点与某个关键帧关键点对应。然后选泽了被关键帧观测次数多的地图点与那个关键点对应
                           // 在 LoopClosing.cc 中 CorrectLoop() 函数中调用，SearchAndFuse()
     // Scale invariance distances 这两个在  MapPoint::UpdateNormalAndDepth() 中更新，但是计算的方式不理解？？？
     float mfMinDistance; // init = 0 尺度不变距离
     float mfMaxDistance; // init = 0

     Map* mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
