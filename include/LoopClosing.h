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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"

#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:

    typedef pair<set<KeyFrame*>,int> ConsistentGroup; // pair<关键帧集，一致性得分>
    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose; // sim3 位姿 ，就按照 map<KeyFrame*, g2o::Sim3>

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested; // init = false，表示是否重置闭环线程
    std::mutex mMutexReset; // 用来保护 mbResetRequested 变量，因为这个变量会在追踪线程时用到，也会在闭环线程中用。所以需要加锁！

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested; // init = false
    bool mbFinished; // init = true，在 run 中 = false
    std::mutex mMutexFinish;

    Map* mpMap; // 地图结构{地图点、关键帧}
    Tracking* mpTracker; // 设置追踪线程，为什么没有用到？？？

    KeyFrameDatabase* mpKeyFrameDB; // 关键帧数据库(用来重定位和闭环检测)
    ORBVocabulary* mpORBVocabulary; // 字典结构.位置识别和特征匹配

    LocalMapping *mpLocalMapper; // 局部建图线程

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;  // 闭环需要的关键帧，在 手动或者初始化失败时，主线程调用 Reset() 就会清零这里的资源

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh; // init = 3,连续 3 次检测到闭环，才认为是真正检测到了闭环。

    // Loop detector variables
    KeyFrame* mpCurrentKF; // 在检测闭环函数中进行赋值的。表示当前正在处理的基准关键帧。
    KeyFrame* mpMatchedKF; // init = NULL,在成功找到闭环帧（检测时也认为成功），之后需要利用这个进行纠正闭环
    std::vector<ConsistentGroup> mvConsistentGroups; // pair<set<KeyFrame*>,int>：<一致群，一致性得分>。当前闭环线程正在处理的，用于检测比较的一致关键帧群。
                                                     // 内部元素对应的关键帧集可能相同，但是后面的得分可能不同
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates; // 每个元素表示，该关键帧满足连续 3 次一致性条件{当然具体含义需要看 LoopClosing::DetectLoop()函数}。
                                                          // 说明该关键帧就是一个一致性闭环帧。会在 ComputeSim3() 中进行验证这个帧是否是真正要找的闭环帧，如果是闭环帧
                                                          // 那么会进行纠正闭环误差）
    std::vector<KeyFrame*> mvpCurrentConnectedKFs; // 闭环正在处理的 mpCurrentKF 关键帧的临近关键帧集（包括自身）
    std::vector<MapPoint*> mvpCurrentMatchedPoints; // 当前成功检测到闭环时，记录的实际有效的匹配点对(包括最后在 sim3 优化后再次寻找潜在匹配点对):
                                                   // mpCurrentKF 关键点 ---> 潜在闭环关键帧的地图点。淡然这里的 mpCurrentKF 有些关键点可能有自己的地图点，也有些可能没有。
    std::vector<MapPoint*> mvpLoopMapPoints; // 使用该值前会 clear() size,保存的是 mpMatchedKF(检测到的真正的闭环帧) 对应的临近关键帧的所有地图点。包含 mpMatchedKF 对应的地图点
    cv::Mat mScw; // 世界到当前正在处理的关键帧 mpCurrentKF 之间的相似变换。cv::Mat 形式的下面的 mg2oScw
    g2o::Sim3 mg2oScw; // 世界到 mpCurrentKF 相机的相似变换（中间利用了检测到的闭环关键帧）,为了将所有 mpMatchedKF 临近关键帧的所有地图点投影到 mpCurrentKF ,再次找更多的匹配关系

    long unsigned int mLastLoopKFid; // init = 0，记录上次检测闭环成功且经过了本质图优化和全局 BA 优化，闭环线程正在处理的关键帧的 id。即：mpCurrentKF->mnId

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA; // init = false 是否正在跑全局 BA 优化
    bool mbFinishedGBA; // init = true,在闭环线程完成全局 BA 后就会置位 ture
    bool mbStopGBA; // init = false
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA; // 线程 init = NULL

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;    // init = :单目 false 单目不是固定尺度，其尺度不确定。
                        // init = :双目 and RGB-D: true 得到的尺度都是真实尺度


    bool mnFullBAIdx; // init = 0
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
