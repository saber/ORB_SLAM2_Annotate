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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release(); // 释放 Local Mapping 存储的关键帧
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;   // 传感器类型：true 表示使用单目

    // 重置系统操作、标志以及对应的锁保护，表示重置系统，但是不退出。仍然运行当前线程，仅仅清理一些资源(指针）
    // 但是指针指向的资源，都是在 Tracking 线程中分配的,然后统一给了 Map 变量，所以释放也是在 Tracking 中调用 Map::clear() 释放的。
    void ResetIfRequested();
    bool mbResetRequested;  // init = false
    std::mutex mMutexReset;

    // 完成操作、标志以及对应的锁保护,表示退出系统，结束线程资源。
    // 在整个系统跑完，需要调用的
    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested; // init = false, 在 System::Shutdown() 函数调用请求完成函数，置该变量为 true
    bool mbFinished; // init = true, 在 run() 中设置为 false
    std::mutex mMutexFinish;

    Map* mpMap; // 地图资源

    LoopClosing* mpLoopCloser;  // 闭环线程，插入闭环关键帧
    Tracking* mpTracker;    // 暂时没用

    std::list<KeyFrame*> mlNewKeyFrames; // 插入所有新的在追踪线程中创建的关键帧 局部建图资源    // 这些资源包含的指针内存，已经包含在 Map 类里面了。通过 mpMap->clear(); 已经释放了。

    KeyFrame* mpCurrentKeyFrame; // 局部建图线程正在处理的当前关键帧。

    std::list<MapPoint*> mlpRecentAddedMapPoints; // 这个变量在两个地方进行添加(保证了所有地图点都会经过剔除操作。因为一些点可能是错误匹配，然后就会错误三角化，这样不利于跟踪)：
                                                  // 1、在 ProcessNewKeyFrame 函数中把最近建立的地图点添加进来
                                                  // 2、在创建新的地图点时，每成功一次。就会把那个地图点增加到这里
                                                  // 这些资源包含的指针内存，已经包含在 Map 类里面了。通过 mpMap->clear(); 已经释放了。

    std::mutex mMutexNewKFs;

    bool mbAbortBA; // init = false , true = 在 BA 优化时，停止当前优化！

    bool mbStopped; // init = false, 这个变量是在 判断当前局部建图线程是否该停止了 true: 表示该停止
    bool mbStopRequested; // init = false, 请求优化。要停止该线程，需要先让这个为 true!
    bool mbNotStop; // init = false, true 表示不要停止,即使请求停止，也不会让 mbStopped = true
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames; // init = true 表示当前局部建图线程不繁忙。
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
