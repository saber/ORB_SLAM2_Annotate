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


#ifndef VIEWER_H
#define VIEWER_H

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

class Viewer
{
public:
    Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking *pTracking, const string &strSettingPath);

    // Main thread function. Draw points, keyframes, the current camera pose and the last processed
    // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

private:

    bool Stop();

    System* mpSystem;
    FrameDrawer* mpFrameDrawer; // cv::Mat 形式，显示图片关键点等等...
    MapDrawer* mpMapDrawer;
    Tracking* mpTracker;

    // 1/fps in ms
    double mT; // 相机两张图像之间时间间隔，周期时间 ms
    float mImageWidth, mImageHeight;    // 对于单目，没有这个值，仅仅针对双目和 RGB-D 相机

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF; // 对于单目 TUM数据集：0, -0.7, -1.8, 500 。前三个参数是相机在世界坐标的位置

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested; // init = false
    bool mbFinished;    // init = true ，在 run() 中修改为 false，不如直接在构造函数中直接进行修改！
    std::mutex mMutexFinish;

    bool mbStopped; // init = true ，在 run() 中修改为 false. true 表示将要停止当前显示线程
    bool mbStopRequested;   // init = false, 在 Tracking 线程中 Reset() 置位时调用 RequestStop() 手动请求停止. 然后内部置 true 表示请求停止，然后显示线程检测到后，就会进行线程停止操作
    std::mutex mMutexStop;

};

}


#endif // VIEWER_H
	

