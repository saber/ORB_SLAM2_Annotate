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

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>


namespace ORB_SLAM2
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    FrameDrawer(Map* pMap);

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame();

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm; // 在跟踪线程中，刚刚处理完毕的图像
    int N; // 图像 ORB 提取的关键点个数
    vector<cv::KeyPoint> mvCurrentKeys; // 图像对应的原始地图点(有可能是未去除畸变的点集，根据数据集图像是否去除畸变来决定)
    vector<bool> mvbMap, mvbVO; // init: N 个 false,mvbMap:在正式追踪过程中标记当前地图点是否被关键帧观测过。 这个与上面的 mvCurrentKeys 对应，然后没被观测的关键点就不显示
                                // mvbVO: 如果地图点没有被观测过，此时这里对应位置为 true。还不知道如何使用？
    bool mbOnlyTracking; // 默认为追踪模式即 false
    int mnTracked, mnTrackedVO; // mnTracked: 当前图像有效的地图点个数(也是用来显示当前匹配的个数)
    vector<cv::KeyPoint> mvIniKeys; // 初始化时，对应的初始参考帧关键点(未去除畸变的)
    vector<int> mvIniMatches; // 很有可能初始化时没有成功，初始化时刻，参考帧和当前帧匹配的对(能够进行三角化的) mvIniMatches[i] = index; 表示参考帧 i 关键点和当前帧关键点 index 是配对的
    int mState; // update跟新上次追踪线程所处状态

    Map* mpMap;

    std::mutex mMutex;
};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
