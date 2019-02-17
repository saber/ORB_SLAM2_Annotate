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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;   // 重置时清理的资源，这里在单目初始化的时候，保存了初始时刻的那个参考关键帧！在 LoopClosing.cc 中用了这个值

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict) // 创建 MapPoint 时需要的锁
    std::mutex mMutexPointCreation;

protected:
    // 下面 4 个连续的变量，在重置时需要清理相应资源
    // 所有的地图点{目前是关键帧节点对应的 3d 点}
    std::set<MapPoint*> mspMapPoints;   // 在重置已经清理指针资源,对应关键帧的地图点，这里直接清理即可
    // 所有关键帧
    std::set<KeyFrame*> mspKeyFrames;   // 已清理过指针资源，直接对应 Map,保存关键帧

    std::vector<MapPoint*> mvpReferenceMapPoints; // 在单目初始化时，通过一个函数给其赋值两个关键帧三角化后的所有 3d MapPoint,用来与追踪后的帧进行再次投影匹配。其实这里不需要在单目化时赋值。因为在
                                                  // Tracking 中局部建图函数会自动更新这个值.这个值就是局部关键帧组

    long unsigned int mnMaxKFid;    // 默认初始化为0，此时地图中保存的最大关键帧 id

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
