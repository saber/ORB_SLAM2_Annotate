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

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>

#include<mutex>

namespace ORB_SLAM2
{

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const string &strSettingPath);

    Map* mpMap;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

private:
    // 在 TUM1.yaml 文件中的 Viewer Parameters
    float mKeyFrameSize; // 0.05
    float mKeyFrameLineWidth; // 1
    float mGraphLineWidth; // 0.9
    float mPointSize; // 2
    float mCameraSize; // 0.08
    float mCameraLineWidth; // 3

    cv::Mat mCameraPose; // 4x4 当前显示的帧的 pose,这个值是在追踪线程中设置的。在 Track() 函数 TrackLocalMap() 之后。以及在单目和双目初始时设置的。

    std::mutex mMutexCamera;
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
