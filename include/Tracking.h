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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;  // Tracking 系统当前状态
    eTrackingState mLastProcessedState; // Tracking 系统上次状态,仅仅在 Tracking::Track() 里面开始部分，进行改变状态

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;    // 当前帧对应的灰度图像

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches; // mvIniMatches[i] = index; 参考帧和当前帧的匹配关系（以参考关键帧为基准）。参考帧 i 关键点，对应匹配 当前帧图像上 index 的关键点。
                                   // 关键点为 mCurrentFrame.unKeysUn[index]，如果 index=-1 表示没有匹配上,在成功三角度化后，这里会把没有成功三角化的位置置位负。
    std::vector<cv::Point2f> mvbPrevMatched; // mvbPrevMatched[i]=point; // 表示参考图像第 i个关键点，对应匹配当前帧图像的关键点 point,如果没有匹配上那么 point是空的。
    std::vector<cv::Point3f> mvIniP3D;  // 三角化的初始地图点{经过多层筛选当前帧和参考帧进行匹配的有效点 3d 坐标} 世界坐标的{相对于参考帧}
    Frame mInitialFrame; // 正常初始化时，就以当前初始参考帧作为本地世界坐标系的起点的。此时设置 pose = 单位 4x4 矩阵

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses; // 每次跟踪成功后，参考关键帧到当前帧的一个变换。之后为下次跟踪做准备
    list<KeyFrame*> mlpReferences;  // 跟踪成功后的，记录所有参考关键帧(包含重复的关键帧)
    list<double> mlFrameTimes;  // 跟踪成功的当前帧时间戳
    list<bool> mlbLost; // 每次跟踪是否丢失,1 :丢失

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;    // 通过 viewer 线程手动选择 gui启动定位模式，之后间接让 Tracking 线程，这里变为 true，然后就变为了定位模式。当然默认情况下，这里是追踪模式的 false
    // 在系统初始化失败，以及跟踪丢失时(刚刚初始化完，5个关键帧以内，如果多于 5 个关键帧完成了，那么此时可能会开启重定位模式)，会进行重置，仅仅清理一下相关资源
    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame(); // 当前帧和追踪过程中的参考关键帧进行匹配
    void UpdateLastFrame();
    bool TrackWithMotionModel(); // 与上一帧，根据匀速模型提供初始 pose。然后与上一帧进行匹配！

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers // 在 system构造函数中直接通过 Tracking::SetLocalMapper() 类似这样的函数设置好的
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight; // left 初始化，对于单目和 rgb
    ORBextractor* mpIniORBextractor;    // 比 mpORBextractorLeft对应的： 2 * nFeatures，初始特征提取比较多

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer; // 构造时 = null,作用：对应论文中 Automatic Map Initialization 部分，在两个线程中计算 F/H 矩阵，根据模型选择 H或者F 来恢复两帧相机运动

    //Local Map 应该是论文 V TRACKING --- D Track Local Map 所说的局部地图
    KeyFrame* mpReferenceKF; // 作用: 在 Tracking::TrackReferenceKeyFrame() 中使用。目的是追踪帧和这个关键帧进行匹配。
                             // 有三个地方会从新赋值，可以从该变量的作用上理解。
                             // 1) 单目初始化时，插入当前关键帧。
                             // 2) 在插入新的关键帧后(CreateNewKeyFrame())，赋值为新的关键帧。
                             // 3) Tracking::UpdateLocalKeyFrames() 函数中更新了这个帧，
                             //   这个函数在这里被调用 Tracking::TrackLocalMap()。
                             //   与当前追踪帧有最强共视关系的关键帧且其隶属于 K1 关键帧集。
                             //   然后用其与下一帧进行匹配！
                             // 总结：三种方式都是为了能够在正常追踪时，在调用 TrackReferenceKeyFrame()
                             //   时，能够让追踪帧和参考帧有最大共视关系。这样
                             //  在理论上会与当前追踪帧更好匹配
    std::vector<KeyFrame*> mvpLocalKeyFrames; // 在单目初始化时，加入了参考关键帧和当前关键帧,这个可能也会用在定位模式？？ 。跟踪时。在第一次进入 Tracking::TrackLocalMap()函数后，
                                            // 进而进入 Tracking::UpdateLocalKeyFrames() 函数，在里面就会立刻清零。然后更新这个值为与当前帧有着共同观测点的关键帧！
                                            // 将 Covisibility Graph 中与上面的关键帧组有链接关系的其他关键帧 再次加入到关键帧组。但是程序中限制了局部关键帧的最大个数！
    std::vector<MapPoint*> mvpLocalMapPoints; // 在单目初始化时，加入了当前所有的地图点,与上面变量一致，根据上面的局部关键帧组，
                                              // 找到所有有效地图点.作为当前局部地图点，进而与追踪帧进行投影匹配+优化
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;   // Viewer 线程
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap; // System 中定义的 Map 变量

    //Calibration matrix
    cv::Mat mK; // 相机内参矩阵 3 x 3(在 )
    cv::Mat mDistCoef;  // 畸变系数 k1 k2 p1 p2 k3
    float mbf;  // 单目：无  双目：f*b  RGB-D: f*b 参照 14 讲双目公式

    //New KeyFrame rules (according to fps)
    int mMinFrames; // init = 0
    int mMaxFrames; // 30 fps 根据相机参数而定,这个条件用在关键帧是否插入的条件中。表示地图中关键帧个数不能超过这个值

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth; // 深度阈值？以基线为尺度，都是基线的倍数值

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers; // 在 TrackingLocalMap() 记录当前追踪帧中内点个数

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame; // 记录上次关键帧, 在单目初始化时，就是当前帧
    Frame mLastFrame;   // 单目初始化之后当前帧(也是关键帧)，然后又从新赋值给他,之后就是每次成功跟踪，然后更新一次为刚刚跟踪的这个普通 Frame 帧,
                        // 这里就是普通帧。通过这个帧提供给下一帧匹配时位姿初始值然后通过 3d-2d 匹配优化
    unsigned int mnLastKeyFrameId; // 在 CreateNewKeyFrame() 关键帧后，更新当前值,在单目初始化时，就是成功初始化后的当前帧的 id （普通帧id）
    unsigned int mnLastRelocFrameId; // init = 0,在跟踪丢失时，下一帧会进行重定位，如果重定位成功。那么这里记录重定位成功的跟踪帧 id

    //Motion Model
    cv::Mat mVelocity;  // 第一次使用时为空,在成功追踪了当前帧之后。需要更新的： 就是前一帧到当前帧(这个是在还没来下一帧前的称呼)的变换矩阵：
                        // 然后在来一帧时就可以用这个来递推，找到下一帧位姿的初始估计值.
                        // 前一帧--->当前帧

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB; // yaml::Camera.RGB: 1

    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
