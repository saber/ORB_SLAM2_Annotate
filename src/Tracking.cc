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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    // 内参矩阵
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 畸变参数，针对单目，双目不起作用
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // baseline * fx
    mbf = fSettings["Camera.bf"];   // 对于文件内没有该项内容，此时读取的值是默认初始值，此时这里为 0
    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;   // 每秒 30 帧图像,针对单目 TUM1 数据集

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 下面两个值表示： 一张图像被分割为多个 grid ，然后每个 grid 中提取的特征点个数阈值
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        // b * fx * ThDepth / fx = b * ThDepth 这里基线的倍数
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

//! \brief 将输入图像(RGB/RGBA)转化为灰度图像并且构造 mCurrentFrame，进行追踪
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

//! \brief 将输入图像(RGB/RGBA)转化为灰度图像并且构造 mCurrentFrame，进行追踪
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;   // 直接对应值拷贝,此时是彩色图像

    // 1) 将 RGB 或 RGBA 图像转为灰度图像
    if(mImGray.channels()==3)   // TUM 数据集本身是 RBG 三通道图像。
    {
        if(mbRGB) {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);  // 此时转换成的 8 位灰度图为单通道！
//            std::cout << "转化后的灰度图像： 通道" << mImGray.channels() << endl;
        }
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
    // 2) 根据 Tracking 状态选择使用 mpIniORBextractor 或者 mpORBextractorLeft，构造当前 Frame
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        // 创建 frame
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    // 3) 正式跟踪
    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    // 系统 system 对象构造时默认为 NO_IMAGES_YET
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState; // 至此在这里一次改变状态

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // 初始化或者继续正常追踪
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();  // 至少进入两次，第一次选定参考帧，第二次用当前帧和参考帧匹配，进行初始化地图点。
                                        // 如果失败的话，那么就从新初始化(选择参考帧。当前帧和参考帧匹配)，重置所有线程。
        // 通过观察显示线程，可以发现。如果第一张图片没有提取出来足够的关键点，那么此时 mvIniMatches 这个值就没有被初始化
        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {   // 正常追踪！ 包含了 TRACKING 论文对应的步骤
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 在局部建图线程中会有地图点创建以及最近地图点的剔除，在使用上一帧匹配时，对应的地图点可能发生变化。
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2) // 在追踪失败时，然后通过重定位模式追踪成功，此时的速度模型就会为空！或者第一次初始化时，速度模型会为空
                {   // 这里这么做是因为在初始化时，refer 帧，和第二个帧，都作为了关键帧，此时我们用当前帧和前一帧匹配就是普通帧和关键帧的匹配.
                    // 之后通过 3d-2d 进行位姿优化。pose 初始值是上一个记录的关键帧
                    // 这个函数内部就是调用的 关键帧和普通帧之间的搜索匹配。
//                    std::cout << "first enter: " << std::endl;
                    bOK = TrackReferenceKeyFrame(); // 这里可能会出现当前帧的匹配失败且pose为空！
                }
                else
                { //std::cout << "second enter: " << std::endl;
                    // 此时速度模型已求出,运用匀速模型，进行当前帧 pose 估计。如果根据运动模型也无法求出当前帧的  pose 。
                    // 那么此时需要用当前帧和前一个参考关键帧进行 bow 匹配。实际上这里需要增加一些额外条件才对，
                    // 因为通过这两次的跟踪匹配无法满足相机突然快速的转向和移动。额外加传感器？还是运用其他模型？
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else    // 此时 mStati == LOST，进入重定位模式！
            {
                bOK = Relocalization();
                //std::cout << "-------------重定位退出！" << std::endl;
            }
        }
        else    // 定位模式，仅仅能够通过 viewer 线程交互改变。否则这里不会执行
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // 追踪后(无关成功与否)。。。
        mCurrentFrame.mpReferenceKF = mpReferenceKF; // 记录当前追踪帧和哪个参考关键帧进行的匹配，为下一次跟踪帧计算 pose 做准备{因为}

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK) // 当前帧追踪成功，跟踪局部地图，将一些帧的地图点投影到当前帧。然后再次优化当前帧的位姿
                bOK = TrackLocalMap();
        }
        else
        { // 定位模式
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        // 到这里之前有两个拦路虎：1、初始跟踪是不是成功 2、TrackLocalMap() 只有这两个同时表示跟踪成功 bOK 才等于 true!
        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model 更新匀速运动模型，这个就是论文 V.B 中说的 a constant velocity motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc; // 前一帧到当前帧的变换： LastTwc 帧上的点，通过 mVelocity 变换到当前帧坐标系
            }
            else // 什么时候执行？？？ 就是上一帧是通过在初始化成功后的第一帧。追踪失败。此时上一帧 pose 为空。
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches  通过上面调用 TrackLocalMap（），可能一些地图点通过调整当前帧 pose ，不会被观测到或者说即使观测到，
            // 但是优化pose后，投影误差大于指定阈值，那么也认为这个地图点匹配错误，需要做一些清理。因为在 TrackLocalMap() 中没有清理
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints // 这个需要待看！在上面的 TrackLocalMap() 过程里面会使用这个值，(在单目的时候不使用！)目前不知道有什么用？？
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear(); // 单目不会用这个变量

            // Check if we need to insert a new keyframe 对应论文 V TRACKING --- E New Keyframe Decision
            if(NeedNewKeyFrame()) // 定位模式直接返回 false.对于单目来说，只要局部建图处于繁忙阶段。这里就不能加入关键帧
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 统一在剔除一次不符合当前追踪帧条件的地图点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5) // 表示刚刚初始化不久，就追踪失败了，此时要进行重新系统初始化！
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF) // 统一更新一下当前帧的参考关键帧，其实在上面创建新关键时或者在第一次初始跟踪时，已经更新过了
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame); // 当前帧 pose 已经优化好！
    }

    // 下面两个分支的结果不会影响跟踪过程！
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty()) // 进入这里的条件：
                                    // 1) 在第一次使用 Tracking::TrackReferenceKeyFrame() 时
                                    // 最后返回是失败的，但是 mCurrentFrame.mTcw 已经有一个初始值。
                                    // 不过从分析来看，这种情况发生的可能性很小。不过理论上确实会发生。
                                    // 2) 经过 1 时，下一帧就会进入重定位模式。然后重定位虽然失败了。
                                    // 但是让处理下一帧时 mCurrentFrame.mTcw 有初始值。
                                    // 3) 确实追踪成功
                                    // 总结： 1) 2) 在理论上是会发生的。不过没有做过试验。

    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse(); // 参考关键帧到当前跟踪帧的一个变换。
        mlRelativeFramePoses.push_back(Tcr); // 这个在跟踪和最后保留轨迹时会用到
        mlpReferences.push_back(mpReferenceKF); // 也就是说 mlpReferences 会加入重复的参考关键帧。

        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else // 进入这里的条件：
         // 1、当正式跟踪时与参考帧进行匹配。也就是上面先调用这个函数进行追踪：
         //     Tracking::TrackReferenceKeyFrame()。此时执行内部函数时，
         //     第一次寻找匹配就失败了。那么 mCurrentFrame.mTcw 一定为空。此时进入这个分支
         //  2、除了 1 的情况外导致的当前追踪失败。那么接下来处理的一个追踪帧就会用重定位模式。
         //    如果重定位「完全」失败，也就是内部没有让 mCurrentFrame.mTcw 有初始值。
         //   此时 mCurrentFrame.mTcw 也是空的。也会进入这个分支来。
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{
    // 系统初始化时赋值 null
    if(!mpInitializer)
    {   // 选择参考帧（为 map Initializaton 做准备）
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size()); // mvKeysUn.size == mvKeys.size 这里是去除畸变
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer) {  // 这里不会执行！ 实际上可以用 CHECK() 来进行检查
                delete mpInitializer;
//                std::cout << "Tracking::MonocularInitialization() 这里会执行吗？" << endl;
            }

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize (当前帧关键点数量要大于 100 才可以与参考帧进行匹配！)
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);

        // 特征描述子匹配    // mvIniMatches 对应的 vector 大小在下面的函数中才会扩充
        // mvIniMatches:   保存的是 mInitialFrame 图像 与 mCurrentFrame 图像配准好的关键点序号（对应 mCurrentFrame 图像）,
        //                 此时 index = mvIniMatches[i] 如果值不是 -1 ，那么说明 mInitialFrame 图像上序号 i 对应的关键点可以在
        //                 mCurrentFrame 图像上找到对应的匹配点，并且匹配点序号是 index。可以通过 mCurrentFrame.mvKeysUn[index]
        //                 获取图像 mCurrentFrame 上的关键点。
        // mvbPrevMatched: 保存的是 mInitialFrame 图像 与 mCurrentFrame 图像配准好的关键点（在 mCurrentFrame 图像上）
        //                 可以这样理解： mvbPrevMatched[i] = point; 此时 i 表示 mInitialFrame 图像上的关键点序号。point就表示
        //                 匹配的关键点 (在图像 mCurrentFrame 上),如果数组值为空，那么说明没有匹配上
        // 对应论文：IV. AUTOMATIC MAP INITIALIZATION 中第一步
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // 匹配数量大于 100 个，才能算是初始化成功，实际上在初始提取特征时。一般都是在 1000多个特征点，
        // 但是在实际跑的过程中初始化阶段成功匹配上的还是很少。说明 orb 特征点还是存在很大的不稳定性。
        // 也许是描述子本身不太好，或者关键点不鲁棒，其实关键点不鲁棒应该是重点。毕竟一个像素即使在微小的运动下，
        // 对于实际物体中的同一个点像素周围区域也会受到一些影响，这些小的因素其实影响了 FAST 关键点阈值。
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        // 经过 H 或 F 分解得到的
        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches) vbTriangulated[i] = true: 表示参考图像 i 关键点是符合的被成功三角化的关键点序号

        // Map Initialization ，选择 H 或 F，然后求解变换矩阵
        // 这里仅仅处理了参考帧和当前帧匹配成功的情况。对于恢复失败的话，程序会自动就获取下一帧图像，和之前参考帧进行匹配了。
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) // 成功初始化对应论文 IV:AUTOMATIC MAP INITIALIZATION
        {   // 不管是 H/F 哪个，只要满足恢复的条件，这里就是初始 pose 成功！// 下面需要细看！ mvIniMatches什么意思，以及vbTriangulated
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i]) // 参考图像有对应的匹配点并且 这对匹配点是无效的，也就是没有对应的 3d 世界坐标，需要进行剔除
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses 包含参考帧（此时作为世界坐标原点！）当前帧 pose 是刚刚通过 F/H 恢复出来的 Rcw,tcw
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);
            // 对于成功初始化
            CreateInitialMapMonocular();
        }
    }
}

// Map 包含如下指针： KeyFrame、MapPoints
// KeyFrame 反过来包含 Map 指针
// MapPoints 也包含 Map 指针
//    通过上面的关系可以发现。三者之间是一个耦合关系
//   创建两个关键帧及其三角化的地图点，并且在 mpMap 中插入。归一化地图点，更新局部建图线程需要的资源等等。最后标志当前初始化完成！
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames // 这里初始化是对于 mInitialFrame 和 mCurrentFrame 仅仅是拷贝相应的资源，对于自己本身的资源没有做配置更新
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);  // 初始化失败后在这些关键帧会在 mpMap.clear() 函数中释放！
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW(); // 计算 bow 向量，为之后的追踪帧和关键帧匹配做准备！
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and associate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++) // 以参考关键帧关键点为基准
    {
        if(mvIniMatches[i]<0)   // 没有被成功初始化的关键点{没有匹配好，以及匹配成功但是没有三角化成功的点},
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        // 添加当前关键帧对应的新建立的地图点
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // 观测实际上就是记录地图点和关键帧之间的联系
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // 地图点被关键帧观测到后一定要更新的量
        pMP->ComputeDistinctiveDescriptors(); // 计算当前地图点对应的最好的描述子
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure // 当前帧添加地图点，但是对参考帧没有做相应处理，因为后面把当前帧作为上次追踪的帧。然后有一系列处理
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections(); // 父关键帧节点是 curFrame
    pKFcur->UpdateConnections(); // 子关键帧节点是 ini

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // 全局 BA 优化，优化刚刚加入的关键帧 pose 和 有效的 3d 路标点
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1  计算当前关键帧中所有地图点{与自己的关键点是对应的}在关键帧所在相机坐标系下的中位数深度值 z
    float medianDepth = pKFini->ComputeSceneMedianDepth(2); // 此时初始关键帧地图点就是自己本身{以初始帧为参考世界坐标系}
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100) // 有效追踪的地图点也要满足 100 个
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // 统一单目全局尺度，更新地图点尺度

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth; // 按照中位数深度对平移向量进行归一化,如果换做 imu 固定尺度如何做？
    pKFcur->SetPose(Tc2w);

    // Scale points // 对初始帧 MapPoint 归一化即可相应的 当前帧 mappoint 也被归一化！
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth); // 对地图点按照中位数深度进行尺度归一化，
        }
    }

    // 局部建图线程插入关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose()); // 更新前面 BA 优化后的新的 pose
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    // 增加局部关键帧和地图点
    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame); // 这里把当前帧更新了参考关键帧，位姿，地图点，标号信息

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints); // 这里仅仅为了显示！跟踪过程的局部地图点
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose()); // 这里其实用 mCurrentFrame.mTcw 即可表达更清晰些

    mpMap->mvpKeyFrameOrigins.push_back(pKFini); // 保存初始参考帧。在 LoopClosing.cc 中用了这个变量

    mState=OK;  // 正式初始化完毕后，才会正式开始追踪！
}

//! \brief 检查上一帧对应的匹配地图点，是不是被其他地图点替换了。
//!    如果上一帧对应的地图点被某个地图点代替，此时需要更新对应位置的地图点
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced(); // 获得被替代的地图点
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

//! \note 只有当前普通帧和上一个参考关键帧(带有地图点的关键点)匹配对数符合条件。才算是追踪成功
//! \details 这里有两次条件筛选才算是追踪成功：
//!     1)通过 SearchByBoW() 匹配成功的对数 >=15
//!     2)经过一匹配后，再次位姿优化后，经过剔除一部分外点后，剩下的有效匹配对数 >=10 对
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW(); // 更新当前帧的 BOw 和特征向量

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;    // 当前帧和上一参考帧进行 3d-2d匹配的 3d 点

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches); // 根据 BoW 匹配当前帧和参考关键帧

    if(nmatches<15) // 只有当前普通帧和上一个关键帧(带有地图点的关键点)最后成功的匹配对 >=15 个，才算是追踪成功
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches; // 此时直接把匹配关键帧对应的地图点给了当前帧的地图点
    mCurrentFrame.SetPose(mLastFrame.mTcw); // 设定当前帧的 pose 初始值，为下面的 pose 优化提供初始值,实际上如果速度过快，
                                            // 以这个 pose 作为初始值可能结果不太好，在这里加入 Imu数据结果

    // 利用匹配关系进行，利用 3d 点和当前 frame 进行 3d-2d 的一元边，Pose 优化
    Optimizer::PoseOptimization(&mCurrentFrame); // 优化更新当前帧的 pose

    // Discard outliers 当前帧的地图点如果是外点，此时会把当前地图点 = null。减少有效匹配个数
    int nmatchesMap = 0; // 与参考关键帧有效匹配的地图点个数
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i]) // 在经过上面的 PoseOptimization() 后，内部的地图点会更新是不是内点/外点
            {   // 外点剔除
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 这里仅仅表示当前帧看不到当前地图点
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

//! \note 对于单目来说这里没必须要对 mLastFrame 的位姿进行计算。
//! 因为在追踪过程中，这个位姿都已经计算过了。这里重复了
//! 满足下面条件就会直接返回。
//!      1）单目
//!      2）正在处于追踪模式
//!      3）上一次追踪帧变为了关键帧
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    // 这个处理对单目来说是不是没有必要？？？？？ 经过分析是
    mLastFrame.SetPose(Tlr*pRef->GetPose()); // 在正式追踪第一帧完毕后，在第二帧执行这里时，实际上这里不需要进行更新，
                                            // 原本正式追踪完的这一帧 pose 就是计算好了的。这里在计算只会有数值误差。不知道其他情况下这里会不会变化？？？
    //
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

//! \see 论文 V TRACKING --- B Initial Pose Estimation from Previous Frame
//! \brief 这里通过假定匀速运动模型，推断出当前帧的初始位姿。然后根据上一帧地图点投影到当前帧图像，进而寻找匹配点。
//!    如果匹配对小于 20 ，还会再次加宽阈值 th = 14。再次投影匹配。只有满足当前匹配的点对 > 20
//!    之后才会进行位姿优化！然后更新有效地图点个数
//! \return 是否成功跟踪。（有效匹配的地图点是否大于 10 个，大于 10 个才算是成功跟踪）
//!    里面对于定位模式的代码还没有看！！！！！！
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true); // 在单目初始化时，这里也是 0.9 ，但是在初始化成功后的第一帧追踪时，使用的 0.7

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame(); // 经过分析可知，这里对于单目来说是不需要的！！！！

    // 这里在速度模型失败
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw); // 根据匀速运动模型，估计当前帧的初始 pose。这个模型如何用 imu 替代？？

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR); // 根据前一帧图像的地图点投影到当前帧来进行 pose 匹配。

    // If few matches, uses a wider window search // 匹配点对比较少。说明之前的匀速运动模型失效。需要用其他的方法。在次进行配准！
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL)); // 在调用下面函数之前，需要清理当前帧对应的地图点，里面会用
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20) // 表示追踪失败
        return false;

    // Optimize frame pose with all matches // 在优化后，有效地图点可能变为了外点，所以在调用完这里后，需要更新 nmatches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers 只有当前帧地图点不是外点才算是匹配的地图点.剔除无效的匹配点
    int nmatchesMap = 0; // 记录最后投影到当前帧的有效地图点！
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false; // 因为这些地图点都在一个存储地图点的类中存储着。所以这里能够写入值。
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--; // 剔除无效匹配点对
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

//! \see 论文 V:TRACKING --- D Track Local Map
//! \brief 更新局部地图{局部地图点、局部关键帧}、找到当前追踪帧中更多的地图点。之后进行 pose 优化，统计内点集。
//!  只有内点集个数满足最后的条件才算是跟踪成功！！如果不满足条件，仍然属于跟踪失败！
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map. 局部地图其实也是包含的{关键帧、地图点}
    // 更新局部地图{局部关键帧集、对应的局部地图点}
    UpdateLocalMap();

    // 在局部地图点中选择能够再次与当前追踪帧进行匹配的地图点。之后将新的地图点插入到当前追踪 Frame 中。提供了 3d-2d 优化位姿必备条件！
    SearchLocalPoints();

    // Optimize Pose 对上面新增加的地图点又再次进行 3d-2d 优化。并进行外点标记
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0; // 记录当前追踪帧中内点个数

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // 表示当前地图点被多少个帧找到{观测到}
                if(!mbOnlyTracking) // 追踪模式
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO) // 为什么单目和 RGB-D 不需要设置为 NULL???? 在 TrackReferenceKeyFrame()、TrackWithMotionModel()
                                            // 中在进行完 Optimizer::PoseOptimization(&mCurrentFrame); 后，都执行了下面这句话？？？？？ 但是在 Track() 函数中也进行了清理！
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50) // 条件限制会不会太宽了？？？
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

// 参考论文中 V TRACKING --- E.New KeyFrame Decision
// 只有满足一定的条件才能作为关键帧！！
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking) // 定位模式不需要。
        return false;

    // If Local Mapping is freezed by a Loop Closure, do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap(); // 返回地图中关键帧个数

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2; // 考虑到单目初始化成功后，正常追踪第一帧时。仅仅有两个关键帧
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs); // 参考关键帧的被跟踪的有效地图点个数 后面对于单目来说：当前帧的追踪点要小于这个值的 90%

    // Local Mapping accept keyframes? // ture: 表示局部建图线程不繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames(); // 这里在局部建图线程在处理的过程中都不能插入关键帧

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR) // 不执行
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70); // 对单目来说 = false

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 表示长时间没有插入关键帧，可能每次判断时，不符合条件太多，导致累计 1s 没有新的关键帧。为了防止追踪丢失。
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15); // 当前追踪帧内点个数小于参考关键帧有效地图点个数的90% 且内点个数 >15

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle) // 此时局部建图不繁忙
        {
            return true;
        }
        else // 单目直接返回 flase
        {   // 这里直接通知局部建图线程空闲，以便插入新的关键帧，防止追踪线程追踪失败
            mpLocalMapper->InterruptBA();// 这里对于单目来说直接停止 局部建图线程的 BA ,这样在局部建图线程使用局部 BA 优化时，
                                         // 每次优化都会检查这个函数内部的变量。只有这个变量为 false 时才会进行优化。这里中断优化是因为追踪过程中其实可以插入关键帧了。
                                        // 但是因为局部建图繁忙没法加入关键帧。所以这里提早结束局部建图线程的 BA 优化。然后尽快挂起局部建图线程。然后插入关键帧。否则会有追踪失败的风险！！！
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true)) // 返回 false 表示局部建图线程已经停止。
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF; // 建立完新的关键帧后，立刻更新参考关键帧
    mCurrentFrame.mpReferenceKF = pKF;

    // 双目和 RGB-D 操作(待看！！！！)
    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false); // 表示可以停止局部建图线程

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF; // 这个值从来没有用过！
}

// 在局部地图点中选择能够再次与当前追踪帧进行匹配的地图点。之后将新的地图点插入到当前追踪 Frame 中。提供了 3d-2d 优化位姿必备条件！
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched {这里其实有一个问题。这些地图点其实不一定能被当前帧观测到，只是在初始时的 pose 下，观测到的。所以下面的显示次数增加其实有待研究！！}
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible(); // 显示次数增加
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false; // 标记，之后在 SearchByProjection() 函数中需要用到
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // 将局部地图点全部投影到当前帧上来。记录哪些地图点隶属于当前帧
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId) // 已经被当前帧记录过的地图点不需要在处理
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5)) // 夹角为 60 °限制,会计算关键点所在的 predicted scale
        { // 地图点属于当前帧。说明 pMP 被当前帧观测到
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th); // 为 mCurrentFrame 中的 mvpMapPoints 增加一些新匹配的地图点！
    }
}

// 更新局部地图{地图点、关键帧}
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update 必须先更新局部关键帧之后，在更新地图点。顺序不能变。之后与当前追踪帧进行匹配 + pose 优化！
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

// 根据局部关键帧组，找到所有的地图点。之后为了和当前追踪帧进行匹配，然后进行 PoseOptimization() 优化追踪帧的位姿
void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear(); // 在单目初始化成功后。加入的是初始化时的地图点

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches(); // 获取关键帧有效地图点

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP); // 加入到局部地图点集中
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

// 下面找到局部关键帧组（总数量限制在 80 个）。就是论文中 V TRACKING --- D Track Local Map 中的说的 K1 关键帧集。以及 K2 关键帧集。
//    将这两个关键帧集都加入到局部关键帧组。之后调用上面哪个函数 UpdateLocalPoints（）取出地图点。然后与当前追踪帧进行再次匹配。然后优化 pose!!!
//    需要注意的是这里 K2 的组合包括了 k1中每一个关键帧的 {临近关键帧、孩子关键帧、父关键帧 }。 这里是取其中的一个作为代表
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter; // {关键帧,观测不同的地图点个数}
    // 参考论文 V TRACKING --- D Track Local Map 对应寻找与当前帧有共同观测地图点的关键帧组
    // 记录当前帧初始地图点被哪些关键帧观测到。因为这些关键帧对应的其他地图点才有可能被当前帧观测到。
    // 并记录那些关键帧观测的当前帧的地图点个数（这里的地图点仅仅是当前帧初始的地图点）也就是说，与当前帧初始共视点个数
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 当前帧有对应的地图点, 单目初始化成功后的下一帧的追踪时，已经有了一些匹配的地图点，但不是全部！
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty()) // 因为当前帧地图点就是根据上一关键帧或者上一普通帧进行匹配的。这个时候，这里不会为空。当然可能在局部建图的时候把地图点删除导致这里为空也有可能！
        return;

    int max=0; // 某个关键帧与当前帧有共同观测点的最多个数
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL); // 对应与当前帧最多观测点个数的关键帧{k1关键帧集合中的}

    mvpLocalKeyFrames.clear(); // 使得内部 size = 0，在初始化成功后的第一帧时，其实这个变量已经包含了初始化时的两个关键帧了
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());
    // 记录与当前帧有着共同观测点的关键帧 称为局部关键帧组，下面的帧属于 K1 关键帧集。这里可以直接在上面写。可能为了代码简洁。分开写了
    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max) // 找到与当前帧有最多共同观测点的关键帧
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first); // 加入局部关键帧组
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId; // 记录当前关键帧与哪个普通帧有共同观测点
    }

    // 将 Covisibility Graph 中与上面的关键帧组有链接关系的其他关键帧 再次加入到关键帧组。下面的帧属于 K2 关键帧集
    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 超过局部关键帧集上限就会自动退出
            break;

        KeyFrame* pKF = *itKF; // 选取 K1 集中的一个关键帧
        // 在 Covisibility 共视图（关键帧组合的）中，找到与关键帧有着最好的共视关系的前 10 个关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10); // 这个关键帧 vector 是之前与 pKF 关键帧共视强度从大到小排列的，所以下面需要一个相邻帧即可
                                                                                 // 这里有一个问题，就是这个函数内部实现中，如果这个关键帧共视图中他有多于 10 个临近关键帧
                                                                                 // 此时仅仅取出前 10 个。后面自动不要了。但是下面的循环中。如果有一种情况，就是取出来的 10 个帧
                                                                                 // 都不符合要求。那么其实还需要在把剩下的那些帧拿出来再次遍历。当然最方便的就是改写这个函数
                                                                                 // 直接取出所有的共视临近关键帧！！！
        // 在隶属于 K1 集的关键帧邻居中只选取一个符合要求的关键帧  {临近关键帧寻找} 其实是与 k1 关键帧有着最强的共视图关系(不包含在 K1 中)
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 因为相等时表示，之前已经将这个关键帧插入到关键帧组了。就不需要加入了
                {
                    mvpLocalKeyFrames.push_back(pNeighKF); // 这个就是论文中说的属于 K2 关键帧集的关键帧
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 这里记录 pNeighKF关键帧可能与当前帧 id 有着共同的共视关系
                    break; // 说明在一个属于 K1 帧集的关键帧。只要他的其中一个符合要求的邻关键帧(属于 K2 关键帧集)
                }
            }
        }
        // 在当前关键帧的孩子关键帧中寻找  {孩子关键帧寻找！} 在新的关键帧插入时，会自动更新共视关系，和孩子关键帧以及父关键帧
        const set<KeyFrame*> spChilds = pKF->GetChilds(); // 获取当前关键帧的孩子关键帧。就是与该关键帧有着最多共视地图点的后来新建立的关键帧.
                                                          // 这个与上面临近关键帧不同。不一定是最强共视关系。其实我们可以变为最强的共视帧。如何做？？
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        // 在当前父关键帧组中寻找 {父关键帧寻找！}
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax) // K1 关键帧集中与当前帧有着最多共视地图点的某个关键帧
    {
        mpReferenceKF = pKFmax; // 更新追踪线程的参考关键帧(用来与下一次新来的帧进行匹配)
        mCurrentFrame.mpReferenceKF = mpReferenceKF; // 记录当前帧对应的参考关键帧，之后在追踪时，调用 Tracking::TrackReferenceKeyFrame() 函数进行匹配
    }
}

//! \brief 当跟踪失败时，需要进行重定位解出跟踪帧位姿。
//! \details 首先根据关键帧数据库找到潜在匹配的关键帧(利用共同单词)。
//! 然后在潜在匹配的关键帧中利用 Bow+PnPsolver 获得 pose 及其匹配对,并对当前帧进行位姿优化。
//! 如果匹配点对不满足条件。那么对关键帧进行投影匹配 SearchByProjection() 在多获得几个地图点。之后在次进行 pose 优化
//! 直到满足匹配的地图点个数 > 50 才算是重新定位成功
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation // 得到潜在的匹配关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty()) // 这里一般不为空，这个是由 上面函数实现方式决定的
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers; // 内部存储与当前帧配对关系。使得后面可以直接进行优化！
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches; // vector<MapPoint*> 存储与当前帧匹配的地图点
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded; // 记录潜在配对关键帧是否丢弃！
    vbDiscarded.resize(nKFs);

    int nCandidates=0; // 对潜在的关键帧集，每个关键帧与当前帧通过 Bow 匹配后，匹配点对大于 15 个，算是一个有效的匹配。这个变量是统计有效关键帧个数

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver; // 保存 pnp 需要的匹配点对！
                nCandidates++; // 有效关键帧个数
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);
        // 这里在没有解决 bug 之前。总是需要等到 nCandidates = 0 才能结束，或者等到满足 50 个内点才能结束！
        // 但是等 nCandidates = 0 这个条件其实需要多次调用 pSolver->iterate() 这个函数。实际上这里能不能结束循环
        // 就看里面的 RANSAC 得到的 pose 结果。得到不好就会在这里一致循环。即使新来的帧数据也不会处理。这对实时运行程序不太友好！！！
    // 所以这个 bug 需要改变 bMatch 来解决！ 循环两次还不能成功就需要直接跳出循环了才对，不能在这里空等了。或者就进行一次查找找不到就直接退出！进行下一帧的处理
    // 实际上这个也不算是 bug ，只是觉得他的实现方式不太好！！！，可以试试增加迭代次数 PnPsolver 初始化时，改变哪个 3 为 4
    while(nCandidates>0 && !bMatch) // 只要其中一个关键帧满足重定位要求就会 break
    { //std::cout << "潜在关键帧个数1： " << nCandidates << std::endl;
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers; // 记录哪个关键点是内点
            int nInliers;
            bool bNoMore; // 记录是否迭代达到了最大次数，达到了，说明此次迭代不好，需要丢弃当前匹配的关键帧

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers); // 返回一个相对好的 pose

            // If Ransac reachs max. iterations discard keyframe 此时其实可以计算处 Tcw ，但是没有达到要求的退出，说明没有得到好的匹配
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--; // 递减有效关键帧个数！
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound; // 记录所有内地图点

                const int np = vbInliers.size(); // 当前帧中关键点个数

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j]) // 关键点是内点
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j]; // 记录匹配的地图点(内点)，为后面位姿优化做准备！
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame); // 位姿优化！

                if(nGood<10) // 要求内点个数
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++) // 清理外地图点.因为在上面位姿优化后，在当前帧中已经标记了那些点是外点。需要清理
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100); // 此时 mCurrentFrame 内部已经有了初始 Pose

                    if(nadditional+nGood>=50) // 新增加的地图点 + 上面通过 BOW+Pnp求解后的地图点
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame); // 再次进行优化

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            // 下面的 for 是自己添加的！
                            for(int io =0; io<mCurrentFrame.N; io++) // 清理外地图点.因为在上面位姿优化后，在当前帧中已经标记了那些点是外点。需要清理
                                if(mCurrentFrame.mvbOutlier[io])
                                    mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);
                            for(int ip =0; ip<mCurrentFrame.N; ip++) // 这里之前其实需要再次对外点进行清理！！！这也算是一个 bug
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++) // 清理资源！
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        } // end for
       // break; // 自己添加的！
       // std::cout << "潜在关键帧个数2： " << nCandidates << std::endl;
    } // end while
    if(!bMatch)
    {
        return false; // 重定位失败！
    }
    else // 重定位成功
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

// 目前针对单目：在系统初始化失败，以及初始化完毕后小于 5 个关键帧跟踪丢失时，这里会进行重置。或者在显示线程中进行手动的重置，最后也会调用 Reset()
void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped()) // 等待显示线程挂起
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames) // 这些资源在闭环和局部建图线程中共享了
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)   // 初始化失败时，自动清理资源
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)    // 唤醒显示线程！
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
