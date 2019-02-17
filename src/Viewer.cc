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

#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>

namespace ORB_SLAM2
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath):
    mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps; // 周期

    mImageWidth = fSettings["Camera.width"];    // 单目参数无此值
    mImageHeight = fSettings["Camera.height"];  // 单目参数无此值
//    std::cout << "Viewer mImageWith ,mImageHeight 值是多少： " << mImageWidth << mImageHeight << std::endl; // 对于文件中无此值的情况。此时为0
    if(mImageWidth<1 || mImageHeight<1) // 此时单目摄像头默认参数，如果自己的单目相机有变化。那么这里需要改变！
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
}
// 显示线程开始
void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST); // 开启深度测试

    // Issue specific OpenGl we might need
    // 两个函数的功能或者混合的概念参考：
    // https://www.cnblogs.com/Clingingboy/archive/2010/10/26/1861261.html
    // 详细介绍 glBlendFunc() 函数： http://www.cnblogs.com/ylwn817/archive/2012/09/07/2675285.html
    glEnable (GL_BLEND); // 使用颜色混合，就是前一个带颜色的物体和另一个带颜色的物体叠加了。此时混合表示两个物体的颜色在叠加部分会变成两种颜色的混合色
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 选择混合选项

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));

    // 对于 bool 类型，这里的真假哪里第一个值，就是是否选择该选项，第二是显示交互按钮还是普通的选择项
    // 新建按钮和选择框，第一个参数为按钮的名字，第二个为默认状态，第三个为是否有选择框
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true,true); // 表示选择该项，第二个 true 表示显示普通项
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);  // 默认为不启动定位模式
    pangolin::Var<bool> menuReset("menu.Reset",false, false);   // 默认不复位，然后后面是交互的按钮.系统初始化选项

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(

                // 参数信息：前两个是窗口大小，接着是与相机内参数差不多的数据，最后两个参数是近点和远点显示
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),

                // 前三个参数：相机在世界坐标的位置 中间三个：相机镜头对准的物体在世界坐标的位置   后三个参数：相机头顶朝向的方向
                // 参考：https://blog.csdn.net/ivan_ljf/article/details/8764737
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ,   0,0,0,   0.0,-1.0, 0.0)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc; // 相机-->世界的矩阵
    Twc.SetIdentity();

    cv::namedWindow("ORB-SLAM2: Current Frame"); // 在使用 imshow 显示图像之前必须先初始化一个窗口

    bool bFollow = true; // 跟随相机视角
    bool bLocalizationMode = false; // 不启动定位模式

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc); // 根据相机位姿调整视角
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc); // 跟随相机视角
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }
        // 默认不启动定位模式，
        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode(); // 激活定位模式
            bLocalizationMode = true;
        }   // 关闭定位模式
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f); // 使得 gui 面板显示白色
        mpMapDrawer->DrawCurrentCamera(Twc); // 根据相机位姿画出相机模型 绿色为相机追踪帧
        if(menuShowKeyFrames || menuShowGraph)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph); // 这里的 Pangolin 变量其实就是模板参数
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints(); // 绘制地图点{局部=红色 全部=黑色}

        pangolin::FinishFrame(); // 完成一次绘制

        cv::Mat im = mpFrameDrawer->DrawFrame(); // 画出当前图像上的关键点、显示基本信息{地图点、关键帧个数等等}
        cv::imshow("ORB-SLAM2: Current Frame",im);
        cv::waitKey(mT); // 等待 ms 仿真视频流

        if(menuReset)   // 手动在 gui 中选择，重新进行系统置位！保持跟踪模式。原来是定位模式，需要进行关闭
        {
            menuShowGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode(); // 关闭定位模式
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->Reset();
            menuReset = false;
        }
        // 只要在其他线程（实际上是在主线程 Tracking 中 Reset()中调用的）中手动调用这个函数：Viewer::RequestStop()，此时当前 Viewer 线程就会检测到，然后就会挂起当前线程。
        // 当线程从休眠状态返回时，仍然会判断 isStopped() ，实际上在调用 Viewer::RequestStop() 后需要再次调用 Viewer::Release()，此时才会退出挂起状态。然后继续执行当前
        // 线程
        if(Stop())
        {
            while(isStopped()) // 不断的检测，并挂起当前线程。直到在 Tracking:;Reset() 函数最后调用 Viewer::Release()。使得返回为 false ，此时当前显示线程不会进入挂起状态
            {
                usleep(3000);   // 挂起 3000 us 后，然后再次启动时，仍然在这里启动再次判断 是否停止
            }
        }

        if(CheckFinish())   // 这个是在主线程中调用 Viewer::RequestFinish() 时才会退出当前线程
            break;
    }

    SetFinish(); // 设置当前线程完成标志
}
// 系统完成时需要调用的+++++++++++++++++++++++++++++++++++++++
    // 请求完成
void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}
// 下面是在系统重置时的函数++++++++++++++++++++++++++++++++++++++
// 请求显示线程停止
void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}
// 返回是否停止当前线程
bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested) // 已经发出让当前线程停止的信号
    {
        mbStopped = true; // 停止当前线程
        mbStopRequested = false;
        return true;
    }

    return false;

}
// 唤醒显示线程
void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

}
