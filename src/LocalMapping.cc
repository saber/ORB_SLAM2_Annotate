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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false); // 在局部建图处理资源期间，在追踪线程中不能插入新的关键帧。
            // 对于单目来说。在初始化后。内部有两个关键帧。但是正常追踪后。这里就只能有一个关键帧。只有局部建图线程处理完一个关键帧。跟踪线程才能插入新的关键帧。
            // 所以这里不会积累关键帧。除了单目初始化时情况。经过测试，这里确实每次执行时，内部都会有一个关键帧。但是为什么没有添加多个关键帧？？？按理说没有这种限制条件？？？

        // Check if there are keyframes in the queue
        // 下面整个过程参照 VI LOCAL MAPPING 部分
        if(CheckNewKeyFrames())
        {   // 存在插入的关键帧
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints 如果当前关键帧序列中，此时还有关键帧。那么内部处理一次三角化后，需要返回。为了加速处理其他关键帧。
            // 因为三角化其他新的地图点什么时候都可以，目前优先处理新的关键帧
            CreateNewMapPoints(); // 对关键帧之间没有对应地图点的关键点，找到匹配关系，然后在进行三角化，产生新的地图点

            if(!CheckNewKeyFrames()) // 关键帧序列没有新关键帧了。那么可以放心的处理当前关键帧和临近关键帧组之间地图点的融合。
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors(); // 融合一些地图点。找到更多的地图点和关键帧之间的匹配关系
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested()) // 只有当前没有新的关键帧时，我们在进行局部 BA 优化。因为局部优化。不需要很及时和频繁优化。
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2) // 在单目初始化完毕后。那里已经进行了局部 BA 优化。所以这里不需要再次优化了。
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap); // 此时，当前关键帧可能已经有了更多的地图点

                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame); // 插入闭环帧，用来检测闭环。但是插入的这个关键帧。但是现在插入的关键帧。到后面可能会被剔除。所以需要 bad flag 检测
        }
        else if(Stop()) // 停止标志置位 true
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000); // 挂起当前线程。进入休眠状态
            }
            if(CheckFinish()) // 这里只有在整个系统关闭时，才会调用请求完成.然后这里才会为真。但是总觉得这里不会调用。这里经测试没有调用过
                break;
        }
        // 判断当前是否需要清理资源（当前主线程 Tracking 跟踪丢失(初始化完毕后)并且地图中关键帧个数小于等于 5、
        // 或者在可视化线程 Viewer 中，当手动在 GUI 中点击 Rest 时）这些条件都会间接导致 系统重置，然后清理相应资源(关键帧以及对应最近添加的地图点！)
        ResetIfRequested(); // 将 mlNewKeyFrames置位空，这里是重置请求，所以只是清理一下资源即可，

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true); // 在跟踪线程中可以插入关键帧,因为后面休眠当前线程 3000ms ，所以追踪线程可以进行插入关键帧

        if(CheckFinish())   // 检查是否停止该线程！在整个系统全部执行完毕时调用 LocalMapping::SetFinish() 时，才会停止
            break;

        usleep(3000); // 这个休眠时间可能会有影响
    }

    SetFinish(); // 设置局部建图线程完成
}
//  在 Tracking 线程中调用,插入新的关键帧，禁止 BA 优化
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

// 判断当前关键帧序列是否有新的关键帧插入(在追踪线程中插入的！)
// true: 有 false: 无
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}
// 记住;跟踪线程仅仅插入关键帧。做了一次匹配。剩下的所有操作都是在局部建图线程中处理的。
// 从新关键帧 list 中取出一个关键帧，计算 BoW 加速匹配，对取出的关键帧加入到对应地图点 observations 变量中，将他们进行关联。
// （因为单目时，有些地图点是在跟踪线程中是直接通过投影得到的匹配，创建关键帧时，没有把关键帧加入到地图点变量中。因此地图点包含关键帧 这种关系当时没有更新，此时需要更新）
// mlpRecentAddedMapPoints 这个变量加入的是最近添加的新三角化的地图点（但是这个变量中包含的地图点会有重复）。局部建图线程中会有地图点的擦除操作。
//    对于单目这个变量包含的内容有：1、单目初始化时的所有地图点 2、在 CreateNewMapPoints 中创建新的地图点
//   更新该关键帧对应的共视图，并将该关键帧插入地图中。（默认重的关键帧不会插入）
void LocalMapping::ProcessNewKeyFrame()
{
    {   // 取出新的关键帧
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front(); // 这里把单目初始化时的两个关键帧中的初始关键帧放在了第一位。这里先取出第二个关键帧
        mlNewKeyFrames.pop_front(); // 删除
    }// 经过试验。这里在处理之前。除了单目初始化。此时 mlNewKeyFrames 里面仅仅有一个关键帧。也就是局部建图线程中不会累积关键帧

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW(); // 计算当前关键帧 bow ，如果之前计算过了，那么此时就不会再次计算！。因为只有在单目初始化的时候，
                                     // 两个关键帧就已经计算了 bow。但是在其他关键帧在创建时，没有进行该计算。都是在局部建图线程中这里进行计算的。
                                     // 当然在还不是关键帧的时候，作为普通帧。也可能在跟踪过程中，初始匹配时就计算了 bow 向量

    // Associate MapPoints to the new keyframe and update normal and descriptor。因为跟踪线程没有做。仅仅做了匹配。但是没有进行双向关联
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 这里的地图点并没有把当前关键帧加入自己的变量里。所以下面需要把关键帧加入到地图点。建立联系

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame)) // mObservations 中是否有这个关键帧变量。地图点与给定关键帧没有构成联系，需要添加关联。
                                                          // 因为当前帧与该地图点在追踪线程中已经构成了匹配联系
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i); // 构建地图点和关键帧之间的联系。
                    pMP->UpdateNormalAndDepth(); // 在地图点加入关键帧联系时，必须要进行调用更新
                    pMP->ComputeDistinctiveDescriptors(); // 更新当前地图点最具有代表性的描述子
                }
                else // this can only happen for new stereo points inserted by the Tracking // 这里对于单目来说这里也会调用，已测试！下面就是原因
                {   // 这个地图点的加入会有重复的。其实可以用 std::set 类型
                    mlpRecentAddedMapPoints.push_back(pMP); // 对于单目初始化时的两个关键帧。那些地图点都会在这里加入。因为那些地图点和关键帧进行了 AddObservation() 关联
                                                            // 对于单目来说。跟踪线程仅仅建立关键帧。除了单目初始化外，其他关键帧在创建时并没有建立新的地图点。
                                                            // 在 CreateNewMapPoints 函数中，再一次对单目初始化的两个关键帧寻找新的地图点。然后三角化新的地图点。
                                        // 需要注意的是，这里还会发生的情况是：当前正在处理的关键帧，在跟踪线程中已经有对应的地图点了。但是在局部建图线程中的
                                        // SearchInNeighbors() 函数，有些比较早的地图点会被新的地图点替换。所以这里即使后出现的关键帧，也会包含了一些最近建立的地图点。！！！
                                        // 这也是为什么除了单目初始化外的2次。其他时候仍然会执行进入这里的原因！！！！！
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections(); // 因为地图点关联的关键帧会发生变化，所以此时共视图之间的权重也会发生变化，可能会增加新的共视关系！

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);  // 插入关键帧，这个对于单目和双目来说，最开始初始化的时候，已经将初始化的关键帧加入到这里来了。
                                            // 因为这个函数内部是 std::set::insert() 重复元素不会插入
}
// 对最近新增加的地图点进行冗余去除.这些点可能是错误的三角化的（由于一些伪数据关联(匹配)）。不好的点就会设为 bad flag。然后在 mlpRecentAddedMapPoints 中擦除。
// 对于没有满足一定要擦除的条件，该地图点就会保留在 mlpRecentAddedMapPoints 中，然后下次在检验。直到满足成为好的地图点。才会在这个变量中擦除（但是不设置 bad flag）
// 成为好点的条件：满足地图点经过了 4 个关键帧，仍然没有被标记 bad flag。
//    这些地图点为什么会变为坏点或者说为什么要被剔除，其实还有下面条件
//        1）关键帧剔除函数，会把不好的关键帧剔除。导致对应的地图点的 observation obs 都会减少。使得地图点被关键帧观测次数下降
//    除了在这里一些地图点被剔除外，其实在局部 BA 优化时，不好的地图点也会被剔除。
//  这些策略，作为剔除一些不良地图点。
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin(); // 最近新增加的地图点
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad()) // 说明当前这个地图点已经在 mlpRe..这个变量里面，然后之前已经被删除了。因为 mlpR..这个变量是 list 可以插入相同的元素。其实把 mlpRecentAddedMapPoints 变为 std::set 就好了
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f ) // ??? 为什么用这个条件？？ 这个条件就是 VI.LOCAL MAPPING B 的条件 1
        {
            pMP->SetBadFlag(); // 标记当前地图点是坏点。并在地图中擦除这个点，以及取消该地图点与对应关键帧之间的联系
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs) // 条件：经过 3 个关键帧了，但是该地图点被关键帧观测的次数却小于等于 2。说明该地图点不好
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3) // 经过了 4 个关键帧。但是该地图点没有成为坏点。所以不需要设置为 badflag? 但是这些策略有没有什么依据？？
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++; // 对于这种情况，说明这个地图点不满足必须擦除的条件。那么就会在这里保留。然后经过下次再次检验。直到满足条件为止
    }
}
// 当前正在处理的关键帧在共视图中找到其对应的临近关键帧组。把临近关键帧组中的每个关键帧和当前正在处理的关键帧，对他们进行匹配（因为两个关键帧的位姿已知），
// 在他们中找到没有匹配上地图点的关键点，尝试找到这些关键点之间的匹配。 之后对这些匹配点对进行三角化，获得新的地图点。（会有一系列检测，比如深度值为正，重投影误差满足阈值条件）
// 把新的地图点和对应的两个关键帧之间进行关联 AddObservation()。更新地图点平均 Normal 向量，以及将新的地图点加入到 mlpRecentAddedMapPoints 这个变量中。
// （下次会在 MapPointCulling 中把不符合要求的地图点给剔除掉）
//   对于单目来说。在初始化阶段。一下子加入了两个关键帧。因为在处理完第二个关键帧后，这里可能三角化了新的地图点（也可能没有新的地图点加入）。然后就退出。
// 再次处理第一个关键帧。那么第一个关键帧就会有新的地图点.然后进行关键这和地图点之间的关联。
//    需要注意的是：这里的地图点的 mnFirstKFid 就是当前局部建图线程正在处理的关键帧的 Id
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn); // 临近关键帧组

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor; // 为什么要乘 1.5f ???

    int nnew=0; // 记录当前新增加的地图点个数，但是并没有作为返回值！

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames()) // 说明这里至少处理一个临近关键帧组！这里针对单目而言。刚刚初始化之后。加入了两个关键帧,所以直接处理一次临近关键帧就退出了。防止关键帧的累积
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1; // 当前正在处理的关键帧和临近关键帧之间的光心向量 即为基线
        const float baseline = cv::norm(vBaseline); // 基线之间的距离

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else // 单目 深度不确定
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2; // 基线/中位数深度  比值！作用？？？？

            if(ratioBaselineDepth<0.01) // 是不是说明两帧之间距离太近，导致后面三角化的误差很大？？应该是，但是这个比例值如何确定是 0.01???
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices; // 当前处理的关键帧和关键帧 pKF2 之间没有对应地图点的关键点之间的匹配关系。为之后的三角化新的地图点做准备！
                                                      // vMatchedIndices[i] = idx; 表示当前关键帧关键点 i 和关键帧 pKF2 关键点 idx 是一个匹配对
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false); // 找到当前关键帧和 pKF2 之间的匹配关系{未三角化的关键点}

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size(); // 将要三角化的匹配的地图点{这些点在当前帧和临近匹配的帧之间都没有对应的地图点，所以下面需要创建新的地图点}
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first; // 当前关键帧对应的关键点序号
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1]; // 双目
            bool bStereo1 = kp1_ur>=0; // 单目来说是 false

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0; // 单目来说是 false

            // Check parallax between rays // 下面是将图像坐标转化为归一化平面上的坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1; // 世界坐标系下的，为什么不加上平移向量？？
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2)); // 与之前计算视差时不同，这里没有利用平移信息，不知道为什么？？？？

            float cosParallaxStereo = cosParallaxRays+1; // [0,2]
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1) // 双目
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2) // 双目
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2); // 对单目来说这里不变，都是 cosParallaxRays + 1

            cv::Mat x3D; // 三角化的新的地图点（世界坐标系下的）
            // 对于单目来说，表明视差角 cos 角度在 0-90 270-360 ，此时条件为真。为什么要设定这种条件？？？ 这种为什么代表视差高？？
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {   // 三角化匹配的关键点得到新的地图点
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F); // 直接查看多视图几何中 218 页给出的 A 矩阵及其三角化的计算方法
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t(); // 得到的 3d 地图点（这个是齐次 4x1 的向量，需要化为 3x1 非齐次坐标）

                if(x3D.at<float>(3)==0) // 如果该值为0，那么后面无法求解欧式坐标了。
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3); // 求出真正的 3x1 欧式坐标，这个是世界坐标系下的坐标点

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t(); // 列存储

            //Check triangulation in front of cameras // 检查新建的地图点是否同时在两个相机的前面
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2); // 只计算 z 坐标。自己可以导出来
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1) // 单目
            {
                float u1 = fx1*x1*invz1+cx1; // 转换到关键帧相机图像坐标系下
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1) // 一个像素误差？？？，误差计算方式？？？但是这里为什么需要乘以 sigma
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor) // 这个又是什么？？？
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap); // 需要注意的是，单目初始化时，这里的 mpCurrentKeyFrame 可能是第一个关键帧，所以地图点 FirsId=0。

            // 地图点增加观测
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1); // 新增加当前关键帧对应的地图点。这个关键点是之前挑选出来的没有对应地图点的
            pKF2->AddMapPoint(pMP,idx2);
            // 这里应该更新一下 pKF2->UpdateCovilibility() 函数？？？？？？方便追踪线程 TrackLocalMap() 函数使用最新的临近关键帧集
            pMP->ComputeDistinctiveDescriptors(); // 计算地图点代表性的描述子，为后期匹配做准备

            pMP->UpdateNormalAndDepth(); // 计算尺度不变性距离，以及 normal 向量

            mpMap->AddMapPoint(pMP); // 在地图结构中增加新的地图点
            mlpRecentAddedMapPoints.push_back(pMP); // 这个地图点加入最近增加的地图点变量。会在这里面进行冗余地图点的筛选

            nnew++;
        } // end for  Triangulate each match
    } // end for Search matches
}
// 1、将当前处理的关键帧对应的地图点与其指定个数的临近关键帧进行融合（投影到临近关键帧图像坐标系）。找到临近关键帧新的地图点匹配关系。更新地图点信息
// 2、将临近关键帧所有地图点与当前正在处理的关键帧进行融合。更新当前正在处理的关键帧的地图点匹配关系。
//    因为当前正在处理的关键帧匹配的地图点发生变化(有些原来关键帧对应的地图点被新的地图点替换了)。所以需要更新一下共视图。以及对应的地图点描述子、normal 向量都会更新下
//    需要注意：在下面会调用 Fuse() 函数。该函数中会把地图点进行一部分更新。所以调用一次。那么相应关键帧匹配的地图点就可能会发生变化。
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    // 找到当前关键帧对应的最好的临近关键帧。
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; // 标记当前帧已经是融合目标了。

        // Extend to some second neighbors // 扩展到临近关键帧的临近关键帧，再次寻找将要融合的目标关键帧
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // 将当前正在处理的关键帧对应的地图点投影到目标关键帧上。在目标关键帧上找到其对应的匹配关键点。然后增加目标关键帧对应的地图点（关键帧的关键点和地图点关联）。
    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches); // 将地图点融合更新到关键帧 pKFi 中。增加或替换新的地图点与关键点之间的联系。或者增加地图点和 pKFi 关键帧之间的联系
    }                                         // 这次融合，对于当前正在处理的关键帧。有些地图点可能已经被清空，且置为坏点。也就是说当前处理的关键帧匹配地图点关系可能会发生变化

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    // 获取所有目标关键帧上的地图点，为后面将地图点与当前正在处理的关键帧融合做准备
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates); // 将所有[隶属于当前关键帧临近关键帧组]地图点，全部尝试投影到当前关键帧上。
                                                      // 更新当前关键帧关键点和地图点之间的联系。或者地图点与关键帧之间的联系


    // Update points // 通过分析上面两个 Fuse() 函数，可以知道这里仅仅更新当前关键帧的匹配的地图点，就可以把上面处理过程中的所有变动的地图点相关变量全部更新了。
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 获取最新的匹配关系
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors(); // 更新一下当前地图点对应的最好的描述子
                pMP->UpdateNormalAndDepth();    // 更新一下 normal 向量,尺度不变距离
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections(); // 此时上面与 mpCurrentKeyFrame 链接的共视图中的其他关键帧都还没有进行更新链接关系
}
// 通过两个关键帧之间的 Pose。计算两个关键帧之间的基础矩阵 F12
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    // 这两个其实就是 T12 = T1w*T2w^T 的旋转和平移部分
    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12); // 获取平移向量 t12 的反对称矩阵

    const cv::Mat &K1 = pKF1->mK; // 两个内参矩阵对于单目来说一样，可能对双目不太一样？？？？？
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv(); // 这里是检查对极约束 t^R 利用 14 讲书上的对极约束部分即可得到
}
// 请求停止局部建图线程。在定位模式时，会调用这个函数来停止局部建图线程。在闭环检测线程中 CorrectLoop() 中，
// 避免在闭环线程中插入新的关键帧。还有在闭环线程中。跑全局的 BA 优化时。都会请求停止当前的局部建图线程
void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}
// 只有在请求停止线程 RequestStop()。以及当前局部建图线程允许停止时。(设置 mbNotStop 变量= false)
// 什么时候允许停止局部建图线程？：就是在跟踪线程中。创建新的关键帧时。
// 在插入新的关键帧之前是不允许停止局部建图线程的。只有在结束插入新的关键帧，才允许停止局部建图线程
bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}
// 清理局部建图资源，以及对应指针内存。让局部建图线程可以跑起来.
//! \note 这里没有，也不要释放最近添加的地图点{mlpRecentAddedMapPoints}！
void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear(); // 清除 list 内部元素，对于指针指向的内容，需要自己释放

    cout << "Local Mapping RELEASE" << endl;
}
// 是否接受关键帧, true: 表示局部建图线程不繁忙
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}
// flag = true; 表示当前局部建图能够接受关键帧(在追踪线程中需要使用这个变量，判断是否需要新建立关键帧)。
// flag = false; 局部建图不接收关键帧
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}
// flag = true，此时如果 mbStopped = true，则表示停止了当前线程。此时设置失败！返回 false
//    flag = false. 此时设定当线程可以停止。返回 true 。设置成功！
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}
// 参考论文 VI LOCAL MAPPING E 部分
// 在当前关键帧对应的局部关键帧集中，剔除冗余的关键帧。当前关键帧此时不会被剔除。
//    冗余关键帧：在这个关键帧观测的地图点中。有些地图点能够被其他关键帧观测到。对于其中一个地图点来说，该地图点在其他关键帧上对应的金字塔层数比较好。
//              如果比较好的次数大于等于 3，也就是对应至少 3 个其他关键帧能比较好的观测到这个地图点。那么这个关键帧对于该地图点来说是冗余的，
//              此时冗余变量 nRedundantObservations++。对当前关键帧观测的地图点都做这个测试。最后如果这个冗余变量 > 0.9*总有效的地图点个数
//              说明当前这个关键帧对所有观测的地图点来说就是冗余的。那么最后会剔除这个关键帧。
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames(); // 获得局部关键帧集

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0) // 参考世界坐标系不处理
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0; // 当前关键帧对应的有效的地图点个数
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs) // 保证当前地图点被关键帧观测的次数大于这个阈值 3，才有可能有较好的尺度>=3
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1) // 条件：in at least other 3 keyframes (in the same or finer scale)
                            {
                                nObs++;
                                if(nObs>=thObs) // 表明至少有 3 个关键帧，该地图点在这些关键帧上的金字塔层数较低。有较好的 scale
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++; // 满足条件：增加冗余次数
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs) // 表明当前关键帧所看到的地图点。有 90% 的点，都能够被其他至少 3 个关键帧很好的观测，则当前处理的关键帧需要剔除
            pKF->SetBadFlag(); // 清理与当前关键帧相关的联系和资源
    }
}
// 返回平移向量 t 的反对称矩阵
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}
// 在系统重置时，需要调用的。该函数在之宗线程中重置系统时调用的
void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000); // 挂起当前调用这个函数的线程
    }
}
// 清理局部建图所用资源。只有系统请求了重置，这个才会执行！
void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {    // 这些资源包含的指针内存，已经包含在 Map 类里面了。通过 mpMap->clear(); 已经释放了。
        mlNewKeyFrames.clear(); // 这里仅仅清除了 list 本身，但是内部指针指向的元素在哪里清除的？这里似乎忘了清除资源了。
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}
// 在整个系统执行完毕时，才会请求完成，然后结束当前局部建图线程
void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}
// 检查当前局部建图线程是否请求完成
bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
