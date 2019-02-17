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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2
{
// 该函数在下面情况会被调用！
// 在单目初始化：CreateInitialMapMonocular函数 优化 20 次
// 闭环优化： RunGlobalBundleAdjustment函数 优化 10 次
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

// 注意：这里面没有去除不好的地图点！
// 这里实际上就是 14 讲上面的重投影误差，不断优化节点 pose 和三维点坐标。然后在从新更新地图点位置和关键帧位姿。
// vpKFs: 包含当前系统中所有的关键帧
// vpMP: 包含当前系统中的所有的地图点
//    对于 G2O 自带的3d-2d 二元边时，位姿节点是相对于世界坐标系的，路标节点也需要是世界坐标系的，因为内部计算误差时，使用的是将 3d 点映射到相机坐标系，然后进行 2d 像素误差计算
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP; // 记录当前地图点是否被加入 BA 优化中， ture: 表示没有被加入优化中
    vbNotIncludedMP.resize(vpMP.size());
    // 求解器设置可以参考 slam 14 讲第七章例子
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>(); // 线性求解器类型 g2o::LinearSolverEigen

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag) // 在单目初始化调用时为 null
        optimizer.setForceStopFlag(pbStopFlag); // 设置停止标志！

    long unsigned int maxKFid = 0; // 记录当前优化，有多少个关键帧参与

    // Set KeyFrame vertices // 加入顶点！
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose())); // 这里用了拷贝 pose 然后去优化，考虑其他线程会操作同一个 pose，
                                                                 // 如果这里直接用原始 Pose,那么这里需要加锁，但是加锁的话，优化过程会很长，其他线程不能访问这个 pose 了
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0); // 第一个关键帧作为参考世界坐标系，不做优化！
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices 及其每个点与关键帧链接的边！(3d-2d 重投影误差)
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1; // 这里 id 为什么要加 1 ？？？
        vPoint->setId(id);
        vPoint->setMarginalized(true); // 设置边缘化
        optimizer.addVertex(vPoint);
        // 当前地图点被哪个关键帧观测过，那么此时就会添加对应的边
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES 当前地图点归属的关键帧的某个对应的关键点(2d)，构成了 3d-2d 投影误差边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid) // 表示在另一个线程中有改动，不符合上面添加的顶点标号。此时不做处理，但是会发生吗？？
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second]; // 去除畸变的关键点{图像坐标系}

            if(pKF->mvuRight[mit->second]<0) // 单目进入
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id))); // 路标节点
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId))); // 关键帧位姿节点
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave]; // 因子选择标准？？
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2); // 关键点所在金字塔层越高，这里的权重越小。依据？？

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else // 双目
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0) // 当前 MapPoint 顶点没有对应的边，需要在优化器中去掉该节点
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations); // 单目初始化时为 20 次迭代

    // Recover optimized data

    //Keyframes // 更新 BA 优化后的关键帧 pose 相关参数
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0) // 单目初始化时，此时默认为 0，
        {
            pKF->SetPose(Converter::toCvMat(SE3quat)); // 更新优化好的 pose
        }
        else // 在闭环线程进行全局 BA 优化时。处理了地图中所有的关键帧。这里没有直接更新 Tcw 的值，
             // 是因为哪个值会在闭环线程中开辟全局 BA 线程时会用到，纠正局部建图新增加的关键帧。之后纠正完毕后才会更新 Tcw
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF; // 标记当前关键帧参与了闭环优化，且对应的关键帧 id 是 nLoopKF
        }
    }

    //Points // 更新 BA 优化后的地图点位置
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i]) // 当前地图点没有加入优化器中，则不需要更新
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1)); // 注意这里的标号要与之前加入优化器的标号一致！

        if(nLoopKF==0) // 在单目初始化时
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else // 闭环成功时,
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}
// 利用 3d - 2d 重投影误差优化 pose（当前帧）.会更新当前帧的 pose。会标记当前帧的某些地图点是外点
//  pFrame: 当前追踪的帧
//    对于一元边，位姿是相对于世界的，3d 点是世界坐标系的，然后内部用 pose * 3d点=相机坐标系 3d 点
//    下面在添加完边和节点后，分为 4 次进行优化。每次优化都会更新内点/外点。
//    注意在调用这个函数之前 pFrame 已经通过与前一帧图像/前一个关键帧匹配好了，然后当前帧 pFrame 就有了地图点3d。
// 返回值： 所有边数(当前帧的地图点个数) - 优化后产生的外点数 = 最后实际有效的边数(地图点个数)
int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0; // 统计有效的边，也是地图点个数

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw)); // 使用 g2o 自带的 3d-2d 一元这个是世界坐标系下的坐标。这里的 pose 实际上是作为一个初始 pose ，一般都是前一阵的 pose 作为初始值
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N; // 这个是关键点个数 < 地图点个数

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono; // 单目方案，统计当前所有加入优化器中的边
    vector<size_t> vnIndexEdgeMono; // vnIndexEdgeMono[i]=index,就是当前帧第 index 个关键点构成了边；size() = vpEdgeMono.size()
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo; // 双目
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);
    // 鲁棒核函数阈值参数,一般使用的第一个参数？？？？
    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();   // 一元边，3d 点投影到当前位姿构成的单边

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos(); // 这个是世界坐标点，内部需要将该点投影到当前相机图像坐标系
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }


    if(nInitialCorrespondences<3) // 实际上在调用这个函数之前就已经判断了 pFrame->mvpMapPoints[i] 的个数要大于 15 个，所以原则上这里不会成立
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier：分类标准是通过 chi2 边的误差来定义
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）？ 如何计算？？？
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10}; // 每次优化次数

    int nBad=0; // 记录不好的边
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0); // 对 level = 0 的边进行优化
        optimizer.optimize(its[it]);

        nBad=0; // 记录当前优化完毕后，外点的个数
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError(); // 计算误差，之后通过 e->chi2() 可以得到一个类似卡方检验的数据？
            }

            const float chi2 = e->chi2(); // ???误差平方
            // 外点
            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true; // 记录 idx 代表的关键点是外点
                e->setLevel(1); // 设置当前边的 level = 1; 之后仅仅优化 level=0 的边
                nBad++; // 记录不好的边
            }
            else // 内点
            {
                pFrame->mvbOutlier[idx]=false; // 对于之前标记为外点的边，此时标记为内点
                e->setLevel(0); // level = 0; 下次进行优化
            }

            if(it==2) // 对于第二次迭代为什么不设置鲁棒核？？ 会不会不稳定？？？使得优化的 pose 有问题？？因为这里的精度实际上受到匹配精度的影响
                e->setRobustKernel(0);
        }
        // 双目 迭代优化
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }
        // 当前优化内点个数小于 10 停止优化，这里的边的个数在最开始的时候就决定了。不会随着外点/内点的变动而变动
        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose); // 更新当前帧的 pose

    return nInitialCorrespondences-nBad; // 经过 4 次优化后，实际有效的边
}
// pbStopFlag = false ,这里对于不好的地图点都剔除了。然后再次更新
// pKF: 当前局部建图线程正在处理的关键帧
// pMap: 地图，在优化后会处理地图点和关键帧，所以我们需要锁保护
//    这个就是把 pKF 关键帧的所有临界关键帧（局部关键帧）。作为 BA 优化的位姿节点。
//    所有这些局部关键帧能够看到的地图点作为 BA 的路标节点。还有那些能够观测到这些路标节点，但不在局部关键帧中。那些节点作为约束。固定不优化。
//    优化完毕后。对边误差较大的地图点和关键帧，去除他们之间的联系。当然可能会把这个地图点从地图中删除。
//    最后恢复地图点坐标和关键帧位姿
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap) // 仅仅在局部建图线程中调用一次
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames; // 局部关键帧集

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames(); // 获得当前关键帧所有临近关键帧组,为后面 BA 优化做准备
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints; // 局部关键帧集中能够看到的所有地图点
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras; // 记录不被优化的关键帧。即不属于局部关键帧集，后面优化时，固定住。然后仅仅作为约束。不参与优化
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId) // 说明当前关键帧不属于局部关键帧组合
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag) // 这个指针不为空。说明外部如果改变这个标志为真。那么每次在优化迭代时，都会进行这个检查，如果为真，那么就会停止迭代
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0; // 局部关键帧和不优化的关键帧中 id 最大号

    // Set Local KeyFrame vertices // 设置局部将要优化的关键帧集
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0); // 最开始建立的第一个关键帧当做世界坐标系。然后固定住即可。不做优化，但是局部关键帧集不一定包含第一个关键帧
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices // 设置仅仅作为约束的局部关键帧，固定住不优化
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono; // 单目参数---边，记录所有的边 ，为后面再次进行优化做准备！
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono; // 记录上面的边对应的关键帧
    vpEdgeKFMono.reserve(nExpectedSize); // 顶点

    vector<MapPoint*> vpMapPointEdgeMono; // 记录上面的边对应的地图点
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo; // 双目参数
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991); // 单目设置的阈值？？？确定的具体算法？
    const float thHuberStereo = sqrt(7.815);
    // 设置路标节点(地图点) 及其对应的边
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit; // 这里并没有检查这个地图点是不是坏点，为什么？？？
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();  // 设置地图点节点
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true); // 将地图点边缘化，加快求解速度
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges 设置边,因为该函数开始已经包含了所有的关键帧。并加入图中
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2); // 这个信息矩阵设置的有问题？？？？？？

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono); // 参数如何确定？？？

                    e->fx = pKFi->fx; // 设置相机内参矩阵
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag) // 存在标志位
        if(*pbStopFlag) // 停止标志为真，则直接退出优化
            return;

    optimizer.initializeOptimization(); // 默认优化 level = 0 的边
    optimizer.optimize(5); // 仅仅优化 5 次，考虑优化时间问题，如果内部退出标志为真，此时会直接结束优化

    bool bDoMore= true; // 表示经过上面优化后再优化一次

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false; // 如果停止标志为真，那么这里不会再进行一次优化

    if(bDoMore)
    {

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i]; // 与边是一一对应的

        if(pMP->isBad()) // 上面没有检查坏点，但是这里却检查了？？？？？
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive()) // 大于一个像素误差,且空间点深度不为正。那么下面就不对他进行优化了
        {
            e->setLevel(1); // level = 1,则下面默认不优化，当然我们可以指定优化哪个 level
        }

        e->setRobustKernel(0); // 该边不设置鲁棒核函数。是因为误差此时很小吗？？？
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++) // 双目
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    // Optimize again without the outliers

    optimizer.initializeOptimization(0); // 对 level = 0 的边进行优化
    optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase; // 将要取消关键帧和地图点之间的联系
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++) // 单目
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP)); // 记录当前地图点不能被 pKFi 关键帧观测到。或者误差很大
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++) // 双目
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    // 对上面不符合要求的地图点和关键帧（构成的边误差大）。去除他们之间的关联
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi); // 取消关键帧和地图点之间的关联
            pMPi->EraseObservation(pKFi); // 取消地图点被当前关键帧观测这个联系
        }
    }

    // Recover optimized data // 在优化后，恢复原始的位姿和地图点位置

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId)); // 说明不需要按照顺序放在优化器中。只要标号唯一即可
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points // 这里面恢复了所有的地图点。当然从上面的去除操作后。很可能一些地图点不在地图中了。此时这里的做法其实不太影响。因为擦除的地图点已经被标记为坏点了。
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth(); // 因为上面优化后。有些边误差很大。则地图点对应的关键帧会发生变化。所以这里需要更新.但是地图点对应的描述子其实也需要进行更新才对？？
            // 这里其实有些地图点可能被置为坏点，那么直接返回不会更新
        pMP->ComputeDistinctiveDescriptors(); // 自己添加！！！！！！
    }
}

//! \note 下面函数对于参考世界坐标系 pose 也进行了优化！！！
//!       这里实际上是一个位姿图优化。只不过对于单目来说是 sim3 所以这里变为 sim3 表达的位姿之间的链接边。顶点是各个位姿
//!         1) 设置地图中所有关键帧节点。（用 sim3 表达）
//!         2) 设置闭环边。就是 pCurKF 的临近关键帧和 pLoopKF 闭环帧临近关键帧之间的链接。（满足 共视点至少 100 个）且 pCurKF 与 pLoopKF 之间的边不连接。
//!         3) 设置 normal eage。下面这些边不能与重复
//!             1) Spanning tree edge // 与父关键帧之间的边
//!             2) Loop edges // 历史链接的闭环边 每次成功检测到一个闭环。就会保存一对一的链接关系
//!             3) Covisibility graph edges
//!         4) 纠正地图中所有关键帧 pose （利用 sim3[SR t] ->SE3[R t/s]）以及所有地图点(利用优化后的 sim3)
//! \brief
//! \param pMap 所有的地图点
//! \param pLoopKF 闭环帧
//! \param pCurKF 闭环线程正在处理的关键帧
//! \param NonCorrectedSim3  pCurKF 的临近关键帧及其 sim3 {scale = 1}，相当于 Tiw
//! \param CorrectedSim3 pCurKF 的临近关键帧(包含自身)及其 sim3iw ，闭环纠正的相似变换
//! \param LoopConnections pCurKF 临近关键帧集（包含 pCurKF）与闭环帧临近关键帧集之间的链接，以 pCurKF 临近关键帧集为基准
//! \param bFixScale 单目 = false
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16); // 这个是做什么的？？？
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1); // 记录地图中现有关键帧对应的 sim3，有些 sim3 是通过 Tiw 直接加了 scale = 1
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1); // 优化后保存的所有关键帧的 sim3wc
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1); // 所有关键帧对应的位姿节点

    const int minFeat = 100; // 对应论文 III SYSTEM OVERVIEW  D Covisibility Graph and Essential Graph 中 Essential Graph 的说明。这个是最小阈值

    // Set KeyFrame vertices 遍历地图中所有关键帧，并设置关键帧节点。对于参考世界坐标帧竟然也优化了!!!为什么？？？
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end()) // 该关键帧存在对应的 sim3（尺度非 1），那么直接设置估计值
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second); // 设置 sim3 顶点估计值
        }
        else // 用 scale = 1，构造一个 sim3 然后设置估计值
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF) // 对于检测到的哪个闭环关键帧固定住不优化
            VSim3->setFixed(true);

        VSim3->setId(nIDi); // 这里说明参考世界关键帧也在优化！！！！！！
        VSim3->setMarginalized(false); // 关键帧节点不边缘化
        VSim3->_fix_scale = bFixScale; // 单目 false，也优化尺度

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;  // 添加 pose---pose 边对应的 id--id 对，pair(min, max)

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges 这里实际上就是将闭环帧的临近关键帧和 pCurrentKF 关键帧的临近关键帧建立   位姿--位姿   边
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second; // 与 pKF 链接的新增的关键帧集
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();
//if ((Siw.translation()(0) == 0) && (Siw.translation()(1) == 0) && (0 == Siw.translation()(2))) 目前还没有验证是不是 LoopConnections 中包含的帧会在局部地图线程中删除，因为数据集闭环很难成功
//    std::cout << " 该闭环链接帧已经被剔除" << std::endl;
        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat) // 对于 pCurKF 和 pLoopKF 链接的边，不添加。剩下的只要满足权重大于 100 就会添加
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3(); // 创建关键帧之间的边
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi))); // 基准边是 pCurKF 的临近关键帧
            e->setMeasurement(Sji); // 测量值的设定是 0--->1 的相似变换 上面计算的就是 Sij

            e->information() = matLambda; // 7x7 的本质矩阵

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj))); // 添加闭环边 pose---pose 边对应的 id--id 对，pair(min, max)
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
        // 实际上这里直接就用 vScw[nIDi].inverse() 就可以
        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse(); // scale = 1
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent(); // 获得父关键帧

        // Spanning tree edge // 与父关键帧之间的边
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi; // 这种测量值计算方式，假定了两个位姿节点之间的相对变换不变（作为测量值）！！！然后优化所有位姿。让总误差最小！！

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges // 历史链接的闭环边 每次成功检测到一个闭环。就会保存一对一的链接关系
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges(); // 获取与当前关键帧链接的闭环边
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId) // 这样保证重复的 loop edges 不被加入。因为 pLKF 和 pKF 如果是闭环对。那么等到两者互换时，这里的作用保证两个关键帧之间的边仅仅加入一次
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat); // 关键帧之间的权重为至少 100
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn)) // 保证上面 Spanning tree edge、Loop edges 包含的边不被加入
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId) // 保证关键帧之间的共视对,仅仅加入一次边
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId)))) // 保证不加入重复的 Set Loop edges 中的边
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1] 为什么可以直接用这种形式替换原来的 T ???
    // 这里更新的 pKFi->SetPose(Tiw) 作为新的 pose 其实是有问题的！因为用这个 Tiw 乘以一个 Pw 变换到相机坐标系 Pc'。
    // 与下面 correctedSwr.map(Srw.map(eigP3Dw)); 这里的 Srw.map(eigP3DW) 是相机坐标 Pc。此时 Pc = s*Pc'。相差一个尺度 s。
    // 这样虽然在匹配的时候没有多大影响。因为投影到图像坐标系是没问题的。但是在匹配的过程中，有的还会检查深度值是不是在最大值和最小值之间。
    // 那么此时这里由于相差尺度其实匹配时会有一定的影响。所以如何更改？或者如何验证自己的想法是对的？？？
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 将原始点云坐标，用优化后的 sim3 更新
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId) // 表示当前的地图点已经被纠正过了。获取在处理哪个关键帧时纠正的
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame(); // 当前地图点在哪个关键帧三角化的.如果这个关键被剔除了。但是这里仍然是有效的。因为关键帧没有真正的删除。所以这里其实是有问题的
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr]; // 保留的是未优化前的变换。scale 可能为 1
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw)); // 先用原始位姿变换点云，然后用更新的 sim3 变换回到世界

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth(); // 位置坐标变化，就需要调用这个
    }
}

//! \brief
//! \param pKF1
//! \param pKF2
//! \param vpMatches1 记录 12 的匹配点对。1 对应 pKF1 上的关键点，2 对应 pKF2 上的地图点. 其 size() = 基准的闭环关键帧也就是 pKF1 的关键点个数。
//!                   随着更新，剔除不好的边，此时这个匹配点对也会相应的更新
//! \param g2oS12 记录的是 sim3_12,在优化后，从新更新了该值。
//! \param th2 10
//! \param bFixScale
//! \return 优化过后，实际匹配的点对数
//! \note 这里的优化方式，仅仅是优化 s12，即两个相机坐标系之间的相似变换。所以需要把世界点转换到相机坐标系下，然后作为边的节点
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale; // 设置是否固定尺度
    vSim3->setEstimate(g2oS12); // 包含 R t S
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2); // cx
    vSim3->_principle_point1[1] = K1.at<float>(1,2); // cy
    vSim3->_focal_length1[0] = K1.at<float>(0,0); // fx
    vSim3->_focal_length1[1] = K1.at<float>(1,1); // fy
    vSim3->_principle_point2[0] = K2.at<float>(0,2); // 针对双目而言，单目的话一个就够
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size(); // 有效的匹配点对数(以关键帧 pKF1 为基准)
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12; // e12 边  这两个边是成对的存取的
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21; /// e21 边
    vector<size_t> vnIndexEdge; // 关键帧 1 对应的添加了边的关键点索引

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2); // 根号(10),鲁棒核阈值

    int nCorrespondences = 0; // 优化器中加入的有效的地图点对
    // 添加 sim3 节点，以及地图节点
    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i]) // 表示 pKF1 对应的关键点没有对应的匹配点
            continue;
        // 获取对应的匹配地图点
        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2); // 地图点 2 在关键帧 2 中的对应关键点索引

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c)); // 转换到相机坐标系下的 3d 点坐标
                vPoint1->setId(id1);
                vPoint1->setFixed(true); // 地图点不优化
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;
        // 接下来设置两条边；参考论文中 Appendix: Relative Sim(3) Optimization ，因为边对应的 e1 e2
        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ(); // 利用两个相机坐标系之间的相似变换 s12 s21，互相计算投影误差
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2))); // 先是地图点 2 经过 S12 投影到关键帧 1 图像上
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); //
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1); //

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ(); // 注意这里是 EdgeInverseSim3ProjectXYZ * 自动表示了 S21

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1))); // 地图点 1
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0; // 记录优化后，对于误差较大的边作为坏点。也是不好的匹配点对个数
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2) // chi2 表示误差平方。阈值为 10 ，实际上相当于 3 个左右的像素误差
        { // 误差太大，非内点
            size_t idx = vnIndexEdge[i]; // 取出对应的关键帧 1 对应的关键点序号
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12); // 在图上移除与边链接的地图顶点（sim3顶点不会移除），为什么不设置边的 level = 0???与 LocalBAOptimization() 一致？？
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    // 有坏点，在次优化次数更高
    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10) // 剩下的有效匹配的点对为 10 个，那么直接就表示这次计算的 S12 相似变换无效
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL); // 误差较大的匹配，再次擦除匹配
        }
        else
            nIn++; // 记录有效匹配点对个数
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate(); // 返回优化的 sim3

    return nIn;
}


} //namespace ORB_SLAM
