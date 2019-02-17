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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>
#include <chrono> // 统计运行时间


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3; // 连续 3 次检测到闭环
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;
    // 整个部分参考：VII LOOP CLOSING
    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop()) // 检测到闭环，得到了初始闭环帧存储在 mvpEnoughConsistentCandidates 变量中
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
//                   std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//                   std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
//                   std::cout << "执行完一次闭环检测相关操作需要时间： " << time_used.count() << " s"<< std::endl;
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }

        }       
        // 有重置请求时，会自动清理资源
        ResetIfRequested();

        if(CheckFinish()) // 在系统结束处理数据集时，会自动请求停止完成标志，然后这里检测到就会退出大的 while() 循环，直接退出闭环线程
            break;

        usleep(5000);
    }

    SetFinish();
}
// 在局部建图线程中调用。在闭环关键帧队列中插入关键帧
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

//! \see VII LOOP CLOSING
//!     A Loop Candidates Detection
//! 检测是否存在闭环。如果最后真正得到了符合一致性的闭环帧就会存放在 mvpEnoughConsistentCandidates 变量里。此时返回 true
//! 具体操作：
//!    1）获得当前闭环线程正在处理的关键帧,和其临近关键帧之间的最低得分 minScore
//!    2) 通过最低得分，在关键帧数据库中找到潜在闭环关键帧(不属于其临近关键帧集)
//!    3）检测一致性群和潜在闭环关键帧对应的群之间的一致性得分。获得初始闭环关键帧。{这个过程比较复杂，可以根据下面图示及其说明进行理解}
//!    4）如果 mvpEnoughConsistentCandidates 包含了满足一致性得分的关键帧。那么说明找到了初始闭环帧。返回 true
bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase(); // 局部建图线程中无法擦除该关键帧
        std::cout << "mlpLoopKeyFrameQueue.size() " << mlpLoopKeyFrameQueue.size() << std::endl;
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    // 地图包含少于 10 个关键帧，或者经过上次成功闭环后不到 10 个关键帧，没有必要进行闭环检测。
    // 因为少于 10 个关键帧之内有闭环的概率很大。这种频繁的闭环会占用 cpu 。导致追踪线程无法跟踪。
    // 还有 10 个关键帧之内累计的误差其实不大。闭环起不到太大的作用，没必要进行闭环
    // （这里没有考虑有些关键帧之前已经被剔除了，仅仅是说，历史上超过 10 个关键帧。）
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF); // 添加到关键帧数据库。为重定位以及闭环检测做准备
        mpCurrentKF->SetErase(); // 允许当前关键帧被当做冗余关键帧被剔除。
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    // 获得当前关键帧与其临近关键帧之间的最低 Bow 得分 minScore
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec); // 计算当前正在处理的关键帧以及共视图中临近关键帧之间的评分。

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore); // 以当前正在处理的关键帧为基准。选择潜在闭环帧集

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty()) // 当前关键帧没有检测到备选的闭环帧集
    {
        mpKeyFrameDB->add(mpCurrentKF); // 将当前关键帧添加到数据库（用在下次闭环和重定位）
        mvConsistentGroups.clear(); // 因为后面检测的是连续的一致关键帧群。所以单次没有检测到，那么就直接把这个变量清零。
        mpCurrentKF->SetErase();
        return false;
    }

    // 下面用图示来表示下面算法的过程。
    //            当前得到的潜在闭环关键帧：       CKF1        CKF2        CKF3        CKF4         ...
    //  当前潜在闭环关键帧一致性群：               |set<KF*>|  |set<KF*>|  |set<KF*>|  |set<KF*>|    ...
    //
    //  上次得到的一致性群(mvConsistentGroups)：  |SET<kF*>|  |set<KF*>|  |set<KF*>|  |set<KF*>|    ...
    //            上次得到的潜在闭环关键帧:        LKF1         LKF2       LKF3        LKF4          ...
    //            当前子群是否产生一致性:          false        false      false       false         ...
    // 说明一下，上面符号是上下对应的。CKF1 对应的 |set<KF*>| 表示一个群中包含的其中一个关键帧集（子群）。这个关键帧集包含 CKF1 的临近关键帧和 CKF1 本身。
    // 下面的过程就是取出一个 CKF1，计算得到对应的关键帧集（子群），然后和 LKF1 LKF2 .... 对应的关键帧集（子群）依次进行比较，查看有没有共同的关键帧。
    // 如果有共同的关键帧，说明两个关键帧集（子群）之间具有一致性。假设 CKF1 和 LKF2 代表的子群检测到具有一致性。
    // 那么会把 LKF2 对应的一致性得分 + 1 赋值到 CKF1 子群对应的一致性得分。然后当前一致性群变量 vCurrentConsistentGroups 会把 CKF1 关键帧集和一致性得分记录下来
    // 之后把 LKF2 标记为一致群(子群)，也就是对应的 false-->true，表示下次即使与其他关键帧比如 CKF2 产生一致性了。但是上面变量 vCurrentConsistentGroups
    // 也不会把 CKF2 关键帧集和一致性得分记录下来（该群已经被传递过了就不会再次记录了）。如果 CKF1 与 LKF2 在有一致性前提下，且 CKF1 当前更新后的一致性得分满足 >= 3，
    // 那么 CKF1 关键帧认为是初始闭环帧，这个帧是以后会在 ComputeSim3() 函数中判断这个帧是不是真正的闭环帧。那么此时 mvpEnoughConsistentCandidates 会添加 CKF1 。
    // 该变量内部只要有一个添加了。然后就表示检测到了初始闭环，然后该函数最后就能返回真了。
    // 为了避免 CKF1 插入多次到 mvpEnoughConsistentCandidates（可能会与 LKF3 ..也产生一致性达到 3 次）。会置位相应的标志使其不在被插入更新。
    // 如果以 CKF1 为基准，遍历了所有 LKFi，都没有检测到一致性/没有共同的关键帧，那么就会把 CKF1 关键帧集和对应一致性得分为 0，然后加入到 mvCurrentConsistentGroups 里。
    // 供下次检测一致性时对比使用。
    // 上面是一次遍历，然后在以 CKF2 为基准。再次从 LKFi 进行遍历，然后同样的操作如上。
    // 最后会：mvConsistentGroups = mnCurrentConsistentGroups; 这样上面图中的 「上次得到的群」就会进行更新。以备下次使用

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it （这里的阈值是 3 ）
    mvpEnoughConsistentCandidates.clear(); // 为下次检测闭环做准备

    vector<ConsistentGroup> vCurrentConsistentGroups; // 当前一致群，每次执行下面的循环可能会加入新的
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false); // 对应标记上次的一致群。每个元素对应上次其中一个一致群。 true: 表示这个群已经被标记过

    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false; // true 一致性: 标记上次得到的其中一个群和当前得到的其中一个潜在群有至少一个共同的关键帧。
                                      // 简单来说，就是标记群和群之间是否至少有一个共同关键帧
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit)) // 当前群和前一个群至少有一个相同的关键帧
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent) // 群和群之间具有一致性：两个群至少有一个共同的关键帧
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second; // 获取一致性得分
                int nCurrentConsistency = nPreviousConsistency + 1; // 一致性得分 + 1
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg); // 将当前的群更新一致性得分后，增加到下一次一致群中。以备下次检测。
                                                            // 但是这里会出现加入相同的重复的元素，可能得分不一样？这样为什么可以？：可能这样也表示一种连续性，
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent) // 对于符合一致性得分的群，说明群对应的关键帧是符合初始闭环帧。表示成功检测到闭环。
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF); // 说明当前得到的备选关键帧 pCandidateKF 满足 3 次连续一致性条件。
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        // 如果当前群和上次所有群检测到不一致。那么此时这个群加入到当前一致性群里面，跟下一次进入该函数后得到的潜在群之间进行检测。
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0); // pair<关键帧集,一致性得分>
            vCurrentConsistentGroups.push_back(cg); // 增加一致性群的检测个数
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups; //更新一致性群。为下次检测一致性做准备

    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty()) // 检测潜在闭环帧失败，允许局部建图线程擦除当前正在处理的闭环关键帧擦除
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else  // 表示当前处理得到的潜在闭环关键帧中，有满足 3 次一致性条件。那么表示当前这次确实检测到了闭环。那么就可以进行接下来的步骤。。。
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

//! \brief 论文引用了两个其他论文。1、单目 7 自由度 2、找相似变换 Sil。具体步骤
//! \details
//!     1) 对于在 DetectLoop() 函数中计算出来的所有潜在闭环关键帧。按照 Bow 方法，找到与当前处理的关键帧 mpCurrentKF 匹配点对。
//!       只有匹配点对大于等于 20 个，才能建立 sim3 求解器 Sim3Solver。{这里用 BoW 方法的原因是，此时关键帧之间的变换关系未知。只能通过词袋的方式找潜在配对点}
//!     2) 对于有求解器的所有潜在闭环关键帧进行 RANSAC 迭代。只有满足条件(下面会说明)的潜在闭环帧才能利用 sim3 方式寻找更多的匹配。
//!       然后进行 OptimizeSim3（）优化。得到比较好的 sim3 变换。优化完成后的内点个数大于 20 ，保留该帧为 mpMatchedKF， 然后退出整个循环：遍历潜在关键帧集。
//!          条件（在 iterate() 函数完成的）： 找到与当前 mpCurrentKF 的相似变换 sim3。然后检查内点集。内点集大于某个值（足够多）才算是符合条件
//!     3) 获取步骤 2 得到的 mpMatchedKF 关键帧的所有临近关键帧对应的地图点（不能重复），然后与 mpCurrentKF 关键帧寻找更多的匹配。
//!        只有匹配的点对足够大于 40 才算是成功检测到了闭环，然后返回 true。此时 mpMatchedKF 会用来纠正闭环误差，即在 CorrectLoop() 函数用到。
//! \return 在 DetectLoop() 函数中检测的潜在闭环关键帧中，如果成功找到了一个关键帧，且通过一系列的条件检验。那么才会表示真正检查出来一个关键帧。
//!         然后直接返回 true。
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size(); // 获得一致性闭环帧个数（在 DetectLoop() 中得到的！）

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded; // 标记当前位置的一致性闭环帧将要被丢弃
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches 有足够的内点，可以进行 sim3 计算的一致性闭环帧个数。统计完之后，后面会进行 iterate() 验证

    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase(); // 禁止在局部建图线程擦除冗余关键帧。只有后面找到了真正的闭环关键帧，
                            // 最后那些不是真正的闭环关键帧/都不是闭环帧，都会 SetErase()，表示该关键帧可以擦除

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]); // 在当前基准关键帧和一致性闭环关键帧中找到匹配点对

        if(nmatches<20)  // 这个条件是如何确定的？？？
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale); // sim3 求解
            pSolver->SetRansacParameters(0.99,20,300); // 最小内点集为 20 ，因为上面的 if(nmatches<20) 已经表示这里至少有 20 个内点了。
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i]) // 不符合要求
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers; // mN1 = size  vbInliers[i] = value; 与基准关键帧关键点一一对应。 value = true; 表示是对应匹配是内点
            int nInliers;
            bool bNoMore; // ture:当前对应的闭环帧不好，应该被丢弃

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers); // 返回是满足条件的变换矩阵 T12 (带有尺度 s)

            // If Ransac reachs max. iterations discard keyframe
            // 不好的关键帧，需要丢弃
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--; // 潜在闭环帧递减
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            // (就是通过上面 iterate() 计算出来的相似变换，然后通过再次投影匹配，找到两帧之间更多的相关性（匹配点对），之后进行 Sim3 优化)
            if(!Scm.empty())
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL)); // size: 是以正在处理的基准闭环关键帧为基准。
                                                                                                                 // 记录 12 的匹配点对。1对应关键点，2对应地图点
                                                                                                                 // 在下面优化后，会改变.
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++) // 遍历所有基准关键帧关键点
                {
                    if(vbInliers[j]) // 判断基准关键帧中当前关键点是否是内点。只有是内点的，才会设定 true。否则其他都是 false
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j]; // 找到有效内点对应的 3d 点，之后进行 3d-2d 优化
                }

                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5); // 再次找到新的匹配点(基准关键帧和检测的潜在闭环帧)对放在 vpMapPointMatches 中

                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s); // 传入下面函数后，会自动更新。 由 pKF2 -->CurrentKF 的变换
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale); // pKF--->mpCurrentKF 的变换

                // If optimization is succesful stop ransacs and continue，说明找到了一个很好的闭环帧,然后直接保存纠正闭环需要的值后，直接退出循环
                if(nInliers>=20) // 阈值选取问题？？？？？
                {
                    bMatch = true; // 不在循环
                    mpMatchedKF = pKF; // 当前潜在闭环帧是真正的闭环
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0); // 这个是 世界 ---> pKF2 的变换(这个是刚体变换)
                    mg2oScw = gScm * gSmw; // 相似变换乘法
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches; // 实际有效的匹配点对 关键帧 1 关键点i ---> pKF2地图点
                    break;
                }
            }
        }
    }

    if(!bMatch) // 表示上面退出循环是没有找到有效的闭环帧。这次检测闭环失败。然后允许局部建图线程擦除潜在闭环关键帧和当前正在处理的关键帧
    {
        for(int i=0; i<nInitialCandidates; i++) // 对所有潜在闭环帧都允许局部建图线程擦除
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    // 获取 mpMatchedKF 的临近关键帧的所有地图点
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames(); // 注意这里是 Loop Keyframe 的 临近关键帧集
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3 利用刚刚计算出来的相似变换计算更多的匹配点对。这个匹配点对更严格些
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0; // 记录最终匹配的点对数
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40) // 这个条件如何确定？？？ 已知，越大越好
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF) // 对于不是检测的真正闭环帧，那么直接允许局部建图线程擦除冗余的关键帧
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else // 此时该检测出来的闭环帧不符合要求
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

//! \note
//! \brief
//!     1) 停止闭环线程，防止添加新的关键帧和地图点
//!     2) 获取闭环线程正在处理的关键帧(mpCurrentKF)的临近关键帧集(包含自身)。计算他们纠正后的 sim3 通过 mpCurrentKF 的 sim3。
//!        将他们对应的地图点都通过对应的 sim3 更新一下位置坐标。然后将他们关键帧本身 pose 设置为 [R t/s]（这个是由 sim3 [sR t] 得到的）
//!        用闭环检测到的关键帧 mpMatched 的地图点代替 mpCurrentKF 对应的地图点（因为闭环后面误差会很大，所以认为最开始的地图点更准确）
//!     3) 将 Loop Keyframe(mpMatched) 临近帧的所有地图点，用上面纠正过的 sim3 投影到 mpCurrentKF 及其临近关键帧上。寻找对应的匹配关系，然后替换其不好的地图点。
//!      （这样 mpMatched 临近关键帧和 mpCurrentKF 的临近关键帧都会共同的地图点了）
//!     4) 保存 mpCurrentKF 的临近关键帧和 mpMatched 临近关键帧之间，新添加的链接关系（以 mpCurrentKF 为基准）
//!     5) 进行本质图优化
//!     6) 开辟新的线程，进行全局 BA 优化
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++; // 仅仅用来表示与上次进入 RunGlobalBundleAdjustment() 函数时，不是一个值，
                       // 那么就是说在经过全局 BA 优化期间，又再次检测到了闭环。那么停止 BA 后，该函数 RunGlobalBundleAdjustment() 会直接返回，
                       // 结束全局 BA 优化线程 mpThreadGBA

        if(mpThreadGBA) // 已经存在线程资源
        {
            mpThreadGBA->detach(); // 分离线程？？？这里会有一个问题，就是如果此时闭环线程先于分离线程结束，那么分离线程会使用释放掉的资源，那么此时会出线程利用了释放的变量错误。
                                   // 解决： 实际上停止了全局 BA 优化后，很快就会结束 mpThreadGBA 线程。所以此时不会发生这种错误
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped()) // 直到局部建图线程停止时，才会继续进行下面操作
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated // 实际上在局部建图线程中可能还没有更新，因为局部建图线程可能已经处理了很多关键帧插入到了闭环线程。但是闭环线程没有处理完。
    mpCurrentKF->UpdateConnections(); // 这个更新应该在处理闭环之前才对？？？并且在这里也需要再次更新

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames(); // 获得最新的临近关键帧
    mvpCurrentConnectedKFs.push_back(mpCurrentKF); // 自己也加入其中！

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3; // CorrectedSim3 当前关键帧 mpCurrentKF 的临近关键帧(包含本身)到世界的相似变换
                                                     // NonCorrectedSim3 当期关键帧的临近关键帧到世界的普通刚体变换。只不过用了 sim3 形式。尺度为 1
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        // 利用 mpCurrentKF 与世界的相似变换结果计算其临近关键帧与世界的相似变换，称为 g2oCorrectedSiw
        // 为什么仅仅要临近的关键帧和世界的相似变换？？？按理说，闭环出现的话纠正结果可以从检测的闭环帧到当前闭环线程正在处理的关键帧。之间的所有关键帧都要求纠正误差
        // 是因为共视图在构建时本身已经尽可能包含了所有关键帧？？？还是说为了减少计算量？？？
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            if(pKFi!=mpCurrentKF)
            {
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0); // 尺度为 1 的相似变换,认为连着的两个帧之间的尺度一致，所以 scale = 1
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw; // 获取世界到 i 关键帧的相似变换
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw; // 到世界的相似变换
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints observed by current keyframe and neighbors, so that they align with the other side of the loop
        // 把 mpCurrentKF 的临近关键帧对应的所有地图点的世界坐标用 sim3 更新一下。具体步骤（把地图点变换到相机坐标系，然后通过对应相机的 Swi 把点变换到世界坐标系下）
        // 用 sim3 更新临近关键帧对应的 pose。就是把 sim3 = [SR t] ---> [R t/s] 等效的刚体变换。
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi]; // 实际上就是 Tiw ()

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++) // pKFi 的所有地图点
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw); // 经过相似变换的世界坐标系下的坐标（上面通过 Tcw 先投影到相机坐标系，然后通过 Swi 投影到世界坐标系）
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId; // 记录该地图点已经纠正过，下次不在纠正
                pMPi->mnCorrectedReference = pKFi->mnId; // 表示是该地图点是在处理 pKFi 帧时纠正的
                pMPi->UpdateNormalAndDepth(); // 为什么需要在这里更新？？？可能是因为后面优化的时候需要这里的函数更新的量。为了减少计算量，这里只把闭环优化的点进行更新
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt); // 这里仅仅是一个等效的变换。使用这个 pose 旋转一个 3d 点，并不是相机坐标系下的真实坐标。
                                                                  // 与经过 sim3 变换到相机坐标系的 3d 点坐标还相差一个尺度因子。不过并不影响最后投影到图像坐标系
                                                                  // 下的坐标

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections(); // 这里与上面的 pMPi->UpdateNormalAndDepth() 函数一样，因为这些关键帧都需要优化，此时仅仅更新需要优化的关键帧。
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        // 用匹配好的地图点代替或新增到当前关键帧 mpCurrent。
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i]; // 真正的闭环关键帧上的地图点
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP); // 为什么要直接代替？不需要判断哪个地图点对应的关键帧多吗？: 是因为 T 累计误差，然后直接认为靠后面关键帧，地图点的匹配关系假的概率比较大。
                else
                {   // 因为在 ComputeSim3() 函数中已经找到了新的匹配关系。对于 mpCurrentKF 的关键点之间没有对应地图点时，此时就要添加与真正闭环帧匹配的地图点 pLoopMP
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                    // 这里少调用了 ->UpdateNormalAndDepth() 函数？？？
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    // 将 Loop Keyframe 临近关键帧（包含自己本身）的所有地图点全部投影到 mpCurrentKF 及其临近关键帧集上。找到匹配的地图点并替换。
    // 实际上，上面的 for 循环哪里其实已经用到了 Loop Keyframe 上的地图点了。在下面函数内部执行 Fuse() 函数时用到的 mvpLoopMapPoints
    // 又再次用到了 Loop Keyframe 上的所有地图点。所以这里其实有些重复！还有一个结果就是可能上面的 for 循环替换完毕后，紧接着在下面函数中再次替换了。
    // 那么在 Replace() 函数内部的增加 mnFound 次数，以及 mnVisible 次数哪里就会出现问题！！！这些都是潜在的 bug!!
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    // 经过上面的融合，其实在共视图中增加了很多新的边。或者替换了很多老的边。下面找到新增加的与闭环关键帧的临近帧链接的边。然后进行 EssentialGraph 优化。
    // 更新后的链接帧集 - 更新前的链接帧集 - mpCurrentKF 的临近关键帧集 = 新增加的闭环两侧的链接关系
    map<KeyFrame*, set<KeyFrame*> > LoopConnections; // 保存的是当前关键帧(mpCurrentKF)的临近关键帧(包含自身)对应的新增的临近关键帧集(与闭环帧的临近关键帧有链接)
                                                    // （不包含 mpCurrentKF 的临近关键帧）

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames(); // 获得更新前的二级临近关键帧

        // Update connections. Detect new links.
        pKFi->UpdateConnections(); // 更新共视图(增加新的链接关系)
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        // 仅仅保留新加入的链接关系并且不能包含 mpCurrentKF 的临近关键帧。这样才能保证新增加的边就是闭环两侧的关键帧之间的形成的边。
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph // 在优化的时候局部建图不会插入新的关键帧
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge // 增加闭环边
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId); // 此时该线程立刻执行

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release(); // 清理局部建图线程新增加的关键帧，然后唤醒局部建图从新可以插入新关键帧了

    mLastLoopKFid = mpCurrentKF->mnId;   
}

//! \brief 将 Loop Keyframe 临近帧的所有地图点，投影到 CorrectedPosesMap(当前闭环线程正在处理的关键帧及其临近关键帧) 包含的关键帧上。
//！       寻找对应的匹配关系，然后替换其不好的地图点。
//! \param CorrectedPosesMap
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints); // vpReplacePoints[i] = MP ; 表示 mvpLoopMapPoints 地图点 i 将要替换 pKF 对应的地图点 MP

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate); // 下面需要改变地图点，因此需要加锁
        // 替换 pKF 对应的误差较大的地图点
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]); // replace
            }
        }
    }
}

//! \brief 请求重置闭环线程。
//! \purpose 清理闭环线程的闭环关键帧队列 mlpLoopKeyFrameQueue。mLastLoopKFid(记录上次成功完成闭环的关键帧 id)
void LoopClosing::RequestReset()
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
        usleep(5000);
    }
}

// 重置闭环线程资源
void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        // 这些资源包含的指针内存，已经包含在 Map 类里面了。通过 mpMap->clear(); 已经释放了。
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false; // 表示已经重置完毕，释放标志
    }
}

//! \note 这里的函数在执行的时候是在一个新的线程。所以在闭环线程执行完 CorrectLoop() 后会让局部建图线程启动，因此会插入新的关键帧。
//!         1）全局 BA 优化
//!         2) 更新纠正局部建图线程在全局 BA 优化后新增加的关键帧位姿和地图点位置
//! \param nLoopKF 闭环线程正在处理的关键帧 id
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF) // 这里应该是 pCurrKF->id
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx; // 初始为 0
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false); // 经过一次全体的 BA 优化,记住这里的参考世界帧 pose，已经在本质图优化中进行优化了。在这里没有进行优化！

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx) // 判断是否又再次检测到了闭环，再次检测到了闭环直接返回，不需要更新地图点和关键帧了
            return;

        if(!mbStopGBA) // 在跑下面纠正过程时，不能让局部建图添加新的关键帧了，否则就白纠正了
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop(); // 停止局部建图线程，防止新的关键帧插入
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());
// std::cout << "lpKFtoCheck.size" << lpKFtoCheck.size() << std::endl;
            // 纠正局部建图线程在上面跑全局 BA 函数期间新增加了关键帧。通过 spanning tree 的方式依次寻找孩子关键帧。把实际上把全部关键帧都遍历了一遍
            // 这里为什么没有用直接用 pMap->GetAllKeyFrames(); 是因为，纠正过程需要一个基准关键帧，对于一个新增加的关键帧。最好的基准当然是与他共视地图点最多的关键帧（父关键帧）。
            // 然后遍历方式应该是从已经修正过的 ---> 没有纠正过。从顶向下依次遍历才可以。如果用 pMap->GetAllKeyFrames() 方式，自己找父关键帧。那么很可能父关键帧还没有纠正过。
            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds(); // 孩子关键帧集
                cv::Mat Twc = pKF->GetPoseInverse(); // Twc ，经过本质图优化后的 Twc。
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF) // 说明孩子关键帧没有参与全局 BA 优化，判断该孩子关键帧是局部建图新增加的关键帧！！
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc; // 由 pKF 关键帧到其孩子关键帧的一个变换.这里假定这个变换固定不变。但是实际上这里应该是变化的！如何解决？？？
                                                                 // 然后用新的 pKF 经过 BA 优化的 pose 来求出孩子关键帧对应的新的 pose
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA; 求出局部建图新增加的关键帧假定经过 BA 优化后对应 pose
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild); // 加入后，遍历其孩子关键帧，直到找到所有的局部建图新增加的关键帧为止。
                }

                pKF->mTcwBefGBA = pKF->GetPose(); // 保存未经过全局 BA 的 Tcw,后面纠正地图点时会用到。
                pKF->SetPose(pKF->mTcwGBA); // 此时孩子关键帧都遍历完了，那么这里就可以放心的更新自己的 Tcw 。因为后面不会在用原来没更新前的 Tcw 了。
                lpKFtoCheck.pop_front(); // 这个孩子关键帧遍历完了，就没有用了，可以直接去除！
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF) // 表示该地图点参与了优化，那么此时更新 地图点 mWorldPos 位置变量
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else // 是在全局 BA 优化时，局部建图线程又新增加的地图点！
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF) // 说明参考关键帧没有经过全局 BA 优化。而且在上面新的关键帧纠正时，也没有纠正过。这种情况很少。不知道有没有？？？
                        continue;

                    // Map to non-corrected camera // 未纠正前的 pose
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse(); // 这个是已经纠正过的/经过全部 BA 优化过的 pos
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange(); // 记录地图有一个大的改变{新增关键帧和地图点被纠正了}

            mpLocalMapper->Release(); // 清理局部建图线程新增加的关键帧，然后唤醒局部建图从新可以插入新关键帧了

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

// 检查是否有外部要求关闭闭环线程
bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

// 设置当前闭环线程已经结束完毕
void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
