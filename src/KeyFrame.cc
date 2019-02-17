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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)//,mRelocScore(0.0)// 自己添加的 在重定位模式时，没有初始值目前不要也行，这个是随时变动的！
{
    mnId=nNextId++;
    // 复制网格（内部包含的是网格内部包含关键点序号）
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}
// 计算 Bow 向量
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        // 将描述子转化为词袋向量和特征向量
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1); // 针对双目?? center??
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}
// 获取当前关键帧光心在世界坐标系下的坐标
cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);    // 这种锁表示，获取同一个资源时，不同线程之间的锁定。
    return Tcw.rowRange(0,3).col(3).clone();    // 选取 4x4 矩阵，的平移部分：最后一列的前三行
}
// 增加当前关键帧和指定关键帧链接关系。并增加两个关键帧共视地图点个数！
//    pKF: 建立联系的关键帧
//    weight: 权重，两关键帧之间共视地图点个数
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF)) // 不存在链接关系，则添加关键帧之间联系
            mConnectedKeyFrameWeights[pKF]=weight; // 增加关键帧之间联系
        else if(mConnectedKeyFrameWeights[pKF]!=weight) // 存在链接关系，但是又有新的地图点(后期又新增加的地图点，恰好当前两帧都能观测)被共同观测。需要修改链接权重
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}
// 将共视关系按照权重，由小到大进行排序
//    更新 mvpOrderedConnectedKeyFrames  mvOrderedWeights
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs; // 方便下面按照第一个 int 类型元素进行排序
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end()); // 按照共视点个数(权重)从 小--->大 排序，下面在存储的时候用了相反的顺序存储的，所以造成下面 mvOrderedWeights 变量是有大--->小排序的
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++) // 分离排好序的关键帧和权重，方便下面插入
    {
        lKFs.push_front(vPairs[i].second); // 在前面插入，所以最后的结果是，大的在前，小的在后
        lWs.push_front(vPairs[i].first);
    }
    // 更新共视关系 大--->小
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}
// 把与当前关键帧链接的所有关键帧返回。用 set<> 保存
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}
// 获得当前关键帧所有临近关键帧组
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}
// 获得与当前关键帧有着共视关系的其他关键帧。指定个数。且里面按照共视强度 大->小 顺序递减排列
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else // 选取指定个数的关键帧
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}
// 取出与当前关键帧共视地图点的备选关键帧。条件：共视地图点个数大于 > w
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();
    // 按照权重 w 值。找到 > w 的最后一个位置。默认排序是升序。但是对于 mvOrderedWeithts 是按照降序排列的。所以这里要加上一个 return a>b 的比较函数！
    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}
// 返回 pKF 关键帧，在共视图中与当前关键帧的共视强度
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}
// 当前关键帧关键点对应的 MapPoint 地图点
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}
// 清理当前关键帧对应的地图点,实际上就是相应位置置位 NULL
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}
// 取消当前关键帧和 pMP 地图点之间的联系
void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}
// 获得当前关键帧关键点对应的所有有效的地图点
set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}
// 获取当前关键帧自己的地图点{与自己提取的关键点对应的}。符合该条件：被关键帧观测次数 >= minObs 的有多少个
//  minObs > 0: 被观测次数 > minObs 的个数才算是被观测过
//  minOBS <=0: 这种情况会发生吗？？？自己的地图点肯定是被自己观测过的？
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs) // 只有被观测过的地图点才算是正在跟踪的地图点
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}
// 获取当前关键帧所有关键点对应的有效地图点{因为有些关键点没有匹配上，有些关键点匹配上了，但是三角化没有成功}
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}
// 获得指定地图点
MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}
// 注意只要 mvpMapPoints 有变化且对应的地图点已经把该关键帧加入到 mObservations 中，那么就必须要调用这个来更新一下共视图
// 记录与当前关键帧有共视关系的关键帧，以及共视强度(存在多少个共同观测的地图点),方法：根据 mvpMapPoints 地图点来做
// 把当前关键帧包含的所有地图点拿来。然后地图点包含了一些关键帧。这样即可 记录与当前关键帧有共同观测的关键帧及其共同观测次数。
// 这里把共同观测的地图点个数作为链接两个关键帧边的权重
// 实际上在维护一个 Covisibility Graph 参考论文： III D
void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter; // 记录与当前关键帧有共视关系的关键帧，以及共视强度(存在多少个共同观测的地图点)

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    // 记录与当前关键帧存在共同地图点的关键帧，及其共同观测的次数。就是有多少个地图点，两个关键帧都能共同看到。
    // 统计后根据 III SYSTEM OVERVIEW 中的 D 部分维护共视图 Covisibility Graph
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP) // 说明该关键点没有对应的 MapPoint
            continue;

        if(pMP->isBad()) // 当前点被剔除，则不需要操作该点
            continue;
        // 获得当前点被那些关键帧观测过
        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId) // 当前关键帧不算
                continue;
            KFcounter[mit->first]++; // 记录共同观测到该地图点的关键帧，及其次数
        }
    }

    // This should not happen 因为在三角化地图点的时候，至少是有两个关键帧能够看到相同的关键点// 利用 CHECK() 代替
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15; // 对应论文中 III SYSTEM OVERVIEW 中 D. Covisibility Graph and Essential Graph 与当前关键帧有共同观测地图点个数至少为 15 个

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax) // 统计最大的与当前帧共同观测次数，以及对应的关键帧
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th) // 满足条件，那么按照论文中 III -D 部分说法，需要在两个帧之间构成边，且权重是共同观测的地图点个数
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second); // 增加链接关系 并对 mit->first 关键帧共视图按照权重由大--->小进行排序
        }
    }

    if(vPairs.empty()) // 说明没有满足 > 15 个共视地图点，那么选择其中共视最多的关键帧进行插入！
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax); // 增加链接关系(双向记录)
    }

    sort(vPairs.begin(),vPairs.end()); // 小 ---> 大 排列，按照 pair 第一个元素
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second); // 按照 大 --> 小
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end()); // 由大 --> 小排列
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0) // 这里实际上 mnId != 0,这个值最小为 1
        {
            mpParent = mvpOrderedConnectedKeyFrames.front(); // 取出最强的共视关系(共同观测的地图点最多)，对应的关键帧
            mpParent->AddChild(this); // 把自己当做父关键帧的孩子
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}
// 当前关键帧改变父亲关键帧为 pKF，然后 pKF 关键帧将当前关键帧添加到自己的孩子关键帧
void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}
// 子关键帧没有按照与当前关键帧共视强度从大到小排列！
set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}
// 每个关键帧仅仅有一个父节点
KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

// 加入与当前关键帧闭上环的关键帧 pKF
void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true; // 表示对于成功闭上环的关键帧，不能擦除。一致保留
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}
//! \brief 设置当前关键帧不能被擦除,如果当前关键帧是冗余的，此时局部建图线程也无法剔除该关键帧。
//!        调用该函数有两种情况：
//!         1）LoopClosing::DetectLoop() 闭环线程正在处理关键帧 mnCurrentKF 时，不能让局部建图线程删除闭环线程当前正在处理的关键帧
//!         2 LoopClosing::ComputeSim3() 在检测初始备选闭环关键帧中有没有真正的闭环关键帧，此时在检测时，都会不让其被擦除
void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}
//! \brief 让当前关键帧可以被擦除。满足下面条件时，仍然不会被擦除：
//!    该关键帧被成功检测为闭环帧或者在闭环线程处理他时，成功到检测到闭环帧
//! \note 这个函数与 Keyframe::SetNotErase() 函数是配对使用的。且仅仅在闭环线程中用到
void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty()) // 表示当前关键帧没有成功检测到闭环关键帧/没有被检测到成为闭环关键帧
        {
            mbNotErase = false; // 允许当前关键帧在局部建图线程中，如果被认为是冗余关键帧，那么就可以直接被擦除。
        }
    }

    if(mbToBeErased) // 说明在局部建图线程中，之前认为该关键帧是冗余的。但是该关键帧被闭环线程用到了。然后没有立刻擦除。那么此时在这里擦除
    {
        SetBadFlag(); // 内部会在关键帧数据库中擦除当前关键帧
    }
}

// 因为当前关键帧要被剔除。所以要将当前关键帧的孩子关键帧找到对应的父关键帧，也就是更新最小生成树。在当前关键帧的共视图中。也需要去掉其他关键帧和当前关键帧之间的关联。
//    在地图和关键帧数据库（闭环检测/重定位）中擦除当前关键帧。置当前关键帧为 bad flag。取消当前关键帧的父亲关键帧和当前关键帧之间的联系。
//    需要注意的是：如果闭环线程函数 ComputeSim3() 和 DetectLoop() 设置了 mbNotErase = true.此时该函数会直接返回。不会清理关键帧。
//    然后该关键帧能够被剔除，实际上是由闭环线程来决定。
void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase) // 如果当前关键帧不是参考世界关键帧。暂时不允许擦除。因为闭环线程正在使用其检测是否存在闭环
        {
            mbToBeErased = true; // 将要被删除的关键帧。当闭环线程决定该关键帧不满足其条件。然后会自动调用 KeyFrame::SetErase() 函数。然后就会清理当前关键帧。
            return;
        }
    }
    // 与当前关键帧链接的其他共视图中的关键帧。要剔除当前链接关系。然后更新对应关键帧的共视图
    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this); // 清理地图点和当前关键帧之间的联系
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree // 更新最小生成树
        set<KeyFrame*> sParentCandidates; // 潜在的父关键帧节点
        sParentCandidates.insert(mpParent); // 当前关键帧的父关键帧，潜在的父关键帧包含了已经找到了父关键帧的孩子关键帧和 当前关键帧的父关键帧

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest？？？
        // 因为当前关键帧是将要被剔除的。所以他的孩子关键帧必须要找到分配一个父关键帧。首先第一次遍历就是看看当前关键帧的父关键帧和自己的孩子关键帧是不是存在 「父---孩子」关系
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1; // 父和孩子关键帧之间的共视强度
            KeyFrame* pC; // 孩子关键帧
            KeyFrame* pP; // 父关键帧
            // 对每个孩子关键帧，都要找到一个父关键帧。潜在的父关键帧是已经找到了父关键帧的孩子关键帧和 当前关键帧的父关键帧
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe 因为父关键帧必须要与当前 pKF 关键帧在共视图中才可以
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId) // 第一次时：说明当前关键帧的孩子关键帧与自己的父关键帧有链接
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP); // 找到了一个孩子关键帧和父关键帧之间的联系。添加「父亲---孩子」关联
                sParentCandidates.insert(pC); // 将刚刚处理过的孩子关键帧作为潜在的父关键帧。但是这有什么依据吗？？是因为自己的孩子关键帧之间也可能是有父亲--孩子这种关系吗？？
                mspChildrens.erase(pC); // 已经在孩子关键帧之中找到了一个父亲--孩子配对关系。那么要剔除其找到的孩子关键帧
            }
            else
                break;
        }
        // 退出上面循环有两种情况：1、所有的孩子关键帧都找到了新的父关键帧。
        // 2、剩下孩子关键帧的共视图中没有包含潜在的父关键帧的，所以导致 bContinue = false 然后剩下的孩子就没有父关键帧（这种几率小些）
        // 此时的做法就是把没有分配父关键帧的孩子关键帧。都把当前关键帧的父关键帧给他。但是这种关联其实是一个弱的联系。当然也可能他们之间有共视的地图点。但是比较少。
        // 但是也可能他们之间没有共视地图点（虽然几率更小些）
        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this); // 父亲关键帧擦除当前关键帧
        mTcp = Tcw*mpParent->GetPoseInverse(); // 系统执行完毕后，轨迹的保留需要这个值
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this); // 在地图中擦除关键帧。但是关键帧内存没有释放
    mpKeyFrameDB->erase(this); // 在数据库中擦除当前关键帧。
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}
// 取消当前关键帧和 pKF 关键帧之间在共视图上的链接关系。之后更新共视图
void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles(); // 擦除链接之后，需要更新一些共视图
}
// 在当前关键帧图像上以 (x,y) 为圆心。半径为 r 的范围内找到一些当前关键帧上的关键点(序号)，存储下来，就是潜在的匹配点。然后返回
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols) // 说明该点{带有半径 r 的区域}不在当前图像的网格中,则无法寻找匹配点对
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
// 判断给定的坐标是否在当前关键帧图像坐标系上
bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}
// 计算当前帧所有 有效地图点(世界坐标)在当前相机坐标系下的深度值 z 。默认情况计算中位数深度
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3); // 旋转矩阵最后一行
    Rcw2 = Rcw2.t(); // 单目初始化时这里是 [0 0 1]^T 向量
    float zcw = Tcw_.at<float>(2,3); // 平移向量 z 坐标
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i]) // 表示当前关键点有对应的地图点
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw; // 这里实际上就是 Rcw * x3Dw + tcw 的第 三行，就是相机坐标系下该地图点的坐标深度 z 值。可以自己拆解来看看
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q]; // 默认参数是中位数
}

} //namespace ORB_SLAM
