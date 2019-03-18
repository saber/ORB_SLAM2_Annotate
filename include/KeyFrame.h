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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
public:
    // 设定一个是否被剔除的变量
    bool keyframe_culling_ = false; // 自己为了验证剔除的关键帧还会不会优化，暂时没有验证成功。数据集时，闭环无法起作用
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame* pKF);

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){ // 因为 covilibility 图中是按照权重大->小顺序存储的。所以这里是 a>b。降序查找
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId; // init = 0
    long unsigned int mnId; // 当前关键帧 id (与 Frame id 不同 )（且永远不会改变，即使有些关键帧会被剔除，但是剩下的关键帧 id 不会更新！）
    const long unsigned int mnFrameId; // Frame 转换为 keyframe 时所带 frame 的标号

    const double mTimeStamp; // frame 转换为 keyframe 所带的时间戳

    // Grid (to speed up feature matching)
    const int mnGridCols;   // 64
    const int mnGridRows;   // 48
    const float mfGridElementWidthInv; //  mfGridElementWidthInv = FRAME_GRID_COLS/(mnMaxX-mnMinX) ;那么一个小网格，固定分配宽度为  1/mfGridElementWidthInv
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame; // init = 0 当前关键帧与哪个普通帧可能有着共同观测的地图点。这个只在 Tracking::UpdateLocalKeyFrames() 中更新写入的
    long unsigned int mnFuseTargetForKF; // init = 0 只在  LocalMapping::SearchInNeighbors() 函数中进行更新,表示当前关键帧是局部建图线程正在处理的关键帧的融合目标

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF; // init = 0 ,记录在局部建图线程中 BA 优化时正在处理的关键帧 id,以这个 ID 为基准进行的局部 BA 优化
    long unsigned int mnBAFixedForKF; // init = 0, 在优化时，记录正在处理的关键帧 id。此时表示当前关键帧在优化时，不属于局部关键帧组，后面优化时，固定住。
                                      // 然后仅仅作为约束。其 pose 不参与优化

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery; // init = 0,在 KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore) 中设定。
                                   // 表示当前关键帧已经记录了当前闭环线程正在处理的关键帧(pKF)。重复找到时不在记录.且暗示当前关键帧不是闭环线程正在处理的关键帧的临近关键帧
    int mnLoopWords; // init = 0,与当前闭环线程正在处理的关键帧 pKF 有着共同单词的个数。此时说明当前关键帧已经不属于闭环线程正在处理的关键帧的临近关键帧
    float mLoopScore; // 当前关键帧与闭环线程处理的关键帧(pKF)之间的相似性得分。此时说明当前关键帧与闭环处理的关键帧之间的共同单词个数是大于最低单词个数（筛选阈值）
    long unsigned int mnRelocQuery; // init = 0, 在追踪失败/或者重定位模式，重定位时进入  KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F) 函数内部，
                                    // 当该关键帧和该变量存储的id对应的 Frame 有着至少共同的单词(需要提前用 DBoW2 来提取特征向量和词袋向量)
    int mnRelocWords; // init = 0 // 与上面的变量有关，表示当前这个关键帧和普通 Frame 有多少共同的单词个数
    float mRelocScore; //并没有初始化。所以如果直接获取这个值，就会是垃圾值。 与上面变量有关，计算当前关键帧与 mnRelocQuery 记录的普通 Frame相似性评分(根据 BoW向量 )

    // Variables used by loop closing
    cv::Mat mTcwGBA; // 在闭环优化时，全局 BA 函数中更新的值，就是该关键帧的 pose。在全局 BA 函数优化时，没有更新 Tcw 的值，而是更新了这里.
                    // 是因为在全部 BA 函数中用其纠正局部建图新增加的关键帧的 Pose。之后会在全局 BA 函数中用其更新 Tcw 值。
    cv::Mat mTcwBefGBA; // 保存未经过全局 BA 的 Tcw 的值。因为在该关键帧的孩子关键帧纠正后，就会用 mTcwGBA 更新 Tcw。这里为了保证本质图优化时的 Tcw 值。
    long unsigned int mnBAGlobalForKF; // init = 0，闭环线程跑全局优化时，当前正在处理的关键帧的 id

    // Calibration parameters // frame 给定的 后面三个是 baseline*fx, baseline, 远近点阈值
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N; // frame 对应的 orb 提取关键点个数

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys; // frame 提取的关键点
    const std::vector<cv::KeyPoint> mvKeysUn; // frame 去除畸变的关键点
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors; // frame 提取的关键点对应的描述子,包括了那些没有匹配上以及三角化好的关键点描述子

    //BoW 需要由一副图像的描述子转换 (在单目初始化时构造的关键帧这里都为空)
    DBoW2::BowVector mBowVec;   // 词袋向量   std::map<WordId, WordValue>: WordId: 字典 id (叶子节点)，WordValue: 指定的权重！
    DBoW2::FeatureVector mFeatVec;  // 用到了 第 4 层的所有树节点，特征向量 std::map<NodeId, std::vector<unsigned int> >对象，NodeId: 指定层的某个节点 id，vector<>存储的是对应的描述子标号
                                    // 是按照节点 id 的小到大的顺序来存储的
    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp; // 保留父关键帧到当前关键帧的变换，为什么要保留？:在保留轨迹的时候，虽然这个关键帧不要了。但是内存地址没有清理。后期还需要进行轨迹的保存

    // Scale
    const int mnScaleLevels; // 8
    const float mfScaleFactor; // 1.2
    const float mfLogScaleFactor; // log(1.2)
    const std::vector<float> mvScaleFactors; // 包含的是尺度因子组合，内部是 vector[8] 8 个元素，每一维元素表示的是该层金字塔图像相对于原始图像的尺度因子
    const std::vector<float> mvLevelSigma2; // [i] = mvScaleFactor[i]*mvScaleFactor[i];
    const std::vector<float> mvInvLevelSigma2;  // 1/mvLevelSigma2[i]，作为全局 BA 优化时，信息矩阵的因子，为什么选择这个？？？

    // Image bounds and calibration  图像的四个角的范围,{x,y}方向,是根据去除畸变后四个角的坐标计算的图像边界范围
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK; // 内参矩阵


    // The following variables need to be accessed through a mutex to be thread safe.
protected:

    // SE3 Pose and camera center 4x4
    cv::Mat Tcw; // 在 Frame 中匹配恢复运动得到的,相对世界的 pose
    cv::Mat Twc; // 相机到世界的变换
    cv::Mat Ow; // 当前 keyframe 在世界坐标系下的坐标

    cv::Mat Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints  // 这个是 frame 初始化提取特征点，大小就是 N。还会在局部建图线程中建立新的地图点。
    std::vector<MapPoint*> mvpMapPoints; // 当前帧所有关键点对应的地图点。如果这个关键点有被成功三角化，那么此时就有 MapPoint * 插入，否则为 NULL。也就是有效的地图点
    // BoW
    KeyFrameDatabase* mpKeyFrameDB; // 构建 KeyFrame 时传入的参数
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;  // 二维数组，数组内部是 vector,vector 保存的网格内部关键点序号(在 mvKeysUn 中的关键点序号)

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights; // 用共视的地图点个数作为权重: 记录与当前关键帧有共视关系的关键帧，以及共视强度(存在多少个共同观测的地图点) 未排序的！
    // 下面两个变量是上面这个变量 mConnectedKeyFrameWeights 变量分开部分。是按照权重 大--->小 进行排序的
    //  KeyFrame::UpdateBestCovisibles() 在这个函数中进行排序的！
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames; // 与当前关键帧链接的其他关键帧(是按照有多少个共同观测的地图点个数，从大--->小排列的),为什么排序，
                            // 是因为在 Tracking::TrackLocalMap()函数内部。对应论文 V---D 。找 K2 关键帧组时，我们需要临近关键帧。
                            // 就是在这个函数Tracking::UpdateLocalKeyFrames()内。里面仅仅要了关键帧临近帧的其中一个。所以我们理所当然的要与当前关键帧有着最强的共视关系的关键帧。
                            // 所以这里就是带有顺序的。并且是按照共视强度从大到小排列的！！！
    std::vector<int> mvOrderedWeights; // 与上面的 mvpOrderedConnectedKeyFrames 序号一一对应，记录的是共同观测地图点个数

    // Spanning Tree and Loop Edges
    bool mbFirstConnection; // 初始化 = true,在单目初始化时，通过增加关键帧后，建立链接,然后变为 false.在函数 KeyFrame::UpdateConnections()
    KeyFrame* mpParent; // init = NULL, 与该关键帧存在最强的共视关系(共同观测的地图点最多)的关键帧作为父关键帧节点，一般都是之前就加入共视图里的老关键帧
    std::set<KeyFrame*> mspChildrens; // 当前关键帧被其他关键帧视为最强的共视关系且视为父关键帧，
                                      // 那么此时当前关键帧的孩子 mspChildrens 就会加入那个关键帧。
                                      // 因为后加入的关键帧都是新来的帧，按照时间顺序只能是作为老关键帧的孩子
                                      // 这个子关键帧并没有进行按照与当前帧共视强度由 大到小排列。这个可能是一个改进地方！！！！！！
    std::set<KeyFrame*> mspLoopEdges; // 以当前关键帧为基准检测到了闭环。或者被其他关键帧检测成为闭环帧。在 LoopClosing::CorrectLoop() 函数执行完本质图优化后，
                                      // 那么此时两个关键帧之间就会添加一个链接。
                                      // 后面会在闭环线程中 OptimizeEssentialGraph() 进行添加边然后更新。

    // Bad flags
    bool mbNotErase; // init = false
    bool mbToBeErased; // init =false
    bool mbBad; // init = false ，标记当前关键帧是否是冗余的关键帧，是的话就会在 Local Map 中去除

    float mHalfBaseline; // Only for visualization baseline/2

    Map* mpMap; // 构造时传入

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
