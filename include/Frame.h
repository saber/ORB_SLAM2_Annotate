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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
    // 对于 TUM 数据集相机参数是 ： 640 X 480 (宽度 X 高度)，但是下面这个参数并不是根据图像参数来定的。
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera  地图点是否能够投影到当前图像坐标系
    // and fill variables of the MapPoint to be used by the tracking 这个函数只在 tracking.cc 中调用了一次
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx; // 1/fx
    static float invfy; // 1/fy
    cv::Mat mDistCoef;  // 畸变去除向量 5x1 k1 k2 p1 p2 k3

    // Stereo baseline multiplied by fx.
    float mbf;  // 双目 b *fx,对于单目这里为 0

    // Stereo baseline in meters.
    float mb;   // 基线长度：mbf/fx，单目这里为 0

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    // 一个点判断是否是近点/远点的阈值， 针对双目和 RGB-D 传感器的可信深度
    float mThDepth;

    // Number of KeyPoints.
    int N; //  orb 特征提取的关键点个数

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;  // 利用 ORBextractor 在 Frame() 构造函数中进行提取关键点(2D图像),与下面的描述子 mDescriptors的行是一一对应的
    std::vector<cv::KeyPoint> mvKeysUn; // 仅仅针对单目和 RGB（双目必须提前去除了畸变），
                                        // 用 opencv 函数对提取的 mvKeys 关键点进行去畸变利用 mDistCoef cv::Mat矩阵

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;    // ??
    std::vector<float> mvDepth;     // ??

    // Bag of Words Vector structures. 通过 Frame::ComputeBoW() 更新计算，用当前 Frame 的描述子
    DBoW2::BowVector mBowVec; // init = 空 std::map<WordId, WordValue>: WordId: 字典 id (叶子节点)，WordValue: 指定的权重！
    DBoW2::FeatureVector mFeatVec; // init =空 ,std::map<NodeId, std::vector<unsigned int> >对象，NodeId: 指定层的某个节点 id，vector<>存储的是对应的描述子标号

    // ORB descriptor, each row associated to a keypoint.   // 每幅图像对应的提取出来的描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;    // 当前帧提取的关键点对应的 MapPoint ，可以在初始化该类的时候用 N 作为初始个数 = NULL，
                                            // 在单目初始化时，currentFrame 赋值的 MapPoint 地图点
                                            // 但是在正常追踪时，这个是当前帧和上一个关键帧匹配好的关键点对应的地图点。就是把关键帧地图点和当前帧关键点配上对的给这里
                                            // 当然这个点要求多于 15 个
                                            // 如果在跟踪丢失，重定位时，这个记录的就是通过重定位得到的地图点
    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;   // 判断 mvpMapPoints 对应的地图点是不是外点,与上面的 mvpMapPoints 一一对应的, true: 表示是外点
                                    // 在进行  Optimizer::PoseOptimization() 时，会记录当前帧地图点最后是不是外点。这里会进行更新
    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    // 这里计算的是根据去除畸变后的 mnMinX...来计算的
    static float mfGridElementWidthInv; //  mfGridElementWidthInv = FRAME_GRID_COLS/(mnMaxX-mnMinX) ;那么一个小网格，固定分配宽度为  1/mfGridElementWidthInv
    static float mfGridElementHeightInv;// FRAME_GRID_ROWS/(mnMaxY-mnMinY)   // 网格高度为 1/...
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];   // 二维数组，数组内部是 vector,vector 保存的网格内部关键点序号(在 mvKeysUn 中的关键点序号)

    // Camera pose.
    cv::Mat mTcw; // 在运动恢复结构后得到的 pose

    // Current and Next Frame id.
    static long unsigned int nNextId;   // 初始为 0 每次来一个新的图像，就会构造一次 ++
    long unsigned int mnId; // 当前帧标号(累计的)构造一次加一次

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;    // 当前帧与之匹配的关键帧!在单目初始化之后/或者在追踪过程中新的关键帧建立时,当前普通帧都指向了自己本身关键帧
                                // 这里在正常追踪的过程中(目前来说是初始化后第一次跟踪，就是与关键帧进行匹配的)，
                                // 这个参考的关键帧其实是在当前帧和该关键帧成功匹配且计算完 pose 后加入的！此时还没有调用 Tracking::TrackLocalMap() 函数
                                // 可以参考 Tracking.cc 中 418 行 mCurrentFrame.mpReferenceKF = mpReferenceKF;
                                // 不过这个是为了追踪时。当前帧和上一帧匀速运动模型失效后。需要与当前帧的参考关键帧再次进行匹配。这个就是该变量的用途！
    // Scale pyramid info.
    int mnScaleLevels;  // 8
    float mfScaleFactor;    // 1.2
    float mfLogScaleFactor; // log(1.2)
    vector<float> mvScaleFactors;   // 包含的是尺度因子组合，内部是 vector[8] 8 个元素，每一维元素表示的是该层金字塔图像相对于原始图像的尺度因子 1.2  1.2*1.2  1.2*1.2*1.2
    vector<float> mvInvScaleFactors;    // 是上面这个变量的倒数：1/mvScaleFactors[i]
    vector<float> mvLevelSigma2;   //  不知道是做什么的？[i] = mvScaleFactor[i]*mvScaleFactor[i];
    vector<float> mvInvLevelSigma2; // 1/mvLevelSigma2[i]

    // Undistorted Image Bounds (computed once). 图像的四个角的范围{x,y}方向
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;  // 初始时为 true,静态变量，归类所有,对第一帧图像计算完毕后，置位 false


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified! 这里对于单目来说也需要去除关键点畸变。
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc
};

}// namespace ORB_SLAM

#endif // FRAME_H
