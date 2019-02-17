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
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>
#include "Frame.h"


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
    typedef pair<int,int> Match;

public:

    // Fix the reference frame
    Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);


private:

    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);

    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);

    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1) （去除畸变的）
    vector<cv::KeyPoint> mvKeys1;   // 初始化时参考图像去除畸变的关键点

    // Keypoints from Current Frame (Frame 2)
    vector<cv::KeyPoint> mvKeys2;   // 当前帧去畸变点集

    // Current Matches from Reference to Current
    vector<Match> mvMatches12;  // 记录参考图像和当前图像匹配的点对 Match: std::pair<int,int> 第一个数是参考图像关键点序号，第二个数是当前图像关键点序号
                                // 这里是有效匹配，vector size() 就是匹配好的点对数
    vector<bool> mvbMatched1;   //size() 是 mvKeys1关键点个数， 哪个位是 ture，表示参考图像关键点有对应的匹配点

    // Calibration
    cv::Mat mK; // 相机内参矩阵

    // Standard Deviation and Variance
    float mSigma, mSigma2;  // 初始化时为 1,此时表示测量误差为 1 个像素！可以参考 mvg 书上 75 页，就明白测量误差是什么了！

    // Ransac max iterations == 200
    int mMaxIterations; // 计算基础矩阵 F，以及单应矩阵 H

    // Ransac sets 用迭代 200 次，每次取 8 对点，所以是：vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));
    vector<vector<size_t> > mvSets;   

};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
