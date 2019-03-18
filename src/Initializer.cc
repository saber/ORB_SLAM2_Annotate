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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{
// sigma = 1 为什么是 1？？？？？？
// iterations = 200
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}
//! \see 对应论文中自动初始化部分:IV. AUTOMATIC MAP INITIALIZATION
//! \brief 根据参考帧和当前帧的匹配关系，计算 H/F 矩阵恢复初始 3d 地图点
//! \param vMatches12: 输入参数：参考帧和当前帧的匹配关系。vMatches12[i] = index ;
//!                    i 是参考图像关键点序号， index: 当前图像关键点序号
//! \param vP3D: 将要得到的 参考图像和当前图像有效匹配点对成功三角化的 3d 点。
//! \param vbTriangulated: vbTriangulated[i] = true: 表示参考图像关键点 i 是有效点（成功三角化的 3d 点）
//! \return 是否成功初始化
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = CurrentFrame.mvKeysUn;

    mvMatches12.clear();    // 清除上次初始化失败建立的资源,使得 size = 0
    mvMatches12.reserve(mvKeys2.size()); // 扩充容量，这里为什么是 mvKeys2???，虽然不影响结果，但是这里是 mvKeys1 比较好
    mvbMatched1.resize(mvKeys1.size());

    // 获取有效的参考帧和当前帧的匹配关系.
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)    // 对于内部值为 -1 表示没有匹配的点对
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }
    // 有效匹配对数
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0); // 使用了 DBOW2 作者写的一个常用工具集

    // 随机选取点对 填充 mvSets
    for(int it=0; it<mMaxIterations; it++)
    {   // 从新赋值，因为需要迭代 200 次 （mMaxIterations）
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;
            // 为了防止 8 次选取相同的点，这里需要清理刚刚选取过的序号点
            vAvailableIndices[randi] = vAvailableIndices.back(); // 把最后的序号放在刚刚选过的位置，之后就可以从 vector 后面剔除元素了
            vAvailableIndices.pop_back();   // 剔除最后一位元素
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    // 下面参数对应论文 IV:AUTOMATIC MAP INITIALIZATION 自动初始化部分，第二步，建立两个线程计算两种模型{平面 H 矩阵，非平面 F 矩阵}
    vector<bool> vbMatchesInliersH, vbMatchesInliersF; // 表示计算相应 H/F 参数时，分别对应的最好配对集。
    float SH, SF; // 两种得分，论文上有标注
    cv::Mat H, F; // 这里是 F=F21: 1->2的变换即，参考图像到当前图像的变换
                  // H=H21: 1->2的单应
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if(RH>0.40)
        // vP3D: 三角化的地图点，vbTriangulated: 参考图像关键点是否有对应的 3d 点。
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;   // 这里虽然不执行，但是上面 if () else 中返回值可能是 false
}

//! \note H 矩阵的自由度是 8.实际上 4 对点即可求解。但是为了与计算 F 矩阵等价。
//!       这里用到了全部的 8 对点。
//! \brief 内部用的是归一化的 DLT + RANSAC 迭代，参考 mvg 中文版书上第 3 章
//!  根据 8 对点，Ransac 迭代求解 H 矩阵。计算最好结果的得分。和记录内点集
//!  vbMatchesInliers: 判断当前的 mvMatches12 中对应点是否是内点
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches 目前最优匹配数量，后面可能根据 RANSAC 在去除一些不好的匹配点
    const int N = mvMatches12.size();

    // Normalize coordinates vPn1 vPn2 是归一化后（通过一个相似变换）的点集， T1 T2 是原始点集到归一化后坐标系点集的相似变换矩阵
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score = 0.0;    // 得分越高，说明某个单应矩阵越符合要求！
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false); // 记录当前点是否参与论文中 AUTOMATIC MAP INITIALIZATION 得分计算，是否是内点集
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 这个 RANSAC 没有使用标准的方法，仅仅用了评分最高。其实正规方法是。包含内点数目最多的模型才是最终确定的模型。
    // 如果最高内点数目有多个。那么才选取其中评分最高的模型。这里需要改进。因为内点越多，评分相应降低。
    // 但是还有一个问题就是在初始化阶段选取 H /F? 论文是通过评分比值。那么其实得分比值这里应该把内点集个数也添加进去
    // 否则内点很少，导致评分很高。另一个内点很多，评分很低。那么论文中评分比值就不能适用这种情况，这样会选择分数高的模型，
    // 但是正确的模型确是另一个。但是如何做？？？？
    // 正规 ransac 参考原始论文以及多视图几何第 3 章，以及下面的博客：
    //  https://blog.csdn.net/fandq1223/article/details/53175964
    //  https://blog.csdn.net/laobai1015/article/details/51682596
    //  https://blog.csdn.net/robinhjwy/article/details/79174914
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j]; // 取出对应点索引

            vPn1i[j] = vPn1[mvMatches12[idx].first];    // 都是归一化后的点集
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        // 计算归一化后点集的单应矩阵
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        // 解除归一化
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            H21 = H21i.clone(); // 当前函数返回值：最好的单应
            vbMatchesInliers = vbCurrentInliers; // 当前函数返回值：最好的单应含有的有效点集
            score = currentScore;
        }
    }
}

//! \see 多视图几何 10.2 归一化 8 点算法。
//! \brief 根据 8 对点，Ransac 迭代求解 F 矩阵。计算最好结果的得分。和记录内点集
//! \param vbMatchesInliers: 表示参考图像关键点是否是符合当前比较好的 F21 参数意义下的内点(投影误差在一个指定范围内)
//! \param score: 在参数 F21 情况下的内点个数
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
//    const int N = vbMatchesInliers.size();  // ??? 这个是原始的代码，有问题，应该与 FindHomography()方法第一行代码一致，如下：
    const int N = mvMatches12.size();
    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // 返回一个秩为 2 的基础矩阵
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        // 解除归一化
        F21i = T2t*Fn*T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

//! \brief 计算图像 1 到图像 2 的单应矩阵，按照多视图几何书上 54 页公式 3.3，然后通过 SVD 分解。求解单应矩阵
//! \param vP1: 隶属图像 1 归一化后的关键点集
//! \param vP2: 隶属图像 2 归一化后的关键点集
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size(); // 8 对点

    // 一对点构造 2 个方程，参考多视图几何中文版 54 页 3.3 公式上面介绍的。
    // 这里有一点需要注意。vP1 vP2 点集是 2D,但是书上的公式是齐次坐标。所以这里默认 w = 1，即在归一化平面上
    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }
    // 通过奇异值分解：u为 2Nx2N vt:9x9
    cv::Mat u,w,vt;

    // 书上已说明：用 SVD 分解计算。解是 A 最小特征值对应的单位特征矢量
    // w 奇异值矩阵， u 左分解 vt 右分解的转置
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 因为 vt 是 v 的转置，所以这里取行 8
    return vt.row(8).reshape(0, 3); // 换算成矩阵的形式 3 行
}
// 计算基本矩阵，参考 mvg 191页，整个步骤参考 193 页算法 10.1
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);
    // 书上 10.3 公式，且输入的点集已经是归一化后的点
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    // 求解基本矩阵 F(线性解法)
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    // 强迫约束！ 为什么让 w(2)=0，就是在 F 范数下最接近 F 的奇异矩阵？？
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;   // 让 w 的最后一行为0，因为 w 是 Fpre 的奇异值构成的对角线矩阵。因次 w 秩为 2

    // 返回值是 p192 页中强迫约束的计算方法！
    return  u*cv::Mat::diag(w)*vt;
}
//! \brief 给定单应矩阵，检查当前计算的单应矩阵的得分以及间接统计有多少对匹配点符合要求
//!        利用对称转移误差。
//! \see 多视图几何第 3 章
//! \param vbMatchesInliers: 对应此时 H 矩阵时，最好的配对点集
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991; // 内点到模型距离平方阈值

    const float invSigmaSquare = 1.0/(sigma*sigma);
    // 计算所有点的总得分
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // 下面两个对称转移误差在书上 57 页说明，以及 3.7 公式
        // Reprojection error in first image
        // x2in1 = H12*x2
        // 投影后的点也是齐次坐标点（x/z,y/z）
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 这里为什么除以sigma^2 可以在 mvg 书上 3.17 式子，看出阈值本身带有 sigma^2 的。
        // 前面 th = 5.991 不带 sigma^2，所以这里要除以这个值
        const float chiSquare1 = squareDist1*invSigmaSquare; // chi ：表示 卡方分布符号

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

//! \brief 检查当前计算的 F 矩阵得分。并记录内点集
//! \see mvg 书上 191 页，计算基本矩阵 F，几何意义参考 8.2 章节 基本矩阵 F
//! \note 对于 x'^T F x = 0的理解问题。 Fx 就是一条直线 I。然后 x'^T I =0 表示的是点在线上。
//!       下面的误差就是先把直线算出来:Fx，之后用传统方法计算点到直线的距离
//! \param vbMatchesInliers: 表示参考图像关键点是否是符合当前 F21 参数意义下的内点(投影误差在一个指定范围内)
//! \return 在当前参数 F21 下，内点个数。
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // 下面误差的计算就是按照： x^T F x = 0 计算的。这个公式意义就是点到对极线的距离，
        // Fx=一条对极线，然后 x^T I = 0 表示点在直线 I 上
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2) 投影后的极线方程三个参数， a2x + b2y + c2 = 0

        const float a2 = f11*u1+f12*v1+f13; // 齐次坐标 3x1
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // 下面就是计算点 u2,v2 到平面极线的距离，然后在开平方
        // 点到直线的距离参考: 1.2 2d 射影平面
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        // 与上同理！
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

//! \brief 由 F 矩阵恢复运动 R t
//! \details 首先将 F 矩阵变为 E 矩阵，然后通过奇异值分解。获得 4 个解。分别验证 4 个解是否符合要求。
//!          记录每种解对应的有效点集及其点的个数.以及最小视差。只有四个解当中有一个解是远优于其他 3 个解，
//!          并且有效三角化点个数高于最低阈值。此时才算成功恢复运动。
//! \see IV. AUTOMATIC MAP INITIALIZATION 的第 4 步。
//! \param vbMatchesInliers: 当前 F 参数对应的内点集。
//! \param minTriangulated: 最小内点集
//! \param minParallax: 最小的视差角，这里是 1°，太小的角度三角化出来的点不准确
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,                            // 1               // 50
                               cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated,
                               float minParallax, int minTriangulated)
{
    int N=0; // 当前计算的 F21 参数下， 对应的内点集
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    // 四种解
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4; // 这个是计算的角度值

    // 检查给定的 R t 条件下，有效点个数和最小视差，内部按照 1 °视差来筛选的，所以一般来说 parallax1 是 > 1°的！
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    // 这些参数可以适当调节或者用其他方法进行判定！
    // 起始点的 90%
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    // 第一：要有足够有效的三角化点，第二个：必须区分哪个解是有效点
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize。
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax) // 这个条件可以适当放宽。让初始化成功率高些。但是视差角度越小，3d 点越不准确！
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}
//! \brief 从给定的 H 矩阵中恢复运动 R,t，由 H 矩阵计算出来 Fougeras 论文中的 A 矩阵，然后通过 A 矩阵进行求解 R ,t。
//!         获得 8 种解后，
//! \note vbTriangulated[i] = true : 表示参考图像关键点 i 是有效点（有正确的匹配点）。
//!       虽然论文中讲解了 3 种情况，但是在程序中，实际上 d1 不等于 d2 不等于 d3
//!       下面代码就是分为 d'>0 以及 d'<0 ，然后分别讨论了论文中的第一种情况，各获得 4 种解。
//!       下面的 t 是单位化了的。
//! \see Motion and structure from motion in a piecewise planar environment
//! \param vbMatchesInliers: 表示计算相应 H 参数时，分别对应的最好配对集。
//! \param vbTriangulated : 当前 H 得到的 R t 时，参考图像 1 对应的关键点是否有对应的三角化 3d 点。
//! \param vP3D : 对应 R t 情况下，有效关键点对对应的三角化的 3d 点。
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;    // 记录内点集，当前 H21 对应的
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i]) // 表示是内点，这里计算内点个数（对应 H 矩阵时，好对的配对点数）
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988
    // 可以观看[23]文献 5 页处。那里给出的 A 是两个相机坐标系之间的映射。H21 是通过两个图像坐标系得到的。
    // 我们要想使用 [23] 的方法。必须得到 A 。所以这里要通过相机内参矩阵做一个变换。
    // 就是在 [23] 文献 5 页，那里 X2=(R+tn^T/d)X1 把 X2 X1 通过内参矩阵替换成对应的图像坐标。即可得到
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K; // 3x3

    // 下面这步直接参考文献 [23] 中 4部分：Solving the decomposition problem
    // 下面求解已经假设了 d1 不等于 d2 不等于 d3 也就是第 1 种方法。 对于第 2 种方法的只有平移情况没有考虑
    cv::Mat U,w,Vt,V; // w:3x1 存储的奇异值
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

//    float s = cv::determinant(U)*cv::determinant(Vt); // 这里标准来说应该是 det(U)*det(V)
    float s = cv::determinant(U)*cv::determinant(V);    // 因为一个矩阵转置的行列式等于自己本身的行列式

    // 奇异值：d1>=d2>=d3
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if(d1/d2<1.00001 || d2/d3<1.00001)  // 说明SVD数值计算有问题，不算
    {
        return false;
    }

    // 记录求解出来的可能的旋转矩阵和平移向量，以及平面法向量（这个存储的是正的法向量）
    vector<cv::Mat> vR, vt, vn; // 这里 vn 获取值后就没在用到
    // 存储 8 种可能的解
    vR.reserve(8);
    vt.reserve(8); // 存储的是单位 t 向量
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3)); // 对应论文中公式 (12) x1，去除前面符号
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3)); // (12) 式中的 x3，同上
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2 暗含 d'>0 因为 d2 为奇异值肯定 >0
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2); // 对应论文公式 (13)

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta}; // 符号与 x1[] x3[]对应位置有关。看论文中 (12)(13)式子

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F); // R'
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt; // 公式（8）
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t)); // 这里平移向量做了单位化！！！

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)    // 这里为什么这么做？？？？？，可能是就是单纯为了得到平面正的法向量
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2 < 0
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2); // sin(fi)

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2); // cos(fi)
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0) // 与上面同理
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // (论文中对应这句话的相关知识点没有看！！！)
    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi; // 记录视差最小值（角度）
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;

        // 会三角化 3d 点
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood) // 成功三角化的点对数
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

//! \see mvg 217 页中关于线性三角形法求解，这里也是用 DLT 方法来计算，然后通过 SVD 求解,最优的方法是 11.3 以后的方法！
//! \param x3D: 4x1 非齐次
//! \param p1: 相机 1 的位姿矩阵
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    // 计算 218 页 A 矩阵
    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t(); // 点为列向量
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3); // 由齐次坐标转换为非齐次坐标点
}
//! see: MVG 中文版本中 67 页。以及吴博提供的 pdf 详解单目 Initializer.cc 部分特征点归一化部分公式
//! DLT 计算 H 矩阵的其中一步：归一化点集
//! 计算相似变换{平移 + 缩放}
//! 函数具体步骤：
//!    1) 计算形心
//!    2）计算缩放的尺度因子
//!    2）计算以形心为坐标系。缩放后的点集
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    // 计算点集的形心坐标（均值点），然后就是以此为坐标系原点，计算其他点在当前坐标系下的坐标。
    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;
    // (根据多视图几何中给出的归一化 DLT 算法步骤中，需要使得归一化后的点到形心的平均距离为根号 2，这样两个方向上。平均距离就是 1
    // 首先计算当前点集到形心的平均距离（分两个方向算），之后根据每个方向经过缩放后的平均距离是 1 的要求。计算缩放的尺度因子

    // 计算当前到形心的平均距离
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    // 计算缩放的尺度因子
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 归一化后的点集
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }
    // 计算两个坐标系的变换 T，因为归一化 DLT 算法计算完结果后，最后需要解除归一化，恢复到原始图像坐标系
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

//! \brief 对当前得到的 R t 参数，对有效匹配点对(在 H 参数下是内点对)进行三角化。返回有效三角化个数
//! \param vbMatchesInliers: 表示计算相应 H 参数时，对应的最好配对集。
//! \param vP3D: 保存在此时 R t 的情况下，有效配对点三角化对应的 3d 点。vP3D[i]:表示参考图像的关键点序号 i 对应的三角化 3d 点
//! \param vbGood: 参考图像 1 对应的所有关键点，是否是三角化了的。如果被三角化，那么说明这个点对应的配对是有效配对(基于当前参数下的)！=true
//! \param parallax: 记录当前点集中，所有符合条件的视差角度中最小值或者索引 = 50 对应的视差角度。
//! \param th2: 这里传进来是 4*sigma2，不知道如何确定是这个值的？
//! \return 返回成功三角化的点对数
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax; // 记录满足条件点对应的 cos(视差角度)
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]  为参考世界坐标系
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F); // 相机 1 光心

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t; // 求出相机 2 光心坐标

    int nGood=0; // 有效点个数

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i]) // 对此时 H 参数来说，不是正确匹配的内点集去掉
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1; // 4x1 列向量，前 3 行是空间点非齐次坐标

        // 三角化空间点。根据匹配的点对。
        Triangulate(kp1,kp2,P1,P2,p3dC1); // 因为此时是参考帧和第一帧三角化，参考帧就是世界坐标系。所以这里直接是 p3dC1

        // 判断当前三角化后的点是否是有效点{±无穷、空 除外}
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false; // 标记参考图像关键点序号
            continue;
        }

        // Check parallax 计算视差角
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        // a . b = |a||b|cos(theta) ===> cos(theta)=a.b/(|a||b|)
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // 视差角太小（三角测量的不确定性，计算的 3d 点很可能是错的！这样情况需要排除），原理见: 14 讲 p157
        // cos 越小，视差角度越大！
        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)    // 这里按照 1 °的视差计算的
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        // 此时是 4 个像素误差内即可
        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        // 保存视差角度，以及记录参考图像关键点对应的 3d 点坐标，并记录三角化成功个数
        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());  // 最大值表示对应的最小的视差角度

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI; // 获取一个相对比较小的视差角度，不知道为什么？？
    }
    else
        parallax=0;

    return nGood;
}

//! \brief 分解 E 为可能的 R t
//! \see MVG p175 页。结论 8.19 以及上面那一段的说明，
//! \note t 就是 U 的最后一列。在 mvg 8.13 式子中有写明 W 的形式
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    // 求平移向量
    u.col(2).copyTo(t);
    t=t/cv::norm(t); // 单位化

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;

    // 保证旋转矩阵行列式为 1(旋转矩阵本身约束就是行列式 = 1)
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
