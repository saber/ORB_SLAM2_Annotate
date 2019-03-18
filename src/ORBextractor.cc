/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15; // 这里是按照半径算的，不是方块 patch
const int EDGE_THRESHOLD = 19;

//! \brief 计算以 pt 为圆心，半径为 HALF_PATCH_SIZE 大小的区域质心。由质心计算角度值。
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];  // u = 0

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();  // 图像每一行包含像素的大小
    // 这里计算的是以选定的点为圆心，半径为 HALF_PATCH_SIZE 的圆形区域的质心
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {   // 这里计算步骤，可以自己手动一步一步代入即可理解！
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);    // 因为 v =1 时，这里计算了 v=0上下 ±1 行的元素，对于 m_01而言，上下两行计算的灰度值分别乘以 ±v ，所以这里取减号
            m_10 += u * (val_plus + val_minus); // 这里表示计算 u 这个 x坐标，对应的上下两行元素
        }
        m_01 += v * v_sum;// 这里实际上是 v*val_plus + (-v) *val_minus 只不过上面循环中已经计算了关于 v=0 对称上下两行的像素值
    }

    return fastAtan2((float)m_01, (float)m_10); // 返回的是角度 0-360
}

const float factorPI = (float)(CV_PI/180.f);

// 根据关键点以及图像，和对应的模式，计算关键点二进制描述子
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;    // 弧度转换
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    // 定义旋转后的对应点像素值，这里是参考关键点为中心获取的像素值。可参考 ORB 原始论文 4.1
    // 实际上，可以根据每个关键点对应的角度得到旋转矩阵R，以关键点为坐标原点，以模式 pattern 为坐标的点对用旋转矩阵 R 进行旋转，之后获取旋转后的点对应的像素值
    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

    // 计算二进制带方向的描述子 256 维
    for (int i = 0; i < 32; ++i, pattern += 16) // desc 是 8 * 32 = 256 位，因为 pattern +=16  表示已经比较过 8 个点对了
    {   // 选取 8 对点，进行模式判别，计算描述子
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);   // 根据模式选取的一对点的像素值
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}


static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         int _iniThFAST, int _minThFAST):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i]; // ??sigma2 是什么
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    // 根据一副图像将要提取的总的特征数量，计算每一层特征数量，这里是按照尺度因子依次递减每层的特征点数量.
    // 比如第一层原始图像特征数量为 a1 ，则第二层特征点数量为 a1/scaleFactor。第三层特征数量为 a1/(scaleFactor*scaleFactor)。
    // 第 n 层特征数量为 a1/(scalFactor^(n-1))。通过 nlevels=8 层。总特征数量等于各个层特征数量之和。
    // 则 nfeatures = a1 + a1/scaleFactor + a1/(scaleFactor*scaleFactor) + ... + a1/(scaleFactor^(nlevels-1))
    // nfeatures = a1*(1-q^n)/(1-q) 其中 q=1/scaleFactor ,n=nlevels
    // 此时可以求出 a1 的大小，a1 = nfeatures *(1-q)/(1-q^n)。
    // 下面两行代码就是计算 a1 的值，不过将 q 替换为 factor 。然后接下来的 for 循环就是分别计算每层对应的特征数量
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);   // 最后一层特征数量

    const int npoints = 512;

    // 因为 cv::Point 类型非继承别的类，内部仅仅存储 _x, _y 二维平面坐标。
    // 所以可以将 int[256x4] 强制转换成 [2*512]，2 表示cv::Point内部成员int类型
    const Point* pattern0 = (const Point*)bit_pattern_31_;  // static int bit_pattern_31_[256*4]
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    // 下面这里选取关键点周围小块的方式其实是以关键点为圆心的半径为 16 的小块区域。实际上这里需要对照 IC_angle() 函数才可以理解！
    // 我们可以选取 16 x 16 的区域计算关键点小块质心。这样其实效果都没什么差别。
    umax.resize(HALF_PATCH_SIZE + 1);   // 16, 扩充 vector大小，然后初始值为 0
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1); // 11.607 => 11
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2); // 10.607 => 11
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;// 15 * 15
    for (v = 0; v <= vmax; ++v) {   // 获取 12 行，元素，还差 4个元素
        umax[v] = cvRound(sqrt(hp2 - v * v));
    }
    // Make sure we are symmetric 这里的选取方式可以不这样，自己测试修改 umax[14]=7其实也没什么不同，只要保证所有关键点的周围小块选取一致即可。
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);  // 计算关键点方向信息
    }
}
//! \details 将一个节点划分为 4 个子节点(田字格)，分别计算 4 个子节点的四个角的坐标，并对每个节点进行关键点分配
//!          需要注意的是；这里没有对那些不包含关键点的区域进行处理，也是说，子节点中很可能没有关键点
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs 格式如下：
    // n1 n2
    // n3 n4
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size()); // 扩充容量到：父节点包含关键点个数

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    // 对父节点的关键点，分配到 4 个子节点中
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x) // n1 n3 半区
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }   // 处理 n2 n4 半区：直接处理 y 坐标即可
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    // 对于无法再次细分的节点（关键点只剩下一个），标志为叶子节点
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}

//! \note 这个函数适用于图像的宽度 > 图像高度,否则需要转换 x y 的输入。
//! \brief 在输入的 vToDistributeKeys 包含的关键点中，均匀提取每个小区域的响应值最大的关键点
//! \details 划分过程如下：首先对这个图像有效区域进行四叉树--田字格形式 划分子节点（每个子节点至少包含一个关键点），
//!  节点个数直到满足该层图像需要提取的关键点个数 N 为止，或者无法继续划分(很可能关键点个数没有达到要求，但是现有节点中关键点都是一个，无法继续划分))
//!  划分完毕后，对最后划分的小区域。每个子区域挑选一个响应值最大的关键点--作为这个小区域的代表关键点。当然如果节点中就一个关键点，就选这个关键点。
//!  然后将所有关键点进行返回
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes   // 这里把整个图像作根据 width/height 的比值来决定，一个图像有几个初始节点
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));  // 初始节点个数 如果 小数 >0.5 则多增加一个节点， 这里要么 1 个要么 2个

    const float hX = static_cast<float>(maxX-minX)/nIni;    // 每个初始节点对应的 width

    list<ExtractorNode> lNodes; // 包含所有节点(内部至少有一个关键点)

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    // （图像宽度大于高度）一般情况下 nIni = 1 | 2 ，此时最多有 2 个节点
    // 保存每个节点的四个角的坐标
    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);    // 一个节点：左上角坐标
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);  // 一个节点：右上角坐标
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY); // 一个节点：对应左下角坐标
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY); // 一个节点：对应右下角坐标
        ni.vKeys.reserve(vToDistributeKeys.size()); // 扩充容量

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs 按照关键点 x 坐标与每个节点的宽度比值，进而确定该点隶属于哪个节点
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    // 对节点进行处理{叶子节点标记、去除不包含关键点的节点}
    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)    //当前节点包含的关键点个数 == 1，表示不可在细分。变为了叶子节点
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty()) // 去除关键点个数为 0 的节点
            lit = lNodes.erase(lit);    // 此时 lit 自动指向下一个节点
        else
            lit++;
    }

    bool bFinish = false;   // 四叉树细分完成标志（直到全部变为叶子节点或者划分的节点个数已经大于我们想在该层图像提取的关键点个数）

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;    // pair{子节点包含关键点个数,对应的子节点指针}
    vSizeAndPointerToNode.reserve(lNodes.size()*4); // 相当于每个节点 * 4，这里像是四叉树

    // 不断细分节点
    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();   // 当前有效节点(包含关键点)个数

        lit = lNodes.begin();

        int nToExpand = 0;  // 记录有效的子节点数量，表示下次划分子节点时，至少会在增加一次一个子节点
                // 自己添加的一句。为了保证每次更新子节点时，能够保证能够容纳所有子节点，防止后期内存不够，动态划分导致速度的降低！
        vSizeAndPointerToNode.reserve(lNodes.size()*4);

        vSizeAndPointerToNode.clear();  // 这里实际上要把 reserve() 跟新一下。因为在第一次子节点划分后， lNodes 内部元素已经增加了。此时要扩充 vector 的容量，防止后期动态内存分配

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                // 划分四个节点,分配关键点，内部没有处理不包含关键点的子节点
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                // 对于 4 个子节点，只有包含关键点，才会加入 lNode list里
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);  // 在前面插入节点
                    if(n1.vKeys.size()>1)   // 再次细分
                    {
                        nToExpand++; // 说明下次增加子节点时，至少会在增加一次
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);  // 一个父节点细分完毕就会自动擦除，进行当前层级的下一个父节点的细分
                continue;
            }
        }   // 根节点已经擦除完毕， lNodes 中仅仅剩下，第一次细分后的子节点

        // Finish if there are more nodes than required features
        // or all nodes contain just one point(此时 lNodes.size() == prevSize ，说明无法划分为子节点)
        // 因为 lNodes 中包含的节点内部都含有关键点。(没有关键点的不会加入 lNodes 中)，尽管可能每个节点内部的关键点个数大于 1 个
        // 但是总体来说，我们划分关键点的区域都是按照四叉树均匀划分的。只要保证总节点个数能够大于等于每层图像要求的关键点个数。我们就可以获得
        // 每层图像要求的关键点
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
            // 表示上次划分子节点完毕后，大概率在从新划分一次就能保证子节点个数大于我们需要在每层图像上获取的关键点数，
            // 当然我们直接进行上面的循环也可以达到要求，但是上面循环是从任意一个子节点开始划分的。没有考虑每个节点包含的关键点数
            // 此时我们为了加速划分结果。选择从包含关键点个数最多的节点处开始。依次递减。当划分好的子节点个数满足每层图像要求的关键点个数时，我们就可以停止了
        else if(((int)lNodes.size()+nToExpand*3)>N) // 说明在划分一次大概率能保证得到 N 个子节点。所以这里用了 while(!bFinish)
                                                    // (每次划分理想情况下，实际上会增加 3 个子节点。然后去掉 1 个父节点。)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();   // 当前有效节点个数

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                // 自己添加一句，扩充容量
                vSizeAndPointerToNode.reserve(lNodes.size() * 4);
                vSizeAndPointerToNode.clear();
                // 默认按照 pair 的第一个元素进行升序排序！（最后的元素对应节点包含的关键点个数最多，可以从那里开始细分子节点）
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);   // 细分（四叉树）当前节点

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    // 相当于在 lNodes 中擦除上层的父节点
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)   // 实际上很大概率，不用遍历完毕就能都等到大于 N 的子节点个数，这样减少了最后一次的遍历的时间！！符合我们要求选取关键点个数
                        break;
                }
                // 标记完成标志
                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        } // else if
    } // while(外层)

    // Retain the best point in each node
    // 在每个节点中选择响应值最大的关键点，代表该层图像需要的备选关键点
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures); // 这里保证 nfeatures 就够了，是因为上面的判断条件是，只要满足当前层要求的关键点个数，就退出。所以我们这里扩充要求的总关键点个数即可
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

//! \details 对金字塔图像中的每一层，首先对其划分一系列网格，然后在这些网格中分别提取 FAST 关键点。
//!      对这些关键点进行均匀选取，选取响应值比较大的关键点。然后对金字塔图像的所有层求出的所有关键点进行特征方向的计算
//! \note 这里虽然在每一层金字塔图像上进行均匀的选取关键点，但是不同层关键点的坐标相对的坐标系是不一样的。都是相对于降采样的图像坐标系
//! 恰好每个关键点都保留了自己所在的层数，可以反推会自己在原始图像坐标系下的坐标
//! 作者在 operator() 函数里面最后通过尺度恢复了降采样图像关键点在原始图像坐标系下的坐标
void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(nlevels);

    const float W = 30; // 这个值本身不能太小，否则在小区域内获取关键点 FAST-12 这种不太方便，其实需要每个小网格能够安全检测指定数量的关键点

    for (int level = 0; level < nlevels; ++level)
    {
        const int minBorderX = EDGE_THRESHOLD-3;    // 16   这里是 16 的原因，就是在算关键点的方向时是按照半径为 15 的圆来计算的。
                                                    // 为了保证能够在起始点是关键点情况下也能计算出来方向，这里选择 16
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3; // 这里的做法，默认 mvImagePyramid 存储的就是原始图像的降采样，没有加边界的东西。
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        vector<cv::KeyPoint> vToDistributeKeys; // 每一层的关键点
        vToDistributeKeys.reserve(nfeatures*10);    // 预保留关键点，这个值其实是一个大概的值。尽量大些保证下面插入数据时，不需要再次分配内存

        // 有效图像区域
        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        // 在有效图像区域上划分网格
        // 粗略统计大概能够有多少网格（按照每个格子 30 像素的间隔算） 这里会出现比如 20.3 = 20 ，此时说明有 0.3*30=9个像素余量
        const int nCols = width/W;
        const int nRows = height/W;
        // 通过粗略统计格子数，反推每个格子实际像素间隔，因为上面说了，会有 9 个余量，那么此时如果分为 nCols nRows 份数。那么每份会有多一些的像素
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        // 在该层图像上，对每一个网格(田字格)进行 FAST 特征提取。网格大小是 WxW
        // 每层图像进行大概 30 x 30 面积大小的网格进行 FAST 检测（尽可能多的检测），检测完毕后，后面会根据 四叉树方式划分关键点
        for(int i=0; i<nRows; i++)
        {
            // 对网格 y 方向有效区域处理
            const float iniY = minBorderY+i*hCell;  // 每个格子的初始 y 值
            float maxY = iniY+hCell+6;  // 每个格子中最大的 y 值（高度）
                                        // 但是这个 6 感觉加上没什么意义！！！
//            if(iniY > maxBorderY)
//                continue;
            if(iniY >= maxBorderY-3)    // 如果该条件成立，那么按照有效区域来说（去除了边界）无法按照定义来计算 FAST 关键点的。但是实际上这里没有什么意义。
                continue;               // 因为去除的边界本身就是自己图像上的东西。即使我这里直接用 iniY > maxBorderY 也是合适的！ 已经验证了！
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            for(int j=0; j<nCols; j++)
            {
                // 对网格 x 方有效区域处理
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;
//                if (iniX > maxBorderX)
//                    continue;
                if(iniX>=maxBorderX-6)    // 这里也同上
                    continue;
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                vector<cv::KeyPoint> vKeysCell;

                // 在指定图像区域内进行 FAST 角点检测，按照初始阈值 iniThFAST = 20
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                     vKeysCell,iniThFAST,true);

                if(vKeysCell.empty())
                {
                    // 降低阈值为 minThFAST 进行检测
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,minThFAST,true);
                }

                if(!vKeysCell.empty())
                {   // 计算特征点实际位置（相对于原始金字塔图像去除边界后的坐标系），
                    // 因为上面是按照网格作为一个图像来进行提取的，提取出的坐标是以网格为坐标系
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level]; // 每层金字塔图像提取出来的相应值比较大的关键点集
        keypoints.reserve(nfeatures);

        // 此时 vToDistributeKeys 中包含的关键点个数 和 当前层图像要求的特征数量存在三种关系 > = <
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {   // 把边界也考虑进去，获取关键点在金字塔图像（未去除边界）坐标系下的坐标，
            // 值得注意的是，关键点的坐标，仅仅是相对于金字塔每层的图像的原始坐标。并不全是相对于原始图像坐标系
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave=level;  // 关键点所在图像金字塔层数
            keypoints[i].size = scaledPatchSize;    // 不同金字塔层，对应 pathch 会差尺度因子
        }
    }

    // compute orientations,为了后面两幅图像匹配，以及计算特征点描述子做准备
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}
// 没有使用
void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        const int nCells = levelRows*levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;


        float hY = cellH + 6;

        for(int i=0; i<levelRows; i++)
        {
            const float iniY = minBorderY + i*cellH - 3;
            iniYRow[i] = iniY;

            if(i == levelRows-1)
            {
                hY = maxBorderY+3-iniY;
                if(hY<=0)
                    continue;
            }

            float hX = cellW + 6;

            for(int j=0; j<levelCols; j++)
            {
                float iniX;

                if(i==0)
                {
                    iniX = minBorderX + j*cellW - 3;
                    iniXCol[j] = iniX;
                }
                else
                {
                    iniX = iniXCol[j];
                }


                if(j == levelCols-1)
                {
                    hX = maxBorderX+3-iniX;
                    if(hX<=0)
                        continue;
                }


                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                FAST(cellImage,cellKeyPoints[i][j],iniThFAST,true);

                if(cellKeyPoints[i][j].size()<=3)
                {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                }


                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if(nKeys>nfeaturesCell)
                {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                }
                else
                {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell-nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }

            }
        }


        // Retain by score

        while(nToDistribute>0 && nNoMore<nCells)
        {
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
            nToDistribute = 0;

            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    if(!bNoMore[i][j])
                    {
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures*2);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates
        for(int i=0; i<levelRows; i++)
        {
            for(int j=0; j<levelCols; j++)
            {
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                if((int)keysCell.size()>nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);


                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    keysCell[k].octave=level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        if((int)keypoints.size()>nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // and compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,// 256 位
                               const vector<Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)   // 计算所有关键点对应的描述子
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

//! \param _mask 没有使用
//! \param _keypoints: 返回的关键点在原始坐标系下的坐标
//! \param _descriptors：关键点对应的描述子
//! \brief 提取关键点及其对应的描述子.
//! \details 且关键点索引号和描述子矩阵的行是一一对应的！（这个是重点！）
//!          对图像进行金字塔处理，每一层分别进行关键点和描述子的计算，
//!          最后统一所有关键点坐标为原始图像坐标系（这里面暗含的步骤，需要注意！）
void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors)
{
    if(_image.empty())
        return;

    Mat image = _image.getMat();    // 将 InputArray 类型转换为 cv::Mat 类型
    assert(image.type() == CV_8UC1 );   // 假定灰度图像

    // Pre-compute the scale pyramid
    // 对图像进行下采样，构建并保存金字塔图像
    ComputePyramid(image);

    vector < vector<KeyPoint> > allKeypoints;

    // 对每一层金字塔图像提取所有符合要求的关键点，之后返回所有层金字塔图像关键点总和.
    // 关键点坐标恰好是降采样图像坐标系下的坐标，需要恢复处在原始图像坐标系下的坐标（关键点间坐标可能会重复理论上不会重复，但是与计算精度有关）
    // 图像金字塔越高，图像越小，说明景物离当前相机的距离越远
    ComputeKeyPointsOctTree(allKeypoints);
    //ComputeKeyPointsOld(allKeypoints);

    Mat descriptors;

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release(); // 此时其实没有必要释放
    else
    {
        _descriptors.create(nkeypoints, 32, CV_8U); // 创建一个 （关键点数）行 x 32列，每列是一个 8 位的矩阵。这样其实描述子就是 32 x 8 = 256 维
        descriptors = _descriptors.getMat(); // 转换成 cv::Mat 类型,数据是引用的格式，修改 descriptors 其实就在修改 _descriptors
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;

    // 计算每层关键点对应的描述子，以及将降采样的图像关键点坐标转换到原始图像坐标系下
    // 这里计算的关键点[i]和描述子的行i 是一一对应的
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();    // 当前层关键点个数

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        Mat workingMat = mvImagePyramid[level].clone();

        // 此时是自己根据 Size(7,7) 2,2自己计算高斯核，然后进行图像的高斯模糊，具体如何进行自定义计算高斯核:
        // 可以参考：https://www.cnblogs.com/tornadomeet/archive/2012/03/10/2389617.html
        // 实际上也需要很多时间！待测量！
        // 这里在计算描述子之前模糊图像是说降低噪声的影响
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors  // 内部包含了多层图像的描述子，从上到下依次存储每层的描述子
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);

        // desc 返回当前层所有关键点对应的描述子
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;  // 下一层图像的描述子对应的行

        // Scale keypoint coordinates
        // 恢复第二层及以后图像关键点在原始图像坐标系下的坐标
        if (level != 0)
        {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;  // 恢复第二层及以后图像关键点在原始图像坐标系下的坐标
        }
        // And add the keypoints to the output  在关键点最后插入新的关键点，多层图像对应的关键点依次排在在后面，然后与描述子的行是一一对应的
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}
//! \brief 对输入图像进行下采样
//! \question 这里有一个问题，虽然把图像复制到了 temp 中，但是 mvImagePyramid[] 仍然指向对应降采样的数据。对于边界的数据没有直接指向
//! 该函数下面是简化版的计算金字塔函数，经过测试发现这里跟猜想的一致。在这个函数内部，他做的一些动作，其实最后 mvImagePyramid 都是指向
//! 降采样的数据。并没有包含一些边界扩充值，可能他在操作提取特征时，实际上忽略了边界的元素。当然他这里实现的方式冗余，或者他想用其他方法。
//! \note 一个简化版本在下面
void ORBextractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));    // 变化的尺度的图像
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        Mat temp(wholeSize, image.type()), masktemp;
        // 这里实际上就是让 mvImagePyramid[level] 指向了 temp() 内部元素，即使 temp 超出范围，这里的指向的数据也不会释放
        mvImagePyramid[level] = temp( Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height)) ;   // Rect 功能是(x,y,width,height)，
                                                                                                    // 以左上角起始坐标 (x,y)，然后行取 width 个
                                                                                                    // 列取 height 个，构成的一个矩阵

        // Compute the resized image
        if( level != 0 )
        {   // 将前一帧图像降采样到下一帧图像（这里的操作实际上都是在原始图像开始的，依次进行下采样）
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR); // [level-1] 图像降采样为 [level]
            // 边界扩充，因为 mvImagePyramid[level] 在 temp 中，此时默认在 mvImagePyramid[level]开始扩充边界，
            // 但是边界值，他会自动使用 mvImagePyramid[level] 边上的值直接作为边界值，并不进行边界值推断，那么此时就不符合要求了，因为我们想让其
            // 以 mvImagePyramid[level] 图像来推断边界，那么一个方法就是后面加上 BORDER_ISOLATED ，此时就是推断类型了。而不是直接拿 mvImagePyramid[level] 边上的值作为边界
            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);     // 超出枚举
        }
        else
        {   // 复制原始图像到 temp 中，并进行边界值的扩充(因为 temp 元素比原始 image 图像要大)
            // 因为 image 不是 temp 的子元素,所以这里只是把 image 复制到 temp 里面,然后在 temp 中填充边界值
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
        }
    }

}

// 简化版本：计算金字塔
//void ORBextractor::ComputePyramid_brief(cv::Mat image) {
//    for (int level = 0; level < nlevels; ++level) {
//        float scale = mvInvScaleFactor[level];
//        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
//
//        // Compute the resized image
//        if (level != 0) {
//            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
//        } else {
//            mvImagePyramid[level] = image;
//        }
//    }
//}

} //namespace ORB_SLAM
