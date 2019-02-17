/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include "MapPoint.h"
#include "Frame.h"

namespace ORB_SLAM2
{

class PnPsolver {
 public:
  PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches);

  ~PnPsolver();

  void SetRansacParameters(double probability = 0.99, int minInliers = 8 , int maxIterations = 300, int minSet = 4, float epsilon = 0.4,
                           float th2 = 5.991);

  cv::Mat find(vector<bool> &vbInliers, int &nInliers);

  cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

 private:

  void CheckInliers();
  bool Refine();

  // Functions from the original EPnP code
  void set_maximum_number_of_correspondences(const int n);
  void reset_correspondences(void);
  void add_correspondence(const double X, const double Y, const double Z,
              const double u, const double v);

  double compute_pose(double R[3][3], double T[3]);

  void relative_error(double & rot_err, double & transl_err,
              const double Rtrue[3][3], const double ttrue[3],
              const double Rest[3][3],  const double test[3]);

  void print_pose(const double R[3][3], const double t[3]);
  double reprojection_error(const double R[3][3], const double t[3]);

  void choose_control_points(void);
  void compute_barycentric_coordinates(void);
  void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);
  void compute_ccs(const double * betas, const double * ut);
  void compute_pcs(void);

  void solve_for_sign(void);

  void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void qr_solve(CvMat * A, CvMat * b, CvMat * X);

  double dot(const double * v1, const double * v2);
  double dist2(const double * p1, const double * p2);

  void compute_rho(double * rho);
  void compute_L_6x10(const double * ut, double * l_6x10);

  void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);
  void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
				    double cb[4], CvMat * A, CvMat * b);

  double compute_R_and_t(const double * ut, const double * betas,
			 double R[3][3], double t[3]);

  void estimate_R_and_t(double R[3][3], double t[3]);

  void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
		    double R_src[3][3], double t_src[3]);

  void mat_to_quat(const double R[3][3], double q[4]);


  double uc, vc, fu, fv; // fu = fx, fv = fy, uc = cx, vc = cy

  double * pws, * us, * alphas, * pcs; // init = 0 pws:存储 mRansacMinSet = 4 个世界坐标系下的点坐标。顺序存储点坐标 x y z
                                       //          us : 存储与上面 pws 中世界坐标点对应的图像坐标点坐标。
                                       //          alphas: 对应论文中计算的 (1) 式的 alphas_ij 。具体计算在 compute_barycentric_coordinates()
                                       //                   alphas[4 * number_of_correspondences] 每个世界参考点都会有 4 个 alphas 参数
                                       //          pcs: 3 * number_of_correspondences: 存储 4 个相机坐标系下点的参考点坐标。根据 alphas 以及相机坐标系的 4 个控制点 ccs
  int maximum_number_of_correspondences; // init = 0，之后为： mRansacMinSet = 4
  int number_of_correspondences; // init = 0 在 add_correspondence() 中增加的，经过 4 次调用，后该值 为 4

  double cws[4][3], ccs[4][3]; // cws： 保存任意 4 个世界坐标点的计算出来的控制点。
                               // ccs: 相机坐标系下的 4 个控制点
  double cws_determinant;

  vector<MapPoint*> mvpMapPointMatches; // [i] = pMP; i: 表示图像关键点序号 pMP :表示图像关键点对应的 3d 地图点，构成 3d-2d 匹配对

  // 2D Points ( size = Frame 有效的关键点(有对应的匹配的点对))
  vector<cv::Point2f> mvP2D; // 输入的 Frame 图像的有对应地图点的关键点
  vector<float> mvSigma2; // size = N,尺度因子相关的： mvScaleFactor[i]*mvScaleFactor[i]; i 已经对应 mvp2D 中的关键点所属金字塔层数

  // 3D Points
  vector<cv::Point3f> mvP3Dw; // 直接存储的 3d 坐标点(与 2D 图像坐标直接匹配的)

  // Index in Frame
  vector<size_t> mvKeyPointIndices; // 存储有效的关键点序号(对应 Frame 图像的),通过 这个值调用 size() 即可知道当前有效的 3d-2d 点对多少个！

  // Current Estimation
  double mRi[3][3];
  double mti[3];
  cv::Mat mTcwi;
  vector<bool> mvbInliersi; // size = N
  int mnInliersi; // init = 0 ,在一次计算 Rt 后验证得到的内点个数

  // Current Ransac State(每次 Ransac 迭代时，当次迭代状态)
  int mnIterations; // init = 0,记录当前迭代次数
  vector<bool> mvbBestInliers; // 记录迭代过程中最好的内点 bool 判断 N
  int mnBestInliers; // init = 0 , 记录迭代中最好的内点个数，与上面这个变量对应
  cv::Mat mBestTcw; // 迭代过程中最好的 R t

  // Refined
  cv::Mat mRefinedTcw; // 提炼好的 r t
  vector<bool> mvbRefinedInliers;
  int mnRefinedInliers; // 根据最好的内点个数。再次进行求解 r t ，然后检查内点。此时得到的内点个数

  // Number of Correspondences
  int N; // init = 有效匹配点对个数(Frame 可以找到匹配地图点的对数)

  // Indices for random selection [0 .. N-1]
  vector<size_t> mvAllIndices; // 用来随机选择 0- mvP3Dw.size()  :内部存储的是有效的匹配点对数量

  // RANSAC probability
  double mRansacProb; // init = 0.99

  // RANSAC min inliers
  int mRansacMinInliers; // init = { N*mRansacEpsilon, minInliers=8 , minSet=4}
                         // 如果 > 8 直接是 N*mRansacEpsilon，如果 4 < N*... <8 则是 8 如果 < 4, 则是 4

  // RANSAC max iterations
  int mRansacMaxIts; // init = 300, 在设置参数 SetRansacParameters 中仍然会调节，这里可能不是 300

  // RANSAC expected inliers/total ratio
  float mRansacEpsilon; // init = 0.4 ,之后在设置参数 PnPsolver::SetRansacParameters 函数内部. = max{0.4, mRansacMinInliers/N}

  // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
  float mRansacTh;

  // RANSAC Minimun Set used at each iteration
  int mRansacMinSet; // 默认 = minSet =4 在设置参数函数中设置

  // Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
  vector<float> mvMaxError; // size = N,有效内点个数。目前存储的是： sigma(level)*th(5.991)

};

} //namespace ORB_SLAM

#endif //PNPSOLVER_H
