/**
 * cubature_kalman_filter.hpp
 * @author wendao
 * 20/12/6
 **/
#ifndef KKL_CUBATURE_KALMAN_FILTER_X_HPP
#define KKL_CUBATURE_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
  namespace alg {

/**
 * @brief Cubature Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class CubatureKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;    //列向量
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  CubatureKalmanFilterX(const System& system, int state_dim, int input_dim, int measurement_dim, const MatrixXt& process_noise, const MatrixXt& measurement_noise, const VectorXt& mean, const MatrixXt& cov)
    : state_dim(state_dim),
    input_dim(input_dim),
    measurement_dim(measurement_dim),
    N(state_dim),
    M(input_dim),
    K(measurement_dim),
    S(2 * state_dim),
    mean(mean),
    cov(cov),
    system(system),
    process_noise(process_noise),
    measurement_noise(measurement_noise),
    normal_dist(0.0, 1.0)
  {
    weights.resize(S, 1);
    cubature_points.resize(S, N);
    ext_weights.resize(2*(N+K), 1);
    ext_cubature_points.resize(2*(N+K), N + K);
    expected_measurements.resize(2*(N+K), K);

    // initialize weights for cubature filter
    for (int i = 0; i < S; i++) {
      weights[i] = 1.0 / S;
    }

    // weights for extended state space which includes error variances
    for (int i = 0; i < 2*(N+K); i++) {
      ext_weights[i] = 1.0 / 2*(N+K);
    }
  }

  /**
   * @brief predict  预测函数
   * @param control  input vector
   */
  void predict(const VectorXt& control) {
    // calculate cubature points
    ensurePositiveFinite(cov);
    computeCubaturePoints(mean, cov, cubature_points); //根据上一时刻的均值和方差计算cubature点
    for (int i = 0; i < S; i++) {
      cubature_points.row(i) = system.f(cubature_points.row(i), control); //根据系统方程传播cubature点
    }

    const auto& Q = process_noise; //系统噪声|过程噪声

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * cubature_points.row(i);   //传播后的cubature点集均值
    }
    for (int i = 0; i < S; i++) 
      cov_pred += weights[i] * cubature_points.row(i).transpose() * cubature_points.row(i);
    cov_pred -= mean_pred.transpose() * mean_pred; //传播后的cubature点集方差
    cov_pred += Q;                                      //加上过程噪声

    //得到预测值和预测协方差
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct      校正函数
   * @param measurement  观测值
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;

    ensurePositiveFinite(ext_cov_pred);
    computeCubaturePoints(ext_mean_pred, ext_cov_pred, ext_cubature_points); //根据预测均值和协方差以及测量噪声计算cubature点
                                                                             //此时测量误差并未添加到cubature主体,而是存放于拓展部分

    // cubature transform
    expected_measurements.setZero();
    for (int i = 0; i < ext_cubature_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_cubature_points.row(i).transpose().topLeftCorner(N, 1));     //观测方程传播cubature点集
      expected_measurements.row(i) += VectorXt(ext_cubature_points.row(i).transpose().bottomRightCorner(K, 1));//添加测量噪声
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_cubature_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);  //传播后的cubature点集均值
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_cubature_points.rows(); i++)
      expected_measurement_cov += ext_weights[i] * expected_measurements.row(i).transpose() * expected_measurements.row(i);        

    expected_measurement_cov -= expected_measurement_mean.transpose() * mean_pred; //传播后的cubature点集方差
    expected_measurement_cov += measurement_noise;  //R = measurement_noise

    // calculated transformed covariance
    MatrixXt cross_cov = MatrixXt::Zero(N, K); //互协方差
    for(int i=0; i<S; ++i)
      cross_cov += ext_weights[i] * ext_cubature_points.row(i).transpose() * expected_measurements.row(i)
    cross_cov -= ext_mean_pred * expected_measurement_mean;

    kalman_gain = cross_cov * cross_cov.inverse(); //卡尔曼增益
    const auto& K = kalman_gain;

    VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean); //最优估计
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();    //最优估计的协方差

    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return cubature_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  CubatureKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  CubatureKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  CubatureKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  CubatureKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;  //状态向量维度
  const int M;  //控制向量维度
  const int K;  //测量向量维度
  const int S;  //采样点个数

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;		  //过程噪声 Q
  MatrixXt measurement_noise;	//测量噪声 R

  VectorXt weights;
  MatrixXt cubature_points;

  VectorXt ext_weights;
  MatrixXt ext_cubature_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute cubature points
   * @param mean          mean
   * @param cov           covariance
   * @param cubature_points  calculated cubature points
   */
  void computeCubaturePoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& cubature_points) {
    const int n = mean.size(); //状态维度
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt(cov);
    MatrixXt P_chol = llt.matrixL();
    MatrixXt l = P_chol * sqrt(n);

    for (int i = 0; i < n; i++) {
      cubature_points.row(  i) = mean + l.col(i);
      cubature_points.row(n+i) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    return;
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt D = solver.pseudoEigenvalueMatrix();
    MatrixXt V = solver.pseudoEigenvectors();
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }

    cov = V * D * V.inverse();
  }

public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

  }
}


#endif
