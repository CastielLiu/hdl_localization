/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>
#include <kalman/kalman_filter.hpp>

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilterX  : public KalmanFilter<T, System>
{
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;    //列向量
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
  using KalmanFilter<T, System>::state_dim;
  using KalmanFilter<T, System>::N;
  using KalmanFilter<T, System>::input_dim;
  using KalmanFilter<T, System>::measurement_dim;
  using KalmanFilter<T, System>::M;
  using KalmanFilter<T, System>::mean;
  using KalmanFilter<T, System>::cov;
  using KalmanFilter<T, System>::system;
  using KalmanFilter<T, System>::process_noise;
  using KalmanFilter<T, System>::measurement_noise;
  using KalmanFilter<T, System>::kalman_gain;
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
  UnscentedKalmanFilterX(const System& _system, int _state_dim, int _input_dim, int _measurement_dim, 
                         const MatrixXt& _process_noise, const MatrixXt& _measurement_noise, 
                         const VectorXt& _mean, const MatrixXt& _cov):
    KalmanFilter<T, System>(_system, _state_dim, _input_dim, _measurement_dim, _process_noise, _measurement_noise, _mean, _cov),
    S(2 * state_dim + 1),
    lambda(1)
  {
    weights.resize(S, 1);
    sigma_points.resize(S, N);
    ext_weights.resize(2 * (N + M) + 1, 1);
    ext_sigma_points.resize(2 * (N + M) + 1, N + M);
    expected_measurements.resize(2 * (N + M) + 1, M);

    // initialize weights for unscented filter
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    ext_weights[0] = lambda / (N + M + lambda);
    for (int i = 1; i < 2 * (N + M) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + M + lambda));
    }
  }

  /**
   * @brief predict  预测函数
   * @param control  input vector
   */
  virtual void predict(const VectorXt& control) override
  {
    // calculate sigma points
    this->ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points); //根据上一时刻的均值和方差计算sigma点
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i), control); //根据系统方程传播sigma点
    }

    const auto& Q = process_noise; //系统噪声|过程噪声

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);   //传播后的sigma点集均值
    }
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose(); //传播后的sigma点集方差
    }
    cov_pred += Q;                                      //加上过程噪声

    //得到预测值和预测协方差
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct      校正函数
   * @param measurement  观测值
   */
  virtual void correct(const VectorXt& measurement) override
  {
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(N + M, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + M, N + M);
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    ext_cov_pred.bottomRightCorner(M, M) = measurement_noise;

    this->ensurePositiveFinite(ext_cov_pred);
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points); //根据预测均值和协方差以及测量噪声计算sigma点
                                                                       //此时测量误差并未添加到sigama主体,而是存放于拓展部分

    // unscented transform
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));     //观测方程传播sigama点集
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(M, 1));//添加测量噪声
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(M);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);  //传播后的sigama点集均值
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(M, M);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();        //传播后的sigama点集协方差
    }

    // calculated transformed covariance
    MatrixXt sigma = MatrixXt::Zero(N + M, M);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    kalman_gain = sigma * expected_measurement_cov.inverse();                       //计算卡尔曼增益

    VectorXt ext_mean = ext_mean_pred + kalman_gain * (measurement - expected_measurement_mean); //最优估计
    MatrixXt ext_cov = ext_cov_pred - kalman_gain * expected_measurement_cov * kalman_gain.transpose();    //最优估计的协方差

    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  const MatrixXt& getSigmaPoints() const { return sigma_points; }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  T lambda;
  const int S; //sigma点个数
  VectorXt weights;
  MatrixXt sigma_points;
  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt;   //Cholesky分解，将对称正定矩阵表示成一个下三角矩阵L和其转置的乘积的分解 M = L*L.transpose()
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }
};


#endif
