#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pclomp/ndt_omp.h>
#include <pcl/filters/voxel_grid.h>

#include <hdl_localization/pose_system.hpp>
#include <kalman/unscented_kalman_filter.hpp>
#include <kalman/cubature_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZI;

  /**
   * @brief constructor
   * @param registration        registration method
   * @param stamp               timestamp
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const ros::Time& stamp, const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration = 1.0)
    : init_stamp(stamp),
      registration(registration),
      cool_time_duration(cool_time_duration)
  {
    process_noise = Eigen::MatrixXf::Identity(16, 16);
    process_noise.middleRows(0, 3) *= 1.0;
    process_noise.middleRows(3, 3) *= 1.0;
    process_noise.middleRows(6, 4) *= 0.5;
    process_noise.middleRows(10, 3) *= 1e-6;
    process_noise.middleRows(13, 3) *= 1e-6;

    Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
    measurement_noise.middleRows(0, 3) *= 0.01;
    measurement_noise.middleRows(3, 4) *= 0.001;

    Eigen::VectorXf mean(16);
    mean.middleRows(0, 3) = pos;
    mean.middleRows(3, 3).setZero();
    mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
    mean.middleRows(10, 3).setZero();
    mean.middleRows(13, 3).setZero();

    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

    PoseSystem system;
    ukf.reset(new UnscentedKalmanFilterX<float, PoseSystem>(system, 16, 6, 7, process_noise, measurement_noise, mean, cov));
  }

  /**
   * @brief predict
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   */
  void predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
    if((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
      prev_stamp = stamp;
      return;
    }

    double dt = (stamp - prev_stamp).toSec();
    prev_stamp = stamp;

    ukf->setProcessNoiseCov(process_noise * dt);
    ukf->getSystem().dt = dt;

    Eigen::VectorXf control(6);
    control.head<3>() = acc;
    control.tail<3>() = gyro;

    ukf->predict(control);
  }

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    init_guess.block<3, 3>(0, 0) = quat().toRotationMatrix();
    init_guess.block<3, 1>(0, 3) = pos();

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    registration->setInputSource(cloud);
    registration->align(*aligned, init_guess);

    Eigen::Matrix4f trans = registration->getFinalTransformation();
    Eigen::Vector3f p = trans.block<3, 1>(0, 3);
    Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

    if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
      q.coeffs() *= -1.0f;
    }

    Eigen::VectorXf observation(7);
    observation.middleRows(0, 3) = p;
    observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());

    ukf->correct(observation);
    return aligned;
  }

  /* getters */
  Eigen::Vector3f pos() const {
    auto mean = ukf->getMean();
    return Eigen::Vector3f(mean[0], mean[1], mean[2]);
  }

  Eigen::Vector3f vel() const {
    auto mean = ukf->getMean();
    return Eigen::Vector3f(mean[3], mean[4], mean[5]);
  }

  Eigen::Quaternionf quat() const {
    auto mean = ukf->getMean();
    return Eigen::Quaternionf(mean[6], mean[7], mean[8], mean[9]).normalized();
  }

  Eigen::Matrix4f matrix() const {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 3>(0, 0) = quat().toRotationMatrix();
    m.block<3, 1>(0, 3) = pos();
    return m;
  }

private:
  ros::Time init_stamp;         // when the estimator was initialized
  ros::Time prev_stamp;         // when the estimator was updated last time
  double cool_time_duration;    //

  Eigen::MatrixXf process_noise;
  std::unique_ptr<UnscentedKalmanFilterX<float, PoseSystem>> ukf;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}

#endif // POSE_ESTIMATOR_HPP
