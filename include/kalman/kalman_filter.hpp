#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <Eigen/Dense>

template<typename T, class System>
class KalmanFilter
{
typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;    //列向量
typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:

    KalmanFilter(const System& _system, int _state_dim, int _input_dim, int _measurement_dim, 
                 const MatrixXt& _process_noise, const MatrixXt& _measurement_noise, 
                 const VectorXt& _mean, const MatrixXt& _cov):
        system(_system),
        state_dim(_state_dim),
        input_dim(_input_dim),
        measurement_dim(_measurement_dim),
        N(_state_dim),
        M(_measurement_dim),
        mean(_mean),
        cov(_cov),
        process_noise(_process_noise),
        measurement_noise(_measurement_noise)
    {

    }

    ~KalmanFilter()
    {
    }

    virtual void predict(const VectorXt& control) = 0;
    virtual void correct(const VectorXt& measurement) = 0;
    
    virtual const VectorXt& getMean() const { return mean; }
    virtual const MatrixXt& getCov() const { return cov; }

    virtual System& getSystem() { return system; }
    virtual const System& getSystem() const { return system; }
    virtual const MatrixXt& getProcessNoiseCov() const { return process_noise; }
    virtual const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }
    virtual const MatrixXt& getKalmanGain() const { return kalman_gain; }

    /*			setter			*/
    virtual void setMean(const VectorXt& m) { mean = m;}
    virtual void setCov(const MatrixXt& s) { cov = s;}

    virtual void setProcessNoiseCov(const MatrixXt& p) { process_noise = p;}
    virtual void setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;}

    /**
    * @brief make covariance matrix positive finite
    * @param cov  covariance matrix
    */
    virtual void ensurePositiveFinite(MatrixXt& cov) 
    {
        return;
        const double eps = 1e-9;

        Eigen::EigenSolver<MatrixXt> solver(cov);
        MatrixXt D = solver.pseudoEigenvalueMatrix();
        MatrixXt V = solver.pseudoEigenvectors();
        for (int i = 0; i < D.rows(); i++) 
        {
            if (D(i, i) < eps) 
            D(i, i) = eps;
        }

        cov = V * D * V.inverse();
    }

public:
    const int state_dim,       N; //状态向量维度
    const int input_dim;
    const int measurement_dim, M; //测量向量维度

    VectorXt mean;                //均值
    MatrixXt cov;                 //协方差

    System system;                //控制系统
    MatrixXt process_noise;		  //过程噪声 Q
    MatrixXt measurement_noise;	  //测量噪声 R

    MatrixXt kalman_gain;         //卡尔曼增益K

};

#endif