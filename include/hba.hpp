#ifndef HBA_HPP
#define HBA_HPP

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
//#include <visualization_msgs/Marker.h>
//#include <visualization_msgs/MarkerArray.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "mypcl.hpp"
#include "tools.hpp"
#include "ba.hpp"

class LAYER
{
public:
  int pose_size;
  int layer_num;
  int max_iter;
  int part_length;
  int left_size;
  int left_h_size;
  int j_upper;
  int tail;
  int thread_num;
  int gap_num;
  int last_win_size;
  int left_gap_num;
  double downsample_size;
  double voxel_size;
  double eigen_ratio;
  double reject_ratio;

  std::string data_path;
  vector<mypcl::pose> pose_vec;
  std::vector<thread*> mthreads;
  std::vector<double> mem_costs;
  std::vector<Eigen::Matrix<double, 6, 1>> hessians;
  std::vector<pcl::PointCloud<PointType>::Ptr> pcds;

  // Default constructor initializes member variables to default values
  LAYER() {
    pose_size = 0;
    layer_num = 1;
    max_iter = 10;
    downsample_size = 0.1;
    voxel_size = 4.0;
    eigen_ratio = 0.1;
    reject_ratio = 0.05;
    
    // Clear all vectors
    pose_vec.clear(); 
    mthreads.clear(); 
    pcds.clear();
    hessians.clear(); 
    mem_costs.clear();
  }

  // Initializes storage vectors based on layer parameters
  void init_storage(int total_layer_num) {
    mthreads.resize(thread_num);
    mem_costs.resize(thread_num);

    pcds.resize(pose_size);
    pose_vec.resize(pose_size);

    #ifdef FULL_HESS
    // Calculate hessian size once
    int hessian_size;
    if (layer_num < total_layer_num) {
      // Avoid repeated calculations by storing intermediate results
      const int win_size_factor = (WIN_SIZE - 1) * WIN_SIZE / 2;
      hessian_size = (thread_num - 1) * win_size_factor * part_length;
      hessian_size += win_size_factor * left_gap_num;
      
      if (tail > 0) {
        hessian_size += (last_win_size - 1) * last_win_size / 2;
      }

    } else {
      hessian_size = pose_size * (pose_size - 1) / 2;
      std::cout << "hessian_size: " << hessian_size << std::endl;
    }

    hessians.reserve(hessian_size);
    hessians.resize(hessian_size);
    std::cout << "hessian_size: " << hessian_size << std::endl;
    #endif
  }

  // Initializes layer parameters based on pose size
  void init_parameter(int pose_size = 0) {
    // Set pose size based on layer number
    this->pose_size = (layer_num == 1) ? pose_vec.size() : pose_size;

    // Calculate key parameters
    tail = (this->pose_size - WIN_SIZE) % GAP;
    gap_num = (this->pose_size - WIN_SIZE) / GAP;
    last_win_size = this->pose_size - GAP * (gap_num + 1);

    // Calculate initial part length
    part_length = ceil((gap_num + 1) / static_cast<double>(thread_num));
    if (gap_num - (thread_num - 1) * part_length < 0) {
      part_length = floor((gap_num + 1) / static_cast<double>(thread_num));
    }

    // Adjust thread count and part length if needed
    const double max_ratio = 2.0;
    while (part_length == 0 || 
           (gap_num - (thread_num - 1) * part_length + 1) / static_cast<double>(part_length) > max_ratio) {
      thread_num--;
      part_length = ceil((gap_num + 1) / static_cast<double>(thread_num));
      if (gap_num - (thread_num - 1) * part_length < 0) {
        part_length = floor((gap_num + 1) / static_cast<double>(thread_num));
      }
    }

    // Calculate remaining parameters
    left_gap_num = gap_num - (thread_num - 1) * part_length + 1;

    if (tail == 0) {
      const int offset = gap_num - (thread_num - 1) * part_length;
      left_size = (offset + 1) * WIN_SIZE;
      left_h_size = offset * GAP + WIN_SIZE - 1;
      j_upper = offset + 1;
    } else {
      const int offset = gap_num - (thread_num - 1) * part_length + 1;
      left_size = offset * WIN_SIZE + GAP + tail;
      left_h_size = offset * GAP + last_win_size - 1;
      j_upper = offset + 1;
    }

    // Log parameters
    printf("init parameter:\n");
    printf("layer_num %d | thread_num %d | pose_size %d | max_iter %d | part_length %d | "
           "gap_num %d | last_win_size %d | left_gap_num %d | tail %d | left_size %d | "
           "left_h_size %d | j_upper %d | downsample_size %f | voxel_size %f | "
           "eigen_ratio %f | reject_ratio %f\n",
           layer_num, thread_num, this->pose_size, max_iter, part_length, gap_num, last_win_size,
           left_gap_num, tail, left_size, left_h_size, j_upper,
           downsample_size, voxel_size, eigen_ratio, reject_ratio);
  }
};


class HBA
{
public:
  int thread_num;
  int total_layer_num;
  std::vector<LAYER> layers;
  std::string data_path;

  HBA(int total_layer_num_, std::string data_path_, int thread_num_)
  {
    total_layer_num = total_layer_num_;
    thread_num = thread_num_;
    data_path = data_path_;

    layers.resize(total_layer_num);
    for(int i = 0; i < total_layer_num; i++)
    {
      layers[i].layer_num = i+1;
      layers[i].thread_num = thread_num;
    }
    layers[0].data_path = data_path;
    layers[0].pose_vec = mypcl::read_pose(data_path + "pose.json");
    layers[0].init_parameter();
    layers[0].init_storage(total_layer_num);

    for (int i = 1; i < total_layer_num; i++) {
      // Calculate pose size for current layer based on previous layer
      int pose_size_ = (layers[i-1].thread_num - 1) * layers[i-1].part_length;
      pose_size_ += layers[i-1].tail == 0 ? 
                   layers[i-1].left_gap_num : 
                   (layers[i-1].left_gap_num + 1);

      // Initialize layer parameters and storage
      layers[i].init_parameter(pose_size_);
      layers[i].init_storage(total_layer_num);
      layers[i].data_path = layers[i-1].data_path + "process1/";
    }
    std::cout << "HBA init done!" << std::endl;
  }

  void update_next_layer_state(int cur_layer_num) {
    const auto& cur_layer = layers[cur_layer_num];
    auto& next_layer = layers[cur_layer_num + 1];
    const int part_length = cur_layer.part_length;
    const int thread_num = cur_layer.thread_num;

    for (int i = 0; i < thread_num; i++) {
      const int loop_limit = (i < thread_num - 1) ? part_length : cur_layer.j_upper;
      
      for (int j = 0; j < loop_limit; j++) {
        const int src_index = (i * part_length + j) * GAP;
        const int dst_index = i * part_length + j;
        next_layer.pose_vec[dst_index] = cur_layer.pose_vec[src_index];
      }
    }
  }

  void pose_graph_optimization() {
    // Get poses and covariances from initial and final layers
    std::vector<mypcl::pose> upper_pose = layers[total_layer_num-1].pose_vec;
    std::vector<mypcl::pose> init_pose = layers[0].pose_vec;
    std::vector<Eigen::Matrix<double, 6, 1>> upper_cov = layers[total_layer_num-1].hessians;
    std::vector<Eigen::Matrix<double, 6, 1>> init_cov = layers[0].hessians;

    // Initialize pose graph
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    
    // Add prior factor on first pose
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
    auto priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);
    
    gtsam::Pose3 firstPose(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), 
                          gtsam::Point3(init_pose[0].t));
    initial.insert(0, firstPose);
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, firstPose, priorModel));

    // Add factors between poses in initial layer
    int cnt = 0;
    for(size_t i = 0; i < init_pose.size(); i++) {
      if(i > 0) {
        initial.insert(i, gtsam::Pose3(gtsam::Rot3(init_pose[i].q.toRotationMatrix()),
                                     gtsam::Point3(init_pose[i].t)));
      }

      if(i % GAP == 0 && cnt < init_cov.size()) {
        for(int j = 0; j < WIN_SIZE-1; j++) {
          for(int k = j+1; k < WIN_SIZE; k++) {
            if(i+j+1 >= init_pose.size() || i+k >= init_pose.size()) break;

            cnt++;
            if(init_cov[cnt-1].norm() < 1e-20) continue;

            // Calculate relative transform between poses
            Eigen::Vector3d t_ab = init_pose[i+j].t;
            Eigen::Matrix3d R_ab = init_pose[i+j].q.toRotationMatrix();
            t_ab = R_ab.transpose() * (init_pose[i+k].t - t_ab);
            R_ab = R_ab.transpose() * init_pose[i+k].q.toRotationMatrix();

            // Create noise model from covariance
            Vector6 << fabs(1.0/init_cov[cnt-1](0)), fabs(1.0/init_cov[cnt-1](1)), 
                      fabs(1.0/init_cov[cnt-1](2)), fabs(1.0/init_cov[cnt-1](3)), 
                      fabs(1.0/init_cov[cnt-1](4)), fabs(1.0/init_cov[cnt-1](5));
            auto odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

            // Add between factor
            graph.push_back(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr(
              new gtsam::BetweenFactor<gtsam::Pose3>(
                i+j, i+k, gtsam::Pose3(gtsam::Rot3(R_ab), gtsam::Point3(t_ab)), odometryNoise)));
          }
        }
      }
    }

    // Add factors between poses in final layer
    cnt = 0;
    const int pose_size = upper_pose.size();
    const int gap_power = pow(GAP, total_layer_num-1);
    
    for(int i = 0; i < pose_size-1; i++) {
      for(int j = i+1; j < pose_size; j++) {
        cnt++;
        if(upper_cov[cnt-1].norm() < 1e-20) continue;

        // Calculate relative transform
        Eigen::Vector3d t_ab = upper_pose[i].t;
        Eigen::Matrix3d R_ab = upper_pose[i].q.toRotationMatrix();
        t_ab = R_ab.transpose() * (upper_pose[j].t - t_ab);
        R_ab = R_ab.transpose() * upper_pose[j].q.toRotationMatrix();

        // Create noise model
        Vector6 << fabs(1.0/upper_cov[cnt-1](0)), fabs(1.0/upper_cov[cnt-1](1)),
                   fabs(1.0/upper_cov[cnt-1](2)), fabs(1.0/upper_cov[cnt-1](3)),
                   fabs(1.0/upper_cov[cnt-1](4)), fabs(1.0/upper_cov[cnt-1](5));
        auto odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

        // Add between factor
        graph.push_back(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr(
          new gtsam::BetweenFactor<gtsam::Pose3>(
            i*gap_power, j*gap_power,
            gtsam::Pose3(gtsam::Rot3(R_ab), gtsam::Point3(t_ab)), 
            odometryNoise)));
      }
    }

    // Optimize pose graph
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 isam(parameters);
    isam.update(graph, initial);
    isam.update();

    // Get optimized poses
    gtsam::Values results = isam.calculateEstimate();
    std::cout << "vertex size " << results.size() << std::endl;

    // Update initial poses with optimized results
    for(size_t i = 0; i < results.size(); i++) {
      gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
      assign_qt(init_pose[i].q, init_pose[i].t,
                Eigen::Quaterniond(pose.rotation().matrix()), 
                pose.translation());
    }

    // Save optimized poses
    mypcl::write_pose(init_pose, data_path);
    std::cout << "pgo complete" << std::endl;
  }
};

#endif