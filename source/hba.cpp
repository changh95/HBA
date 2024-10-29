#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
//#include <sensor_msgs/Imu.h>
//#include <sensor_msgs/PointCloud2.h>
//#include <geometry_msgs/PoseArray.h>
//#include <tf/transform_broadcaster.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl_conversions/pcl_conversions.h>

#include "ba.hpp"
#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"
#include "cxxopts.hpp"

using namespace std;
using namespace Eigen;

int pcd_name_fill_num = 5;

void cut_voxel(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map,
               const pcl::PointCloud<PointType>& feat_pt,
               const Eigen::Quaterniond& q, const Eigen::Vector3d& t, 
               const int fnum, const double voxel_size, 
               const int window_size, const float eigen_ratio)
{
  for(const auto& point : feat_pt.points)
  {
    const Eigen::Vector3d pvec_orig(point.x, point.y, point.z);
    const Eigen::Vector3d pvec_tran = q * pvec_orig + t;

    std::array<float, 3> loc_xyz;
    for(int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    const VOXEL_LOC position(static_cast<int64_t>(loc_xyz[0]), 
                            static_cast<int64_t>(loc_xyz[1]), 
                            static_cast<int64_t>(loc_xyz[2]));

    auto [iter, inserted] = feat_map.try_emplace(position, nullptr);
    if(!inserted) 
    {
      // Existing voxel
      auto* tree = iter->second;
      tree->vec_orig[fnum].push_back(pvec_orig);
      tree->vec_tran[fnum].push_back(pvec_tran);
      tree->sig_orig[fnum].push(pvec_orig);
      tree->sig_tran[fnum].push(pvec_tran);
    }
    else
    {
      // New voxel
      auto* tree = new OCTO_TREE_ROOT(window_size, eigen_ratio);
      tree->vec_orig[fnum].push_back(pvec_orig);
      tree->vec_tran[fnum].push_back(pvec_tran);
      tree->sig_orig[fnum].push(pvec_orig);
      tree->sig_tran[fnum].push(pvec_tran);

      tree->voxel_center[0] = (0.5 + position.x) * voxel_size;
      tree->voxel_center[1] = (0.5 + position.y) * voxel_size;
      tree->voxel_center[2] = (0.5 + position.z) * voxel_size;
      tree->quater_length = voxel_size / 4.0;
      tree->layer = 0;
      iter->second = tree;
    }
  }
}

void parallel_comp(LAYER& layer, int thread_id, LAYER& next_layer) {
  const int part_length = layer.part_length;
  const int layer_num = layer.layer_num;
  const int start_idx = thread_id * part_length;
  const int end_idx = (thread_id + 1) * part_length;

  for(int i = start_idx; i < end_idx; i++) {
    // Initialize point cloud vectors
    std::vector<pcl::PointCloud<PointType>::Ptr> src_pc(WIN_SIZE);
    std::vector<pcl::PointCloud<PointType>::Ptr> raw_pc(WIN_SIZE);

    // Initialize pose buffers
    double residual_cur = 0, residual_pre = 0;
    std::vector<IMUST> x_buf(WIN_SIZE);
    for(int j = 0; j < WIN_SIZE; j++) {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    // Load point clouds for non-first layer
    if(layer_num != 1) {
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++) {
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
      }
    }

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++) {
      // Load point clouds for first layer
      if(layer_num == 1) {
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++) {
          if(loop == 0) {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();
        }
      }

      // Process point clouds
      std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
      
      for(int j = 0; j < WIN_SIZE; j++) {
        if(layer.downsample_size > 0) {
          downsample_voxel(*src_pc[j], layer.downsample_size);
        }
        cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
      }

      // Recut and optimize
      for(auto& [_, tree] : surf_map) {
        tree->recut();
      }
      
      VOX_HESS voxhess(WIN_SIZE);
      for(const auto& [_, tree] : surf_map) {
        tree->tras_opt(voxhess);
      }

      // Optimize poses
      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      // Cleanup
      for(auto& [_, tree] : surf_map) {
        delete tree;
      }
      const double residual_change = std::abs(residual_pre - residual_cur);
      const double residual_ratio = residual_change / std::abs(residual_cur);
      
      if((loop > 0 && residual_ratio < 0.05) || loop == layer.max_iter-1) {
        if(layer.mem_costs[thread_id] < mem_cost) {
          layer.mem_costs[thread_id] = mem_cost;
        }
        
        const int hess_offset = i * (WIN_SIZE-1) * WIN_SIZE / 2;
        for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++) {
          layer.hessians[hess_offset + j] = hess_vec[j];
        }
        break;
      }
      residual_pre = residual_cur;
    }

    // Create keyframe point cloud
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    const Eigen::Matrix3d R0_inv = x_buf[0].R.inverse();
    const Eigen::Vector3d p0 = x_buf[0].p;
    
    for(int j = 0; j < WIN_SIZE; j++) {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, 
                Quaterniond(R0_inv * x_buf[j].R),
                R0_inv * (x_buf[j].p - p0));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
}

void parallel_tail(LAYER& layer, int thread_id, LAYER& next_layer)
{
  int& part_length = layer.part_length;
  int& layer_num = layer.layer_num;
  int& left_gap_num = layer.left_gap_num;

  double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0,
    tran_t = 0, sol_t = 0, save_t = 0;
  
  if(layer.gap_num-(layer.thread_num-1)*part_length+1!=left_gap_num) printf("THIS IS WRONG!\n");

  for(uint i = thread_id*part_length; i < thread_id*part_length+left_gap_num; i++)
  {
    printf("parallel computing %d\n", i);
    double t0, t1;
    double t_begin = time_now();
    
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE); raw_pc.resize(WIN_SIZE);
    
    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for(int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }
    
    if(layer_num != 1)
    {
      t0 = time_now();
      for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
      load_t += time_now()-t0;
    }

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      if(layer_num == 1)
      {
        t0 = time_now();
        for(int j = i*GAP; j < i*GAP+WIN_SIZE; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();
        }
        load_t += time_now()-t0;
      }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      for(size_t j = 0; j < WIN_SIZE; j++)
      {
        t0 = time_now();
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        dsp_t += time_now()-t0;

        t0 = time_now();
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
        cut_t += time_now()-t0;
      }

      t0 = time_now();
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      recut_t += time_now()-t0;

      t0 = time_now();
      VOX_HESS voxhess(WIN_SIZE);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);
      tran_t += time_now()-t0;

      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      t0 = time_now();
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      sol_t += time_now()-t0;

      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
            
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        if(i < thread_id*part_length+left_gap_num)
          for(int j = 0; j < WIN_SIZE*(WIN_SIZE-1)/2; j++)
            layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }
    
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for(size_t j = 0; j < WIN_SIZE; j++)
    {
      t1 = time_now();
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
      save_t += time_now()-t1;
    }
    t0 = time_now();
    downsample_voxel(*pc_keyframe, 0.05);
    dsp_t += time_now()-t0;

    t0 = time_now();
    next_layer.pcds[i] = pc_keyframe;
    save_t += time_now()-t0;
    
    total_t += time_now()-t_begin;
  }
  if(layer.tail > 0)
  {
    int i = thread_id*part_length+left_gap_num;

    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(layer.last_win_size); raw_pc.resize(layer.last_win_size);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(layer.last_win_size);
    for(int j = 0; j < layer.last_win_size; j++)
    {
      x_buf[j].R = layer.pose_vec[i*GAP+j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i*GAP+j].t;
    }

    if(layer_num != 1)
    {
      for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        src_pc[j-i*GAP] = (*layer.pcds[j]).makeShared();
    }

    size_t mem_cost = 0;
    for(int loop = 0; loop < layer.max_iter; loop++)
    {
      if(layer_num == 1)
        for(int j = i*GAP; j < i*GAP+layer.last_win_size; j++)
        {
          if(loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j-i*GAP] = pc;
          }
          src_pc[j-i*GAP] = (*raw_pc[j-i*GAP]).makeShared();          
        }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

      for(size_t j = 0; j < layer.last_win_size; j++)
      {
        if(layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, layer.last_win_size, layer.eigen_ratio);
      }
      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      
      VOX_HESS voxhess(layer.last_win_size);
      for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      VOX_OPTIMIZER opt_lsv(layer.last_win_size);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;
      
      if(loop > 0 && abs(residual_pre-residual_cur)/abs(residual_cur) < 0.05 || loop == layer.max_iter-1)
      {
        if(layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        for(int j = 0; j < layer.last_win_size*(layer.last_win_size-1)/2; j++)
          layer.hessians[i*(WIN_SIZE-1)*WIN_SIZE/2+j] = hess_vec[j];
        
        break;
      }
      residual_pre = residual_cur;
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for(size_t j = 0; j < layer.last_win_size; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
  printf("total time: %.2fs\n", total_t);
  printf("load pcd %.2fs %.2f%% | undistort pcd %.2fs %.2f%% | "
   "downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | trans %.2fs %.2f%% | solve %.2fs %.2f%% | "
   "save pcd %.2fs %.2f%%\n",
    load_t, load_t/total_t*100, undis_t, undis_t/total_t*100,
    dsp_t, dsp_t/total_t*100, cut_t, cut_t/total_t*100, recut_t, recut_t/total_t*100, tran_t, tran_t/total_t*100,
    sol_t, sol_t/total_t*100, save_t, save_t/total_t*100);
}

void global_ba(LAYER& layer) {
  const int window_size = layer.pose_vec.size();
  
  // Initialize pose buffers
  std::vector<IMUST> x_buf(window_size);
  for(int i = 0; i < window_size; i++) {
    x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
    x_buf[i].p = layer.pose_vec[i].t;
  }

  // Initialize point clouds
  std::vector<pcl::PointCloud<PointType>::Ptr> src_pc(window_size);
  for(int i = 0; i < window_size; i++) {
    src_pc[i] = layer.pcds[i]->makeShared();
  }

  double residual_cur = 0, residual_pre = 0;
  size_t mem_cost = 0, max_mem = 0;
  double dsp_t = 0, cut_t = 0, recut_t = 0, tran_t = 0, sol_t = 0;

  for(int loop = 0; loop < layer.max_iter; loop++) {
    std::cout << "--------------------- \nIteration " << loop << std::endl;

    std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

    // Process each frame
    for(int i = 0; i < window_size; i++) {
      const auto t0 = time_now();
      
      if(layer.downsample_size > 0) {
        downsample_voxel(*src_pc[i], layer.downsample_size);
      }
      dsp_t += time_now() - t0;

      const auto t1 = time_now();
      cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i,
                layer.voxel_size, window_size, layer.eigen_ratio * 2);
      cut_t += time_now() - t1;
    }

    // Recut and transform
    const auto t2 = time_now();
    for(const auto& [_, tree] : surf_map) {
      tree->recut();
    }
    recut_t += time_now() - t2;
    
    const auto t3 = time_now();
    VOX_HESS voxhess(window_size);
    for(const auto& [_, tree] : surf_map) {
      tree->tras_opt(voxhess);
    }
    tran_t += time_now() - t3;
    
    // Optimize
    const auto t4 = time_now();
    VOX_OPTIMIZER opt_lsv(window_size);
    opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
    PLV(6) hess_vec;
    opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
    sol_t += time_now() - t4;

    // Cleanup
    for(auto& [_, tree] : surf_map) {
      delete tree;
    }
    
    const double residual_change = std::abs(residual_pre - residual_cur);
    const double residual_ratio = residual_change / std::abs(residual_cur);
    
    std::cout << "Residual absolute: " << residual_change 
              << " | percentage: " << residual_ratio << std::endl;
    
    if((loop > 0 && residual_ratio < 0.05) || loop == layer.max_iter - 1) {
      max_mem = std::max(max_mem, mem_cost);
      
      #ifdef FULL_HESS
      std::copy(hess_vec.begin(), 
                hess_vec.begin() + window_size * (window_size - 1) / 2,
                layer.hessians.begin());
      #else
      for(int i = 0; i < window_size - 1; i++) {
        Matrix6d hess = Hess_cur.block(6*i, 6*i+6, 6, 6);
        for(int row = 0; row < 6; row++) {
          for(int col = 0; col < 6; col++) {
            hessFile << hess(row, col) << ((row*col==25) ? "" : " ");
          }
        }
        if(i < window_size - 2) hessFile << "\n";
      }
      #endif
      break;
    }
    residual_pre = residual_cur;
  }

  // Update final poses
  for(int i = 0; i < window_size; i++) {
    layer.pose_vec[i].q = Quaterniond(x_buf[i].R);
    layer.pose_vec[i].t = x_buf[i].p;
  }

  std::printf("Downsample: %.3f, Cut: %.3f, Recut: %.3f, Tras: %.3f, Sol: %.3f\n", 
              dsp_t, cut_t, recut_t, tran_t, sol_t);
}

void distribute_thread(LAYER& layer, LAYER& next_layer) {
  const int thread_num = layer.thread_num;
  const auto t0 = time_now();

  // Use vector of unique_ptr to manage thread lifetime
  std::vector<std::unique_ptr<std::thread>> threads;
  threads.reserve(thread_num);

  // Launch threads
  for (int i = 0; i < thread_num; i++) {
    if (i < thread_num - 1) {
      threads.emplace_back(std::make_unique<std::thread>(
        parallel_comp, std::ref(layer), i, std::ref(next_layer)));
    } else {
      threads.emplace_back(std::make_unique<std::thread>(
        parallel_tail, std::ref(layer), i, std::ref(next_layer)));
    }
  }

  // Join all threads
  for (auto& thread : threads) {
    thread->join();
  }
}

int main(int argc, char** argv)
{
  cxxopts::Options options("hba", "Hierarchical LiDAR bundle adjustment");

  options.add_options()
      ("total_layer_num", "Total number of layers", cxxopts::value<int>()->default_value("3"))
      ("pcd_name_fill_num", "Filling in PCD name 0 for park, 5 for kitti07", cxxopts::value<int>()->default_value("5"))
      ("data_path", "Data path", cxxopts::value<std::string>()->default_value("/data/kitti07/"))
      ("thread_num", "Number of CPU threads", cxxopts::value<int>()->default_value("16"))
  ;

  auto result = options.parse(argc, argv);

  std::cout << "total_layer_num: " << result["total_layer_num"].as<int>() << std::endl;
  std::cout << "pcd_name_fill_num: " << result["pcd_name_fill_num"].as<int>() << std::endl;
  std::cout << "data_path: " << result["data_path"].as<std::string>() << std::endl;
  std::cout << "thread_num: " << result["thread_num"].as<int>() << std::endl;

  const auto total_layer_num = result["total_layer_num"].as<int>();
  const auto pcd_name_fill_num = result["pcd_name_fill_num"].as<int>();
  const auto data_path = result["data_path"].as<std::string>();
  const auto thread_num = result["thread_num"].as<int>();

  HBA hba(total_layer_num, data_path, thread_num);
  for(int i = 0; i < total_layer_num-1; i++)
  {
    std::cout<<"---------------------"<<std::endl;
    distribute_thread(hba.layers[i], hba.layers[i+1]);
    hba.update_next_layer_state(i);
  }
  global_ba(hba.layers[total_layer_num-1]);
  hba.pose_graph_optimization();
  printf("iteration complete\n");
}