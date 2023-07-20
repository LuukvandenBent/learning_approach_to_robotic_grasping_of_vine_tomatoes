#include "ros/ros.h"
#include <ros/package.h>
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "geometry_msgs/Point.h"
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <iostream>

#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/io/ply_io.h>

#include <cstdlib>

#include "grasp/create_map_command.h"
#include "pointcloud/create_map_from_pointclouds.h"


void Map::init(ros::NodeHandle n){
    map_pub = n.advertise<sensor_msgs::PointCloud2>("map", 1);
    n.getParam("/planning_frame", planning_frame);
}

bool Map::create_map(grasp::create_map_command::Request &req, grasp::create_map_command::Response &res){
    reset();
    bboxes = req;//Set to be used later

    double max_wait_time = 20.0;
    double sleep_time = 0.2;
    start = true;
    while ((aligned_pointclouds.size() < max_pointclouds)){
        ros::Duration(sleep_time).sleep();
        ros::spinOnce();
        max_wait_time -= sleep_time;
        if (max_wait_time < 0.0){
            ROS_WARN("Max waiting time for a pointcloud map reached");
            return false;
        }
    }
    res.map = combine_pointclouds();
    return true;
}

void Map::reset(){//todo fix forall
    start = false;
    aligned_pointclouds.clear();
}

sensor_msgs::PointCloud2 Map::combine_pointclouds(){
    pcl::PointCloud<pcl::PointXYZRGB> combined_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    combined_cloud = aligned_pointclouds[0];
    for (auto cloud = aligned_pointclouds.begin() + 1; cloud != aligned_pointclouds.end(); ++cloud){
        combined_cloud += *cloud;
    }
    *combined_cloud_ptr = combined_cloud;
    // Filter combined pointcloud using a gridfilter to reduce size
    //pcl::VoxelGrid<pcl::PointXYZRGB> voxelGridFilter;
    //voxelGridFilter.setInputCloud(combined_cloud_ptr);
    //voxelGridFilter.setLeafSize(voxel_size, voxel_size, voxel_size); // Set voxel size
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    //voxelGridFilter.filter(*filtered_cloud);
    sensor_msgs::PointCloud2 ros_cloud;
    //pcl::toROSMsg(*filtered_cloud, ros_cloud);
    pcl::toROSMsg(*combined_cloud_ptr, ros_cloud);
    ros_cloud.header.frame_id = planning_frame;
    //Publish for debug
    map_pub.publish(ros_cloud);

    return ros_cloud;
}


void Map::cloud_cb(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input){
    if ((aligned_pointclouds.size() >= max_pointclouds) || (!start)){
        return;
    }
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*input,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_icp(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);

    //Transform point cloud
    tf::StampedTransform transform;
    try{
        //listener.waitForTransform("panda_link0", input->header.frame_id, input->header.stamp, ros::Duration(0.2));
        //Use latest instead of stamp of message since message stamp is wrong (sometimes?)!!
        listener.lookupTransform(planning_frame, input->header.frame_id, ros::Time(0), transform);
    }
    catch (tf::TransformException ex){
        std::cout << "transform failed" << std::endl;
        ROS_ERROR("%s",ex.what());
        return;
    }

    pcl_ros::transformPointCloud(*temp_cloud, *cloud_transformed, transform);

    std::string packageName = "grasp";
    std::string graspPackagePath = ros::package::getPath(packageName);
    size_t pos = graspPackagePath.find_last_of("/\\");//Find parent
    graspPackagePath = graspPackagePath.substr(0, pos);
    pos = graspPackagePath.find_last_of("/\\");//Find parent
    graspPackagePath = graspPackagePath.substr(0, pos);
    pos = graspPackagePath.find_last_of("/\\");//Find parent
    graspPackagePath = graspPackagePath.substr(0, pos);

    std::string map_dir = "/experiments/map/";
    std::string full_map_dir = graspPackagePath + map_dir;

    std::string command = "mkdir -p " + full_map_dir;

    std::cout << command << std::endl;
    system(command.c_str());

    pcl::io::savePCDFile(full_map_dir + "raw.pcd", *cloud_transformed);
    //Crop region of interest of truss based on the detected bbox
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr roi(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull(new pcl::PointCloud<pcl::PointXYZRGB>);
    roi->points.resize(8);
    roi->points[0].x = bboxes.poses.poses[1].position.x;
    roi->points[0].y = bboxes.poses.poses[1].position.y;
    roi->points[0].z = bboxes.poses.poses[0].position.z;
    roi->points[1].x = bboxes.poses.poses[2].position.x;
    roi->points[1].y = bboxes.poses.poses[2].position.y;
    roi->points[1].z = bboxes.poses.poses[0].position.z;
    roi->points[2].x = bboxes.poses.poses[3].position.x;
    roi->points[2].y = bboxes.poses.poses[3].position.y;
    roi->points[2].z = bboxes.poses.poses[0].position.z;
    roi->points[3].x = bboxes.poses.poses[4].position.x;
    roi->points[3].y = bboxes.poses.poses[4].position.y;
    roi->points[3].z = bboxes.poses.poses[0].position.z;
    roi->points[4] = roi->points[0];
    roi->points[5] = roi->points[1];
    roi->points[6] = roi->points[2];
    roi->points[7] = roi->points[3];
    roi->points[0].z += 0.30;
    roi->points[1].z += 0.30;
    roi->points[2].z += 0.30;
    roi->points[3].z += 0.30;
    roi->points[4].z -= 0.15;
    roi->points[5].z -= 0.15;
    roi->points[6].z -= 0.15;
    roi->points[7].z -= 0.15;
    
    pcl::ConvexHull<pcl::PointXYZRGB> hull_calculator;
    std::vector<pcl::Vertices> polygons;
    hull_calculator.setInputCloud (roi);
    hull_calculator.reconstruct (*hull, polygons);

    pcl::CropHull<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud_transformed);
    pass.setHullCloud(hull);

    pass.setHullIndices(polygons);
    pass.setCropOutside(true);
    pass.filter(*cloud_filtered);

    try{
        if (!cloud_filtered->size()){//If no points left, something went wrong... return
            return;
        }
    pcl::io::savePCDFile(full_map_dir + "bbox_filtered.pcd", *cloud_filtered);
    //Find the plane and then filter based on depth from this plane
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                        << coefficients->values[1] << " "
                                        << coefficients->values[2] << " " 
                                        << coefficients->values[3] << std::endl;
    if (inliers->indices.size () == 0){
        std::cerr << "Plane could not be found... Something wrong with the pc ";
        return;
    }
    pcl::ModelOutlierRemoval<pcl::PointXYZRGB> plane_filter;
    plane_filter.setModelCoefficients (*coefficients);
    plane_filter.setThreshold (0.05);
    plane_filter.setModelType (pcl::SACMODEL_PLANE);
    plane_filter.setInputCloud (cloud_filtered);
    plane_filter.filter (*cloud_filtered);
    }
    catch (...){
        std::cerr << "Plane could not be found... Something wrong with the pc ";
        return;
    }
    //Add StatisticalOutlierRemoval
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);

    pcl::io::savePCDFile(full_map_dir + "depth_filtered.pcd", *cloud_filtered);
    *cloud_final = *cloud_filtered;

    aligned_pointclouds.push_back(*cloud_final);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapping_node");
    ros::NodeHandle n;
    Map map;
    map.init(n);

    ros::ServiceServer create_map_service = n.advertiseService("create_map", &Map::create_map, &map);
    ros::Subscriber pointcloud_sub = n.subscribe("camera/depth/color/points", 1, &Map::cloud_cb, &map);
    
    ros::spin();

    return 0;
}
