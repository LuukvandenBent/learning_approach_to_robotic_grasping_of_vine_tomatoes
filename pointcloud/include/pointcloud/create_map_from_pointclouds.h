#pragma once

#include "ros/ros.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include <tf/tf.h>
#include "grasp/create_map_command.h"

class Map{
    public:
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>> aligned_pointclouds;
        int max_pointclouds = 1;
        float truss_height = 0.05f;

        float voxel_size = 0.0001f;

        tf::TransformListener listener;
        bool start = false;
        grasp::create_map_command::Request bboxes;

        bool create_map(grasp::create_map_command::Request &req, grasp::create_map_command::Response &res);
        void reset();
        void init(ros::NodeHandle n);
        sensor_msgs::PointCloud2 combine_pointclouds();
        void cloud_cb(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input);

        ros::Publisher map_pub;
        std::string planning_frame;
};


