#pragma once

#include <open3d/Open3D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr open3d_to_pcl(const std::shared_ptr<open3d::geometry::PointCloud> open3d_cloud);
std::shared_ptr<open3d::geometry::PointCloud> pcl_to_open3d( const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud);