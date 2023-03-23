#include <open3d/Open3D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "pointcloud/convert_pcl_open3d.h"

#include <vector>
#include <memory>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr open3d_to_pcl(const std::shared_ptr<open3d::geometry::PointCloud> open3d_cloud){
    const uint32_t size = open3d_cloud->points_.size();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl_cloud->width = size;
    pcl_cloud->height = 1;
    pcl_cloud->is_dense = false;
    pcl_cloud->points.resize( size );

    if( open3d_cloud->HasColors() ){
        #pragma omp parallel for
        for( int32_t i = 0; i < size; i++ ){
            pcl_cloud->points[i].getVector3fMap() = open3d_cloud->points_[i].cast<float>();
            const auto color = ( open3d_cloud->colors_[i] * 255.0 ).cast<uint32_t>();
            uint32_t rgb = color[0] << 16 | color[1] << 8 | color[2];
            pcl_cloud->points[i].rgb = *reinterpret_cast<float*>( &rgb );
        }
    }
    else{
        #pragma omp parallel for
        for( int32_t i = 0; i < size; i++ ){
            pcl_cloud->points[i].getVector3fMap() = open3d_cloud->points_[i].cast<float>();
            uint32_t rgb = 0x000000;
            pcl_cloud->points[i].rgb = *reinterpret_cast<float*>( &rgb );
        }
    }

    return pcl_cloud;
};

std::shared_ptr<open3d::geometry::PointCloud> pcl_to_open3d( const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud){
    const uint32_t size = pcl_cloud->size();

    std::shared_ptr<open3d::geometry::PointCloud> open3d_cloud = std::make_shared<open3d::geometry::PointCloud>();
    open3d_cloud->points_.resize( size );
    open3d_cloud->colors_.resize( size );

    constexpr double normal = 1.0 / 255.0;
    #pragma omp parallel for
    for( int32_t i = 0; i < size; i++ ){
        open3d_cloud->points_[i] = pcl_cloud->points[i].getVector3fMap().cast<double>();
        const uint32_t color = *reinterpret_cast<uint32_t*>( &pcl_cloud->points[i].rgb );
        open3d_cloud->colors_[i] = Eigen::Vector3d( ( color >> 16 ) & 0x0000ff, ( color >> 8 ) & 0x0000ff, color & 0x0000ff ) * normal;
    }

    return open3d_cloud;
};