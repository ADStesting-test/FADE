import sys
# 将上级目录添加到 sys.path
sys.path.append("/apollo/modules/myTest")

import numpy as np
from PIL import Image, ImageDraw
import random
import sys
import os
import open3d as o3d
import lidar_operation_vis


# 函数：从点云文件保存图片
def save_pointcloud_as_image(pcd_file, image_file):


    VIS = o3d.visualization.Visualizer()
    VIS.create_window()
    # 设置视点和背景颜色
    VIS.get_render_option().background_color = [1, 1, 1]  # 白色背景
    VIS.get_render_option().point_size = 2.0  # 设置点大小
    
    # 加载点云数据
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 使用 Open3D 的内置函数显示点云
    o3d.visualization.draw_geometries([pcd])

    # # 将点云添加到渲染器中
    # VIS.add_geometry(pcd)

    # # 渲染并保存为图片
    # VIS.poll_events()
    # VIS.update_renderer()
    # # 渲染并保存为图片
    # VIS.capture_screen_image(image_file)
    # VIS.remove_geometry(pcd)

    # print(f"图片已保存：{image_file}")

# 可视化点云，并标记故障点
def visualize_point_cloud(path,title,width,height):
    pcd = o3d.io.read_point_cloud(path)  # 替换为你的点云文件路径
    VIS = o3d.visualization.Visualizer()
    VIS.create_window(window_name=title, width=width, height=height)
    # 设置视点和背景颜色
    VIS.get_render_option().background_color = [1, 1, 1]  # 白色背景
    VIS.get_render_option().point_size = 2.0  # 设置点大小

    VIS.add_geometry(pcd)
    # # 使用 Open3D 的内置函数显示点云
    # o3d.visualization.draw_geometries([pcd],title)
    # 渲染并显示
    VIS.poll_events()
    VIS.update_renderer()
    VIS.run()

# 可视化点云，并显示噪点和其原始点的连线
def visualize_point_cloud_with_faults_and_lines(pcd, fault_indices, original_points):
    points = np.asarray(pcd.points)
    
    # 设置颜色，默认所有点为蓝色
    colors = np.zeros_like(points)
    colors[:] = [0, 0, 1]  # 正常点为蓝色

    # 将故障点标为红色
    colors[fault_indices] = [1, 0, 0]  # 故障点为红色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建线段，表示原始点与故障点的连线
    lines = []
    for i, fault_index in enumerate(fault_indices):
        lines.append([fault_index, len(points) + i])  # 原始点与噪点之间的连线

    # 将原始点加入到点云中，作为新点
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    
    # 将原始点的颜色设为绿色
    original_colors = np.zeros_like(original_points)
    original_colors[:] = [0, 1, 0]  # 原始点为绿色
    original_pcd.colors = o3d.utility.Vector3dVector(original_colors)

    # 组合新的点云数据
    combined_pcd = pcd + original_pcd

    # 创建线段对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((points, original_points)))  # 故障点与原始点
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])  # 线段为绿色

    # 可视化
    o3d.visualization.draw_geometries([combined_pcd, line_set])

# 可视化点云，并标记故障点
def visualize_point_cloud_with_faults(pcd, fault_indices):
    points = np.asarray(pcd.points)
    
    # 设置颜色，默认所有点为蓝色
    colors = np.zeros_like(points)
    colors[:] = [0, 0, 1]  # 蓝色

    # 将故障点标为红色
    colors[fault_indices] = [1, 0, 0]  # 红色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])


# 可视化点云，并标记故障点
def visualize_point_cloud_with_remove(pcd, fault_indices):
    points = np.asarray(pcd.points)
    
    # 设置颜色，默认所有点为蓝色
    colors = np.zeros_like(points)
    colors[:] = [0, 0, 1]  # 蓝色

    # 将故障点标为红色
    colors[fault_indices] = [1, 0, 0]  # 红色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])

# 可视化点云，并标记噪点和丢失点
def visualize_em_interference(pcd, noisy_indices, lost_indices):
    points = np.asarray(pcd.points)
    
    # 设置颜色，默认所有点为蓝色
    colors = np.zeros_like(points)
    colors[:] = [0, 0, 1]  # 正常点为蓝色

    # 将噪声影响的点标为红色
    colors[noisy_indices] = [1, 0, 0]  # 噪声影响的点为红色

    # 将丢失的点标为绿色
    colors[lost_indices] = [0, 1, 0]  # 丢失的点为绿
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])

# 可视化点云，并标记虚假目标和丢失点
def visualize_lidar_interference(pcd, ghost_targets, lost_indices,noise_points):
    points = np.asarray(pcd.points)

    # 设置颜色，默认所有点为蓝色
    colors = np.zeros_like(points)
    colors[:] = [0, 0, 1]  # 正常点为蓝色

    # 虚假目标设为红色
    colors[-len(ghost_targets)-len(noise_points):-len(ghost_targets)+1] = [1, 0, 0]  # 虚假目标为红色

    # 丢失的点设为绿色
    colors[lost_indices] = [0, 1, 0]  # 丢失的点为绿色

    # 噪声点设为黄
    colors[-len(noise_points):] = [1, 1, 0]  # 噪声点为黄

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])

# 可视化点云，并标记噪点和丢失点
def visualize_rain_snow(pcd,noisy_points):
    points = np.asarray(pcd.points)
    
    # # # 设置颜色，默认所有点为蓝色
    colors = pcd.colors

    # # 将噪声影响的点标为红色
    # colors[-len(noisy_points):] = [1, 0, 0]  # 噪声影响的点为红色

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])

# 可视化点云
def visualize_strong_light(pcd):
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    dir_path = '/apollo/modules/myTest/lidar/'
    # SCENE_STR = 'leftTurn_right_'  
    SCENE_STR = ''  
    pcd_name ='163'

    pcd_path = '/apollo/data/raw_data/lidar/' + "CENTER/1691998431675.pcd"    
    visualize_point_cloud(pcd_path,"CENTER",640,480)
    pcd_path = '/apollo/data/raw_data/lidar/' + "LEFT/1691998431634.pcd"
    visualize_point_cloud(pcd_path,"LEFT",640,480)
    pcd_path = '/apollo/data/raw_data/lidar/' + "RIGHT/1691998431644.pcd"
    visualize_point_cloud(pcd_path,"RIGHT",640,480)
    pcd_path = '/apollo/data/raw_data/lidar/' + "STITCHED/1691998431675.pcd"
    visualize_point_cloud(pcd_path,"STITCHED",640,480)
    # # 加载点云文件 (.pcd, .ply 等格式)
    # pcd = o3d.io.read_point_cloud(pcd_path)  # 替换为你的点云文件路径
    # save_pointcloud_as_image(dir_path + pcd_name+".pcd", dir_path+pcd_name+'.jpeg')
    # save_pointcloud_as_image(pcd_path, dir_path+'lidar/lidar1.jpeg')
    # save_pointcloud_as_image(pcd_path, dir_path+'lidar/lidar2.jpeg')
    # visualize_point_cloud(pcd_path,"CENTER")

    # pcd = o3d.io.read_point_cloud(dir_path + SCENE_STR+pcd_name+".pcd")
    # visualize_point_cloud(pcd)

    # 引入 10% 的 Line Fault 故障，噪声强度为 5
    # faulty_pcd, fault_indices, original_points = lidar_operation_vis.introduce_line_fault(pcd, fault_ratio=0.1, noise_level=5, car_radius=100)
    # # 可视化，显示故障点（红色）和正常点（蓝色）
    # visualize_point_cloud_with_faults_and_lines(faulty_pcd, fault_indices, original_points)

    # # 移除 对应通道的点
    # channels_to_remove = random.sample(range(128),50)
    # faulty_pcd, filter_points,reomve_points = lidar_operation_vis.remove_channels_test(pcd, channels_to_remove)

    # visualize_point_cloud_with_faults(faulty_pcd, reomve_points)


    # # 引入电磁干扰故障
    # # 引入电磁干扰，10% 点受到噪声影响，5% 点丢失，噪声强度为 0.05
    # faulty_pcd, noisy_indices, lost_indices = lidar_operation_vis.introduce_electromagnetic_interference(pcd, noise_ratio=0.1, noise_level=1, loss_ratio=0.1)
    # # 可视化，显示噪声影响点（红色）、丢失点（绿色）和正常点（蓝色）
    # visualize_em_interference(faulty_pcd, noisy_indices, lost_indices)

    # # 模拟激光雷达干扰，5% 点丢失
    # faulty_pcd, ghost_targets, lost_indices,noise_points = lidar_operation_vis.simulate_lidar_interference(pcd,max_distance=10.0, copy_distance=10.0, concentration_radius=3.0, loss_ratio=0.1)
    # # 可视化
    # visualize_lidar_interference(faulty_pcd, ghost_targets, lost_indices,noise_points)

    # # 模拟雨雪干扰
    # faulty_pcd,noisy_points = lidar_operation_vis.simulate_rain_snow_interference(pcd, alpha=0.1, noise_std=0.2, noise_density=0.2)
    # visualize_rain_snow(faulty_pcd,noisy_points)

    # # 模拟强光干扰
    # light_source= lidar_operation_vis.generate_random_light_source(max_distance=30)
    # faulty_pcd = lidar_operation_vis.simulate_strong_light_interference(pcd, light_source=light_source,distance_threshold=30, reduction_ratio=0.5, angle_threshold=45)
    # visualize_strong_light(faulty_pcd)

    while True:
        pass