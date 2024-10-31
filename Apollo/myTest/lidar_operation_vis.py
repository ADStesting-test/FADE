import random
import numpy as np
import copy
import math
from scipy.spatial import KDTree
import open3d as o3d
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于颜色映射

# import sensor_msg_extractor
from cyber.python.cyber_py3 import cyber
from modules.drivers.proto.pointcloud_pb2 import PointCloud

OUT_FOLDER = "/apollo/data/lidar/"

def calculate_vector_from_origin(point):
    """
    计算从激光雷达（假设位于原点）到点的向量
    :param point: 单个点 (x, y, z)
    :return: 向量 (vx, vy, vz)
    """
    return point.x, point.y, point.z

def normalize_vector(vx, vy, vz):
    """
    将向量归一化
    :param vx: 向量的x分量
    :param vy: 向量的y分量
    :param vz: 向量的z分量
    :return: 归一化后的向量 (nx, ny, nz)
    """
    magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
    if magnitude == 0:
        return 0, 0, 0
    return vx / magnitude, vy / magnitude, vz / magnitude

def get_perpendicular_vector(vx, vy, vz):
    """
    获取与给定向量垂直的一个向量（模拟侧面的鬼影偏移）
    :param vx: 向量的x分量
    :param vy: 向量的y分量
    :param vz: 向量的z分量
    :return: 垂直向量 (px, py, pz)
    """
    # 简单的获取一个与原向量垂直的向量，可以通过交换x和y，并反转y来得到
    return -vy, vx, vz  # 这是一种简单方式生成垂直向量

def add_ghosts_model_with_direction(pointcloud, intensity_threshold=0.9, num_ghosts=3, max_offset=1.0, intensity_scale=0.9):
    """
    添加鬼影效应：通过计算方向向量，生成出现在物体后面或侧面的鬼影点
    :param pointcloud: PointCloud protobuf 对象
    :param intensity_threshold: 生成鬼影的反射率阈值
    :param num_ghosts: 每个高反射率点生成的鬼影点数
    :param max_offset: 鬼影点与真实点之间的最大偏移范围
    :param intensity_scale: 鬼影点反射率的缩放比例
    :return: 添加鬼影后的 PointCloud 对象
    """
    new_pointcloud = copy.deepcopy(pointcloud)

    for point in pointcloud.point:
        if point.intensity >= intensity_threshold:
            # 计算从原点（激光雷达位置）到该点的向量
            vx, vy, vz = calculate_vector_from_origin(point)
            # 归一化向量，得到方向向量
            nx, ny, nz = normalize_vector(vx, vy, vz)

            for _ in range(num_ghosts):
                ghost_point = copy.deepcopy(point)

                # 在物体后面生成鬼影点（沿着反向偏移）
                offset_x = -nx * random.uniform(0, max_offset)
                offset_y = -ny * random.uniform(0, max_offset)
                offset_z = -nz * random.uniform(0, max_offset)

                # 更新鬼影点的位置
                ghost_point.x += offset_x
                ghost_point.y += offset_y
                ghost_point.z += offset_z

                # 模拟鬼影点的强度，略低于真实点
                ghost_point.intensity *= intensity_scale
                
                # 将生成的鬼影点添加到新的点云中
                new_pointcloud.point.append(ghost_point)
    del pointcloud
    return new_pointcloud

def add_blooming_effect(pointcloud, intensity_threshold=0.9, num_expanded_points=5, max_offset=0.5, intensity_scale=0.95):
    """
    添加膨胀效应：为高反射率点周围生成虚假的膨胀点
    :param pointcloud: PointCloud protobuf 对象
    :param intensity_threshold: 生成膨胀效应的反射率阈值
    :param num_expanded_points: 每个高反射率点生成的膨胀点数
    :param max_offset: 膨胀点与真实点之间的最大偏移范围（膨胀半径）
    :param intensity_scale: 膨胀点的反射率缩放比例
    :return: 添加膨胀效应后的 PointCloud 对象
    """
    new_pointcloud = copy.deepcopy(pointcloud)

    for point in pointcloud.point:
        if point.intensity >= intensity_threshold:
            for _ in range(num_expanded_points):
                expanded_point = copy.deepcopy(point)
                
                # 生成随机的偏移量，模拟膨胀效应
                offset_x = random.uniform(-max_offset, max_offset)
                offset_y = random.uniform(-max_offset, max_offset)
                offset_z = random.uniform(-max_offset, max_offset)
                
                # 更新膨胀点的位置
                expanded_point.x += offset_x
                expanded_point.y += offset_y
                expanded_point.z += offset_z

                # 模拟膨胀点的强度，略低于真实点
                expanded_point.intensity *= intensity_scale
                
                # 将生成的膨胀点添加到新的点云中
                new_pointcloud.point.append(expanded_point)
    del pointcloud
    return new_pointcloud

def add_point_cloud_adhesion_kdtree(pointcloud, distance_threshold=1.0, num_bridge_points=10, intensity_scale=0.7):
    """
    添加点云粘连效应：使用 KD-Tree 结构查找相邻的两个物体之间生成虚假连接点
    :param pointcloud: PointCloud protobuf 对象
    :param distance_threshold: 用于判定两个物体是否靠近的距离阈值
    :param num_bridge_points: 在两个物体之间生成的虚假连接点数量
    :param intensity_scale: 虚假连接点的反射率缩放比例
    :return: 添加点云粘连效应后的 PointCloud 对象
    """
    new_pointcloud = copy.deepcopy(pointcloud)

    points = list(pointcloud.point)
    num_points = len(points)

    # 将点云的 (x, y, z) 坐标转换为 KD-Tree 的输入格式
    point_coords = [(point.x, point.y, point.z) for point in points]
    
    # 构建 KD-Tree
    kdtree = KDTree(point_coords)

    # 遍历点云中的每个点，找到距离小于阈值的点对
    for i, point_a in enumerate(points):
        # 查询与 point_a 距离小于 distance_threshold 的点
        indices = kdtree.query_ball_point([point_a.x, point_a.y, point_a.z], distance_threshold)
        
        # 遍历找到的点对，生成虚假连接点
        for j in indices:
            if j != i:  # 避免和自身配对
                point_b = points[j]

                for _ in range(num_bridge_points):
                    bridge_point = copy.deepcopy(point_a)  # 以 point_a 为模板创建虚假连接点

                    # 使用线性插值公式生成虚假连接点
                    alpha = random.uniform(0, 1)
                    bridge_point.x = alpha * point_a.x + (1 - alpha) * point_b.x
                    bridge_point.y = alpha * point_a.y + (1 - alpha) * point_b.y
                    bridge_point.z = alpha * point_a.z + (1 - alpha) * point_b.z

                    # 将虚假连接点的反射率降低，模拟不真实的粘连点
                    bridge_point.intensity *= intensity_scale

                    # 将虚假连接点添加到新的点云中
                    new_pointcloud.point.append(bridge_point)
    del pointcloud

    return new_pointcloud


NUM_CHANNELS = 128  # LiDAR 128 通道
# Velodyne 128 仰角表
VERTICAL_ANGLES = [
    -11.742,-1.99,3.4,-5.29,-0.78,4.61,-4.08,1.31,-6.5,-1.11,4.28,-4.41,0.1,6.48,-3.2,2.19,
    -3.86,1.53,-9.244,-1.77,2.74,-5.95,-0.56,4.83,-2.98,2.41,-6.28,-0.89,3.62,-5.07,0.32,7.58,
    -0.34,5.18,3.64,1.75,-25,2.43,2.96,-5.73,0.54,9.7,-2.76,2.63,-7.65,-1.55,3.84,-4.85,
    3.18,-5.51,0.12,5.73,-4.3,1.09,-16.042,-2.21,4.06,-4.63,0.76,15,-3.42,1.97,-6.85,1.33,
    -5.62,-0.23,5.43,-3.53,0.98,-19.582,-2.32,3.07,-4.74,0.65,11.75,-2.65,1.86,-7.15,-1.44,3.95,
    -2.1,3.29,-5.4,-0.01,4.5,-4.19,1.2,13.565,-1.22,4.17,-4.52,0.87,6.08,-3.31,2.08,-6.65,
    1.42,10.346,-1.88,3.51,-6.06,-0.67,4.72,-3.97,2.3,-6.39,-1,4.39,-5.18,0.21,6.98,-3.09,
    4.98,-3.75,1.64,-8.352,-2.54,2.85,-5.84,-0.45,8.43,-2.87,2.52,-6.17,-1.66,3.73,-4.96,0.43,
]

def get_channel_from_angle(angle, vertical_angles=VERTICAL_ANGLES):
    """
    根据仰角找到最近的通道
    :param angle: 当前点的仰角
    :param vertical_angles: 通道的垂直仰角列表
    :return: 通道号
    """
    # 计算点云中每个点的仰角与表中垂直仰角的差值，找到最接近的通道
    closest_channel = min(range(len(vertical_angles)), key=lambda i: abs(vertical_angles[i] - angle))
    return closest_channel

def get_angle_from_z(x, y, z):
    """
    根据点的 x, y, z 坐标计算仰角
    :param x: 点的 x 坐标
    :param y: 点的 y 坐标
    :param z: 点的 z 坐标
    :return: 仰角
    """
    # 计算点的仰角（根据 LiDAR 坐标系的定义，通常为 arctan(z/√(x^2 + y^2))）
    horizontal_distance = math.sqrt(x**2 + y**2)
    angle = math.degrees(math.atan2(z, horizontal_distance))  # 返回角度值
    return angle

def remove_channels(pointcloud, channels_to_remove, vertical_angles=VERTICAL_ANGLES):
    """
    解析 pointcloud 对象并移除指定通道的点
    :param pointcloud: PointCloud protobuf 对象
    :param channels_to_remove: 要移除的通道列表
    :param vertical_angles: LiDAR 的垂直角度分布表
    :return: 移除指定通道后的 PointCloud 对象
    """
    filtered_points = []

    for point in pointcloud.point:
        x = point.x
        y = point.y
        z = point.z
        intensity = point.intensity
        timestamp = point.timestamp
        
        # 根据点的 x, y, z 坐标计算仰角
        angle = get_angle_from_z(x, y, z)
        
        # 根据仰角查找对应的通道号
        channel = get_channel_from_angle(angle, vertical_angles)

        # 过滤掉不在要删除的通道中的点
        if channel not in channels_to_remove:
            filtered_points.append(point)
    # 构造一个新的 PointCloud 对象，并更新点信息
    new_pointcloud = pointcloud

    # 清空新点云的 point 列表
    # 删除原有的点云数据
    del new_pointcloud.point[:]

    # 使用 extend() 方法将过滤后的点添加到新点云
    new_pointcloud.point.extend(filtered_points)

    return new_pointcloud

def remove_channels_test(pointcloud, channels_to_remove, vertical_angles=VERTICAL_ANGLES):
    points = np.asarray(pointcloud.points)
    filtered_points = []
    remove_points = []
    index = 0

    for point in points:
        x = point[0]
        y = point[1]
        z = point[2]
        
        # 根据点的 x, y, z 坐标计算仰角
        angle = get_angle_from_z(x, y, z)
        
        # 根据仰角查找对应的通道号
        channel = get_channel_from_angle(angle, vertical_angles)

        # 过滤掉不在要删除的通道中的点
        if channel not in channels_to_remove:
            filtered_points.append(index)
        else:
            remove_points.append(index)
        index +=1

    # return new_pointcloud,remove_points
    return pointcloud,filtered_points,remove_points


# 引入Line Fault故障干扰
# 选取自车附近的点并引入Line Fault故障
def introduce_line_fault(pcd, fault_ratio=0.1, noise_level=5, car_radius=5):
    points = np.asarray(pcd.points)
    
    # 计算每个点与自车(0,0,0)的距离
    distances = np.linalg.norm(points, axis=1)
    
    # 选取自车附近的点
    near_car_indices = np.where(distances < car_radius)[0]
    num_faulty_points = int(fault_ratio * len(near_car_indices))
    fault_indices = np.random.choice(near_car_indices, num_faulty_points, replace=False)
    
    # 记录原始点的位置
    original_points = points[fault_indices].copy()
    
    # 为故障点引入噪声
    noise_x = np.random.normal(0, noise_level, num_faulty_points)
    noise_y = np.random.normal(0, noise_level, num_faulty_points)
    noise_z = np.random.normal(0, noise_level, num_faulty_points)
    
    points[fault_indices, 0] += noise_x
    points[fault_indices, 1] += noise_y
    points[fault_indices, 2] += noise_z

    # 更新点云对象中的点
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd, fault_indices, original_points



# 模拟电磁干扰故障
def introduce_electromagnetic_interference(pcd, noise_ratio=0.1, noise_level=0.05, loss_ratio=0.05):
    points = np.asarray(pcd.points)
    num_points = len(points)

    # 1. 引入高频随机噪声到部分点
    num_noisy_points = int(noise_ratio * num_points)
    noisy_indices = np.random.choice(num_points, num_noisy_points, replace=False)
    
    # 随机噪声扰动
    noise_x = np.random.normal(0, noise_level, num_noisy_points)
    noise_y = np.random.normal(0, noise_level, num_noisy_points)
    noise_z = np.random.normal(0, noise_level, num_noisy_points)

    points[noisy_indices, 0] += noise_x
    points[noisy_indices, 1] += noise_y
    points[noisy_indices, 2] += noise_z

    # 2. 随机丢失部分点
    num_lost_points = int(loss_ratio * num_points)
    lost_indices = np.random.choice(num_points, num_lost_points, replace=False)
    
    # 将丢失的点设置为 NaN
    points[lost_indices] = np.nan

    # 更新点云对象中的点
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd, noisy_indices, lost_indices


# 计算距离函数
def compute_distance(points, origin):
    return np.linalg.norm(points - origin, axis=1)

# 模拟周边车载激光雷达干扰，集中生成虚假目标
def simulate_lidar_interference(pcd, max_distance=10.0, copy_distance=5.0, concentration_radius=3.0, loss_ratio=0.05):
    points = np.asarray(pcd.points)
    num_points = len(points)

    # 假设车辆中心位于(0, 0, 0)，你可以根据需要修改
    vehicle_origin = np.array([0, 0, 0])

    # 1. 筛选出范围内的点
    distances = compute_distance(points, vehicle_origin)
    nearby_points = points[distances < max_distance]

     # 2. 选择一个中心点，在该中心点周围生成集中区域
    # 随机选择一个点作为中心
    center_index = np.random.choice(len(nearby_points))
    center_point = nearby_points[center_index]

    # 3. 选取在集中区域内的点
    concentrated_points = nearby_points[compute_distance(nearby_points, center_point) < concentration_radius]

     # 4. 随机选择一部分集中点进行复制
    num_to_copy = int(0.3 * len(concentrated_points))  # 复制30%的点
    copy_indices = np.random.choice(len(concentrated_points), num_to_copy, replace=False)
    points_to_copy = concentrated_points[copy_indices]

    # 对虚假目标点进行轻微的扰动，使其分布集中，但不完全重合
    perturbation = np.random.normal(scale=0.1, size=points_to_copy.shape)  # 可调节扰动大小
    points_to_copy = points_to_copy + perturbation

    # 5. 将复制的点沿某个方向移动到新的位置
    # 这里将点沿z轴正方向移动copy_distance，可以根据需要调整
    offer_set = np.random.normal(scale=copy_distance, size=3) 
    new_positions = points_to_copy + np.array([offer_set[0],offer_set[1],0])

    # # 6. 加入一些噪声点
    num_noise_points = int(0.05 * len(points))
    copy_indices = np.random.choice(len(points), num_noise_points, replace=False)
    points_to_copy = points[copy_indices]
    perturbation = np.random.normal(scale=0.5, size=points_to_copy.shape)  # 可调节扰动大小
    noise_points = points_to_copy + perturbation

    # 7. 随机丢失部分点
    num_lost_points = int(loss_ratio * num_points)
    lost_indices = np.random.choice(num_points, num_lost_points, replace=False)

    # 将丢失的点设置为 NaN
    # points[lost_indices] = np.nan

    # 将虚假目标加入到点云
    new_points = np.vstack((points, new_positions,noise_points))

    # 更新点云对象
    pcd.points = o3d.utility.Vector3dVector(new_points)
    
    return pcd, new_positions, lost_indices,noise_points


# 模拟雨雪天干扰
def simulate_rain_snow_interference(pcd, alpha=0.1, noise_std=0.2, noise_density=0.4, radius=20):
    points = np.asarray(pcd.points)

    # 信号衰减
    d = np.linalg.norm(points, axis=1)  # 到原点的距离
    attenuated_intensities = np.exp(-alpha * d)

    # 将强度标准化到 [0, 1]
    normalized = (attenuated_intensities - np.min(attenuated_intensities)) / (np.max(attenuated_intensities) - np.min(attenuated_intensities))
    colors = plt.cm.viridis(normalized)[:, :3]  # 使用viridis色图，只取RGB通道

    # 2. 生成噪点

    # 获取点云的边界范围
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)  # 每个轴的最小值
    max_bound = points.max(axis=0)  # 每个轴的最大值
    num_noise_points = int(noise_density * len(points))
    # 在范围内生成随机噪声点
    noise_points = np.random.uniform(min_bound, max_bound, size=(num_noise_points, 3))
     # 确保噪声点的 z 轴非负（如果生成的点有负值，则调整）
    noise_points[:, 2] = np.abs(noise_points[:, 2]) + min_bound[2]
    new_points = np.vstack((points, noise_points))

    noise_colors = np.zeros_like(noise_points)
    noise_colors[:] = [1,0,0]  # 红色噪点
    new_colors = np.vstack((colors, noise_colors))

    # 生成在球形范围内的随机噪声点
    num_noise_r_points=int(num_noise_points*0.5)
    noise_r_points = np.zeros((num_noise_r_points, 3))

    for i in range(num_noise_r_points):
        # 生成球坐标系中的随机点
        r = radius * np.cbrt(np.random.uniform(0, 1))  # 距离 r 在 [0, radius] 之间
        theta = np.random.uniform(0, 2 * np.pi)        # 水平角度 theta 在 [0, 2π] 之间
        phi = np.random.uniform(0, np.pi / 2)          # 垂直角度 phi 在 [0, π/2] 之间，确保z轴非负
        
        # 转换为笛卡尔坐标系
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        noise_r_points[i] = [x, y, z + min_bound[2]]  # 确保z轴非负
    
    new_points = np.vstack((new_points, noise_r_points))
    noise_r_colors = np.zeros_like(noise_r_points)
    # noise_r_colors[:] = [199/255,21/255,133/255]  # 中紫色噪点
    noise_r_colors[:] = [1,0,0]  # 黄色噪点
    new_colors = np.vstack((new_colors, noise_r_colors))

     # 4. 更新点云对象
    pcd.points = o3d.utility.Vector3dVector(new_points)

    # 更新点云颜色
    pcd.colors = o3d.utility.Vector3dVector(new_colors)

    
    return pcd,noise_points

# 随机生成光源位置
def generate_random_light_source(max_distance=50):
    # 在车辆周围 max_distance 米范围内随机生成光源
    light_source = np.random.uniform(-max_distance, max_distance, size=3)
    light_source[2] = np.abs(light_source[2])  # 确保z轴非负
    return light_source

# 模拟强光干扰，减少点云密度和测量距离
def simulate_strong_light_interference(pcd, distance_threshold=30, reduction_ratio=0.5, angle_threshold=45):
    points = np.asarray(pcd.points)
    light_source = generate_random_light_source(max_distance=30)
    # 计算每个点与光源的距离和夹角
    light_direction = light_source / np.linalg.norm(light_source)  # 光源方向向量
    angles = np.degrees(np.arccos(np.dot(points, light_direction) / (np.linalg.norm(points, axis=1) * np.linalg.norm(light_direction))))

    # 找到受影响的点（假设光源影响一定角度范围内的点）
    affected_indices = np.where(angles < angle_threshold)[0]

    # 1. 减少测量距离：将距离超过阈值的点剔除
    distances = np.linalg.norm(points[affected_indices], axis=1)
    valid_indices = affected_indices[distances <= distance_threshold]
 
    invalid_indices= affected_indices[distances > distance_threshold]

    # 2. 随机剔除一定比例的点以减少点云数量
    num_points_to_keep = int(len(valid_indices) * (1 - reduction_ratio))
    reduced_indices = np.random.choice(valid_indices, num_points_to_keep, replace=False)

    # 获取被剔除的点
    all_valid_indices = set(valid_indices)
    kept_indices = set(reduced_indices)
    removed_indices = np.array(list(all_valid_indices - kept_indices),dtype=int)

    removed_points = points[removed_indices]  # 被剔除的点

    colors = np.zeros_like(points)
    colors[:] = [0,0,1]  

    colors[invalid_indices] = [1,1,0]  

    colors[removed_indices] = [0,1,0]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# 函数：从点云文件保存图片
def save_pointcloud_as_image(pointcloud,image_str):

    # 提取点数据
    points = []
    colors = []

    for point in pointcloud.point:
        # 添加坐标
        points.append([point.x, point.y, point.z])
        # 假设使用强度值作为颜色（归一化处理）
        intensity = point.intensity / 255.0  # 将强度值归一化到[0, 1]
        colors.append([intensity, intensity, intensity])  # 灰度值
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    global VIS  # 声明全局变量

    if VIS == None:
        VIS = o3d.visualization.Visualizer()
        VIS.create_window()
        # 设置视点和背景颜色
        VIS.get_render_option().background_color = [1, 1, 1]  # 白色背景
        VIS.get_render_option().point_size = 2.0  # 设置点大小

    # 将点云添加到渲染器中
    VIS.add_geometry(pcd)

    # 渲染并保存为图片
    VIS.poll_events()
    VIS.update_renderer()

    tstamp = pointcloud.measurement_time

    temp_time = str(tstamp).split('.')
    if len(temp_time[1]) == 1:
        temp_time1_adj = temp_time[1] + '0'
    else:
        temp_time1_adj = temp_time[1]
    image_time = temp_time[0] + '_' + temp_time1_adj
    image_filename = image_str + image_time + ".jpeg"

    VIS.capture_screen_image(image_filename)
    VIS.remove_geometry(pcd)

