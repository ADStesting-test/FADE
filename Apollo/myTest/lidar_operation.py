import random
import numpy as np
import copy
from pypcd import pypcd
import math
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于颜色映射

# import sensor_msg_extractor
from cyber.python.cyber_py3 import cyber
from modules.drivers.proto.pointcloud_pb2 import PointCloud,PointXYZIT


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
        # intensity = point.intensity
        # timestamp = point.timestamp
        
        # 根据点的 x, y, z 坐标计算仰角
        angle = get_angle_from_z(x, y, z)
        
        # 根据仰角查找对应的通道号
        channel = get_channel_from_angle(angle, vertical_angles)

        # 过滤掉不在要删除的通道中的点
        if channel not in channels_to_remove:
            filtered_points.append(point)
    # 清空新点云的 point 列表
    # 删除原有的点云数据
    del pointcloud.point[:]

    # 使用 extend() 方法将过滤后的点添加到新点云
    pointcloud.point.extend(filtered_points)

    return pointcloud

# 引入Line Fault故障干扰
# 选取自车附近的点并引入Line Fault故障
def introduce_line_fault(pointcloud, fault_ratio=0.1, noise_level=5, car_radius=5):

    # 提取点云数据
    points = np.array([[p.x, p.y, p.z] for p in pointcloud.point])  # 将 pointcloud 中的点转为 numpy 数组
    
    # 计算每个点与自车(0,0,0)的距离
    distances = np.linalg.norm(points, axis=1)
    
    # 选取自车附近的点
    near_car_indices = np.where(distances < car_radius)[0]
    num_faulty_points = int(fault_ratio * len(near_car_indices))
    fault_indices = np.random.choice(near_car_indices, num_faulty_points, replace=False)

    
    # 为故障点引入噪声
    noise_x = np.random.normal(0, noise_level, num_faulty_points)
    noise_y = np.random.normal(0, noise_level, num_faulty_points)
    noise_z = np.random.normal(0, noise_level, num_faulty_points)
    
    points[fault_indices, 0] += noise_x
    points[fault_indices, 1] += noise_y
    points[fault_indices, 2] += noise_z

    # 更新 pointcloud 对象的点
    for i, idx in enumerate(fault_indices):
        pointcloud.point[idx].x = points[idx, 0]
        pointcloud.point[idx].y = points[idx, 1]
        pointcloud.point[idx].z = points[idx, 2]
    
    return pointcloud

# 模拟电磁干扰故障
def introduce_electromagnetic_interference(pointcloud, noise_ratio=0.1, noise_level=0.05, loss_ratio=0.05):
     # 提取点云数据
    points = np.array([[p.x, p.y, p.z] for p in pointcloud.point])  # 将 pointcloud 中的点转为 numpy 数组
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

    # 更新 pointcloud 对象中的点
    for i, p in enumerate(pointcloud.point):
        if i in noisy_indices:
            p.x, p.y, p.z = points[i]  # 更新加入噪声后的坐标
        if i in lost_indices:
            p.x, p.y, p.z = np.nan, np.nan, np.nan  # 将丢失的点设置为 NaN

    return pointcloud

# 计算距离函数
def compute_distance(points, origin):
    return np.linalg.norm(points - origin, axis=1)

# 模拟周边车载激光雷达干扰，集中生成虚假目标
def simulate_lidar_interference(pointcloud, max_distance=10.0, copy_distance=5.0, concentration_radius=3.0, loss_ratio=0.05):
     # 提取点云数据
    points = np.array([[p.x, p.y, p.z] for p in pointcloud.point])
    intensities = np.array([p.intensity for p in pointcloud.point])
    timestamps = np.array([p.timestamp for p in pointcloud.point])
    num_points = len(points)

    # 车辆中心位于(0, 0, 0)
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

    # 6. 为新点分配强度和时间戳
    copied_intensities = intensities[copy_indices]  # 保持与原点相同的强度
    copied_timestamps = timestamps[copy_indices]  # 保持与原点相同的时间戳

    # 7. 加入一些随机噪声点
    num_noise_points = int(0.05 * len(points))
    noise_indices = np.random.choice(len(points), num_noise_points, replace=False)
    noise_points = points[noise_indices] + np.random.normal(scale=0.5, size=(num_noise_points, 3))  # 可调节噪声大小
    noise_intensities = intensities[noise_indices]  # 给噪声点赋予相应强度
    noise_timestamps = timestamps[noise_indices]  # 给噪声点赋予相应的时间戳

    # 8. 随机丢失部分点
    num_lost_points = int(loss_ratio * num_points)
    lost_indices = np.random.choice(num_points, num_lost_points, replace=False)

     # 9. 将虚假目标和噪声点加入到点云
    new_points = np.vstack((points, new_positions, noise_points))
    new_intensities = np.concatenate((intensities, copied_intensities, noise_intensities))
    new_timestamps = np.concatenate((timestamps, copied_timestamps, noise_timestamps))

    # 更新点云对象
    # 清空新点云的 point 列表
    # 删除原有的点云数据
    del pointcloud.point[:]

    for i in range(len(new_points)):
        point = PointXYZIT()
        point.x,point.y,point.z =  new_points[i]
        point.intensity = new_intensities[i]
        point.timestamp = new_timestamps[i]
        pointcloud.point.append(point)  # 将点加入到点云中
    
    return pointcloud


# 模拟雨雪天干扰
def simulate_rain_snow_interference(pointcloud, alpha=0.1, noise_std=0.2, noise_density=0.4, radius=20):

    # 提取点云数据
    points = np.array([[p.x, p.y, p.z] for p in pointcloud.point])
    intensities = np.array([p.intensity for p in pointcloud.point])
    timestamps = np.array([p.timestamp for p in pointcloud.point])

     # 1. 信号衰减
    d = compute_distance(points, np.array([0, 0, 0]))  # 到原点的距离
    attenuated_intensities = intensities*np.exp(-alpha * d)

    # 2. 生成噪点

    # 获取点云的边界范围
    min_bound = points.min(axis=0)  # 每个轴的最小值
    max_bound = points.max(axis=0)  # 每个轴的最大值
    num_noise_points = int(noise_density * len(points))
    # 在范围内生成随机噪声点
    noise_points = np.random.uniform(min_bound, max_bound, size=(num_noise_points, 3))
     # 确保噪声点的 z 轴非负（如果生成的点有负值，则调整）
    noise_points[:, 2] = np.abs(noise_points[:, 2]) + min_bound[2]
    new_points = np.vstack((points, noise_points))

     # 3. 生成在球形范围内的随机噪声点
    num_noise_r_points = int(num_noise_points * 0.5)
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
    
     # 合并所有点
    new_points = np.vstack((new_points, noise_r_points))

    # 生成新的强度值和时间戳
    noise_intensities = np.random.uniform(np.min(intensities), np.max(intensities), num_noise_points + num_noise_r_points)
    noise_timestamps = np.random.uniform(np.min(timestamps), np.max(timestamps), num_noise_points + num_noise_r_points)

    # 4. 更新点云对象并赋值
    del pointcloud.point[:] 
    for i in range(len(new_points)):
        point = PointXYZIT()
        point.x,point.y,point.z =  new_points[i]
        if i < len(intensities):  # 原有点
            point.intensity = intensities[i]
            point.timestamp = timestamps[i]
        else:  # 新生成的噪声点
            point.intensity = int(noise_intensities[i - len(intensities)])
            point.timestamp = int(noise_timestamps[i - len(timestamps)])
        
        
        pointcloud.point.append(point)  # 将点加入到点云中

    return pointcloud


# 随机生成光源位置
def generate_random_light_source(max_distance=50):
    # 在车辆周围 max_distance 米范围内随机生成光源
    light_source = np.random.uniform(-max_distance, max_distance, size=3)
    light_source[2] = np.abs(light_source[2])  # 确保z轴非负
    return light_source

# 模拟强光干扰，减少点云密度和测量距离
def simulate_strong_light_interference(pointcloud, light_source,distance_threshold=30, reduction_ratio=0.5, angle_threshold=45):

    # 提取点云中的位置、强度和时间戳数据
    points = np.array([[p.x, p.y, p.z] for p in pointcloud.point])
    intensities = np.array([p.intensity for p in pointcloud.point])
    timestamps = np.array([p.timestamp for p in pointcloud.point])

    # 计算每个点与光源的距离和夹角
    light_direction = light_source / np.linalg.norm(light_source)  # 光源方向向量
    angles = np.degrees(np.arccos(np.dot(points, light_direction) / (np.linalg.norm(points, axis=1) * np.linalg.norm(light_direction))))

    # 找到受影响的点（假设光源影响一定角度范围内的点）
    affected_indices = np.where(angles < angle_threshold)[0]

    # 1. 减少测量距离：将距离超过阈值的点剔除
    distances = np.linalg.norm(points[affected_indices], axis=1)
    valid_indices = affected_indices[distances <= distance_threshold]
    invalid_indices = affected_indices[distances > distance_threshold]

    # 2. 随机剔除一定比例的点以减少点云数量
    num_points_to_keep = int(len(valid_indices) * (1 - reduction_ratio))
    reduced_indices = np.random.choice(valid_indices, num_points_to_keep, replace=False)

    # 获取被剔除的点
    all_valid_indices = set(valid_indices)
    kept_indices = set(reduced_indices)
    removed_indices = np.array(list(all_valid_indices - kept_indices), dtype=int)


    # 3. 更新点云，保留剩余的点
    for i, p in enumerate(pointcloud.point):
        if i in invalid_indices or i in removed_indices:
            p.x, p.y, p.z = np.nan, np.nan, np.nan  # 将丢失的点设置为 NaN


    return pointcloud


