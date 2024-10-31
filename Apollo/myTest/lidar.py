import multiprocessing
from multiprocessing import Process, Queue,Event
import signal
import random
import numpy as np
import copy
from pypcd import pypcd

import lidar_operation
from cyber.python.cyber_py3 import cyber
from modules.drivers.proto.pointcloud_pb2 import PointCloud

MSG_TYPE = PointCloud

q_lidar128 = Queue(maxsize=200)
# 定义一个事件，用于控制进程的停止
stop_event = Event()

output_path="/apollo/data/lidar/"
channels_to_remove = random.sample(range(128),30)


def callback_128(msg):
    """
    Reader message callback.
    """
    # print(msg)

    # 对点云进行处理

    # 移除特定通道
    channels_to_remove = random.sample(range(128),30)
    msg = lidar_operation.remove_channels(msg, channels_to_remove)


    # # 引入 10% 的 Line Fault 故障，噪声强度为 5
    # msg = lidar_operation.introduce_line_fault(msg, fault_ratio=0.1, noise_level=5, car_radius=100)


    # # 引入电磁干扰，10% 点受到噪声影响，5% 点丢失，噪声强度为 0.05
    # msg = lidar_operation.introduce_electromagnetic_interference(msg, noise_ratio=0.1, noise_level=1, loss_ratio=0.1)


    # # 模拟雷达干扰，5% 点丢失
    # msg = lidar_operation.simulate_lidar_interference(msg,max_distance=10.0, copy_distance=10.0, concentration_radius=3.0, loss_ratio=0.1)


    # # 模拟雨雪干扰
    # msg = lidar_operation.simulate_rain_snow_interference(msg, alpha=0.1, noise_std=0.2, noise_density=0.2)
    

    # # 模拟强光干扰
    # light_source= lidar_operation.generate_random_light_source(max_distance=30)
    # msg = lidar_operation.simulate_strong_light_interference(msg, light_source=light_source,distance_threshold=30, reduction_ratio=0.5, angle_threshold=45)
    

    # q_lidar128.put(add_noise_pc_msg)
    if not q_lidar128.full():
        q_lidar128.put(msg)

def lidar_128_listener_class(q):
    """
    Reader message.
    """
    print("read lidar128 PointCloud2:")
    cyber.init()
    lidar_128_node = cyber.Node("lidar_128_listener")
    lidar_128_node.create_reader("/apollo/sensor/zlding/lidar128/compensator/PointCloud2", MSG_TYPE, callback_128)
    lidar_128_node.spin()

def lidar_128_talker_class(q):
    """
    Test talker.
    """
    print("talker_lidar128")
    cyber.init()
    talker_lidar128_node = cyber.Node("lidar_128_talker")
    writer = talker_lidar128_node.create_writer("/apollo/sensor/lidar128/compensator/PointCloud2", MSG_TYPE)

    while not cyber.is_shutdown() and not stop_event.is_set():
        # message = q_lidar128.get()
        # writer.write(message)
        try:
            # 从队列中获取多个消息
            batch = []
            while len(batch) < 1:  # 批量大小可以调整
                try:
                    message = q_lidar128.get(timeout=1)  # 设置超时，避免长时间阻塞
                    batch.append(message)
                except Exception as e:
                    break  # 如果获取失败，退出循环
            
            if batch:
                # 处理批量消息
                for message in batch:
                    writer.write(message)
        except Exception as e:
            print(f"Error in talker: {e}")

def signal_handler(sig, frame):
    print("Signal received, stopping processes...")
    stop_event.set()  # 通知所有进程停止

if __name__ == '__main__':

     # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)

    listen_lidar128 = multiprocessing.Process(target=lidar_128_listener_class, args=(q_lidar128,))
    talker_lidar128 = multiprocessing.Process(target=lidar_128_talker_class, args=(q_lidar128,))

    listen_lidar128.start()
    talker_lidar128.start()

    try:
        # 主进程等待，直到按下 Ctrl+C
        listen_lidar128.join()
        talker_lidar128.join()
    except Exception as e:
        print(f"Error in main process: {e}")

    finally:
        print("Terminating all processes...")
        stop_event.set()  # 设置停止事件
        # 终止所有子进程
        listen_lidar128.terminate()
        talker_lidar128.terminate()

        # 等待所有子进程结束
        listen_lidar128.join()
        talker_lidar128.join()


    
    