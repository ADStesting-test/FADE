import multiprocessing
from multiprocessing import Process, Queue,Event
import signal
import add_noise 


from cyber.python.cyber_py3 import cyber
from modules.drivers.proto.sensor_image_pb2 import CompressedImage

import random

MSG_TYPE = CompressedImage

q_6mm = Queue(maxsize=100)
q_12mm = Queue(maxsize=100)

# 定义一个事件，用于控制进程的停止
stop_event = Event()

def callback_6mm(msg):
    """
    Reader message callback.
    """
    # print(msg)
    # print("callback_6mm")
    # save_image(msg,"")
    # q_6mm.put(msg)

    # msg = add_noise.process_image(msg,"occlusion",num = random.randint(30,50))

    # msg = add_noise.process_image(msg,"occlusion")

    # msg = add_noise.process_image(msg,"gaussian")

    # msg = add_noise.process_image(msg,"scatter")

    msg = add_noise.process_image(msg,"ice")

    # msg = add_noise.process_image(msg,"rain_effect")

    # msg = add_noise.process_image(msg,"dust_effect")

    # msg = add_noise.process_image(msg,"snow_effect",light_position=True)

    # msg = add_noise.process_image(msg,"overexposure")

    # msg = add_noise.process_image(msg,"white_balance")

    # msg = add_noise.process_image(msg,"cracks",num=100)

    # msg = add_noise.process_image(msg,"large_cracks")

    # msg = add_noise.process_image(msg,"radiating_cracks")

    add_noise.save_image(msg,"ice")
    if not q_6mm.full():
        q_6mm.put(msg)

    

def callback_12mm(msg):
    """
    Reader message callback.
    """
    # print(data)
    # print("callback_12mm")
    if not q_12mm.full():
        q_12mm.put(msg)
    
    # msg_ir = add_noise.process_image(msg,"occlusion",num = random.randint(30,50))

    # msg_ir = add_noise.process_image(msg,"occlusion")

    # msg_gauss = add_noise.process_image(msg,"gaussian")

    # msg_scatter = add_noise.process_image(msg,"scatter")

    # msg_rain_effect = add_noise.process_image(msg,"rain_effect")

    # msg_dust_effect = add_noise.process_image(msg,"dust_effect")

    # msg_snow_effect = add_noise.process_image(msg,"snow_effect",light_position=True)

    # msg_overexposure = add_noise.process_image(msg,"overexposure")

    # msg_white_balance = add_noise.process_image(msg,"white_balance")

    # msg_cracks = add_noise.process_image(msg,"cracks",num=100)

    # msg_large_cracks = add_noise.process_image(msg,"large_cracks")

    # msg_radiating_cracks = add_noise.process_image(msg,"radiating_cracks")
    
    # if not q_12mm.full():
    #     q_12mm.put(msg_ir)



def camera_6mm_listener_class(q):
    """
    Reader message.
    """
    print("read 6mm image:")
    cyber.init()
    camera_6mm_node = cyber.Node("camera_6mm_listener")
    camera_6mm_node.create_reader("/apollo/sensor/zlding/camera/front_6mm/image/compressed", MSG_TYPE, callback_6mm)
    camera_6mm_node.spin()

def camera_12mm_listener_class(q):
    print("read 12mm image:")
    cyber.init()
    camera_12mm_node = cyber.Node("camera_12mm_listener")
    camera_12mm_node.create_reader("/apollo/sensor/zlding/camera/front_12mm/image/compressed", MSG_TYPE, callback_12mm)
    camera_12mm_node.spin()

def camera_6mm_talker_class(q):
    """
    Test talker.
    """
    print("talker_6mm")
    cyber.init()
    talker_6mm_node = cyber.Node("camera_6mm_talker")
    writer = talker_6mm_node.create_writer("/apollo/sensor/camera/front_6mm/image/compressed", MSG_TYPE)

    while not cyber.is_shutdown() and not stop_event.is_set():
        # message = q_lidar128.get()
        # writer.write(message)
        try:
            # 从队列中获取多个消息
            batch = []
            while len(batch) < 1:  # 批量大小可以调整
                try:
                    message = q_6mm.get(timeout=0.5)  # 设置超时，避免长时间阻塞
                    batch.append(message)
                except Exception as e:
                    break  # 如果获取失败，退出循环
            
            if batch:
                # 处理批量消息
                for message in batch:
                    writer.write(message)
            # message = q_6mm.get(timeout=0.5)  # 设置超时，避免长时间阻塞
            # writer.write(message)
        except Exception as e:
            print(f"Error in talker: {e}")



def camera_12mm_talker_class(q):
    """
    Test talker.
    """
    print("talker_12mm")
    cyber.init()
    camera_12mm_node = cyber.Node("camera_12mm_talker")
    writer = camera_12mm_node.create_writer("/apollo/sensor/camera/front_12mm/image/compressed",MSG_TYPE)

    while not cyber.is_shutdown() and not stop_event.is_set():
        # message = q_lidar128.get()
        # writer.write(message)
        try:
            # 从队列中获取多个消息
            batch = []
            while len(batch) < 3:  # 批量大小可以调整
                try:
                    message = q_12mm.get(timeout=0.1)  # 设置超时，避免长时间阻塞
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

    listen_6mm = multiprocessing.Process(target=camera_6mm_listener_class, args=(q_6mm,))
    talker_6mm = multiprocessing.Process(target=camera_6mm_talker_class, args=(q_6mm,))

    # listen_12mm = multiprocessing.Process(target=camera_12mm_listener_class, args=(q_12mm,))
    # talker_12mm = multiprocessing.Process(target=camera_12mm_talker_class, args=(q_12mm,))

    
    listen_6mm.start()
    talker_6mm.start()
    
    # listen_12mm.start()
    # talker_12mm.start()
    

    try:
        # 主进程等待，直到用户按下 Ctrl+C
        listen_6mm.join()
        talker_6mm.join()
        # listen_12mm.join()
        # talker_12mm.join()

    except KeyboardInterrupt:
        print("Terminating all processes...")
        # 终止所有子进程
        listen_6mm.terminate()
        talker_6mm.terminate()
        # listen_12mm.terminate()
        # talker_12mm.terminate()

        # 等待所有子进程结束
        listen_6mm.join()
        talker_6mm.join()
        # listen_12mm.join()
        # talker_12mm.join()

    