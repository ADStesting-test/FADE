import os
import random
import lgsvl
from environs import Env

import math
import time


def disnable_modules(dreamview, modules):
    """
    Disables a list of modules in the provided object (dv) one by one.

    Args:
        dreamview: The object that has a method `disable_module` to disable modules.
        modules (list): A list of module names (strings) to be disabled.

    Example:
        disable_modules(apollo, ['planning', 'control', 'localization'])

    Each module is disabled with a 1-second pause between actions.
    """
    not_all = True
    while not_all:
        not_all = False
        module_status = dreamview.get_module_status()
        for module, status in module_status.items():
            if status and (module in modules):
                dreamview.disable_module(module)
                not_all = True
        time.sleep(1)

def enable_modules(dreamview, modules):
    """
    Enables a list of modules in the provided object (dv) one by one.
    Args:
        dreamview: The object that has a method `enable_module` to enable modules.
        modules (list): A list of module names (strings) to be enabled.
    Example:
    enable_modules(apollo, ['planning', 'control', 'localization'])

    Each module is enabled with a 1-second pause between actions.
    """
    # try 5 times
    not_all = True
    while not_all:
        not_all = False
        module_status = dreamview.get_module_status()
        for module, status in module_status.items():
            if (not status) and (module in modules):
                dreamview.enable_module(module)
                not_all = True
        time.sleep(1)

def update_camera(sim, vehicle_state):
    """
    Updates the camera position and rotation to match the vehicle's current position and rotation.
    Args:
        sim: The simulator object.
        vehicle_state: The vehicle's current state.
    """
    position = vehicle_state.transform.position + lgsvl.Vector(-1, 4, -5)  # 相机在车辆后方4米高
    rotation = vehicle_state.transform.rotation + lgsvl.Vector(25,0,0,) # 相机在车辆后方 向前倾斜25度
    tr = lgsvl.Transform(position, rotation)
    sim.set_sim_camera(tr)


def save_lidar(filename,vehicle):
    """
    Saves the lidar data from the simulator to a file.
    Args:
        sim: The simulator object.
        filename: The name of the file to save the lidar data to.
        vehicle: The vehicle object to get the lidar data from.
    """
    sensors = vehicle.get_sensors()
    for s in sensors:
        if s.name == "Lidar":
            s.save(filename+".pcd")
            break

def save_image(filename,vehicle):
    """
    Saves the image from the simulator to a file.
    Args:
        sim: The simulator object.
        filename: The name of the file to save the image to.
        vehicle: The vehicle object to get the image from.
    """
    sensors = vehicle.get_sensors()
    for s in sensors:
        if s.name == "Main Camera":
            # print("Saving image to {}".format(filename+".png"))
            s.save(filename+".png",compression=0)
            break