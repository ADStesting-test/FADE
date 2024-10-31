import lgsvl
from environs import Env

import lgsvl.wise
from scence.StraightDrivingScene import StraightDrivingScene
from scence.LaneChangeScene import LaneChangeScene
from scence.TrafficLightIntersectionScene import TrafficLightIntersectionScene
from scence.RightTurnScene import RightTurnScene
from scence.LeftTurnScene import LeftTurnScene
from scence.CrossIntersectionScene import CrossIntersectionScene
from scence.SceneBase import *
from tools import *
import config 


import time

"""
This function `run` sets up and executes a simulation scenario using the CARLA simulator. It configures the scene based on the specified parameters, creates an ego vehicle, resets Apollo and Dreamview, and starts the simulation. After the simulation completes, it resets the scene and cleans up resources.

- **Parameters**:
  - `sim`: The CARLA simulation object.

- **Scene Configuration**:
  - The scene is configured to be a left turn scene with pedestrian interactions on the San Francisco map.

- **Post-Simulation Actions**:
  - The function waits for 5 seconds after resetting the scene to ensure proper cleanup and resource deallocation.
"""
def run(sim):

    spawns = sim.get_spawn()

    ## 地图为 Borregas Ave
    # scence = StraightDrivingScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE)
    # scence = StraightDrivingScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type = "pedestrian")

    # scence = TrafficLightIntersectionScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE)

    ## 地图为 San Francisco 
    # scence = LaneChangeScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE)
    # scence = LaneChangeScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type = "overtake")
    # scence = LaneChangeScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type = "npc_lanechange")

    # scence = RightTurnScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE) （舍弃）
    # scence = RightTurnScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type="right_line")
    # scence = RightTurnScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type="pedestrian")

    # scence = LeftTurnScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE)
    scence = LeftTurnScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type="right_turn")  #(舍弃)
    # scence = LeftTurnScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type="pedestrian")

    # scence = CrossIntersectionScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE)
    # scence = CrossIntersectionScene(sim,env,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type="follow_npc")

    scence.create_ego()
    # sim.set_sim_camera_state(sim.SimulatorCameraState.DRIVER)
    spawns = sim.get_spawn()

     # 重置Apollo和Dreamview
    dv = scence.reset_apollo_dreamview(sim,scence.ego,scence.destination,config.MODULES)
    scence.create_scence()

    scence.start_simulation()

    # 断开Bridge和清理资源    
    scence.reset()
    time.sleep(5)

if __name__ == "__main__":


    env = Env()

    sim = lgsvl.Simulator(
        env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host),
        env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port)
    )

    # print("Current scene is: ", sim.current_scene)
    if sim.current_scene == config.LGSVL_HD_MAP:
        sim.reset()
    else:
        sim.load(config.LGSVL_HD_MAP)

    for _ in range(config.LOOP_NUM):
        sim.reset()
        time.sleep(1)
        run(sim)



        
