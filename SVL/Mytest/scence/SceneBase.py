import lgsvl
from environs import Env
import time 
import config
import tools

class SceneBase:
    
    def __init__(self, simulation_environment, environment_variables, ego_vehicle_type=config.EGO_VEHILCE_TYPE,scene_type = None,destination=None):
        """
        :param simulation_environment: 仿真环境    
        :param environment_variables: 环境变量
        :param ego_vehicle_type: ego车辆类型
        :param scene_type: 场景类型
        :param destination: 目的地
        """
        self.sim = simulation_environment
        self.env = environment_variables
        self.ego_type = ego_vehicle_type
        self.ego = None
        self.npcs = []
        self.pedestrians = []
        self.dv = None
        self.stop = False
        self.spawns = self.sim.get_spawn()
        self.scene_type = scene_type
        self.destination = destination

        # 确保每次运行后都正确复位Apollo和Dreamview
    def reset_apollo_dreamview(self,sim,ego, destination,modules):
        """

        """
        times = 0
        success = False
        while times < 3:
            try:
                dv = lgsvl.dreamview.Connection(sim,ego, self.env.str("LGSVL__AUTOPILOT_0_HOST", "127.0.0.1"))
                dv.set_hd_map(config.APOLLO_HD_MAP)
                dv.set_vehicle(config.LGSVL_VEHICLE)
                dv.setup_apollo(destination.position.x, destination.position.z, modules)
                success = True
                break
            except:
                print('Fail to spin up Apollo, try again!')
                times += 1  
        if success:
            self.dv = dv
            return dv
        else:
            raise RuntimeError('Fail to spin up apollo')

    def create_ego(self, transform):
        ego_state = lgsvl.AgentState()
        ego_state.transform = transform
        self.ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", self.ego_type), lgsvl.AgentType.EGO, ego_state)
        # tools.update_camera(self.sim, self.ego.state)
        self.ego.connect_bridge(
            self.env.str("LGSVL__AUTOPILOT_0_HOST", lgsvl.wise.SimulatorSettings.bridge_host),
            self.env.int("LGSVL__AUTOPILOT_0_PORT", lgsvl.wise.SimulatorSettings.bridge_port)
        )

        def on_collision(agent1, agent2, contact):
            name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
            name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
            print(str(name1) + " collided with " + str(name2) + " at " + str(contact))
            self.sim.stop()
            self.stop = True 
            time.sleep(1) # 等待仿真器处理
            print("Simulation stopped")

        self.ego.on_collision(on_collision)

    def create_npc(self, npc_type,transform,waypoints=None,follow_lane=False,speed=9.1,isRepeat=False):
        npc_state = lgsvl.AgentState()
        npc_state.transform = transform
        npc = self.sim.add_agent(npc_type, lgsvl.AgentType.NPC, npc_state)
        if waypoints: 
            npc.follow(waypoints,isRepeat)

        if follow_lane:
            # If the passed bool is False, then the NPC will not moved
            # The float passed is the maximum speed the NPC will drive
            # 9.1 m/s is ~32.8 km/h
            npc.follow_closest_lane(True, speed)
        self.npcs.append(npc)

    def create_pedestrian(self, pedestrian_type,transform,waypoints=None,speed=1,isRepeat=False):
        ped_state = lgsvl.AgentState()
        ped_state.transform = transform
        ped = self.sim.add_agent(pedestrian_type, lgsvl.AgentType.PEDESTRIAN, ped_state)
        if waypoints:
            ped.follow(waypoints, isRepeat)
        self.pedestrians.append(ped)
    
    def create_scence():
        raise NotImplementedError("You must implement create_scence in your subclass")
    
    def create_pedestrians(self):
        raise NotImplementedError("You must implement create_pedestrians in your subclass")

    def reset(self):
        self.sim.reset()

    def stop_simulation(self):
        self.sim.stop()

    def start_simulation(self):
        # self.sim.run(2)
        # time.sleep(5)
        total_sim_time = config.TOTAL_SIM_TIME
        time_slice_size = config.TIME_SLICE_SIZE
        action_change_freq = 1.0 / time_slice_size
        time_index = 0

        if config.ENABLE_RECORD:
            tools.enable_modules(self.dv, config.DV_MODULES)
            time.sleep(1)

        for t in range(0, int(total_sim_time)):
            for j in range(0, int(time_slice_size)):
                    if not self.stop:
                        self.sim.run(0.1)
                        if config.SAVE_LIDAR:
                            tools.save_lidar(config.FILE_PATH+"/lidar/"+config.SCENE_STR+str(time_index),self.ego)
                        if config.SAVE_IMAGE:
                            tools.save_image(config.FILE_PATH+"/images/"+config.SCENE_STR+str(time_index),self.ego)
                        time_index += 1
                        # tools.update_camera(self.sim, self.ego.state)
                    else:
                        break

        if config.ENABLE_RECORD:
            tools.disnable_modules(self.dv, config.DV_MODULES)
            time.sleep(1)

        tools.disnable_modules(self.dv,config.MODULES) # 关闭apollo相关模块,并清理模块中占用的内存