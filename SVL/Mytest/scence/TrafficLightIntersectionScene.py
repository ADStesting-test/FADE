import lgsvl
from environs import Env
from scence.SceneBase import SceneBase
import config


class TrafficLightIntersectionScene(SceneBase):
    
    def __init__(self,simulation_environment, environment_variables,ego_vehicle_type =config.EGO_VEHILCE_TYPE):

        super().__init__(simulation_environment, environment_variables,ego_vehicle_type)

    def create_ego(self, transform=None):
        self.destination = self.sim.map_point_on_lane(lgsvl.Vector(125.275100708008, -4.158447265625, -57.6787948608398))  
        # Ego
        transform =  self.sim.get_spawn()[0]
        super().create_ego(transform = transform)
        return self.ego

    def create_scence(self):
        # self.sim.add_random_agents(lgsvl.AgentType.NPC)

        # self.create_pedestrian()
        pass
    
    def create_pedestrian(self):
        self.sim.add_random_agents(lgsvl.AgentType.PEDESTRIAN)
