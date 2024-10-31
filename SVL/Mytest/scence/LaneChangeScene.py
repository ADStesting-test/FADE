import lgsvl
from environs import Env
import config
from scence.SceneBase import SceneBase


class LaneChangeScene(SceneBase):

    def __init__(self,simulation_environment, environment_variables,ego_vehicle_type = config.EGO_VEHILCE_TYPE,scene_type = None):

        super().__init__(simulation_environment, environment_variables,ego_vehicle_type,scene_type = scene_type)
        


    def create_ego(self, transform=None):
        self.sim.set_time_of_day(0.0)

        if self.scene_type == None or self.scene_type == 'default':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-538.360107421875, 10.1981086730957, -97.723648071289))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-565.556274414063, 10.1981029510498, 18.4931106567383))

        elif self.scene_type == 'overtake':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-540.360107421875, 10.1981086730957, -97.723648071289))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-561.556274414063, 10.1981029510498,  28.4931106567383))
        elif self.scene_type == 'npc_lanechange':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-538.360107421875, 10.1981086730957, -97.723648071289))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-558.556274414063, 10.1981029510498, 8.4931106567383))
        
        super().create_ego(transform = transform)
        return self.ego

    def create_scence(self):

        if self.scene_type == None or self.scene_type == 'default':
            # NPC 1 front
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-565.556274414063, 10.1981029510498, -20.4931106567383)) 
            super().create_npc(npc_type ="Sedan",transform = transform,follow_lane = True,speed = 4.5)

            # NPC 2 left front
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-564.556274414063, 10.1981029510498,  33.4931106567383))
            super().create_npc(npc_type ="SUV",transform = transform,follow_lane = True,speed=8.0)

            # NPC 3 right behind
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-568.556274414063, 10.1981029510498,  33.4931106567383))
            super().create_npc(npc_type ="SUV",transform = transform,follow_lane = True,speed=6.0)

        elif self.scene_type == 'overtake':
            # NPC 1 front and right
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-561.556274414063, 10.1981029510498, -36.4931106567383)) 
            super().create_npc(npc_type ="Sedan",transform = transform,follow_lane = True,speed = 3.5)
            # super().create_npc(npc_type ="Sedan",transform = transform,speed = 0)

            # NPC 2 left
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-559.556274414063, 10.1981029510498,  10.4931106567383))
            super().create_npc(npc_type ="SUV",transform = transform,follow_lane = True,speed=2.5)
            # super().create_npc(npc_type ="SUV",transform = transform,speed=0)

        elif self.scene_type == 'npc_lanechange':

            wp = []
            speed = 8.5
            
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-565.556274414063, 10.1981029510498, 28.4931106567383)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-561.556274414063, 10.1981029510498, -28.4931106567383))
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-545.556274414063, 10.1981029510498, -48.4931106567383)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-531.880920410156, 10.1983346939087, -124.162315368652))
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )

            super().create_npc(npc_type ="Sedan",transform = transform_first,waypoints=wp)
            # super().create_npc(npc_type ="Sedan",transform = transform,speed = 0)


    def create_pedestrians(self):
        # pedestrian one
        pass