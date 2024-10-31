import lgsvl
from environs import Env
import config
from scence.SceneBase import SceneBase

class RightTurnScene(SceneBase):

    def __init__(self,simulation_environment, environment_variables,ego_vehicle_type =config.EGO_VEHILCE_TYPE,scene_type = None):

        super().__init__(simulation_environment, environment_variables,ego_vehicle_type,scene_type = scene_type)


    def create_ego(self, transform=None):
        self.sim.set_time_of_day(5.38)

        if self.scene_type == None or self.scene_type == 'default':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-600.360107421875, 10.1981086730957, -165.723648071289))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-540.360107421875, 10.1981086730957, -97.723648071289))

        elif self.scene_type == 'right_line':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-600.360107421875, 10.1981086730957, -165.723648071289))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-540.360107421875, 10.1981086730957, -97.723648071289))
        elif self.scene_type == 'pedestrian':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-600.360107421875, 10.1981086730957, -165.723648071289))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-540.360107421875, 10.1981086730957, -97.723648071289))

        super().create_ego(transform = transform)
        return self.ego

    def create_scence(self):

        controllables = self.sim.get_controllables("signal")
        # print("\n# List of controllable objects ")
        control_policy = "trigger=150;green=30;yellow=0;red=0;loop"
        for c in controllables:
            if c.type == "signal" and abs(c.transform.position.x+535)<80 and abs(c.transform.position.z+161)<80:
                # print(c.transform)
                c.control(control_policy)

        if self.scene_type == None or self.scene_type == 'default':
            # NPC 1 
            wp = []
            speed = 7.0
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-450.360107421875, 10.1981086730957, -138.723648071289)) 
            wp.append(lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-600.360107421875, 10.1981086730957, -168.723648071289))
            wp.append(lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0))
            super().create_npc(npc_type ="SchoolBus",transform = transform_first,waypoints=wp)

        elif self.scene_type == 'right_line':
            # NPC 1 front
            wp = []
            speed = 8.0

            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-465.360107421875, 10.1981086730957, -139.723648071289)) 
            wp.append(lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-600.360107421875, 10.1981086730957, -165.723648071289))
            wp.append(lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0))
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)

            # NPC 2 
            wp = []
            speed = 8.0
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-450.360107421875, 10.1981086730957, -135.723648071289)) 
            wp.append(lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-600.360107421875, 10.1981086730957, -165.723648071289))
            wp.append(lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0))
            super().create_npc(npc_type ="SchoolBus",transform = transform_first,waypoints=wp)

        elif self.scene_type == 'pedestrian':

            # pedestrian
            self.create_pedestrians()
 
    def create_pedestrians(self):
        # pedestrian one
        
        # wp = []
        # speed = 0.7
        
        # # transform_first =lgsvl.Transform(lgsvl.Vector(-540.282775878906, 10.2076635360718, -147.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        # # wp.append( lgsvl.WalkWaypoint(transform_first.position, speed = speed,idle=0) )
        # # transform = lgsvl.Transform(lgsvl.Vector(-530.282775878906, 10.2076635360718, -147.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        # # wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        # # transform = lgsvl.Transform(lgsvl.Vector(-520.282775878906, 10.2076635360718, -145.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        # # wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        # # transform = lgsvl.Transform(lgsvl.Vector(-510.282775878906, 10.2076635360718, -143.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        # # wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )

        # transform_first =lgsvl.Transform(lgsvl.Vector(-538.282775878906, 10.2076635360718, -140.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        # wp.append( lgsvl.WalkWaypoint(transform_first.position, speed = speed,idle=0) )
        # transform = lgsvl.Transform(lgsvl.Vector(-528.282775878906, 10.2076635360718, -138.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        # wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        # transform = lgsvl.Transform(lgsvl.Vector(-508.282775878906, 10.2076635360718, -138.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        # wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )

        # super().create_pedestrian(pedestrian_type = 'Pamela',transform=transform_first,waypoints = wp)

        wp = []
        speed = 1.1

        transform_first =lgsvl.Transform(lgsvl.Vector(-539.282775878906, 10.2076635360718, -138.7101135253906), lgsvl.Vector(0, 170.072402954102, 0))
        wp.append( lgsvl.WalkWaypoint(transform_first.position, speed = speed,idle=0) )
        transform = lgsvl.Transform(lgsvl.Vector(-540.282775878906, 10.2076635360718, -147.7101135253906), lgsvl.Vector(0, 170.072402954102, 0))
        wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        transform = lgsvl.Transform(lgsvl.Vector(-537.282775878906, 10.2076635360718, -158.7101135253906), lgsvl.Vector(0, 170.072402954102, 0))
        wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        transform = lgsvl.Transform(lgsvl.Vector(-534.282775878906, 10.2076635360718, -178.7101135253906), lgsvl.Vector(0, 170.072402954102, 0))
        wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )

        super().create_pedestrian(pedestrian_type = 'Johny',transform=transform_first,waypoints = wp)

        transform_first =lgsvl.Transform(lgsvl.Vector(-540.282775878906, 10.2076635360718, -141.7101135253906), lgsvl.Vector(0, 80.072402954102, 0))
        super().create_pedestrian(pedestrian_type = 'Pamela',transform=transform_first,waypoints = wp)
