import lgsvl
from environs import Env
import config
from scence.SceneBase import SceneBase

class LeftTurnScene(SceneBase):

    def __init__(self,simulation_environment, environment_variables,ego_vehicle_type =config.EGO_VEHILCE_TYPE,scene_type = None):

        super().__init__(simulation_environment, environment_variables,ego_vehicle_type,scene_type = scene_type)

    def create_ego(self, transform=None):
        self.sim.set_time_of_day(3.0)

        if self.scene_type == None or self.scene_type == 'default':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-466.310150146484, 10.1981267929077, 328.197021484375))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-355.692626953125, 10.1980972290039, 339.164947509766))  

        elif self.scene_type == 'right_turn':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-466.310150146484, 10.1981267929077, 328.197021484375))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-355.692626953125, 10.1980972290039, 339.164947509766))

        elif self.scene_type == 'pedestrian':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-466.310150146484, 10.1981267929077, 328.197021484375))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-355.692626953125, 10.1980972290039, 339.164947509766))

        super().create_ego(transform = transform)
        return self.ego

    def create_scence(self):

        controllables = self.sim.get_controllables("signal")
        # print("\n# List of controllable objects ")
        control_policy = "trigger=150;green=30;yellow=0;red=0;loop"
        for c in controllables:
            if c.type == "signal" and abs(c.transform.position.x+403)<50 and abs(c.transform.position.z-385)<50:
                # print(c.transform)
                c.control(control_policy)

        if self.scene_type == None or self.scene_type == 'default':

            # NPC 1 front
            wp = []
            speed = 7.0
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-476.680541992188, 10.2076644897461, 444.474182128906)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-362.004119873047, 10.2076635360718, 336.102600097656)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)


            # NPC 2
            wp = []
            speed = 7.0
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-479.7470703125, 10.2076644897461, 447.404541015625)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-362.004119873047, 10.2076635360718, 336.102600097656)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)


        elif self.scene_type == 'right_turn':
            # NPC 1 front
            wp = []
            speed = 7.5
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-470.791687011719, 10.198055267334, 443.404327392578)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-362.004119873047, 10.2076635360718, 340.102600097656)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)


            # NPC 2
            wp = []
            speed = 7.5
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-485.791687011719, 10.198055267334, 458.404327392578)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = lgsvl.Transform(lgsvl.Vector(-414.412719726563, 10.1980905532837, 389.865875244141), lgsvl.Vector(0.0174325574189425, 133.731521606445, 0.0126252500340343))
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            transform = lgsvl.Transform(lgsvl.Vector(-400.412719726563, 10.1980905532837, 389.865875244141), lgsvl.Vector(0.0174325574189425, 43.731521606445, 0.0126252500340343))
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-360.406219482422, 10.1981182098389, 431.676666259766)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)

        elif self.scene_type == 'pedestrian':

            # pedestrian
            self.create_pedestrians()
 
    def create_pedestrians(self):
        # pedestrian one
        
        # wp = []
        # speed = 1.2
        
        # transform_first =lgsvl.Transform(lgsvl.Vector(-390.412719726563, 10.1980905532837, 389.865875244141), lgsvl.Vector(0.0174325574189425, -137.731521606445, 0.0126252500340343))
        # wp.append( lgsvl.WalkWaypoint(transform_first.position, speed = speed,idle=0) )
        # transform = lgsvl.Transform(lgsvl.Vector(-400.412719726563, 10.1980905532837, 380.865875244141), lgsvl.Vector(0.0174325574189425, -137.731521606445, 0.0126252500340343))
        # wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        # transform = lgsvl.Transform(lgsvl.Vector(-405.412719726563, 10.1980905532837, 370.865875244141), lgsvl.Vector(0.0174325574189425, -137.731521606445, 0.0126252500340343))
        # wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
    
        # super().create_pedestrian(pedestrian_type = 'Pamela',transform=transform_first,waypoints = wp)

        wp = []
        speed = 1.0

        transform_first =lgsvl.Transform(lgsvl.Vector(-408.412719726563, 10.1980905532837, 363.865875244141), lgsvl.Vector(0.0174325574189425, -47.731521606445, 0.0126252500340343))
        wp.append( lgsvl.WalkWaypoint(transform_first.position, speed = speed,idle=0) )
        transform = lgsvl.Transform(lgsvl.Vector(-410.412719726563, 10.1980905532837, 369.865875244141), lgsvl.Vector(0.0174325574189425, -47.731521606445, 0.0126252500340343))
        wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        transform = lgsvl.Transform(lgsvl.Vector(-424.412719726563, 10.1980905532837, 384.865875244141), lgsvl.Vector(0.0174325574189425, -47.731521606445, 0.0126252500340343))
        wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        super().create_pedestrian(pedestrian_type = 'Pamela',transform=transform_first,waypoints = wp)

        wp = []
        speed = 0.8
        transform_first = lgsvl.Transform(lgsvl.Vector(-410.412719726563, 10.1980905532837, 369.865875244141), lgsvl.Vector(0.0174325574189425, -47.731521606445, 0.0126252500340343))
        wp.append( lgsvl.WalkWaypoint(transform_first.position, speed = speed,idle=0) )
        transform = lgsvl.Transform(lgsvl.Vector(-424.412719726563, 10.1980905532837, 384.865875244141), lgsvl.Vector(0.0174325574189425, -47.731521606445, 0.0126252500340343))
        wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )

        super().create_pedestrian(pedestrian_type = 'Johny',transform=transform_first,waypoints = wp)