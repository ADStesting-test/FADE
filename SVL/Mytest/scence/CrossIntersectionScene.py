import lgsvl
from environs import Env
import config
from scence.SceneBase import SceneBase


class CrossIntersectionScene(SceneBase):

    def __init__(self,simulation_environment, environment_variables,ego_vehicle_type =config.EGO_VEHILCE_TYPE,scene_type = None):

        super().__init__(simulation_environment, environment_variables,ego_vehicle_type,scene_type = scene_type)

    def create_ego(self, transform=None):
        self.sim.set_time_of_day(3.0)

        if self.scene_type == None or self.scene_type == 'default':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-447.657623291016, 10.1981105804443, 427.845642089844))
            # Ego
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-355.692626953125, 10.1980972290039, 339.164947509766))  

        elif self.scene_type == 'follow_npc':
            self.destination = self.sim.map_point_on_lane(lgsvl.Vector(-447.657623291016, 10.1981105804443, 427.845642089844))
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
            speed = 8.0
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-460.310150146484, 10.1981267929077, 328.197021484375)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-357.373626708984, 10.1979598999023, 433.919097900391)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)


            # NPC 2
            wp = []
            speed = 8.0
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-464.48974609375, 10.2076635360718, 324.394958496094)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-357.373626708984, 10.1979598999023, 433.919097900391)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)


        elif self.scene_type == 'follow_npc':
            # NPC 1 front
            wp = []
            speed = 7.0
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-375.459320068359, 10.1987104415894, 358.826782226563)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-466.406402587891, 10.1979265213013, 445.737762451172)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)


            # NPC 2
            wp = []
            speed = 8.1
            transform_first = self.sim.map_point_on_lane(lgsvl.Vector(-464.48974609375, 10.2076635360718, 324.394958496094)) 
            wp.append( lgsvl.DriveWaypoint(transform_first.position, speed = speed,idle=0) )
            transform = lgsvl.Transform(lgsvl.Vector(-410.075408935547, 10.1971187591553, 380.042999267578), lgsvl.Vector(0.0171416606754065, 14.3875885009766, 0.0141508243978024))
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            transform = lgsvl.Transform(lgsvl.Vector(-408.1005859375, 10.2004127502441, 387.740997314453), lgsvl.Vector(0.0536392852663994, -24.3988704681396, 0.0037133835721761))
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            transform = lgsvl.Transform(lgsvl.Vector(-410.558898925781, 10.2016515731812, 393.160766601563), lgsvl.Vector(0.0535594671964645, 313.702117919922, 0.003557049902156))
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            transform = self.sim.map_point_on_lane(lgsvl.Vector(-459.605743408203, 10.1979970932007, 440.044036865234)) 
            wp.append( lgsvl.DriveWaypoint(transform.position, speed = speed,idle=0) )
            super().create_npc(npc_type ="SUV",transform = transform_first,waypoints=wp)

    def create_pedestrians(self):
        # pedestrian one
        wp = []

        transform_first = lgsvl.Transform(lgsvl.Vector(200.511444091797, -5.04701614379883, -82.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp.append( lgsvl.WalkWaypoint(transform.position, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(204.511444091797, -5.04701614379883, -75.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp.append( lgsvl.WalkWaypoint(transform.position, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(206.511444091797, -5.04701614379883, -70.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp.append( lgsvl.WalkWaypoint(transform.position, idle=0) )

        transform=lgsvl.Transform(lgsvl.Vector(208.511444091797, -5.04701614379883, -65.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp.append( lgsvl.WalkWaypoint(transform.position, idle=0) )

        super().create_pedestrian(pedestrian_type = 'Pamela',transform=transform_first,waypoints = wp)
        
        # pedestrian two
        wp_two = []
     
        transform_first = lgsvl.Transform(lgsvl.Vector(204.511444091797, -5.04701614379883, -80.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp_two.append( lgsvl.WalkWaypoint(transform_first.position, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(202.511444091797, -5.04701614379883, -75.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp_two.append( lgsvl.WalkWaypoint(transform.position, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(200.511444091797, -5.04701614379883, -70.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp_two.append( lgsvl.WalkWaypoint(transform.position, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(198.511444091797, -5.04701614379883, -63.6100692749023), lgsvl.Vector(0.910983741283417, 15.099830627441, 357.892608642578))
        wp_two.append( lgsvl.WalkWaypoint(transform.position, idle=0) )
        
        super().create_pedestrian(pedestrian_type = 'Johny',transform=transform_first,waypoints = wp_two)