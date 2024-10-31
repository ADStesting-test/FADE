import lgsvl
from environs import Env
from scence.SceneBase import SceneBase
import config


class StraightDrivingScene(SceneBase):
    
    def __init__(self,simulation_environment, environment_variables,ego_vehicle_type =config.EGO_VEHILCE_TYPE,scene_type = None):

        super().__init__(simulation_environment, environment_variables,ego_vehicle_type,scene_type=scene_type)

    def create_ego(self, transform=None):
        self.sim.set_time_of_day(12.0)
        if self.scene_type == None or self.scene_type == 'default':
            self.destination =  self.spawns[0].destinations[0]
            # Ego
            
            transform = self.sim.map_point_on_lane(lgsvl.Vector(353.796813964844, -7.61491346359253, -43.4674797058105))

        elif self.scene_type == 'pedestrian':
            self.destination =  self.spawns[0].destinations[0]
            # Ego
            # transform = self.sim.map_point_on_lane(lgsvl.Vector(353.796813964844, -7.61491346359253, -43.4674797058105))
            transform = self.sim.map_point_on_lane(lgsvl.Vector(358.372253417969, -7.64456605911255, -26.1277732849121))
        super().create_ego(transform = transform)
        return self.ego

    def create_scence(self):

        if self.scene_type == None or self.scene_type == 'default':
            # behind
            transform = lgsvl.Transform(lgsvl.Vector(372.004833984375, -7.8576831817627, 18.3557281494141), lgsvl.Vector(0.345935463905334, 14.6914482116699, 359.644744873047))
            super().create_npc(npc_type = "Sedan",transform = transform)
            # front
            transform = lgsvl.Transform(lgsvl.Vector(377.004833984375, -7.8576831817627, 37.3557281494141), lgsvl.Vector(0.345935463905334, 14.6914482116699, 359.644744873047))
            super().create_npc(npc_type = "SUV",transform = transform)

        elif self.scene_type == 'pedestrian':

            self.create_pedestrian()
    
    def create_pedestrian(self):

        # pedestrian front
        wp = []
        speed = 1.0
        transform_first = lgsvl.Transform(lgsvl.Vector(389.004833984375, -7.8776831817627, 40.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        wp.append( lgsvl.WalkWaypoint(transform_first.position,speed = speed, idle=0) )
        transform = lgsvl.Transform(lgsvl.Vector(382.004833984375, -7.8776831817627, 40.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        wp.append( lgsvl.WalkWaypoint(transform.position,speed = speed, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(377.004833984375, -7.8776831817627, 40.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        # p = self.sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, ped_state)
        wp.append( lgsvl.WalkWaypoint(transform.position,speed = speed, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(372.004833984375, -7.8776831817627, 42.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        # p = self.sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, ped_state)
        wp.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(367.004833984375, -7.7276831817627, 43.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        # p = self.sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, ped_state)
        wp.append( lgsvl.WalkWaypoint(transform.position,speed = speed, idle=0) )

        transform=lgsvl.Transform(lgsvl.Vector(365.44833984375, -7.5876831817627, 40.3557281494141), lgsvl.Vector(0.345935463905334, -166.6914482116699, 359.644744873047))
        # p = self.sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, ped_state)
        wp.append( lgsvl.WalkWaypoint(transform.position,speed = speed, idle=0) )

        super().create_pedestrian(pedestrian_type = 'Pamela',transform=transform_first,waypoints = wp)

        # pedestrian behind
        wp_two = []
        speed = 1.2
        transform_first = lgsvl.Transform(lgsvl.Vector(382.004833984375, -7.6776831817627, 21.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        wp_two.append( lgsvl.WalkWaypoint(transform_first.position, speed = speed,idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(372.004833984375, -7.8776831817627, 26.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        wp_two.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(367.004833984375, -7.8776831817627, 31.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        wp_two.append( lgsvl.WalkWaypoint(transform.position,speed = speed, idle=0) )

        transform = lgsvl.Transform(lgsvl.Vector(364.004833984375, -7.7276831817627, 36.3557281494141), lgsvl.Vector(0.345935463905334, -76.6914482116699, 359.644744873047))
        wp_two.append( lgsvl.WalkWaypoint(transform.position, speed = speed,idle=0) )
        
        super().create_pedestrian(pedestrian_type = 'Johny',transform=transform_first,waypoints = wp_two)