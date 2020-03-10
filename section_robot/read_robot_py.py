#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from ideal_robot import *




def main():            
    world = World(30, 1)
    
    m = Map()
    m.append_landmark(Landmark(2,-2))
    m.append_landmark(Landmark(-1,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)
    
    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0/180*math.pi)
    robot1 = IdealRobot(np.array([2,3,math.pi/6]).T, sensor=IdealCamera(m), agent=straight)
    robot2 = IdealRobot(np.array([-2,-1,math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
    # robot3 = IdealRobot(np.array([0,0,0]).T, color="blue")
    world.append(robot1)
    world.append(robot2)
    # world.append(robot3)
    
    cam = IdealCamera(m)
    # p = cam.data(robot2.pose)
    # print(p)
    
    world.draw()
    

            
if __name__ == '__main__':
    main()        

