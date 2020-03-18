#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../scripts/')
from kld_mcl import *

        

def trial(animation):   
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    m = Map()
    for ln in [(-4,2), (2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)
    
    initial_pose = np.array([np.random.uniform(-5.0,5.0),
                             np.random.uniform(-5.0,5.0),
                             np.random.uniform(-math.pi,math.pi)]).T
    robot_pose = np.array([np.random.uniform(-5.0,5.0),
                           np.random.uniform(-5.0,5.0),
                           np.random.uniform(-math.pi,math.pi)]).T
    pf = KldMcl(m, initial_pose, 10000)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(robot_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)
    
    world.draw()

    return (r.pose, pf.ml.pose)


    
    
if __name__ == '__main__':
    trial(True)
    
    # ok = 0
    # for i in range(10):
    #     actual, estm = trial(False)
    #     diff = math.sqrt((actual[0]-estm[0])**2 + (actual[1]-estm[1])**2)
    #     print(i, "真値:", actual, "推定値:", estm, "誤差:", diff)
    #     if diff <= 1.0:
    #         ok += 1

    # print(ok)
    
