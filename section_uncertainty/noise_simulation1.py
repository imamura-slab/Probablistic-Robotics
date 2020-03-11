#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from ideal_robot import *
from scipy.stats import expon, norm, uniform
import copy

class Robot(IdealRobot):

    def __init__(self, pose, agent=None, sensor=None, color="black",
                 noise_per_meter=5, noise_std=math.pi/60.0,
                 bias_rate_stds=(0.1,0.1),
                 expected_stuck_time=1e100, expected_escape_time=1e-100,
                 expected_kidnap_time=1e100, kidnap_range_x=(-5.0,5.0), kidnap_range_y=(-5.0,5.0)):
        super().__init__(pose, agent, sensor, color)
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale=noise_std)
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False
        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0],ry[0],0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi))

    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()

        return pose

    def bias(self, nu, omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega

    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck)

    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose
        
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)


    


world = World(30, 0.1)

### ランダムな雑音
# for i in range(100):
#     circling = Agent(0.2, 10.0/180*math.pi)
#     r = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray")
#     world.append(r)

### バイアス
# circling = Agent(0.2, 10.0/180*math.pi)
# nobias_robot = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray")
# world.append(nobias_robot)
# biased_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling,
#                      color="red", noise_per_meter=0, bias_rate_stds=(0.2,0.2))
# world.append(biased_robot)

### スタック
# circling = Agent(0.2, 10.0/180*math.pi)
# for i in range(100):
#     r = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray",
#               noise_per_meter=0, bias_rate_stds=(0.0,0.0),
#               expected_stuck_time=60.0, expected_escape_time=60.0)
#     world.append(r)

# r = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="red")
# world.append(r)

### 誘拐
circling = Agent(0.2, 10.0/180*math.pi)
for i in range(100):
    r = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="gray",
              noise_per_meter=0, bias_rate_stds=(0.0,0.0), expected_kidnap_time=5)
    world.append(r)

r = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="red")
world.append(r)

world.draw()


























# def main():            
#     world = World(30, 1)
    
#     m = Map()
#     m.append_landmark(Landmark(2,-2))
#     m.append_landmark(Landmark(-1,-3))
#     m.append_landmark(Landmark(3,3))
#     world.append(m)
    
#     straight = Agent(0.2, 0.0)
#     circling = Agent(0.2, 10.0/180*math.pi)
#     robot1 = IdealRobot(np.array([2,3,math.pi/6]).T, sensor=IdealCamera(m), agent=straight)
#     robot2 = IdealRobot(np.array([-2,-1,math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
#     # robot3 = IdealRobot(np.array([0,0,0]).T, color="blue")
#     world.append(robot1)
#     world.append(robot2)
#     # world.append(robot3)
    
#     cam = IdealCamera(m)
#     # p = cam.data(robot2.pose)
#     # print(p)
    
#     world.draw()
    

            
# if __name__ == '__main__':
#     main()        






