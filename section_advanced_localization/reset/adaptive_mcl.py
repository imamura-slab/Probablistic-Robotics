#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../scripts/')
from mcl import *


        
class ResetMcl(Mcl):
    def __init__(self, envmap, init_pose, num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05,
                 amcl_params={"slow":0.001, "fast":0.1, "nu":3.0}):
        super().__init__(envmap, init_pose, num, motion_noise_stds, distance_dev_rate, direction_dev)
        self.amcl_params = amcl_params
        self.slow_term_alpha, self.fast_term_alpha = 1.0, 1.0


    def sensor_resetting_draw(self, particle, landmark_pos, ell_obs, phi_obs):
        ##パーティクルの位置を決める##
        psi = np.random.uniform(-np.pi, np.pi) #ランドマークからの方角を選ぶ
        ell = norm(loc=ell_obs, scale=(ell_obs*self.distance_dev_rate)**2).rvs() #ランドマークからの距離を選ぶ
        particle.pose[0] = landmark_pos[0] + ell*math.cos(psi)
        particle.pose[1] = landmark_pos[1] + ell*math.sin(psi)
        
        ##パーティクルの向きを決める##
        phi = norm(loc=phi_obs, scale=(self.direction_dev)**2).rvs() #ランドマークが見える向きを決める
        particle.pose[2] = math.atan2(landmark_pos[1]-particle.pose[1], landmark_pos[0]-particle.pose[0]) - phi
        
        particle.weight = 1.0/len(self.particles)
        
        
    ## adaptive MCL
    def adaptive_resetting(self, observation):
        if len(observation) == 0: return

        alpha = sum([p.weight for p in self.particles])
        self.slow_term_alpha += self.amcl_params["slow"]*(alpha-self.slow_term_alpha)
        self.fast_term_alpha += self.amcl_params["fast"]*(alpha-self.fast_term_alpha)
        sl_num = len(self.particles)*max([0, 1.0-self.amcl_params["nu"]*self.fast_term_alpha/self.slow_term_alpha])

        self.resampling()

        nearest_obs = np.argmin([obs[0][0] for obs in observation])
        values, landmark_id = observation[nearest_obs]
        for n in range(int(sl_num)):
            p = random.choices(self.particles)[0]
            self.sensor_resetting_draw(p, self.map.landmarks[landmark_id-1].pos, *values)
            
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)

        self.set_ml()
        self.adaptive_resetting(observation)
        
        

def trial():   
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([0,0,0]).T
    pf = ResetMcl(m, initial_pose, 1000)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    # r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red") # 誘拐なし
    r = Robot(np.array([0,0,0]), sensor=Camera(m), agent=circling, expected_kidnap_time=10.0, color="red") # 誘拐あり
    world.append(r)
    
    world.draw()

    return pf




if __name__ == '__main__':
    pf = trial()

    # for num in pf.alphas:
    #     print("landmarks:", num, "particles:", len(pf.particles),
    #           "min:", min(pf.alphas[num]), "max:", max(pf.alphas[num]))


    
