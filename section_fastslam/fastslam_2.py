#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from mcl import *
from kf import *



class EstimatedLandmark(Landmark):
    def __init__(self):
        super().__init__(0,0)
        #self.cov = np.array([[1,0],[0,2]])
        self.cov = None
        
    def draw(self, ax, elems):
        if self.cov is None:
            return

        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="blue")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))

        e = sigma_ellipse(self.pos, self.cov, 3)
        elems.append(ax.add_patch(e))


        
class MapParticle(Particle):
    def __init__(self, init_pose, weight, landmark_num):
        super().__init__(init_pose, weight)
        self.map = Map()

        for i in range(landmark_num):
            self.map.append_landmark(EstimatedLandmark())

            
    def init_landmark_estimation(self, landmark, z, distance_dev_rate, direction_dev):
        landmark.pos = z[0]*np.array([np.cos(self.pose[2] + z[1]), np.sin(self.pose[2] + z[1])]).T + self.pose[0:2]
        H = matH(self.pose, landmark.pos)[0:2,0:2]
        Q = matQ(distance_dev_rate*z[0], direction_dev)
        landmark.cov = np.linalg.inv(H.T.dot(np.linalg.inv(Q)).dot(H))

        
    def observation_update_landmark(self, landmark, z, distance_dev_rate, direction_dev):
        estm_z = IdealCamera.observation_function(self.pose, landmark.pos)
        if estm_z[0] < 0.01:
            return

        H = -matH(self.pose, landmark.pos)[0:2,0:2]
        Q = matQ(distance_dev_rate*estm_z[0], direction_dev)
        K = landmark.cov.dot(H.T).dot(np.linalg.inv(Q + H.dot(landmark.cov).dot(H.T)))

        Q_z = H.dot(landmark.cov).dot(H.T) + Q
        self.weight *= multivariate_normal(mean=estm_z, cov=Q_z).pdf(z)
        
        # landmark.pos = K.dot(z - estm_z) + landmark.pos
        # landmark.cov = (np.eye(2) - K.dot(H)).dot(landmark.cov)

        
    def observation_update(self, observation, distance_dev_rate, direction_dev):
        for d in observation:
            z = d[0]
            landmark = self.map.landmarks[d[1]-1]

            if landmark.cov is None:
                self.init_landmark_estimation(landmark, z, distance_dev_rate, direction_dev)
            else:
                self.observation_update_landmark(landmark, z, distance_dev_rate, direction_dev)

                
    def drawing_params(self, hat_x, landmark, distance_dev_rate, direction_dev):
        ell = np.hypot(*(hat_x[0:2] - landmark.pos))
        Qhat_zt = matQ(distance_dev_rate*ell, direction_dev)
        hat_zt = IdealCamera.observation_function(hat_x, landmark.pos)
        H_m = -matH(hat_x, landmark.pos)[0:2,0:2]
        H_xt = matH(hat_x, landmark.pos)

        Q_zt = H_m.dot(landmark.cov).dot(H_m.T) + Qhat_zt

        return hat_zt, Q_zt, H_xt

    
    def gauss_for_drawing(self, hat_x, R_t, z, landmark, distance_dev_rate, direction_dev):
        hat_zt, Q_zt, H_xt = self.drawing_params(hat_x, landmark, distance_dev_rate, direction_dev)
        K = R_t.dot(H_xt.T).dot(np.linalg.inv(Q_zt + H_xt.dot(R_t).dot(H_xt.T)))

        return K.dot(z-hat_zt) + hat_x, (np.eye(3) - K.dot(H_xt)).dot(R_t)

    
    def motion_update2(self, nu, omega, time, motion_noise_stds, observation, distance_dev_rate, direction_dev):
        M = matM(nu, omega, time, motion_noise_stds)
        A = matA(nu, omega, time, self.pose[2])
        R_t = A.dot(M).dot(A.T)
        hat_x = IdealRobot.state_transition(nu, omega, time, self.pose)

        for d in observation:
            hat_zt, Q_zt, H_xt = self.drawing_params(hat_x, self.map.landmarks[d[1]-1], distance_dev_rate, direction_dev)
            Sigma_zt = H_xt.dot(R_t).dot(H_xt.T) + Q_zt
            self.weight *= multivariate_normal(mean=hat_zt, cov=Sigma_zt).pdf(d[0])
        
        for d in observation:
            hat_x, R_t = self.gauss_for_drawing(hat_x, R_t, d[0], self.map.landmarks[d[1]-1], distance_dev_rate, direction_dev)

        self.pose = multivariate_normal(mean=hat_x, cov=R_t + np.eye(3)*1.0e-10).rvs()

                
                
class FastSlam(Mcl):
    def __init__(self, init_pose, particle_num, landmark_num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05):
        super().__init__(None, init_pose, particle_num, motion_noise_stds, distance_dev_rate, direction_dev)
        
        self.particles = [MapParticle(init_pose, 1.0/particle_num, landmark_num) for i in range(particle_num)]
        self.ml = self.particles[0]
        self.motion_noise_stds = motion_noise_stds

        
    def motion_update(self, nu, omega, time, observation):
        not_first_obs = []
        for d in observation:
            if self.particles[0].map.landmarks[d[1]-1].cov is not None:
                not_first_obs.append(d)

        if len(not_first_obs) > 0:
            for p in self.particles:
                p.motion_update2(nu, omega, time, self.motion_noise_stds, not_first_obs,
                                 self.distance_dev_rate, self.direction_dev)
        else:
            for p in self.particles:
                p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

                
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
        self.resampling()

        
    def draw(self, ax, elems):
        super().draw(ax, elems)
        self.ml.map.draw(ax, elems)



class FastSlam2Agent(EstimationAgent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(time_interval, nu, omega, estimator)

        
    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval, observation)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega
        
        
        
def trial():   
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)
    
    init_pose = np.array([0,0,0]).T
    pf = FastSlam(init_pose, 100, len(m.landmarks))
    a = FastSlam2Agent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)
    
    world.draw()
        
    
if __name__ == '__main__':
    trial()

    



    
