#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from mcl import *
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse




### 誤差楕円 ############################################################################################
# p   :
# cov : 共分散行列
# n   :
#-------------------------------
# RET : 誤差楕円のオブジェクト
def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)                          #固有値, 固有ベクトル
    ang = math.atan2(eig_vec[:,0][1], eig_vec[:,0][0]/math.pi*180)  #楕円の傾き
    return Ellipse(p, width=2*n*math.sqrt(eig_vals[0]), height=2*n*math.sqrt(eig_vals[1]),
                   angle=ang, fill=False, color="blue", alpha=0.5)


### nu, omega 空間の共分散行列 M を計算 ####################################################################
# nu    : 速度
# omega : 角速度
# time  : デルタt
# stds  : ロボットの動きに生じる雑音の標準偏差
#       : nn : 直進1[m]   で生じる 道のり のばらつきの標準偏差
#       : no : 回転1[rad] で生じる 道のり のばらつきの標準偏差
#       : on : 直進1[m]   で生じる   向き のばらつきの標準偏差
#       : oo : 回転1[rad] で生じる   向き のばらつきの標準偏差
#----------------------------------------------------------------
# RET   : 共分散行列 M
def matM(nu, omega, time, stds):
    return np.diag([stds["nn"]**2*abs(nu)/time + stds["no"]**2*abs(omega)/time,
                    stds["on"]**2*abs(nu)/time + stds["oo"]**2*abs(omega)/time])


### 遷移後の分布をガウス分布に近づけるための線形化に使用する行列 A を計算 ################################
# nu    : 速度
# omega : 角速度
# theta : 向き
#-------------------
# RET   : 行列 A
def matA(nu, omega, time, theta):
    ## st  : 's'in('t'heta)
    ## ct  : 'c'os('t'heta)
    ## stw : 's'in('t'heta + 'omega'*time) : omegaの記号が w みたいだから?
    ## ctw : 'c'os('t'heta + 'omega'*time) : omegaの記号が w みたいだから?
    st,  ct  = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega*time), math.cos(theta + omega*time)
    return np.array([[( stw-st)/omega, -nu/(omega**2)*( stw-st)+nu/omega*time*ctw],
                     [(-ctw+ct)/omega, -nu/(omega**2)*(-ctw+ct)+nu/omega*time*stw],
                     [              0,                                       time]])


### fの線形化に使用するヤコビ行列 F を計算 ##############################################################
# nu    : 速度
# omega : 角速度
# time  : デルタt
# theta : 向き
#---------------------------
# RET   : F : ヤコビ行列 F
def matF(nu, omega, time, theta):
    F = np.diag([1.0,1.0,1.0])
    F[0,2] = nu/omega*(math.cos(theta+omega*time)-math.cos(theta))
    F[1,2] = nu/omega*(math.sin(theta+omega*time)-math.sin(theta))
    return F


### 観測方程式の線形化に使用する行列 H を計算 ###########################################################
# pose         : 姿勢
# landmark_pos : ランドマークの位置
#----------------------------------------------------
# RET          : 観測方程式の線形化に使用する行列 H
def matH(pose, landmark_pos):
    mx,  my       = landmark_pos
    mux, muy, mut = pose
    q = (mux-mx)**2 + (muy-my)**2
    return np.array([[(mux-mx)/np.sqrt(q), (muy-my)/np.sqrt(q),  0.0],
                     [        (my-muy)/q ,         (mux-mx)/q , -1.0]])


### センサ値のばらつきの共分散行列(線形化) Q を計算 #####################################################
# distance_dev  : 距離に加える雑音の標準偏差
# direction_dev : 方角に加える雑音の標準偏差
#-----------------------------------------------------------
# RET           : センサ値のばらつきの共分散行列(線形化) Q
def matQ(distance_dev, direction_dev):
    return np.diag(np.array([distance_dev**2, direction_dev**2]))





'''----- カルマンフィルタクラス -----'''
# envmap            : 地図のオブジェクト
# init_pose         : 初期姿勢  np.array([x, y, theta])
# motion_noise_stds : ロボットの動きに生じる雑音の標準偏差
#                   : nn : 直進1[m]   で生じる 道のり のばらつきの標準偏差
#                   : no : 回転1[rad] で生じる 道のり のばらつきの標準偏差
#                   : on : 直進1[m]   で生じる   向き のばらつきの標準偏差
#                   : oo : 回転1[rad] で生じる   向き のばらつきの標準偏差
# distance_dev_rate : 距離に加える雑音の標準偏差
# direction_dev     : 方角に加える雑音の標準偏差
class KalmanFilter:
    def __init__(self, envmap, init_pose, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05):
        self.belief = multivariate_normal(mean=init_pose, cov=np.diag([1e-10,1e-10,1e-10]))
        self.motion_noise_stds = motion_noise_stds
        self.pose = self.belief.mean
        self.map  = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev     = direction_dev


    ### 観測したセンサ値を使って処理 #####################################################################
    # observation : センサ値
    def observation_update(self, observation):
        for d in observation:
            z      = d[0]      #センサ値
            obs_id = d[1] - 1  #ID

            H = matH(self.belief.mean, self.map.landmarks[obs_id].pos)
            estimated_z = IdealCamera.observation_function(self.belief.mean, self.map.landmarks[obs_id].pos)
            Q = matQ(estimated_z[0]*self.distance_dev_rate, self.direction_dev)
            K = self.belief.cov.dot(H.T).dot(np.linalg.inv(Q + H.dot(self.belief.cov).dot(H.T))) #カルマンゲイン

            self.belief.mean += K.dot(z-estimated_z)
            self.belief.cov = (np.eye(3)-K.dot(H)).dot(self.belief.cov)
            self.pose = self.belief.mean


    ### ロボットを動かす ################################################################################
    # nu    : 速度
    # omega : 角速度
    # time  : デルタt
    def motion_update(self, nu, omega, time):
        if abs(omega) < 1e-5:
            omega = 1e-5

        M = matM(nu, omega, time, self.motion_noise_stds)
        A = matA(nu, omega, time, self.belief.mean[2])
        F = matF(nu, omega, time, self.belief.mean[2])

        self.belief.cov  = F.dot(self.belief.cov).dot(F.T) + A.dot(M).dot(A.T)
        self.belief.mean = IdealRobot.state_transition(nu, omega, time, self.belief.mean)
        self.pose = self.belief.mean


    ### 誤差楕円とtheta方向の誤差の見積もりを描画 #######################################################
    # ax    : サブプロット
    # elems : 描画する図形のリスト
    def draw(self, ax, elems):
        ## xy平面上の誤差の3シグマ範囲
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2,0:2], 3) #共分散行列のx,y部分のみ渡す
        elems.append(ax.add_patch(e))

        ## theta方向の誤差の3シグマ範囲
        x, y, c = self.belief.mean
        sigma3  = math.sqrt(self.belief.cov[2,2])*3
        xs      = [x + math.cos(c-sigma3), x, x + math.cos(c+sigma3)]
        ys      = [y + math.sin(c-sigma3), y, y + math.sin(c+sigma3)]
        elems  += ax.plot(xs, ys, color="blue", alpha=0.5)


        

def main():
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)
    
    # initial_pose = np.array([0,0,0]).T
    # estimator = Mcl(m, initial_pose, 100)
    # a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    # r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    # world.append(r)


    initial_pose = np.array([0,0,0]).T
    kf = KalmanFilter(m, initial_pose)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)
    
    world.draw()



    
if __name__ == '__main__':
    main()


    
