#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from mcl import *
from scipy.stats import chi2

        

'''----- KLD(Kullback-Leibler divergence)リサンプリングを行うMCL(Monte Carlo localization)クラス -----'''
# envmap            : 地図のオブジェクト
# init_pose         : 初期姿勢  np.array([x, y, theta])
# max_num           : パーティクル数の最大値
# motion_noise_stds : ロボットの動きに生じる雑音の標準偏差
#                   : nn : 直進1[m]   で生じる 道のり のばらつきの標準偏差
#                   : no : 回転1[rad] で生じる 道のり のばらつきの標準偏差
#                   : on : 直進1[m]   で生じる   向き のばらつきの標準偏差
#                   : oo : 回転1[rad] で生じる   向き のばらつきの標準偏差
# distance_dev_rate : 距離に加える雑音の標準偏差
# direction_dev     : 方角に加える雑音の標準偏差
# widths            : 各ビンのx, y, thetaの幅
# epsilon           : 真の信念分布をパーティクルの分布で近似するときの誤差の許容値
# delta             : (たとえパーティクル数が十分にあっても)近似したい分布に対してパーティクルの分布が偏って
#                     KL情報量が epsilon 以内に達しない状況になる確率
class KldMcl(Mcl):
    def __init__(self, envmap, init_pose, max_num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05,
                 widths=np.array([0.2,0.2,math.pi/18]).T, epsilon=0.1, delta=0.01):
        super().__init__(envmap, init_pose, 1, motion_noise_stds, distance_dev_rate, direction_dev)
        self.widths  = widths
        self.max_num = max_num
        self.epsilon = epsilon
        self.delta   = delta
        self.binnum  = 0


    ### リサンプリングしながら状態遷移 ###################################################################
    # nu    : 速度
    # omega : 角速度
    # time  : デルタt
    def motion_update(self, nu, omega, time):
        ws = [e.weight for e in self.particles]  #重みのリスト
        if sum(ws) < 1e-100:                     #重みの和がゼロにならないように
            ws = [e + 1e-100 for e in ws]
            
        new_particles = [] #新しいパーティクルのリスト(最終的にself.particlesになる)
        bins = set()       #ビンのインデックスを登録しておくセット
        for i in range(self.max_num):
            chosen_p = random.choices(self.particles, weights=ws) #1つだけ選ぶ
            p = copy.deepcopy(chosen_p[0])
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf) #移動
            bins.add(tuple(math.floor(e) for e in p.pose/self.widths))   #ビンのインデックスをsetに登録
            new_particles.append(p)                                      #新しいパーティクルのリストに追加

            self.binnum = len(bins) if len(bins) > 1 else 2 #ビンの数が1の場合, 2にしないと次の行の計算ができない
            if len(new_particles) > math.ceil(chi2.ppf(1.0-self.delta, self.binnum-1)/(2*self.epsilon)):
                break

        self.particles = new_particles
        for i in range(len(self.particles)):  #正規化
            self.particles[i].weight = 1.0/len(self.particles)


    ### 観測したセンサ値を使って処理 ###################################################################
    # observation : センサ値
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()


    ### パーティクル数, ビン数を描画 ####################################################################
    # ax    : サブプロット
    # elems : 描画する図形のリスト
    def draw(self, ax, elems):
        super().draw(ax, elems)
        elems.append(ax.text(-4.5,-4.5,"particle:{}, bin:{}".format(len(self.particles), self.binnum),
                             fontsize=10))


        

##### このままじゃ矢印の大きさが可変になってしまうけど気にしない
def main():   
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)
    
    initial_pose = np.array([0,0,0]).T
    pf = KldMcl(m, initial_pose, 1000)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)
    
    world.draw()
        
    
if __name__ == '__main__':
    main()

    



    
