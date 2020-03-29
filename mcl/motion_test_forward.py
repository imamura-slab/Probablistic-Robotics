#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from robot import *
import copy
import pandas as pd


def main():
    time_interval = 0.1
    world = World(40.0, time_interval)    

    
    ### 雑音
    ## バイアスを固定した場合における '直線' で生じる '道のり', '向き' のばらつきの標準偏差 "nn","on" が求まる
    # initial_pose = np.array([0,0,0]).T
    # robots = []
    # r = Robot(initial_pose, sensor=None, agent=Agent(0.1,0.0))
    # for i in range(100):
    #     copy_r = copy.copy(r) #copyしてるのでバイアスのかかり方が同じ
    #     copy_r.distance_until_noise = copy_r.noise_pdf.rvs() #最初に雑音が発生するタイミングを変える
    #     world.append(copy_r)  #worldに登録することでアニメーションの際に動く
    #     robots.append(copy_r) #オブジェクトの参照のリストにロボットのオブジェクトを登録

    
    ### バイアス
    initial_pose = np.array([0,0,0]).T
    robots = []

    for i in range(100):
        r = Robot(initial_pose, sensor=None, agent=Agent(0.1,0.0)) #毎回ロボットを作るのでバイアスはばらばら
        world.append(r)
        robots.append(r)

    world.draw()

    
    poses = pd.DataFrame([[math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] for r in robots],
                         columns=['r', 'theta'])
    print(poses.transpose())
    # print(poses["theta"].var())
    # print(poses["r"].mean())
    # print(math.sqrt(poses["theta"].var()/poses["r"].mean()))

    print(poses["r"].var())
    print(poses["r"].mean())
    print(math.sqrt(poses["r"].var()/poses["r"].mean()))

    

    
if __name__ == '__main__':
    main()

