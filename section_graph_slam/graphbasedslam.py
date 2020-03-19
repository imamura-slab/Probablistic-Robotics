#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from kf import * #誤差楕円を描くのに利用
import itertools



def make_ax():
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    return ax
    

def draw_trajectory(xs, ax): #軌跡の描画
    poses = [xs[s] for s in range(len(xs))]
    ax.scatter([e[0] for e in poses], [e[1] for e in poses], s=5, marker=".", color="black")
    ax.plot([e[0] for e in poses], [e[1] for e in poses], linewidth=0.5, color="black")


def draw_observations(xs, zlist, ax): #センサ値の描画
    for s in range(len(xs)):
        if s not in zlist:
            continue

        for obs in zlist[s]:
            x, y, theta = xs[s]
            ell, phi = obs[1][0], obs[1][1]
            mx = x + ell*math.cos(theta + phi)
            my = y + ell*math.sin(theta + phi)
            ax.plot([x, mx], [y, my], color="pink", alpha=0.5)


def draw_edges(edges, ax):
    for e in edges:
        ax.plot([e.x1[0], e.x2[0]], [e.x1[1], e.x2[1]], color="red", alpha=0.5)
            

def draw(xs, zlist, edges):
    ax = make_ax()
    draw_observations(xs, zlist, ax)
    draw_edges(edges, ax)
    draw_trajectory(xs, ax)
    plt.show()



def read_data():
    hat_xs = {} #軌跡のデータ(ステップ数をキーにして姿勢を保存)
    zlist  = {} #センサ値のデータ(ステップ数をキーにして, さらにその中にランドマークのIDとセンサ値をタプルで保存)

    with open("log.txt") as f:
        for line in f.readlines():
            tmp = line.rstrip().split()

            step = int(tmp[1])
            if tmp[0] == "x":
                hat_xs[step] = np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]).T
            elif tmp[0] == "z":
                if step not in zlist:
                    zlist[step] = []
                zlist[step].append((int(tmp[2]), np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]).T))

    return hat_xs, zlist
    


class ObsEdge():
    def __init__(self, t1, t2, z1, z2, xs):
        assert z1[0] == z2[0] #ランドマークのIDが違ったら処理を止める

        self.t1, self.t2 = t1, t2
        self.x1, self.x2 = xs[t1], xs[t2]
        self.z1, self.z2 = z1[1], z2[1]


def make_edges(hat_xs, zlist):
    landmark_keys_zlist = {}

    for step in zlist:
        for z in zlist[step]:
            landmark_id = z[0] - 1
            if landmark_id not in landmark_keys_zlist:
                landmark_keys_zlist[landmark_id] = []
    
            landmark_keys_zlist[landmark_id].append((step, z))

    edges = []
    for landmark_id in landmark_keys_zlist:
        step_pairs = list(itertools.combinations(landmark_keys_zlist[landmark_id], 2))
        edges += [ObsEdge(xz1[0], xz2[0], xz1[1], xz2[1], hat_xs) for xz1, xz2 in step_pairs]

    return edges

        

def main():
    # time_interval = 3
    # world = World(180, time_interval, debug=False)

    # m = Map()
    # landmark_positions = [(-4,2), (2,-3), (3,3), (0,4), (1,1), (-3,-1)]
    # for p in landmark_positions:
    #     m.append_landmark(Landmark(*p))
    # world.append(m)
    
    # init_pose = np.array([0,-3,0]).T
    # a = LoggerAgent(0.2, 5.0/180*math.pi, time_interval, init_pose)
    # r = Robot(init_pose, sensor=PsiCamera(m), agent=a, color="red")
    # world.append(r)
    
    # world.draw()


    hat_xs, zlist = read_data()
    edges = make_edges(hat_xs, zlist)
    draw(hat_xs, zlist, edges)
    
    
    
    
if __name__ == '__main__':
    main()

    



    
