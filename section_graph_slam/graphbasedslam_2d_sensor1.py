#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

### log2.txt

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


def draw_landmarks(ms, ax):
    #ax.scatter([m[0] for m in ms], [m[1] for m in ms], s=100, marker="*", color="blue", zorder=100)
    ax.scatter([ms[k][0] for k in ms], [ms[k][1] for k in ms], s=100, marker="*", color="blue", zorder=100)
        

def draw(xs, zlist, edges, ms=[]):
    ax = make_ax()
    draw_observations(xs, zlist, ax)
    #draw_edges(edges, ax)
    draw_trajectory(xs, ax)
    draw_landmarks(ms, ax)
    plt.show()



def read_data():
    hat_xs = {} #軌跡のデータ(ステップ数をキーにして姿勢を保存)
    zlist  = {} #センサ値のデータ(ステップ数をキーにして, さらにその中にランドマークのIDとセンサ値をタプルで保存)
    delta = 0.0
    us ={}
    
    with open("log2.txt") as f:
        for line in f.readlines():
            tmp = line.rstrip().split()

            step = int(tmp[1])
            if tmp[0] == "x":
                hat_xs[step] = np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]).T
            elif tmp[0] == "z":
                if step not in zlist:
                    zlist[step] = []
                zlist[step].append((int(tmp[2]), np.array([float(tmp[3]), float(tmp[4])]).T))
            elif tmp[0] == "delta":
                delta = float(tmp[1])
            elif tmp[0] == "u":
                us[step] = np.array([float(tmp[2]), float(tmp[3])]).T
                
        return hat_xs, zlist, us, delta
    


class ObsEdge():
    def __init__(self, t1, t2, z1, z2, xs, sensor_noise_rate=[0.14, 0.05]):
        assert z1[0] == z2[0] #ランドマークのIDが違ったら処理を止める

        self.t1, self.t2 = t1, t2
        self.x1, self.x2 = xs[t1], xs[t2]
        self.z1, self.z2 = z1[1], z2[1]

        s1 = math.sin(self.x1[2] + self.z1[1])
        c1 = math.cos(self.x1[2] + self.z1[1])
        s2 = math.sin(self.x2[2] + self.z2[1])
        c2 = math.cos(self.x2[2] + self.z2[1])

        ## 残差の計算
        hat_e = self.x2[0:2] - self.x1[0:2] + np.array([
            self.z2[0]*c2 - self.z1[0]*c1,
            self.z2[0]*s2 - self.z1[0]*s1
        ])
        
        Q1 = np.diag([(self.z1[0]*sensor_noise_rate[0])**2, sensor_noise_rate[1]**2])
        R1 = -np.array([[c1, -self.z1[0]*s1],
                        [s1,  self.z1[0]*c1]])

        Q2 = np.diag([(self.z2[0]*sensor_noise_rate[0])**2, sensor_noise_rate[1]**2])
        R2 = np.array([[c2, -self.z2[0]*s2],
                       [s2,  self.z2[0]*c2]])

        Sigma = R1.dot(Q1).dot(R1.T) + R2.dot(Q2).dot(R2.T)
        #self.Omega = np.linalg.inv(Sigma)
        Omega = np.linalg.inv(Sigma)
        # print(Sigma)

        B1 = -np.array([[1, 0, -self.z1[0]*s1],
                        [0, 1,  self.z1[0]*c1]])
        B2 =  np.array([[1, 0, -self.z2[0]*s2],
                        [0, 1,  self.z2[0]*c2]])

        self.omega_upperleft   = B1.T.dot(Omega).dot(B1)
        self.omega_upperright  = B1.T.dot(Omega).dot(B2)
        self.omega_bottomleft  = B2.T.dot(Omega).dot(B1)
        self.omega_bottomright = B2.T.dot(Omega).dot(B2)

        self.xi_upper  = -B1.T.dot(Omega).dot(hat_e)
        self.xi_bottom = -B2.T.dot(Omega).dot(hat_e)



class MotionEdge:
    def __init__(self, t1, t2, xs, us, delta, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}):
        self.t1, self.t2 = t1, t2
        self.hat_x1, self.hat_x2 = xs[t1], xs[t2]

        nu, omega = us[t2]
        if abs(omega) < 1e-5:
            omega = 1e-5

        M = matM(nu, omega, delta, motion_noise_stds)
        A = matA(nu, omega, delta, self.hat_x1[2])
        F = matF(nu, omega, delta, self.hat_x1[2])

        self.Omega = np.linalg.inv(A.dot(M).dot(A.T) + np.eye(3)*0.0001)

        self.omega_upperleft   =  F.T.dot(self.Omega).dot(F)
        self.omega_upperright  = -F.T.dot(self.Omega)
        self.omega_bottomleft  = -self.Omega.dot(F)
        self.omega_bottomright =  self.Omega

        x2 = IdealRobot.state_transition(nu, omega, delta, self.hat_x1)
        self.xi_upper  = F.T.dot(self.Omega).dot(self.hat_x2 - x2)
        self.xi_bottom = -self.Omega.dot(self.hat_x2-x2)
    
        
        
def add_edge(edge, Omega, xi):
    f1, f2 = edge.t1*3, edge.t2*3
    t1, t2 = f1+3, f2+3
    Omega[f1:t1, f1:t1] += edge.omega_upperleft
    Omega[f1:t1, f2:t2] += edge.omega_upperright
    Omega[f2:t2, f1:t1] += edge.omega_bottomleft
    Omega[f2:t2, f2:t2] += edge.omega_bottomright
    xi[f1:t1] += edge.xi_upper
    xi[f2:t2] += edge.xi_bottom
    

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

    return edges, landmark_keys_zlist



###{
hat_xs, zlist, us, delta = read_data()
dim = len(hat_xs)*3

for n in range(1, 10000):
    ## エッジ, 大きな精度行列, 係数ベクトルの作成
    edges, _ = make_edges(hat_xs, zlist)
    Omega = np.zeros((dim, dim))
    xi = np.zeros(dim)
    Omega[0:3,0:3] += np.eye(3)*1000000

    ## 行動エッジの追加
    for i in range(len(hat_xs)-1):
        for j in range(100): #100回同じエッジを足して移動エッジを増強
            edges.append(MotionEdge(i, i+1, hat_xs, us, delta))

    ## 軌跡を動かす量(差分)の計算
    for e in edges:
        add_edge(e, Omega, xi)

    delta_xs = np.linalg.inv(Omega).dot(xi)

    ## 推定値の更新
    for i in range(len(hat_xs)):
        hat_xs[i] += delta_xs[i*3:(i+1)*3]
    
    ## 終了判定
    diff = np.linalg.norm(delta_xs)
    print("{}回目の繰り返し: {}".format(n, diff))
    if diff < 0.01:
        #draw(hat_xs, zlist, edges) #収束したら描画. drawでdraw_edgeを呼び出している行はコメントアウト
        break

    _, zlist_landmark = make_edges(hat_xs, zlist)
    #print(zlist_landmark)
###}


        
class MapEdge:
    def __init__(self, t, z, head_t, head_z, xs, sensor_noise_rate=[0.14, 0.05]):
        self.x = xs[t]
        self.z = z

        self.m = self.x[0:2] + np.array([
            z[0]*math.cos(self.x[2]+z[1]),
            z[0]*math.sin(self.x[2]+z[1])
        ]).T

        Q1 = np.diag([(self.z[0]*sensor_noise_rate[0])**2, sensor_noise_rate[1]**2])

        s1 = math.sin(self.x[2] + self.z[1])
        c1 = math.cos(self.x[2] + self.z[1])
        R  = np.array([[-c1, self.z[0]*s1],
                       [-s1, self.z[0]*c1]])

        self.Omega = R.dot(Q1).dot(R.T)
        self.xi = self.Omega.dot(self.m)



###{
ms = {}
for landmark_id in zlist_landmark:
    edges = []
    head_z = zlist_landmark[landmark_id][0]
    for z in zlist_landmark[landmark_id]:
        edges.append(MapEdge(z[0], z[1][1], head_z[0], head_z[1][1], hat_xs))

    Omega = np.zeros((2,2))
    xi = np.zeros(2)
    for e in edges:
        Omega += e.Omega
        xi += e.xi

    #ms.append(np.linalg.inv(Omega).dot(xi))    
    ms[landmark_id] = np.mean([e.m for e in edges], axis=0)

draw(hat_xs, zlist, edges, ms)
###}     

   


    
