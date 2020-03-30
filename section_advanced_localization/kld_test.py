#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from robot import *
from scipy.stats import norm, chi2

        
### パーティクル数を求める ###############################################################################
# epsilon : 真の信念分布をパーティクルの分布で近似するときの誤差の許容値
# delta   : (たとえパーティクル数が十分にあっても)近似したい分布に対してパーティクルの分布が偏って
#           KL情報量が epsilon 以内に達しない状況になる確率
# binnum  : ビン数
#---------------------------------------------------------------------------------------------------------
# RET     : パーティクル数(N)
#           N > y / 2*epsilon を満たす最小のN
def num(epsilon, delta, binnum):
    return math.ceil(chi2.ppf(1.0-delta, binnum-1) / (2*epsilon))  #ceil関数 : 整数になるよう切り上げ


### パーティクル数を求める(ウィルソン-フィルファーティ変換を用いて) #####################################
# 入出力は上記のnum関数と同じ
def num_wh(epsilon, delta, binnum):
    dof = binnum - 1
    z   = norm.ppf(1.0-delta)
    return math.ceil(dof/(2*epsilon)*(1.0-2.0/(9*dof) + math.sqrt(2.0/(9*dof))*z)**3)


def main():
    for binnum in 2, 4, 8, 1000, 10000, 1000000:
        print("ビン:", binnum, "epsilon=0.1, delta=0.01", num(0.1,0.01,binnum), num_wh(0.1,0.01,binnum))
        print("ビン:", binnum, "epsilon=0.5, delta=0.01", num(0.5,0.01,binnum), num_wh(0.5,0.01,binnum))
        print("ビン:", binnum, "epsilon=0.5, delta=0.05", num(0.5,0.05,binnum), num_wh(0.5,0.05,binnum))


    fig, (axl, axr) = plt.subplots(ncols=2, figsize=(10,4))
    bs = np.arange(2, 10)
    n = [num(0.1, 0.01, b) for b in bs]
    axl.set_title("bin: 2-10")
    axl.plot(bs, n)

    bs = np.arange(2, 100000)
    n = [num(0.1, 0.01, b) for b in bs]
    axr.set_title("bin: 2-100000")
    axr.plot(bs, n)

    plt.show()

    
if __name__ == '__main__':
    main()

    
