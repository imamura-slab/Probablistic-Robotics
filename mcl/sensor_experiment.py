#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scripts/')
from robot import *
import pandas as pd

        

def main():

    m = Map()
    m.append_landmark(Landmark(1,0))

    distance  = []
    direction = []
    
    for i in range(1000):
        c = Camera(m) #バイアスの影響も考慮するため毎回カメラを新規作成
        d = c.data(np.array([0.0,0.0,0.0]).T)
        if len(d) > 0:
            distance.append(d[0][0][0])
            direction.append(d[0][0][1])
            
    
    df = pd.DataFrame()
    df["distance"]  = distance
    df["direction"] = direction
    #print(df)
    print(df.std())
    print(df.mean())

    
if __name__ == '__main__':
    main()



    
