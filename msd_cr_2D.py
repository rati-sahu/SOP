#  This code calculates the cage relative MSD for a dense colloidal suspension in 2D.
#  Here we have a binary system. 
#  Inputs = x y coordinates in all time frames
#  Outputs = MSD_cr

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

af = '55'
dia_big = 3.34
path = '/media/hdd2/P2-Entropy_2d/pos_binary/0.'+af
big_x = np.load(path+'/big_af-'+af+'_x.npy')*(145/512)/dia_big
big_y = np.load(path+'/big_af-'+af+'_y.npy')*(145/512)/dia_big
small_x = np.load(path+'/small_af-'+af+'_x.npy')*(145/512)/dia_big
small_y = np.load(path+'/small_af-'+af+'_y.npy')*(145/512)/dia_big

print(np.shape(big_x))
big_x = big_x[:,::10].copy()
big_y = big_y[:,::10].copy()
small_x = small_x[:,::10].copy()
small_y = small_y[:,::10].copy()
print(np.shape(big_x))

nb = len(big_x)
ns = len(small_x)
N =nb+ns
x = np.vstack((big_x,small_x))
y = np.vstack((big_y,small_y))

print(np.shape(x))
frames=np.shape(x)[1]


def MSD_cr(x0, x1, y0, y1,disp_crx,disp_cry):
    sumx = np.square((x1-x0)-disp_crx)
    sumy = np.square((y1-y0)-disp_cry)
    msd = np.mean(sumx + sumy)
    return msd
    
def cr(x0,x1,y0,y1):    
	# if tag is on big then len is nb, otherwise ns
    disp_crx = np.zeros(nb)
    disp_cry = np.zeros(nb)
    bulk = np.array([x0, y0])
    bulk = bulk.T
    bulk_df = pd.DataFrame(bulk[:,0:2],columns=['x','y'])
    tree = KDTree(bulk_df[['x', 'y']])
    dist_bulk, idxs = tree.query(bulk_df, k= 10, distance_upper_bound = 1.6)
    for i in range(nb):  # range is nb if tag is on big otherwise from nb to N
        drx = 0
        dry = 0
        c = 0
        for k in range(10):
            if dist_bulk[i,k] != np.Inf and i!=k:
                drx += (x1[idxs[i,k]]-x0[idxs[i,k]])
                dry += (y1[idxs[i,k]]-y0[idxs[i,k]])
                c = c+1
        disp_crx[i] = drx/c
        disp_cry[i] = dry/c   # index=i if tag on big, otherwise i-nb
    return disp_crx, disp_cry
    
def CalcMSD(tagx,tagy,allx,ally):
    msd = np.zeros(frames)
    for w in range(0,frames):   # w = window averaging
        lc = 0
        msd1 = 0
        print('w:', w)
        for ti in range(0,(frames-w)+1):
            if (ti+w<frames):
                disp_crx, disp_cry = cr(allx[:,ti],allx[:,ti+w],ally[:,ti],ally[:,ti+w])
                msd0 = MSD_cr(tagx[:,ti],tagx[:,ti+w],tagy[:,ti],tagy[:,ti+w],disp_crx,disp_cry) 
                
                msd1 += msd0
                lc = lc +1
        msd[w] = msd1/(lc)
        print('msd[' + str(w) + ']:', msd[w])
    return msd
    
start_time = time.time()
msd_big = CalcMSD(big_x, big_y,x,y)    

fps = 1/21
t = np.arange(len(msd_big))*10*fps
data = np.vstack((t,msd_big)).T
print(np.shape(data))
end_time = time.time()
print("seconds elapsed big:"+ str(end_time-start_time))
np.savetxt('msd_cr_10neib_rmax=1.5_af='+af+'_all_big_tag.txt',data)

plt.plot(data[:,0],data[:,1],label='all')
plt.xscale('log')
plt.yscale('log')
plt.show()
