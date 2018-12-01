# %%
from collections import defaultdict
import itertools as it
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from skimage import io
from scipy import fftpack

#%%
# 間違い探しの画像を読み込む
file_='on_vscode/Qiita/img/seek_different.jpeg'
image=io.imread(file_)
io.imshow(image)



#%%
# 画像を分割
width=image.shape[1]
image_A=image[:,:width//2]
image_B=image[:,width//2:]

_,ax=plt.subplots(1,2)
ax[0].imshow(image_A)
ax[1].imshow(image_B)

#%%
# 変換せずに差分を出力する
io.imshow(image_A-image_B)
#%%
# 二次元に変換
image_A_=image_A.mean(axis=2)
image_B_=image_B.mean(axis=2)
plt.imshow(image_A_)
#%%
# forier変換
def convertF(image,center=150,f_peak=98):
    M,N=image.shape
    F=fftpack.fftn(image)
    F_magnitude=np.abs(F)

    F_magnitude=fftpack.fftshift(F_magnitude)
    # ぼんやりとした部分をフィルタの対象から除外する
    F_magnitude[
        M//2-center:M//2+center,
        N//2-center:N//2+center
    ]=0
    F_magnitude=fftpack.fftshift(F_magnitude)

    peak=np.percentile(F_magnitude,f_peak)
    F_filtered=np.where(F_magnitude<peak,F,0)


    return  F_filtered

#%% 
# 差分を抽出

def diff(F_A,F_B,center=150,f_peak=98):
    M,N=F_A.shape[:2]

    F_magnitude=abs(F_A-F_B)

    F_magnitude=fftpack.fftshift(F_magnitude)
    # # ぼんやりとした部分をフィルタの対象から除外する
    # F_magnitude[
    #     M//2-center:M//2+center,
    #     N//2-center:N//2+center
    # ]=0
    F_magnitude=fftpack.fftshift(F_magnitude)

    peak=np.percentile(F_magnitude,f_peak)
    F_filter=np.where(F_magnitude>peak,1,0)
    
    
    return F_filter
#%%
# 逆変換
def convertI(F,center=150,f_peak=98):


    image_filtered=fftpack.ifftn(F_filtered)
    image_filtered=np.real(image_filtered)

    return image_filtered

#%%
# 差分を可視化
F_A=convertF(image_A_)
F_B=convertF(image_B_)


F_diff=diff(F_A,F_B)
F_A[F_diff]
# I_diff=convertI(I_diff)

# plt.imshow(I_diff)

