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
file_A='./on_vscode/Qiita/img/book.jpg'
file_B='./on_vscode/Qiita/img/bottle.jpg'
image_A=io.imread(file_A)
image_B=io.imread(file_B)
# image_A=image_A[image_B.shape]
io.imshow(image_A)

#%%
# file_A='./on_vscode/Qiita/img/seek_different.jpeg'
# image=io.imread(file_A)
# W=image.shape[1]
# image_A=image[:,:W//2]
# image_B=image[:,W//2:]
# plt.imshow(image_A)
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
def convertF(image,center=500,f_peak=98):
    M,N=image.shape
    F=fftpack.fftn(image)
    F_magnitude=np.abs(F)

    F_magnitude=fftpack.fftshift(F_magnitude)
    # ぼんやりとした部分をフィルタの対象から除外する
    F_magnitude[
        M//2-center:M//2+center,
        N//2-center:N//2+center
    ]=0

    
    peak=np.percentile(F_magnitude,f_peak) 
    filter_=F_magnitude<peak
    filter_=fftpack.ifftshift(filter_)

    return  F*filter_.astype(int)

#%% 
# 差分を抽出

def diffF(F_A,F_B,center=500,f_peak=98):
    M,N=F_A.shape[:2]

    F_diff=F_A-F_B

    F_magnitude=np.abs(F_diff)
    F_magnitude=fftpack.fftshift(F_diff)

    F_magnitude[
        M//2-center:M//2+center,
        N//2-center:N//2+center
    ]=0


    peak=np.percentile(F_magnitude,f_peak) 
    filter_=F_magnitude>peak
    filter_=fftpack.ifftshift(filter_)


    return F_diff*filter_.astype(int)
#%%
# 逆変換
def convertI(F,center=150,f_peak=98):

    image=fftpack.ifftn(F)
    image=np.real(image)

    return image

#%%
plt.imshow(convertI(convertF(image_A_,0,f_peak=100),0,0))
#%%
# 変換F_A=convertF(image_A_,1000,f_peak=98)
F_B=convertF(image_B_,1000,f_peak=98)


#%%
F_diff=diffF(F_A,F_B,1000,f_peak=30)

I_diff=convertI(F_diff)

io.imshow(np.where(I_diff>100,100,0))