# %%
from collections import defaultdict
import itertools as it
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%%
# 正弦波
f=10 # 周波数
# 位置秒あたりの測定回数
f_s=100

t=np.linspace(0,2,2*f_s,endpoint=False)
x=np.cos(f*2*np.pi*t)
fig,ax=plt.subplots()
ax.plot(t,x,marker='o',)

#%%
# DFTをかける
# 時間領域から周波数領域に変換する
from scipy import fftpack
X=fftpack.fft(x)
freqs=fftpack.fftfreq(len(x))*f_s
fig,ax=plt.subplots()
ax.stem(freqs,np.abs(X))

#%%
from scipy.io import wavfile
# 鳥のさえずりのスペクトログラム
file='./dataset/elegant-scipy/data/nightingale.wav'
rate,audio=wavfile.read(file)
# モノラルに変換
audio=np.mean(audio,axis=1)
N=audio.shape[0]
L=N/rate # 秒数

#%%
# 可視化
plt.plot(np.arange(N)/rate,audio)

#%%
# 短時間フーリエ変換を行うため、1024個のサンプルを100づつ
# windowさせながら作成
from skimage import util
M=1024
slices=util.view_as_windows(audio,window_shape=(M,),step=100)
print(slices.shape)

# 窓関数を生成し、信号に掛け合わせる
win=np.hanning(M+1)[:-1]
slices=slices*win
# 列ごとに一つのスライスの形にする
slices=slices.T
print(slices.shape)

#%%
# 各スライス(1024の観測点)ごとにDFTを行う
spectrum=np.fft.fft(slices,axis=0)[:M//2+1:-1]

#%%
# 可視化
# スペクトログラムをデシベルに変換
S=abs(spectrum)
S=np.log10(S/np.max(S))

plt.imshow(S,origin='lower',cmap='viridis')

#%%
# ノイズが多い画像の周波数成分を調べる
file='./dataset/elegant-scipy/images/moonlanding.png'
image=io.imread(file)
M,N=image.shape
plt.imshow(image)

#%%
# スペクトルを可視化
F=fftpack.fftn(image)
F_magnitude=np.abs(F)
F_magnitude=fftpack.fftshift(F_magnitude)
# 中有部分は低い周波数(=画像の滑らかなぼんやりとした部分)
plt.imshow(np.log(1+F_magnitude))

#%%
# ノイズを除去する
K=40
# ぼんやりとした部分をフィルタの対象から除外する
F_magnitude[
    M//2-K:M//2+K,
    N//2-K:N//2+K
]=0

# 98%タイルよりも低い値のみを取ってくる
peaks=F_magnitude<np.percentile(F_magnitude,98)
# shift前に戻す
peaks=fftpack.ifftshift(peaks)

F_dim=F.copy()
# ピークのみに絞る
F_dim=F_dim*peaks.astype(int)

# 画像をもとに戻し、実部だけ出力
image_filtered=np.real(fftpack.ifftn(F_dim))
fig,ax=plt.subplots(2,1,figsize=(4.8,7))
ax[0].imshow(image)
ax[1].imshow(image_filtered)
