#%%
import numpy as np
import pandas as pd
from scipy import stats
from skimage import io
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%%
# ホワイトノイズ画像を作成
random_image=np.random.rand(500,500)
plt.imshow(random_image)

#%%
# 逆に画像から配列に変換
url_coins = ('https://raw.githubusercontent.com/scikit-image/scikit-image/'
'v0.10.1/skimage/data/coins.png')

coins=io.imread(url_coins)
print(f'type{type(coins)}shape:{coins.shape}')
io.imshow(coins)




#%% 
# カラー画像
url_astronaut = ('https://raw.githubusercontent.com/scikit-image/scikit-image/' 'master/skimage/data/astronaut.png')
astro=io.imread(url_astronaut)
print(f'type{type(astro)}shape:{astro.shape}')
io.imshow(astro)

#%%
# 一部のセルを加工
astro_sq=astro.copy()
astro_sq[50:100,50:100]=[255,0,0]
io.imshow(astro_sq)

#%%
# 画像にグリッド線を追加する関数
def overlay_grid(image,spacing=128):
    image_gridded=image.copy()

    # 追加
    image_gridded[spacing:-1:spacing, :] = [0, 0, 255]
    image_gridded[:, spacing:-1:spacing] = [0, 0, 255]

    return image_gridded

io.imshow(overlay_grid(astro,20))

#%%
# 信号処理のフィルタ
#　一次元
sig=np.zeros(100,np.float)
sig[30:60]=1 # 30msから60msまで光が観測された

fig,ax=plt.subplots()
ax.plot(sig)

#%%
# 光がいつ点灯したか調べる
sigdelta=sig[1:]
sigdiff=sigdelta-sig[:-1]
# clipで最大と最小を決める
sigon=np.clip(sigdiff,0,np.inf)
# plt.plot(sigon)
print(f"SIGNAL on at {1+np.flatnonzero(sigon)[0]}")


#%%
# 関数を使用した畳み込み
diff=np.array([1,0,-1])
from scipy import ndimage as ndi
dsig=ndi.convolve(sig,diff)
plt.plot(dsig)






