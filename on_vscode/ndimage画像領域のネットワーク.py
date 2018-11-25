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

#%%
# ノイズ処理
sig=sig+np.random.normal(0,0.3,size=sig.shape)
plt.plot(sig)


#%%
# 畳み込みを利用して、近傍で重み付き平均をとることで
# ノイズを除去する
# ガウシアンカーネルで平坦化する
def gaussian_kernel(size,sigma):
    positions=np.arange(size)-size//2
    kernel_raw=np.exp(-positions**2/(2*sigma**2))
    kernel_normalized=kernel_raw/np.sum(kernel_raw)

    return kernel_raw


fig,ax=plt.subplots(4,1)
# もとの信号
ax[0].plot(sig)
# 平坦化せず畳み込み
ax[1].plot(ndi.convolve(sig,diff))

# 差分フィルタ平坦化カーネルを結合　(結合律を利用)
smooth_diff=ndi.convolve(gaussian_kernel(30,5),diff)
sdsig=ndi.convolve(sig,smooth_diff)

# 平坦化
ax[2].plot(ndi.convolve(sig,gaussian_kernel(30,5)))
# 平坦化してから差分をとる
ax[3].plot(sdsig)
plt.tight_layout()

#%%
# 画像のフィルタリング(畳み込み)
coins=coins.astype(float)
diff2d=np.array([[0,1,0],[1,0,1],[0,1,0]])
coins_edge=ndi.convolve(coins,diff2d)
fig,ax=plt.subplots(1,2)
ax[0].imshow(coins)
ax[1].imshow(coins_edge)

#%%
#ノイズを除去して輪郭を際立出せる

# ソーベルフィルタ
hsovel=np.array(
[ 
[1,2,1],
[0,0,0],
[-1,-2,-1]
 ]
)
vsovel=hsovel.T

def reduce_xaxis_label(ax,factor):
    plt.setp(ax.xaxis.get_ticklabels(),visible=False)

    for label in ax.xaxis.get_ticklabels()[::factor]:
        label.set_visible(True)


coin_h=ndi.convolve(coins,hsovel)
coin_v=ndi.convolve(coins,vsovel)

fig,ax=plt.subplots(1,2)
ax[0].imshow(coin_h,cmap=plt.cm.RdBu)
ax[1].imshow(coin_v,cmap=plt.cm.RdBu)

for a in ax:
    reduce_xaxis_label(a,2)

#%%
#水平と垂直のフィルタリングを重ね合わせる
plt.imshow(np.sqrt(coin_h**2+coin_v**2))

#%%
# generic_filterを使う
# 一次元に変換
hsobel_r=np.ravel(hsovel)
vsobel_r=np.ravel(vsovel)

def sobel_magnitude_filter(values):
        h_edge=values@hsobel_r
        v_edge=values@vsobel_r
        # 三平方の和を取る
        return np.hypot(h_edge,v_edge)
        
sobel_mag=ndi.generic_filter(coins,sobel_magnitude_filter,size=3)
plt.imshow(sobel_mag)
#%%
from skimage import morphology
# 住宅の価格のヒートマップを税率のマップに変換する
# 1ピクセル=100mとする。値はその区画での価格の中央値
house_price=(0.5+np.random.rand(100,100))*1e6
# taxの適用範囲(半径1km=100*10)
footprint=morphology.disk(radius=10)

# 特定の領域の値段のうち90%目の値段に対して傾斜がかかる
def tax(prices):
    return 10000+0.05*np.percentile(prices,90)

# taxを計算
tax_rate_map=ndi.generic_filter(house_price,tax,footprint=footprint)
# 畳み込みにより近傍の値段から
fig,ax=plt.subplots(1,2)
ax[0].imshow(house_price)
ax[1].imshow(tax_rate_map)

#%%
#畳み込みにより、Conwayのライフゲームを構築
# 増殖、死滅のルール

def next_gen_filter(values):
        center=values[len(values)//2]
        neighbors_count=np.sum(values)-center
        # 隣接するcellが3つならば生き返るか存続し、2つならば存続する
        if neighbors_count==3 or (center and neighbors_count==2):
                return 1.
        
        # それ以外の場合死滅 
        else:
                return 0.

# 畳み込むを行う関数
def next_generation(board,mode='constant'):
        return ndi.generic_filter(
                board,
                next_gen_filter,
                size=3,
                mode=mode
        )

# 初期条件
random_board=np.random.randint(0,2,size=(50,50))

# 100世代に渡り実験
n_generations=200
# import time
for gen in range(n_generations):
        random_board=next_generation(random_board,mode='wrap')
        plt.imshow(random_board)
        # time.sleep(1)



#%%
a=[1,23,4]
b=[1,23,4]
np.hypot(a,b)


#%%

# 領域隣接グラフ
url = ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/' 'bsds/BSDS300/html/images/plain/normal/color/108073.jpg')
tiger=io.imread(url)
# SLICで分割
from skimage import segmentation,color
seg=segmentation.slic(tiger,n_segments=30,
compactness=40.0,enforce_connectivity=True,
sigma=3)

io.imshow(color.label2rgb(seg,tiger))

#%%
# セグメント同士の色の差を可視化
from skimage.future import graph
g=graph.rag_mean_color(tiger,seg)
graph.show_rag(seg,g,tiger)

#%%
import networkx as nx

def add_edge_filter(values,graph):
        center=values[len(values)//2]
        for neighbor in values:
                # neighborが中心ではなく、かつcenterとneighborが同じedgeに所属してないならば、
                # edgeとして踏力する
                if neighbor!=center and not graph.has_edge(center,neighbor):
                        graph.add_edge(center,neighbor)
        return 0.0
g
def build_rag(labels,image):
        g=nx.Graph()
        footprint=ndi.generate_binary_structure(
                labels.ndim,
                connectivity=1
        )

        _=ndi.generic_filter(
                labels,
                add_edge_filter,
                footprint=footprint,
                mode='nearest',
                extra_arguments=(g,)
        )

        for n in g:
                g.node[n]['totalcolor']=np.zeros(3,np.double)
                g.node[n]['pixelcount']=0
        
        for index in np.ndindex(labels.shape):
                n=labels[index]
                g.node[n]['totalcolor']+=image[index]
                g.node[n]['pixelcount']+=1

        return g

#%%
# 実行
g=build_rag(seg,tiger)
for n in g:
        node=g.node[n]
        node['mean']=node['totalcolor']/node['pixelcount']
for u,v in g.edges():
        d=g.node[u]['mean']-g.node[v]['mean']
        g[u][v]['weight']=np.linalg.norm(d)

def threshold_graph(g,t):
        to_remove=[
                (u,v) for (u,v,d) in g.edges(data=True)
                if d['weight']>t
        ]

        g.remove_edges_from(to_remove)
# 各セグメントの平均の色の差の情報を使用し、エッジの閾値を決定
threshold_graph(g,80)

map_array=np.zeros(np.max(seg)+1,int)
for i,segment in enumerate(nx.connected_components(g)):
        for initial in segment:
                map_array[int(initial)]=i
segmented=map_array[seg]
plt.imshow(color.label2rgb(segmented,tiger))