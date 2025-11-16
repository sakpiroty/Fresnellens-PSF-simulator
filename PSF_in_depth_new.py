### 三次元投影対象の場合
### 投影対象表面上の代表点 → 光源面上の像点 → PSF simu → 切り取って最適化
import os
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
import numpy as np
import time
from PIL import Image
from torch.optim import Adam
from pytorch_memlab import MemReporter
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import math

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
reporter = MemReporter()

refractiveindex = 1.492
diameter = 500.0 #(mm)
pitch = 0.5 #(mm)
split_number = int(diameter / pitch / 2)
image_z = -500.0 #(mm)
pixel_size = 0.3125 #(mm)
IMAGESIZE = 512 #(pix)
MASKSIZE = 64 #(pix)
pixel_size_MASK = diameter/MASKSIZE #(mm)
NRAYS = 1000

# PSFNUM = 25
targetmodel = "thru-focus"
file_path = "./data/" + targetmodel + ".csv"
# CSVファイルを読み込む
df = pd.read_csv(file_path)
# 必要な列（x, y, z）を抽出してtorch.tensorに変換
targetloc = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32, device = device)
PSFNUM =  targetloc.size()[0]

PSFSIZE = 200
PSFCENTER = int((PSFNUM + 1)/2)
EPOCHS = 1
sep_mask = 4
lr = 0.05
black_base, white_base = 0, 1 #マスクの最大透過率

df = pd.read_csv("../raytracing/data/womask_intensity_depth.csv")
ref_intensity = torch.tensor(df[["0", "1", "2","3","4", "5", "6","7", "8", "9","10"]].values, dtype=torch.float32, device = device).t()


def save_tensorimage(tensor, min, max, filepath):
    # テンソルが2次元であることを確認
    if tensor.ndim != 2:
        raise ValueError("Dimension of the saved tensor must be 2.")
    
    # # 値を0〜255に正規化
    # tensor_min = min
    # tensor_max = max
    # normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * 255.0
    normalized_tensor = tensor / max * 255.0
    # テンソルをuint8型に変換
    tensor_uint8 = normalized_tensor.to(torch.uint8)
    
    # PIL.Imageに変換
    Image.fromarray(tensor_uint8.cpu().numpy()).save(filepath)

## 本来の正規化
def prev_normalize_image(img):
    img_max = img.max()
    return img / img_max

### 提案する正規化
def normalize_image(img):
    img_max = img.max()
    img_min = img.min()
    return (img - img_min) / (img_max- img_min)

def save_loss_logger_and_graph(log_dir, loss_logger):
    # loss履歴情報を保管しつつ、グラフにして画像としても書き出す
    torch.save(loss_logger, os.path.join(log_dir, "./log/loss_logger.pt"))
    fig, ax = plt.subplots(1,1)
    epoch = range(len(loss_logger))
    ax.plot(epoch, loss_logger, label="train_loss")
    ax.set_ylim(min(loss_logger)-0.1, max(loss_logger)+0.1)
    ax.legend()
    fig.savefig(os.path.join(log_dir, "./log/loss_history.jpg"))
    plt.clf()
    plt.close()


###
####　目標MTF画像の作成
### 
# CSF計算 (Mannos-Sakrison model)
def csf(f):
    """ コントラスト感度関数（CSF）のモデル """
    # C0 = 1.5  # 正規化係数
    # alpha = 0.2  # 周波数減衰係数
    # return C0 * f * torch.exp(-alpha * f)
    return 2.6 * (0.0192 + 0.114 * f) * torch.exp(- torch.pow(0.114*f, 1.1))

def calc_targetmtf(object_z, pix_size):
    # 画像の物理サイズ (mm)
    W = PSFSIZE * pix_size

    # 画像の視角 (度)
    theta = (2 * torch.rad2deg(torch.arctan(W / (2 * object_z)))).to(device)

    # 空間周波数軸 (サイクル/mm と サイクル/度)
    freqs_mm = torch.fft.fftfreq(PSFSIZE, d=pix_size).to(device)  # 物理周波数 (cycle/mm)
    freqs_deg = freqs_mm * (W / theta)  # 視角周波数 (cycle/degree)

    # Torch tensorsでの実装
    u, v = torch.meshgrid(freqs_deg, freqs_deg, indexing='ij')
    freq_map = torch.sqrt(u**2 + v**2)

    # 目標MTF (正規化)
    target_mtf = csf(freq_map)
    target_mtf = torch.fft.fftshift(target_mtf)
    target_mtf = normalize_image(target_mtf).to(device)  # 最大値を1に正規化

    return target_mtf

def morans_I(image: torch.Tensor) -> float:
    """
    Compute Moran's I for a grayscale image tensor,
    restricted to a circular region centered in the image.

    Args:
        image (torch.Tensor): 2D tensor of shape (H, W), with float values.

    Returns:
        float: Moran's I statistic
    """
    if image.dim() != 2:
        raise ValueError("Input image must be a 2D tensor.")

    H, W = image.shape
    N = H * W
    x = image

    # Create circular mask
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H / 2, W / 2
    radius = min(H, W) / 2
    circular_mask = (((X - center_x)**2 + (Y - center_y)**2) <= radius**2).to(device)

    # Mean over the circular region only
    x_bar = x[circular_mask].mean()

    # Weight matrix using 4-neighbor connectivity
    kernel = torch.tensor([[0.0, 1.0, 0.0],
                           [1.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0]], dtype=torch.float32, device = device).unsqueeze(0).unsqueeze(0)

    # Center the image
    x_centered = x - x_bar
    x_centered = x_centered.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    # Convolve to compute spatial lag (weighted sum of neighbors)
    spatial_lag = F.conv2d(x_centered, kernel, padding=1)

    # Compute valid weights with same kernel
    mask = circular_mask.float()
    weight_sum = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()

    valid = (weight_sum > 0) & circular_mask

    numerator = (x_centered.squeeze() * spatial_lag.squeeze())[valid].sum()
    denominator = (x_centered.squeeze() ** 2)[circular_mask].sum()
    W = weight_sum[valid].sum()
    N_valid = circular_mask.sum()

    I = (N_valid / W) * (numerator / denominator)
    return I



def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = torch.nanmean(tensor)
    tmp = torch.pow((tensor - tensor_mean), 2)
    output = torch.nanmean(tmp)
    return output


#####
# ここまで
#####



class Ray:
    # レイを表現する
    # origin: レイの始点
    # direction: レイの方向ベクトル
    def __init__(self, origin: torch.tensor, direction: torch.tensor, roi: torch.tensor, intensity: torch.tensor):
        self.origin = origin
        self.direction = direction
        self.roi = roi
        self.intensity = intensity
    
    def __repr__(self):
        return "origin: {0}, direction: {1}".format(self.origin, self.direction)

    # レイ上の位置を計算する
    # t: 始点からの距離
    def position(self, t: torch.tensor) -> torch.tensor:
        p = self.origin + t * self.direction
        return p

# ベクトルを正規化する
def normalize(v: torch.tensor) -> torch.tensor:
    norm = torch.norm(v, dim = 2, keepdim = True)
    return v / norm

# テンソルの各要素(3次元ベクトル)について内積を計算する
def dot_tensors(v: torch.tensor, n: torch.tensor) -> torch.tensor:
    dotmat = torch.sum(v * n, dim=2, keepdim=True)
    return dotmat

# レイの方向に応じて法線を逆転させる
# v: レイの方向ベクトル
# n: 法線
def flip_normal(v: torch.tensor, n: torch.tensor) -> torch.tensor:
    dotmat = dot_tensors(v,n)
    n = torch.where(dotmat < 0, n, -n)
    return n

# 屈折ベクトルを計算する# 全反射の場合はNoneを返す
# v: 入射ベクトル
# n: 法線
# n1: 入射側媒質の屈折率
# n2: 出射側媒質の屈折率

# 全反射考慮無し
def refract(v: torch.tensor, n: torch.tensor, n1: torch.tensor, n2: torch.tensor) -> Optional[torch.tensor]:
    # 屈折ベクトルの水平方向
    t_h = -n1 / n2 * (v - dot_tensors(v, n)*n)

    # 全反射
    total_reflect = (torch.norm(t_h, dim = 2) > 1)
    ratio = total_reflect.sum() / total_reflect.numel()
    # if ratio > 0:
    #     print(f"TOTAL REFLECTION: {ratio}")
    t_h = torch.where(torch.norm(t_h, dim = 2, keepdim = True) > 1, float('nan'), t_h)

    # 屈折ベクトルの垂直方向
    t_p = -torch.sqrt(1 - torch.sum(t_h**2,dim=2,keepdim = True)) * n
    
    # 屈折ベクトル
    t = t_h + t_p

    return normalize(t)

class LensSurface:
    # レンズ面を表現する
    # r=0で平面を表現する
    # r: 曲率半径
    # h: 開口半径
    # d: 次の面までの距離
    # ior: 屈折率

    def __init__(self, filepath: str, ior: float):
        self.z = 0
        self.theta = []
        # レンズデータの読み込み
        self.df = pd.read_csv(filepath)
        for i in range(len(self.df)):
            self.theta.append(torch.tensor(self.df.iloc[i]["theta"], dtype = torch.float32))
        self.ior = ior

    # レイと円錐の交差位置, 法線を計算する
    # 交差しない場合はNoneを返す
    # ray: レイ, n_cone: 何個目の円錐か
    def intersect_cone(self, ray: Ray, index_of_cone: int) -> Optional[Tuple[torch.tensor, torch.tensor]]:
        # 円錐の底面半径/頂点
        radius = (split_number - index_of_cone) * pitch
        
        tan = torch.ones_like(ray.roi) * torch.tan(self.theta[index_of_cone])
        center = -radius/tan
        tmp_zeros = torch.where(torch.isnan(center),float('nan'),0.0)
        center = torch.cat((tmp_zeros,tmp_zeros, center), dim = 2)
        
        # 判別式
        d0 = (ray.direction[:,:,0]).unsqueeze(dim = 2)
        d1 = (ray.direction[:,:,1]).unsqueeze(dim = 2)
        d2 = (ray.direction[:,:,2]).unsqueeze(dim = 2)
        o0 = (ray.origin[:,:,0]).unsqueeze(dim = 2)
        o1 = (ray.origin[:,:,1]).unsqueeze(dim = 2)
        o2 = (ray.origin[:,:,2]).unsqueeze(dim = 2)
        a = d0**2 + d1**2 - (d2*tan)**2
        b = o0 * d0 + o1 * d1 - (o2 * (tan**2) + radius*tan) * d2
        c = o0**2 + o1**2 - (o2*tan)**2 - 2 * o2 * radius * tan - radius**2

        D = b**2 - a*c
        # D < 0の場合は交差しない
        D = torch.where(D < 0, float('nan'), D)

        # tの候補
        t_1 = ((-b + torch.sqrt(D))/a)
        ## 減算での桁落ちを防ぐために，t_2 = ((-b - torch.sqrt(D))/a)とは計算しない．
        t_2 = c / (a * t_1)
        t_1 = t_1.squeeze(dim = 2)
        t_2 = t_2.squeeze(dim = 2)

        # 適切な交差位置を選択
        p_1 = ray.position(torch.stack([t_1]*3, axis = -1))
        p_2 = ray.position(torch.stack([t_2]*3, axis = -1))

        mask1 = (-pitch/tan <= p_1[:,:,2].unsqueeze(dim = 2)) & (p_1[:,:,2].unsqueeze(dim = 2) <= 0)
        mask2 = (-pitch/tan <= p_2[:,:,2].unsqueeze(dim = 2)) & (p_2[:,:,2].unsqueeze(dim = 2) <= 0)

        p = torch.where((~torch.isnan(p_1) & mask1), p_1, torch.where((~torch.isnan(p_2) & mask2), p_2, float('nan')))
        t_1 = torch.stack([t_1]*3, axis = -1)
        t_2 = torch.stack([t_2]*3, axis = -1)
        t = torch.where((~torch.isnan(p_1) & mask1), t_1, torch.where((~torch.isnan(p_2) & mask2), t_2, float('nan')))

        # 法線ray_direction = ray_direction / torch.norm(ray_direction, dim=0, keepdim=True)
        n = (p-(1 - (dot_tensors(p-center,p-center))/(torch.sum(center**2, dim =2, keepdim = True) - dot_tensors(p, center)))*center)
        n = normalize(n)
        n = flip_normal(ray.direction, n)
        
        return p, n, t
        
    # レイと円柱の交差位置, 法線を計算する
    # 交差しない場合はNoneを返す
    # ray: レイ, n_cylin: 何個目の円柱か
    def intersect_cylin(self, ray: Ray, index_of_cylin: int) -> Optional[Tuple[torch.tensor, torch.tensor]]:
        # 円錐の底面半径/頂点
        radius = (split_number - index_of_cylin - 1) * pitch
        
        # 判別式
        # odash = ray.origin * (torch.tensor([1,1,0], dtype = torch.float32, device = device).expand_as(ray.origin))
        # ddash = ray.direction * (torch.tensor([1,1,0], dtype = torch.float32, device = device).expand_as(ray.direction))
        # a = torch.norm(ddash**2, dim = 2, keepdim = True)
        # b = dot_tensors(odash,ddash)
        # c = torch.norm(odash**2, dim = 2, keepdim = True) - radius**2
        d0 = (ray.direction[:,:,0]).unsqueeze(dim = 2)
        d1 = (ray.direction[:,:,1]).unsqueeze(dim = 2)
        d2 = (ray.direction[:,:,2]).unsqueeze(dim = 2)
        o0 = (ray.origin[:,:,0]).unsqueeze(dim = 2)
        o1 = (ray.origin[:,:,1]).unsqueeze(dim = 2)
        o2 = (ray.origin[:,:,2]).unsqueeze(dim = 2)
        a = d0**2 + d1**2
        b = o0 * d0 + o1 * d1
        c = o0**2 + o1**2  - radius**2
        D = b**2 - a * c

        # D < 0の場合は交差しない
        D = torch.where(D < 0, float('nan'), D)

        # tの候補
        t_1 = ((-b + torch.sqrt(D))/a)
        ## 減算での桁落ちを防ぐために，t_2 = ((-b - torch.sqrt(D))/a)とは計算しない．
        t_2 = c / (a * t_1)
        t_1 = t_1.squeeze(dim = 2)
        t_2 = t_2.squeeze(dim = 2)

        # 交差位置
        p_1 = ray.position(torch.stack([t_1]*3, axis = -1))
        p_2 = ray.position(torch.stack([t_2]*3, axis = -1))
        
        tan = torch.ones_like(ray.roi) * torch.tan(self.theta[index_of_cylin])

        mask1 = (-pitch/tan < p_1[:,:,2].unsqueeze(dim = 2)) & (p_1[:,:,2].unsqueeze(dim = 2) < 0)
        mask2 = (-pitch/tan < p_2[:,:,2].unsqueeze(dim = 2)) & (p_2[:,:,2].unsqueeze(dim = 2) < 0)
        p = torch.where((~torch.isnan(p_1) & mask1), p_1, torch.where((~torch.isnan(p_2) & mask2), p_2, float('nan')))
        t_1 = torch.stack([t_1]*3, axis = -1)
        t_2 = torch.stack([t_2]*3, axis = -1)
        t = torch.where((~torch.isnan(p_1) & mask1), t_1, torch.where((~torch.isnan(p_2) & mask2), t_2, float('nan')))

        center = p * (torch.tensor([0,0,1], dtype = torch.float32, device = device).expand_as(p))
  
        # 法線
        n = flip_normal(ray.direction, normalize(center-p))

        return p, n, t
    
    # レイと円形平面の交差位置, 法線を計算する
    # 交差しない場合はNoneを返す
    # ray: レイ, z: 対象平面のz座標
    def intersect_plane(self, ray: Ray, z, minr, maxr) -> Optional[Tuple[torch.tensor, torch.tensor]]:
        # 平面との交差
        # 交差位置

        t = -(ray.origin[:,:,2] - z) / ray.direction[:,:,2]
        p = ray.position(torch.stack([t]*3, axis = -1))

        # 交差位置が開口半径以上なら交差しない
        r = p[:,:,0]**2 + p[:,:,1]**2
        mask = (minr**2 <= r) & (r <= maxr**2)

        mask = torch.stack([mask] * 3, axis = -1)
        t = torch.stack([t]*3, axis = -1)
        t = torch.where(mask, t, float('nan'))
        p = torch.where(mask, p, torch.tensor(float('nan'), device = device))
        tmp = torch.tensor([0.0, 0.0, -1.0],device = device)
        n = flip_normal(ray.direction, tmp.repeat(p.size()[0],p.size()[1],1))
        n = torch.where(mask, n, torch.tensor(float('nan'), device = device))

        return p, n, t

# 極座標グリッドを作成
def polar_grid(H = int(MASKSIZE), W = int(MASKSIZE), R = int(MASKSIZE / 2), Theta = 360):
    """ 極座標のサンプリンググリッドを作成 """
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    r = torch.linspace(0, 1, R).view(-1, 1)  # 半径方向
    theta = torch.linspace(0, 2*torch.pi, Theta).view(1, -1)  # 角度方向

    X = r * torch.cos(theta)
    Y = r * torch.sin(theta)

    X = X.view(1, R, Theta)
    Y = Y.view(1, R, Theta)

    return torch.stack([X, Y], dim=-1)  # (1, R, Theta, 2)

# 高周波成分を強調する損失関数
def high_freq_loss(polar_image, Theta = 360):
    """ 各rごとのθ方向の高周波成分を強調する損失 """
    fft = torch.fft.fft(polar_image, dim=-1)  # θ方向のFFT
    freq = torch.fft.fftfreq(Theta, d=1/Theta).to(polar_image.device)  # 周波数軸

    high_freq_mask = (freq > 0.8 * freq.max()).float()  # 高周波部分をマスク
    loss = torch.mean(torch.abs(fft * high_freq_mask))  # 高周波成分の強化

    return -loss  # 高周波成分を最大化

class LensSystem:
    # レンズ系を表現する
    # filepath: csvファイルのファイルパス
    def __init__(self, filepaths: list):

        # レンズ面の生成
        self.lenses = []
        for i in range(1):
            self.lenses.append(LensSurface(filepaths[i], refractiveindex))

    # 物体側から像側に向かってレイトレーシングを行い、光線経路を返す
    # ray_in: 入射レイ
    def raytrace_from_object(self, psfloc, n_rays = NRAYS) -> torch.tensor:
        lens = self.lenses[0]
        N_loop = 20

        # ベクトル化された実装
        x, y = torch.meshgrid(torch.arange(n_rays, device = device), torch.arange(n_rays, device = device), indexing = 'ij') # 全てのx, y座標を生成
        x = (x - n_rays / 2) / n_rays
        y = (y - n_rays / 2) / n_rays
        z = torch.sqrt(torch.tensor(0.3, device = device)).expand_as(x)
        ray_roi = torch.ones_like(x).unsqueeze(dim = 2)
        ray_intensity = torch.ones_like(x).unsqueeze(dim = 2)

        ray_direction = torch.stack([x, y, z], dim = 0)
        ray_direction = ray_direction / torch.norm(ray_direction, dim=0, keepdim=True)
        ray_direction = torch.permute(ray_direction, (1,2,0))

        tmptargetloc = targetloc[psfloc,:].unsqueeze(dim = 0).unsqueeze(dim = 0)
        tmpintensity = torch.tensor(1.0, device = device).unsqueeze(dim = 0).unsqueeze(dim = 0)
        ray_from_target = Ray(tmptargetloc, -tmptargetloc, tmpintensity, tmpintensity)
        
        ray_origin, n, t = lens.intersect_plane(ray = ray_from_target, z = image_z, minr = 0, maxr = 500)
        if (ray_origin[:,:,0] < -(IMAGESIZE / 2 * pixel_size) or ray_origin[:,:,0] > (IMAGESIZE / 2 * pixel_size) or ray_origin[:,:,1] < -(IMAGESIZE / 2 * pixel_size) or ray_origin[:,:,1] > (IMAGESIZE / 2 * pixel_size)):
            print(f"ERROR: PSF No.{psfloc} is out of projectable area.")
            exit()
        # ray_origin = torch.tensor([(PSFX - PSFCENTER)/2*IMAGESIZE/2*pixel_size,(PSFY - PSFCENTER)/2*IMAGESIZE/2*pixel_size, image_z], device = device).expand_as(ray_direction)
        print(ray_origin)
        ray_origin = ray_origin.expand_as(ray_direction)

        ray = Ray(ray_origin, ray_direction, ray_roi, ray_intensity)

        #######
        ### 一回目のフレネルレンズ入射を想定する．
        #######
        ## 底面
        p, n, t = lens.intersect_plane(ray, 0.0, 0.0, diameter/2)

        ## 負の方向に進んでいる．
        if torch.any(t < 0):
            print("some t are minus.")
        
        ## 円錐
        for i in range(split_number):
            tmp_p, tmp_n, tmp_t = lens.intersect_cone(ray, i)
            ## より近い位置で衝突するならそれに更新
            p = torch.where(~torch.isnan(tmp_p) & (0 < tmp_t) & (tmp_t < t), tmp_p, p)
            t = torch.where(~torch.isnan(tmp_t) & (0 < tmp_t) & (tmp_t < t), tmp_t, t)
        
        ## 円柱
        for i in range(split_number-1):
            tmp_p, tmp_n, tmp_t = lens.intersect_cylin(ray, i)
            ## より近い位置で衝突するならそれに更新
            p = torch.where(~torch.isnan(tmp_p) & (0 < tmp_t) & (tmp_t < t), tmp_p, p)
            t = torch.where(~torch.isnan(tmp_t) & (0 < tmp_t) & (tmp_t < t), tmp_t, t)
        
        ## 元々フレネルレンズに当たらないレイはfinished
        finished = torch.where((torch.isnan(p)), True, False)
        ####
        ## ここまで
        ####

        # print(f'Out of Fresnel: {torch.sum(finished)/finished.numel()}')
        finishedratio = [(torch.sum(finished)/finished.numel()).cpu()]

        ## レイを一定回数トレースする．
        for loop in range(N_loop):
            if (finishedratio[-1] > 0.95):
                break

            ###　次の媒質の屈折率を求める．まず全反射しないと仮定すると，次の媒質の屈折率が1と1.492のどちらになるか．
            inverse = torch.where(ray.roi == 1.0, torch.tensor(refractiveindex, device = device), torch.tensor(1.0, device =device))
            
            ## 底面に当たるか？
            p, n, t = lens.intersect_plane(ray, 0.0, 0.0, diameter/2)
            hitbottom = (~torch.isnan(t)) & (0.0 <= t) ## これから底面に当たる
            zero_hitbottom = (~torch.isnan(t)) & (0.0 == t) ## 今底面にいる（全反射）

            ## 屈折（全反射を考慮）
            refvec = refract(-ray.direction, n, ray.roi, inverse) 
            ## 底面で全反射しているレイはhit判定ではない
            totalref_zero_hitbottom = (zero_hitbottom) & (torch.isnan(refvec))
            hitbottom = hitbottom & (~totalref_zero_hitbottom)
            
            # どの面にも当たらないレイは終了
            no_hit_anything = ~hitbottom

            ## 円錐
            for i in range(split_number):
                tmp_p, tmp_n, tmp_t = lens.intersect_cone(ray, i)
                ## より近い位置で衝突するならそれに更新
                updated = (~finished) & (~torch.isnan(tmp_t)) & (0 < tmp_t) & (tmp_t < t)
                p = torch.where(updated, tmp_p, p)
                n = torch.where(updated, tmp_n, n)
                t = torch.where(updated, tmp_t, t)
                hitbottom = torch.where(updated & hitbottom, False, hitbottom)
                no_hit_anything = torch.where(no_hit_anything & (~torch.isnan(tmp_t)) & (0 < tmp_t), False, no_hit_anything)
            
            ## 円柱
            for i in range(split_number-1): 
                tmp_p, tmp_n, tmp_t = lens.intersect_cylin(ray, i)
                ## より近い位置で衝突するならそれに更新
                updated = (~finished) & (~torch.isnan(tmp_t)) & (0 < tmp_t) & (tmp_t < t)
                p = torch.where(updated, tmp_p, p)
                n = torch.where(updated, tmp_n, n)
                t = torch.where(updated, tmp_t, t)
                hitbottom = torch.where(updated & hitbottom, False, hitbottom)
                no_hit_anything = torch.where(no_hit_anything & (~torch.isnan(tmp_t)) & (0 < tmp_t), False, no_hit_anything)

            #####
            ## どの面にも当たらないなら終了
            #####
            ray.origin = torch.where(no_hit_anything, float('nan'), ray.origin)
            ray.direction = torch.where(no_hit_anything, float('nan'), ray.direction)
            finished = torch.where((~finished) & no_hit_anything, True, finished)
            #####
            ## ここまで
            #####

            #####
            ## 底面に当たったレイのうち全反射しないものの終了処理
            #####
            ## 屈折（全反射を考慮）
            refvec = refract(-ray.direction, n, ray.roi, inverse)
            ## 全反射してないものは終了
            finish_in_this_loop = (~finished) & (hitbottom) & (~torch.isnan(refvec)) # このループで終了したか？
            ## 最後の更新
            ray.origin = torch.where(finish_in_this_loop, p, ray.origin)
            ray.direction = torch.where(finish_in_this_loop, refvec, ray.direction)
            ray.roi =  torch.where(finish_in_this_loop[:,:,0].unsqueeze(dim = 2), inverse, ray.roi)
            finished = torch.where(finish_in_this_loop, True, finished) # 平面部から出射したものは終了
            #####
            ## ここまで
            #####


            #####
            ## 終わってないレイの屈折
            #####
            ## 屈折（全反射を考慮）
            refvec = refract(-ray.direction, n, ray.roi, inverse)
            ray.origin = torch.where(~finished, p, ray.origin)
            ray.direction = torch.where(~finished, torch.where(torch.isnan(refvec), normalize(ray.direction - 2 * dot_tensors(ray.direction, n) * n), refvec), ray.direction)
            ray.roi =  torch.where(~finished[:,:,0].unsqueeze(dim = 2), torch.where(torch.isnan(refvec[:,:,0].unsqueeze(dim=2)), ray.roi, inverse), ray.roi)
            #####
            ## ここまで
            #####

            # rays.append(ray)

            # print(f'LOOP {loop+1}: {torch.sum(finished)/finished.numel()}')
            finishedratio.append((torch.sum(finished)/finished.numel()).cpu())
        
        #plt.plot(finishedratio)
        #plt.savefig("finishedratio.png")
        
        ##　ループ回数最大にしてもfinishedにならなかったものはNANとする．
        ray.origin = torch.where(finished, ray.origin ,float('nan'))
        ray.direction = torch.where(finished, ray.direction ,float('nan'))
        ray.intensity[:,:,0] = torch.where(torch.isnan(ray.origin[:,:,0]), 0, ray.intensity[:,:,0])

        del p,n,t,x,y,z,tmp_p, tmp_n, tmp_t
        torch.cuda.empty_cache()

        return ray

    def simulatePSF(self, ray: Ray, mask: torch.tensor, psfloc) -> torch.tensor:
        s_time = time.time()

        tmp_ray = copy.deepcopy(ray)
        # WORLD 座標から (マスク)画像座標への変換
        origin_inMASKpixel = tmp_ray.origin / pixel_size_MASK
        x_index = torch.where(torch.isnan(origin_inMASKpixel[:,:,1]), -1,(-(origin_inMASKpixel[:,:,1] - MASKSIZE/2) - 1 + 0.5).to(torch.long))
        y_index = torch.where(torch.isnan(origin_inMASKpixel[:,:,0]), -1,((origin_inMASKpixel[:,:,0] + MASKSIZE/2) - 1 + 0.5).to(torch.long))

        # 空間マスクの値に応じて光の強度が減衰 (白の透過率90%，黒の透過率10%とする．)
        mask = black_base + mask * (white_base - black_base)
        tmp_intensity = tmp_ray.intensity[:,:,0].clone()
        tmp_ray.intensity[:,:,0] = torch.where(x_index == -1, torch.tensor(0.0, device = device), tmp_intensity * mask[x_index, y_index])
        # transmission = (torch.nansum(tmp_ray.intensity[:,:,0]))/(torch.nansum(tmp_intensity)) # 光線の減衰比率
        # print(f"transmission rate: {transmission}")

        lens = self.lenses[0]

        # 結像位置まで計算
        object_z = targetloc[psfloc, 2]
        p, n, t = lens.intersect_plane(ray, object_z, 0, diameter)
        # ray.origin = p

        MAGNIFY = object_z / (-image_z)
        # レイトレース計算
        adj_pixel_size = pixel_size * MAGNIFY
        pix_world_xys = (p - targetloc[psfloc, :]) / adj_pixel_size
        result_xys = (p - targetloc[psfloc, :]) / adj_pixel_size
        # print(torch.cuda.memory_allocated())
        # del ray, origin_inMASKpixel, x_index, y_index, tmp_intensity 
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())

        # Target MTFの計算
        target_mtf = calc_targetmtf(object_z, adj_pixel_size)

        # WORLD 座標から 画像座標への変換
        result_xys[:,:,0] = -(pix_world_xys[:,:,1] - (PSFSIZE/2 - 1))
        result_xys[:,:,1] = pix_world_xys[:,:,0] + (PSFSIZE/2 - 1)

        # nan mask (迷光など)
        xmin = 0
        xmax = PSFSIZE -1 
        ymin = 0
        ymax = PSFSIZE - 1
        # nan mask (迷光など)
        nanmask = (~torch.isnan(result_xys[:,:,0])) & (result_xys[:,:,0] >= xmin) & (result_xys[:,:,0] < xmax) & (result_xys[:,:,1] >= ymin) & (result_xys[:,:,1] < ymax)

        ######
        ### differentiable ray tracing (Gaussian), GPU実装
        ######
        psftensor = torch.zeros((PSFSIZE, PSFSIZE), dtype=torch.float, device = device)
        intensitymap = torch.zeros((MASKSIZE, MASKSIZE), dtype=torch.float, device = device)
        coords = torch.stack([result_xys[:, :, 0][nanmask], result_xys[:, :, 1][nanmask]], dim=1)
        coords_MASK = torch.stack([x_index[nanmask], y_index[nanmask]], dim=1)
        # print(coords.size())
        intensities = tmp_ray.intensity[:,:,0][nanmask]
        # print(intensities.size())

        # CALCULATION FOR TRANSMISSION
        # including out of fresnel lens, totally reflection in Fresnel lens (not passing through the lens), mask attenuation, and stray lights in Fresnel lens (going away from the target after passing the lens)
        bef = NRAYS * NRAYS # intensities that light source originally had
        aft = torch.nansum(intensities)  # intensities that reaches the target plane
        transmission = aft/bef
        # print(f"Entire transmission rate: {transmission}")
        

        # i_indices = torch.arange(PSFSIZE).view(-1, 1).expand(PSFSIZE, PSFSIZE)  # 行インデックス
        # j_indices = torch.arange(PSFSIZE).expand(PSFSIZE, PSFSIZE)            # 列インデックス
        grid = torch.stack((torch.arange(PSFSIZE).view(-1, 1).expand(PSFSIZE, PSFSIZE), torch.arange(PSFSIZE).expand(PSFSIZE, PSFSIZE))).to(device)
        sigma = torch.tensor((2.0 * pixel_size * pixel_size / 9.0), device = device)

        grid_MASK = torch.stack((torch.arange(MASKSIZE).view(-1, 1).expand(MASKSIZE, MASKSIZE), torch.arange(MASKSIZE).expand(MASKSIZE, MASKSIZE))).to(device)
        sigma_MASK = torch.tensor((2.0 * pixel_size_MASK * pixel_size_MASK  / 9.0), device = device)

        batch_size = 1000
        for i in range(0, coords.size(0), batch_size):
            batch_coords = coords[i:i + batch_size,:]
            batch_coords_MASK = coords_MASK[i:i + batch_size,:]
            batch_intensities = intensities[i:i + batch_size]


            distx = grid[0, :, :].unsqueeze(0) - batch_coords[:, 0].view(-1, 1, 1)  # distxを計算
            disty = grid[1, :, :].unsqueeze(0) - batch_coords[:, 1].view(-1, 1, 1)  # distyを計算

            # PSF計算
            psf = (batch_intensities.view(-1, 1, 1) / torch.sqrt(2 * 3.141592 * sigma) *
                torch.exp(-(distx ** 2 + disty ** 2) / (2 * sigma)))

            # メモリ効率化のために逐次的に加算
            psftensor = psftensor + psf.sum(dim=0)


            distx_MASK = grid_MASK[0, :, :].unsqueeze(0) - batch_coords_MASK[:, 0].view(-1, 1, 1)  # distxを計算
            disty_MASK = grid_MASK[1, :, :].unsqueeze(0) - batch_coords_MASK[:, 1].view(-1, 1, 1)  # distyを計算

            # PSF計算
            imap = (batch_intensities.view(-1, 1, 1) / torch.sqrt(2 * 3.141592 * sigma_MASK) *
                torch.exp(-(distx_MASK ** 2 + disty_MASK ** 2) / (2 * sigma_MASK)))

            # メモリ効率化のために逐次的に加算
            intensitymap = intensitymap + imap.sum(dim=0)

            # メモリクリア
            del batch_coords, batch_intensities, distx, disty, psf, imap
            torch.cuda.empty_cache()

        del coords, intensities, grid
        torch.cuda.empty_cache()

        #print(torch.cuda.memory_allocated())

        psftensor = psftensor / torch.sum(psftensor)
        #intensitymap = normalize_image(intensitymap)

        #######
        ### ここまで
        #######

        save_tensorimage(psftensor, min = torch.min(psftensor), max = torch.max(psftensor), filepath = f"./psfs/optimized_keypoints_{psfloc+1}.png")

        return psftensor, transmission, intensitymap, target_mtf
    
    def optim_mask(self, tau):
        # # 初期値　ランダムマスク
        # ini_mask = torch.randint(0, 2, (MASKSIZE, MASKSIZE), device = device, dtype = torch.float)

        # # # 画像マスク （途中経過から）
        # ini_mask = (Image.open(f"Mask_SA_{tau}.png").convert("L"))
        ini_mask = (Image.open(f"64thru-focusMT_new2.png").convert("L"))
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(MASKSIZE),
            ]
        )
        ini_mask = (normalize_image(transform(ini_mask)).squeeze(0)).to(device)

        # # ピンホール
        # ini_mask = torch.full((MASKSIZE, MASKSIZE), 0.0, device = device, dtype = torch.float)
        # ini_mask[MASKSIZE // 2 - 1, MASKSIZE // 2 - 1] = 1.0
        # ini_mask[MASKSIZE // 2 - 1, MASKSIZE // 2] = 1.0
        # ini_mask[MASKSIZE // 2, MASKSIZE // 2 - 1] = 1.0
        # ini_mask[MASKSIZE // 2, MASKSIZE // 2] = 1.0
        
        # # # マスクの初期値を全開放1
        # ini_mask = torch.full((MASKSIZE, MASKSIZE), 1, device = device, dtype = torch.float)
        
        mask_open = torch.full((MASKSIZE, MASKSIZE), 1, device = device, dtype = torch.float)

        xx, yy = torch.meshgrid(torch.arange(MASKSIZE), torch.arange(MASKSIZE), indexing='ij')
        center = MASKSIZE // 2
        distance = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2).to(device)
        inside = (distance <= (MASKSIZE/2)).float().to(device)
        ini_mask = ini_mask * inside

        distance_norm = distance * inside
        distance_norm = distance_norm / (torch.max(distance_norm))
        # save_tensorimage(distance_norm, min = 0, max= torch.max(distance_norm), filepath = "distance_norm.png")

        x = ini_mask.clone().requires_grad_(True).to(device)
        # optimizer_x = Adam([x], lr=lr)
        # save_tensorimage(ini_mask, min = 0, max = 1,  filepath = f"./optimized_mask/{MASKSIZE}_initial_mask_0.png")
        # torch.autograd.set_detect_anomaly(True)
        # loss_logger = []

        # # 分割用の半径を計算
        # radii = torch.sqrt(torch.arange(0, sep_mask + 1) / sep_mask) * (MASKSIZE//2)
        # print(radii)
        # # 分割領域を決定
        # regions = torch.zeros_like(ini_mask, dtype=torch.int32)
        # for i in range(radii.size()[0] - 1):
        #     regions[(radii[i] <= distance) & (radii[i+1] <= distance)] = i + 1
        # regions.to(device)

        if os.path.exists("./log/ray_behind_Fresnel_" + targetmodel + ".pt"):
            ray_behind_Fresnel = torch.load("./log/ray_behind_Fresnel_" + targetmodel + ".pt", map_location=device)

        else:
            ray_behind_Fresnel = []
            
            for psfloc in range(PSFNUM):
                tmp = self.raytrace_from_object(psfloc = psfloc)
                ray_behind_Fresnel.append(tmp)
            
            print("Tracing to the bottom surface of the Fresnel Lens is successfully finished.")
            torch.save(ray_behind_Fresnel, os.path.join("./log/ray_behind_Fresnel_" + targetmodel + ".pt"))

        firsttime = 1
        breaked = False
        tmp_trans_value = 0
        tmp_mtf_value = 0
        
        psflist = list(range(PSFNUM))
        for epoch in range(EPOCHS):
            for ite in range(PSFNUM):
                # optimizer_x.zero_grad()

                psfloc = psflist[ite]
                
                # # x = torch.sigmoid(6 * (x - 0.5))
                # if ((psfloc % 5) == 0):
                #     if (firsttime == 1):
                #         firsttime += 1
                #     else:
                #         print("Sigmoid")
                #         x = torch.sigmoid(6 * (x - 0.5))

                psftensor, transmission, intensitymap, target_mtf = self.simulatePSF(ray_behind_Fresnel[psfloc], mask = x, psfloc = psfloc)
                ref = self.simulatePSF(ray_behind_Fresnel[psfloc], mask = mask_open, psfloc = psfloc)
                transmission_ref, intensitymap_ref = ref[1], ref[2]
                # print(f'transmission_ref: {transmission_ref.item()}')
                # reporter.report()

                # save_tensorimage(psftensor, min = torch.min(psftensor), max = torch.max(psftensor),  filepath = f"{MASKSIZE}_withmaskPSF.png")

                # # パッチごとに分割
                # psfpatches = (
                #     psftensor.unfold(0, int(PSFSIZE), int(PSFSIZE))  # 高さ方向に100ずつ分割
                #         .unfold(1, int(PSFSIZE), int(PSFSIZE))  # 幅方向に100ずつ分割
                # )
                # psfpatches = psfpatches.contiguous().view(-1, int(PSFSIZE), int(PSFSIZE)) 

                # PSFをフーリエ変換してMTFを取得
                norm_psfpatch = prev_normalize_image(psftensor)
                fft_psfpatch = torch.fft.fft2(norm_psfpatch)  # フーリエ変換
                fft_psfpatch = torch.fft.fftshift(fft_psfpatch)  # 中心を移動
                magnitude_spectrum = torch.abs(fft_psfpatch)

                #　正規化？
                magnitude_spectrum = prev_normalize_image(magnitude_spectrum)

                # for i, spectrum in enumerate(magnitude_spectrums):
                save_tensorimage(magnitude_spectrum, min = 0, max = torch.max(magnitude_spectrum), filepath = f"./MTF/{psfloc+1}.png")
                
                print(ite)
                print(f"MTF: {magnitude_spectrum.mean().item()}")
                print(f"TRANS: {(transmission / transmission_ref).item()}")
                tmp_mtf_value += magnitude_spectrum.mean().item()
                tmp_trans_value += (transmission / transmission_ref).item()
                # # for i, spectrum in enumerate(magnitude_spectrums):
                # #     tmpspectrum = spectrum
                # #     ## 周波数0はロスに含めたくないのでカット
                # #     max_idx = torch.argmax(tmpspectrum)
                # #     # 2次元インデックスに変換 (行, 列)
                # #     max_row, max_col = divmod(max_idx.item(), tmpspectrum.shape[1])
                # #     magnitude_spectrums[i, max_row, max_col] = 0

                # # ロス1：振幅スペクトルの平均値を最大化
                # lossAveMTF = -magnitude_spectrum.mean()

                # # ロス2：振幅スペクトルの分散を最小化
                # lossVarMTF = magnitude_spectrum.var()

                # # ロス：TragetMTFとの誤差
                # ### 単純な MSE 
                # # lossMTF = torch.mean((target_mtf - magnitude_spectrum) * (target_mtf - magnitude_spectrum))
                # ### Target MTF に足りてない成分のみをロスに使用
                # lacked_freq = torch.clamp(target_mtf - magnitude_spectrum, min = 0)
                # lossMTF = torch.mean(lacked_freq)

                # # ロス3：光線透過率を最大化
                # # lossTRANS = ((transmission / transmission_ref) - 0.5)**2
                # TRANS = (torch.mean(intensitymap) / torch.mean(intensitymap_ref))
                # lossTRANS = -(1.0/(1+torch.exp(-100*(TRANS-tau))))
                # # lossTRANS = -(TRANS)**(0.5)
                # # lossTRANS = ((torch.mean(intensitymap) / torch.mean(intensitymap_ref)) - tau)**2
                # # print((transmission / transmission_ref))
                # # print((torch.sum(intensitymap) / torch.sum(intensitymap_ref)))
                # # lossTRANS = -intensitymap.mean()
                
                # lossMean = -intensitymap.mean()
                # save_tensorimage(intensitymap, min = torch.min(intensitymap), max = torch.max(intensitymap), filepath = f"./intensitymap/intensitymap_{psfloc}.png")
                # # intensitymap = torch.where(intensitymap >= 0.1, intensitymap, float('nan'))
                # # nanmean = torch.nanmean(intensitymap)

                # # lossNanmean = -nanmean
                # # diff = torch.where(torch.isnan(intensitymap), torch.zeros_like(intensitymap), intensitymap - nanmean)
                # # lossNanStd = torch.sqrt (torch.sum(diff**2) / (torch.isnan(intensitymap)).sum())

                # # ロス4：intensitymap の同心円状領域間での絶対誤差
                # intensitymap = intensitymap / torch.max(intensitymap)
                # lossShadow = torch.tensor(0, device = device)
                # sums = torch.zeros(sep_mask, device = device)
                # for mask_i in range(sep_mask):
                #     sums[mask_i] = intensitymap[regions == mask_i].sum()
                # for i, s in enumerate(sums):
                #     lossShadow = lossShadow + torch.pow(s - (torch.sum(sums)/ sep_mask), 2)
                
                # # # ロス5：intensitymap の 同心円状領域間での絶対誤差
                # # lossVarSS = sums.var()
                
                # # ロス6：intensity map の極座標表現における theta 方向の高周波成分 OR 分散
                # grid = polar_grid().to(device)
                # polar_image = F.grid_sample(intensitymap.unsqueeze(0).unsqueeze(0), grid, align_corners=True)
                # # lossMapPolar = high_freq_loss(polar_image)
                # polar_image = polar_image.squeeze(0).squeeze(0)
                # save_tensorimage(polar_image, min = torch.min(polar_image), max = torch.max(polar_image), filepath = f"./polar/intensity_polar_{psfloc}.png")
                # lossMapPolar = torch.mean(torch.var(polar_image, dim = 0, unbiased = False)) + torch.mean(torch.var(polar_image, dim = 1, unbiased = False))

                # # # ロス7: Moran's I 空間的な散らばり
                # # loss_moran = morans_I(intensitymap)
                # # print(f"Moran: {loss_moran.item()}")
                
                # # ロス8: intensitymap の中心から離れた画素ほど高い評価
                # lossSS = -torch.mean(intensitymap * distance_norm)
                # # print(lossSS.item())

                # # if (TRANS < 0.1 or TRANS > 0.9):
                # #     breaked = True
                # #     break
                # loss = lossAveMTF + 0.01 *lossTRANS #+ 0.1 * lossSS # + 0.0001 * lossMean
                # loss.backward(retain_graph=True)

                # loss_logger.append(loss.item())
                # save_loss_logger_and_graph("", loss_logger)
                # optimizer_x.step()
                # x.data.clamp_(0, 1)
                # x.where(~torch.isnan(inside), torch.tensor(0, device = device))

                # print(f"Epoch: {epoch+1}, Iter: {ite+1}, PSFLOC: {psfloc}, Loss: {loss.item():.4f}, AveMTF: {lossAveMTF.item():.4f}, TRANS: {lossTRANS.item():.4f}, SS: {lossSS.item():.4f}")#, LossVarSS: {lossVarSS.item():.4f},
                # del psftensor, norm_psfpatch, fft_psfpatch, magnitude_spectrum, transmission, lossAveMTF, lossVarMTF, lossTRANS, loss 
                # torch.cuda.empty_cache()

                # ### for now
                # save_tensorimage(x, min = 0, max = 1,  filepath = f"./optimized_mask/{MASKSIZE}_{tau}_forNow.png")
            
            # ### for each epoch
            # save_tensorimage(x, min = 0, max = 1,  filepath = f"./optimized_mask/{MASKSIZE}_initial_mask_{epoch+1}.png")
            # if breaked:
            #     break
        transmittances.append(tmp_trans_value/PSFNUM)
        ave_of_mtf.append(tmp_mtf_value/PSFNUM)

        # # Save the optimized image
        # save_tensorimage(x, min = 0, max = 1, filepath =  f"./Result_optimized_mask_{MASKSIZE}.png")

#tau_list = range(1, 33)
tau_list = [0.1]
transmittances = []
ave_of_mtf= []
# tau_list = [0.021, 0.022, 0.023,0.024,0.025,0.026,0.027,0.028,0.029]
for tau in tau_list:
    print(tau)
    lsys = LensSystem(['data/fresnel.csv'])
    lsys.optim_mask(tau)

import csv
result_list = [transmittances, ave_of_mtf]
with open("MTFvsLT_ours.csv", 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerows(result_list)