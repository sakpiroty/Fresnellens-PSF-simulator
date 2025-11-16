## 全平面をオブジェクトとして格納
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# refractiveindex = 1.492
# diameter = 500.0 #(mm)
# pitch = 0.5 #(mm)
# split_number = int(diameter / pitch / 2)
# image_z = -500.0 #(mm)
# object_z = 500.0 #(mm)
# pixel_size = 0.625 #(mm)
# IMAGESIZE = 256 #(pix)
# MASKSIZE = 128  #(pix)
# pixel_size_MASK = diameter/MASKSIZE #(mm)
# NRAYS = 1000
# PSFSIZE = 320
# PSFNUM = 5
# PSFCENTER = int((PSFNUM + 1)/2)
# MAGNIFY = object_z / (-image_z)
# NSTEPS = 500
# lr = 0.1


refractiveindex = 1.492
diameter = 500.0 #(mm)
pitch = 0.5 #(mm)
split_number = int(diameter / pitch / 2)
image_z = -500.0 #(mm)
object_z = 500.0 #(mm)
pixel_size = 4e-1 #(mm)
IMAGESIZE = 400 #(pix)
MASKSIZE = 128  #(pix)
pixel_size_MASK = diameter/MASKSIZE #(mm)
NRAYS = 1000
PSFSIZE = 500
PSFNUM = 5
PSFCENTER = int((PSFNUM + 1)/2)
MAGNIFY = object_z / (-image_z)
NSTEPS = 500
lr = 0.1

def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)


def save_tensorimage(tensor, min, max, filepath):
    # テンソルが2次元であることを確認
    if tensor.ndim != 2:
        raise ValueError("Dimension of the saved tensor must be 2.")
    
    # 値を0〜255に正規化
    tensor_min = min
    tensor_max = max
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * 255.0
    
    # テンソルをuint8型に変換
    tensor_uint8 = normalized_tensor.to(torch.uint8)
    
    # PIL.Imageに変換
    Image.fromarray(tensor_uint8.cpu().numpy()).save(filepath)

def save_loss_logger_and_graph(log_dir, loss_logger):
    # loss履歴情報を保管しつつ、グラフにして画像としても書き出す
    torch.save(loss_logger, os.path.join(log_dir, "loss_logger.pt"))
    fig, ax = plt.subplots(1,1)
    epoch = range(len(loss_logger))
    ax.plot(epoch, loss_logger, label="train_loss")
    ax.set_ylim(0, loss_logger[-1]*5)
    ax.legend()
    fig.savefig(os.path.join(log_dir, "loss_history.jpg"))
    plt.clf()
    plt.close()

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

# NaNの割合を計算
def ratio_of_nan(x: torch.tensor):
    nan_mask = torch.isnan(x)
    nan_count = nan_mask.sum().item()
    total_elements = x.numel()
    nan_ratio = nan_count / total_elements
    return nan_ratio

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
    def raytrace_from_object(self, ray_in: Ray, mask: torch.tensor) -> torch.tensor:
        ray = ray_in
        rays = [ray]
        lens = self.lenses[0]
        N_loop = 20

        ####
        ## 一回目のフレネルレンズ入射を想定する．
        ####
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

        print(f'Out of Fresnel: {torch.sum(finished)/finished.numel()}')
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

            rays.append(ray)

            print(f'LOOP {loop+1}: {torch.sum(finished)/finished.numel()}')
            finishedratio.append((torch.sum(finished)/finished.numel()).cpu())
        
        #plt.plot(finishedratio)
        #plt.savefig("finishedratio.png")

        ##　ループ回数最大にしてもfinishedにならなかったものはNANとする．
        ray.origin = torch.where(finished, ray.origin ,float('nan'))
        ray.direction = torch.where(finished, ray.direction ,float('nan'))

        # 空間マスク
        origin_inMASKpixel = ray.origin / pixel_size_MASK
        x_index = torch.where(torch.isnan(origin_inMASKpixel[:,:,0]), -1,((origin_inMASKpixel[:,:,0] + MASKSIZE/2 - 1) + 0.5).to(torch.long))
        y_index = torch.where(torch.isnan(origin_inMASKpixel[:,:,0] ), -1,((origin_inMASKpixel[:,:,1] + MASKSIZE/2 - 1) + 0.5).to(torch.long))
        # 空間マスクの値に応じて光の強度が減衰
        tmp_intensity = ray.intensity[:,:,0].clone()
        black_base, white_base = 0.1, 0.9
        mask = black_base + mask * (white_base - black_base)
        ray.intensity[:,:,0] = torch.where(x_index == -1, torch.tensor(0.0, device = device), tmp_intensity * mask[x_index, y_index])
        transmission = (ray.intensity[:,:,0].sum())/(tmp_intensity.sum()) # 光線の減衰比率
        print(f"transmission rate: {transmission}")

        # 結像位置まで計算
        p, n, t = lens.intersect_plane(ray, object_z, 0, 500)
        ray.origin = p
        rays.append(ray)

        return rays

    def simulatePSF(self, mask: torch.tensor, n_rays = NRAYS) -> torch.tensor:
        s_time = time.time()

        psftensor = torch.zeros((PSFSIZE, PSFSIZE), dtype=torch.float, device = device)
        
        rayin_origin = torch.zeros((n_rays*PSFNUM, n_rays*PSFNUM, 3), device = device)
        rayin_direction = torch.zeros((n_rays*PSFNUM, n_rays*PSFNUM, 3), device = device)
        rayin_roi = torch.zeros((n_rays*PSFNUM, n_rays*PSFNUM, 1), device = device)
        rayin_intensity = torch.zeros((n_rays*PSFNUM, n_rays*PSFNUM, 1), device = device)

        for PSFX in range(1, PSFNUM+1):
            for PSFY in range(1, PSFNUM+1):
                # ベクトル化された実装
                x, y = torch.meshgrid(torch.arange(n_rays, device = device), torch.arange(n_rays, device = device), indexing = 'ij') # 全てのx, y座標を生成
                x = (x - n_rays / 2) / n_rays
                y = (y - n_rays / 2) / n_rays
                z = torch.sqrt(torch.tensor(1, device = device)).expand_as(x)
                ray_roi = torch.ones_like(x).unsqueeze(dim = 2)
                ray_intensity = torch.ones_like(x).unsqueeze(dim = 2)

                ray_direction = torch.stack([x, y, z], dim = 0)
                ray_direction = ray_direction / torch.norm(ray_direction, dim=0, keepdim=True)
                ray_direction = torch.permute(ray_direction, (1,2,0))
                ray_origin = torch.tensor([(PSFX - PSFCENTER)/2*IMAGESIZE/2*pixel_size,(PSFY - PSFCENTER)/2*IMAGESIZE/2*pixel_size, image_z], device = device).expand_as(ray_direction)

                start_row, end_row = (PSFX - 1) * (n_rays), (PSFX) * (n_rays)
                start_col, end_col = (PSFY - 1) * (n_rays), (PSFY) * (n_rays)
                rayin_origin[start_row:end_row, start_col:end_col, :] = ray_origin
                rayin_direction[start_row:end_row, start_col:end_col, :] = ray_direction
                rayin_roi[start_row:end_row, start_col:end_col, :] = ray_roi
                rayin_intensity[start_row:end_row, start_col:end_col, :] = ray_intensity

        ray_in = Ray(rayin_origin, rayin_direction, rayin_roi, rayin_intensity)

        # レイトレース計算
        result_rays = self.raytrace_from_object(ray_in, mask)
        result_xys = result_rays[-1].origin/pixel_size/MAGNIFY
        result_intensity = result_rays[-1].intensity[:,:,0]

        result_xys[:,:,0] = result_xys[:,:,0] + (PSFSIZE/2 - 1)
        result_xys[:,:,1] = result_xys[:,:,1] + (PSFSIZE/2 - 1)
        # nan mask (迷光など)
        nanmask = ~torch.isnan(result_xys[:,:,0])
        # 有効なx座標とy座標を取得して四捨五入
        x_coords = (result_xys[:,:,0][nanmask] + 0.5).to(torch.long)
        y_coords = (result_xys[:,:,1][nanmask] + 0.5).to(torch.long)
        intensities = result_intensity[nanmask]

        # PSF画像サイズ外は排除
        valid = (x_coords > 0) & (x_coords < PSFSIZE) & (y_coords > 0) & (y_coords < PSFSIZE)
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        intensities = intensities[valid]

        psftensor.index_put_((x_coords, y_coords), torch.ones_like(x_coords) * intensities, accumulate=True)

        psftensor = psftensor / torch.sum(psftensor)

        return psftensor

    def optim_mask(self):
        ## ランダム画像の場合
        #ini_mask = torch.randint(0, 2, (MASKSIZE, MASKSIZE), device = device, dtype = torch.float)

        ini_mask = (Image.open(f"MTF+TRANS+SS.png").convert("L"))
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        ini_mask = (normalize_image(transform(ini_mask)).squeeze(0)).to(device)

        save_tensorimage(ini_mask, min = 0, max = 1, filepath =  f"{MASKSIZE}_initial_mask.png")

        psftensor = self.simulatePSF(ini_mask)
        save_tensorimage(psftensor, min = torch.min(psftensor), max = torch.max(psftensor), filepath =  f"{MASKSIZE}_withmaskPSF.png")

        # パッチごとに分割
        psfpatches = (
            psftensor.unfold(0, int(PSFSIZE/PSFNUM), int(PSFSIZE/PSFNUM))  # 高さ方向に100ずつ分割
                .unfold(1, int(PSFSIZE/PSFNUM), int(PSFSIZE/PSFNUM))  # 幅方向に100ずつ分割
        )
        psfpatches = psfpatches.contiguous().view(-1, int(PSFSIZE/PSFNUM), int(PSFSIZE/PSFNUM)) 

        for i, psf in enumerate(psfpatches):
            save_tensorimage(psf, min = torch.min(psf), max = torch.max(psf),  filepath = f"./psfs/optimized_{i+1}.png")

lsys = LensSystem(['data/fresnel.csv'])
lsys.optim_mask()