import cv2
import numpy as np

# 画像を読み込み
a = cv2.imread('64thru-focusMT_new.png')

# 外縁部を黒くする幅
border = 20

# 四辺を黒に
a[:border, :, :] = 0           # 上
a[-border:, :, :] = 0          # 下
a[:, :border, :] = 0           # 左
a[:, -border:, :] = 0          # 右

# 保存または表示
cv2.imwrite('64thru-focusMT_new2.png', a)


# import os
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from typing import List, Tuple, Optional
# import numpy as np
# import time
# from PIL import Image
# from torch.optim import Adam
# from pytorch_memlab import MemReporter
# import copy
# import torch.nn as nn
# import torch.nn.functional as F

# refractiveindex = 1.492
# diameter = 500.0 #(mm)
# pitch = 0.5 #(mm)
# split_number = int(diameter / pitch / 2)
# image_z = -500.0 #(mm)
# pixel_size = 0.3125 #(mm)
# IMAGESIZE = 512 #(pix)
# MASKSIZE = 64 #(pix)
# pixel_size_MASK = diameter/MASKSIZE #(mm)
# NRAYS = 1000

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# def save_tensorimage(tensor, min, max, filepath):
#     # テンソルが2次元であることを確認
#     if tensor.ndim != 2:
#         raise ValueError("Dimension of the saved tensor must be 2.")
    
#     # 値を0〜255に正規化
#     tensor_min = min
#     tensor_max = max
#     normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * 255.0
    
#     # テンソルをuint8型に変換
#     tensor_uint8 = normalized_tensor.to(torch.uint8)
    
#     # PIL.Imageに変換
#     Image.fromarray(tensor_uint8.cpu().numpy()).save(filepath)

# def normalize_image(img):
#     img_min = img.min()
#     img_max = img.max()
#     return (img - img_min) / (img_max - img_min)

#######
## SIGMOID
#######
# filename = "64thru-focusMT006SS"

# ini_mask = (Image.open(filename + ".png").convert("L"))
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Resize(MASKSIZE),
#     ]
# )
# ini_mask = (normalize_image(transform(ini_mask)).squeeze(0)).to(device)

# ini_mask = torch.sigmoid(100000 * (ini_mask - 0.5))
# save_tensorimage(ini_mask, min = torch.min(ini_mask), max = torch.max(ini_mask), filepath = filename + "_Sigm.png")



#####
## Moran's I
#####
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def compute_morans_I(image):
#     """
#     image: [1, 1, H, W] (grayscale image)
#     """
#     N, C, H, W = image.shape
#     x = image.view(-1)
#     x_mean = x.mean()
#     x_diff = x - x_mean

#     # Spatial weights: 4-neighborhood
#     kernel = torch.tensor([
#         [0.0, 1.0, 0.0],
#         [1.0, 0.0, 1.0],
#         [0.0, 1.0, 0.0]
#     ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

#     weights_sum = kernel.sum()

#     # Local sum of neighbors * (x_i - x̄)
#     neighbor_sum = F.conv2d(image, kernel, padding=1)
#     numerator = ((image - x_mean) * (neighbor_sum - x_mean)).sum()

#     denominator = (x_diff ** 2).sum()

#     I = (H * W / weights_sum) * (numerator / denominator)
#     return I

# # --- Optimization ---

# device = "cuda" if torch.cuda.is_available() else "cpu"
# H, W = 64, 64

# # Initialize random image
# image = torch.randn(1, 1, H, W, device=device, requires_grad=True)

# optimizer = torch.optim.Adam([image], lr=0.05)

# for step in range(4000):
#     optimizer.zero_grad()
#     I = compute_morans_I(image)
#     loss = I  # minimize Moran's I
#     loss.backward()
#     optimizer.step()

#     if step % 50 == 0:
#         print(f"Step {step}: Moran's I = {I.item():.5f}")

# # Show resulting image
# result_img = image.detach().cpu().squeeze().clamp(0, 1)
# plt.imshow(result_img, cmap='gray')
# plt.title("Resulting Image with Low Moran's I")
# plt.colorbar()
# plt.savefig("test.png")


# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# # 1. Cartesian -> Polar
# def cartesian_to_polar(img, N_r=256, N_theta=360):
#     """
#     img: (1, 1, H, W) のテンソル, H=W=513
#     N_r: 半径サンプル数
#     N_theta: 角度サンプル数
#     return: (1, 1, N_r, N_theta) の Polar 表現
#     """
#     B, C, H, W = img.shape
#     # 中心を (0,0) に、座標を [-1,1] へ正規化
#     yy, xx = torch.meshgrid(
#         torch.linspace(-1, 1, H, device=img.device),
#         torch.linspace(-1, 1, W, device=img.device),
#         indexing='ij'
#     )
#     # 半径マップ, 角度マップ
#     r = torch.sqrt(xx**2 + yy**2)
#     theta = torch.atan2(yy, xx)  # [-π, π]
#     # 角度を [0, 2π]
#     theta = theta % (2 * torch.pi)
#     # r を [0,1] にクランプ
#     r = torch.clamp(r, 0.0, 1.0)

#     # サンプル座標のグリッド
#     r_lin = torch.linspace(0, 1, N_r, device=img.device)
#     th_lin = torch.linspace(0, 2*torch.pi, N_theta, device=img.device)
#     rr, tt = torch.meshgrid(r_lin, th_lin, indexing='ij')  # (N_r, N_theta)

#     # 逆マッピング: Polar -> Cartesian 正規化座標
#     x_s = rr * torch.cos(tt)
#     y_s = rr * torch.sin(tt)
#     # grid_sample 用に (N_r,N_theta,2) 格納、順序は (x,y)
#     grid = torch.stack([x_s, y_s], dim=-1)[None]  # (1, N_r, N_theta, 2)

#     # サンプリング
#     polar = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#     return polar  # (1,1,N_r,N_theta)


# # 2. Polar -> Cartesian
# def polar_to_cartesian(polar, H=513, W=513):
#     """
#     polar: (1,1,N_r,N_theta)
#     H,W: 出力 Cartesian 解像度
#     return: (1,1,H,W)
#     """
#     B, C, N_r, N_theta = polar.shape

#     # Cartesian 出力座標グリッドを作成 (正規化済み[-1,1])
#     yy, xx = torch.meshgrid(
#         torch.linspace(-1, 1, H, device=polar.device),
#         torch.linspace(-1, 1, W, device=polar.device),
#         indexing='ij'
#     )
#     # 半径と角度（[0,1] と [0,2π]）
#     r = torch.sqrt(xx**2 + yy**2)
#     theta = torch.atan2(yy, xx) % (2 * torch.pi)

#     # 正規化 r_th 座標
#     r_norm = r  # 既に [-1,1]→[0,1]
#     th_norm = theta / (2 * torch.pi)  # [0,1]

#     # grid_sample 用に極座標グリッドを作成
#     # polar の grid_sample は入力が (1,1,N_r,N_theta)
#     #  grid_sample の Grid は (-1,1) 範囲で指定 → まず [0,1]→[-1,1]
#     r_s = 2 * r_norm - 1
#     th_s = 2 * th_norm - 1

#     grid = torch.stack([th_s, r_s], dim=-1)[None]  # (1,H,W,2) 注意：x=theta, y=radius
#     # 注意: polar の shape は (1,1,N_r,N_theta) なので、x 軸が θ 軸、y 軸が r 軸
#     img_rec = F.grid_sample(polar, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#     return img_rec  # (1,1,H,W)


# # --- 簡単な最適化例 ---
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 513×513 のターゲット画像を用意（ここでは同心円パターン）
#     H = W = 513
#     yy, xx = torch.meshgrid(
#         torch.linspace(-1,1,H,device=device),
#         torch.linspace(-1,1,W,device=device),
#         indexing='ij'
#     )
#     target = torch.sin(10 * torch.sqrt(xx**2 + yy**2))[None,None]  # (1,1,H,W)

#     # 初期の Polar パラメータを学習可能に用意
#     N_r, N_theta = 128, 360
#     polar_param = torch.randn(1,1,N_r,N_theta, device=device, requires_grad=True)

#     optimizer = torch.optim.Adam([polar_param], lr=5e-2)

#     for it in range(200):
#         optimizer.zero_grad()
#         img_rec = polar_to_cartesian(polar_param, H, W)
#         loss = F.mse_loss(img_rec, target)
#         loss.backward()
#         optimizer.step()
#         if it % 20 == 0:
#             print(f"iter {it:03d}, loss = {loss.item():.6f}")

#     # 結果を可視化
#     import matplotlib.pyplot as plt
#     with torch.no_grad():
#         rec = img_rec.cpu().squeeze().numpy()
#         tgt = target.cpu().squeeze().numpy()

#     fig, axes = plt.subplots(1,2,figsize=(8,4))
#     axes[0].imshow(tgt, cmap='gray')
#     axes[0].set_title("Target")
#     axes[0].axis('off')
#     axes[1].imshow(rec, cmap='gray')
#     axes[1].set_title("Reconstructed")
#     axes[1].axis('off')
#     plt.savefig("test.png")
