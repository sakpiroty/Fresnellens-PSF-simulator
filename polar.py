# import torch
# import torch.fft
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def normalize_image(img):
#     img_min = img.min()
#     img_max = img.max()
#     return (img - img_min) / (img_max - img_min)

# def cartesian_to_polar(img):
#     """ 画像を極座標変換する """
#     h, w = img.shape
#     cx, cy = w // 2, h // 2  # 画像中心

#     # 極座標の最大半径（画像の対角線の半分）
#     max_radius = 64

#     # 極座標変換
#     polar_img = cv2.warpPolar(img, (max_radius, 360), (cx, cy), max_radius, cv2.WARP_POLAR_LINEAR)
    
#     return polar_img

# def compute_fft(img):
#     """ 画像のフーリエ変換を計算し、対数スケールで返す """
#     img_tensor = normalize_image(torch.tensor(img, dtype=torch.float32))
    
#     # 2Dフーリエ変換
#     fft_result = torch.fft.fft2(img_tensor)
    
#     # 中心をシフト
#     fft_shifted = torch.fft.fftshift(fft_result)
    
#     # 振幅スペクトルを対数スケールで計算
#     magnitude = torch.log1p(torch.abs(fft_shifted))
    
#     return magnitude.numpy()

# def save_fft_image(fft_img, filename):
#     """ フーリエ変換の結果を画像として保存 """
#     plt.figure(figsize=(6,6))
#     plt.imshow(fft_img, cmap='gray')
#     plt.axis('off')
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # 画像の読み込み (グレースケール)
# image_path = "intensitymap_20.png"  # 任意の画像を指定
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # 1. 元の画像のフーリエ変換
# fft_original = compute_fft(img)
# save_fft_image(fft_original, "fft_original.png")

# # 2. 極座標変換
# polar_img = cartesian_to_polar(img)
# cv2.imwrite("polar_image.png", polar_img)

# # 3. 極座標変換後の画像のフーリエ変換
# fft_polar = compute_fft(polar_img)
# save_fft_image(fft_polar, "fft_polar.png")

# print("フーリエ変換画像を保存しました！")



import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
# 画像サイズと極座標サイズ
H, W = 128, 128  # 元画像サイズ
R, Theta = 64, 360  # 極座標変換後のサイズ

def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)

# サンプル画像 (ランダムノイズ)
ini_mask = (Image.open(f"intensitymap_20.png").convert("L"))
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
image = (transform(ini_mask).unsqueeze(0)).to("cuda:0")
image = image.clone().detach().requires_grad_(True)

# image = torch.rand(1, 1, H, W, requires_grad=True)  # 学習可能な画像

# 極座標グリッドを作成
def polar_grid(H, W, R, Theta):
    """ 極座標のサンプリンググリッドを作成 """
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    r = torch.linspace(0, 1, R).view(-1, 1)  # 半径方向
    theta = torch.linspace(0, 2*np.pi, Theta).view(1, -1)  # 角度方向

    X = r * torch.cos(theta)
    Y = r * torch.sin(theta)

    X = X.view(1, R, Theta)
    Y = Y.view(1, R, Theta)

    return torch.stack([X, Y], dim=-1)  # (1, R, Theta, 2)

grid = polar_grid(H, W, R, Theta).to(image.device)

# 画像を極座標に変換
polar_image = F.grid_sample(image, grid, align_corners=True)

# 高周波成分を強調する損失関数
def high_freq_loss(polar_image):
    """ 各rごとのθ方向の高周波成分を強調する損失 """
    fft = torch.fft.fft(polar_image, dim=-1)  # θ方向のFFT
    freq = torch.fft.fftfreq(Theta, d=1/Theta).to(polar_image.device)  # 周波数軸

    high_freq_mask = (freq > 0.1 * freq.max()).float()  # 高周波部分をマスク
    loss = torch.sum(torch.abs(fft * high_freq_mask))  # 高周波成分の強化

    return -loss  # 高周波成分を最大化

# 最適化
optimizer = torch.optim.Adam([image], lr=0.01)

for i in range(500):
    optimizer.zero_grad()
    polar_image = F.grid_sample(image, grid, align_corners=True)
    loss = high_freq_loss(polar_image)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}: Loss = {loss.item()}")

# 結果を表示
plt.subplot(1,2,1)
plt.imshow(image.detach().squeeze().cpu(), cmap='gray')
plt.title("Optimized Image")

plt.subplot(1,2,2)
plt.imshow(polar_image.detach().squeeze().cpu(), cmap='gray')
plt.title("Polar Image")

plt.show()
plt.savefig("test.png")


# from PIL import Image
# import torch 


# # img = (Image.open(f"64_initial_mask.png").convert("L"))
# # img = img.resize((2048*3, 2048*3), Image.NEAREST)
# # img.save('resized.png')



# a = torch.rand(100, 100)  # 100×100のランダムなテンソル
# b = torch.tensor(1.0)     # スカラー値のテンソル

# c = a - b

# print(a)
# print(b)
# print(c)

# import torch

# # Define the secondary moment of inertia loss
# def moment_of_inertia_loss(image):
#     # Get image dimensions
#     H, W = image.shape
    
#     # Create meshgrid for indices
#     y_coords, x_coords = torch.meshgrid(
#         torch.arange(H, dtype=torch.float32, device=image.device),
#         torch.arange(W, dtype=torch.float32, device=image.device),
#         indexing="ij"
#     )
    
#     # Compute the total mass (sum of pixel intensities)
#     total_mass = image.sum()
    
#     # Avoid division by zero
#     if total_mass == 0:
#         return torch.tensor(0.0, device=image.device)
    
#     # Compute centers of mass
#     center_x = W/2
#     center_y = H/2

#     # Compute the moment of inertia around x and y axes
#     I_x = ((y_coords - center_y) ** 2 * image).sum()
#     I_y = ((x_coords - center_x) ** 2 * image).sum()
    
#     # Total loss is the sum of both moments of inertia
#     return I_x + I_y

# # Example optimization
# def optimize_image():
#     # Initialize a random grayscale image (0~1) with requires_grad=True
#     H, W = 64, 64
#     image = torch.rand((H, W), requires_grad=True)
    
#     # Optimizer
#     optimizer = torch.optim.Adam([image], lr=0.1)
    
#     # Optimization loop
#     for step in range(100):
#         optimizer.zero_grad()
        
#         # Compute the loss
#         loss = moment_of_inertia_loss(image)
        
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
        
#         # Clamp image to be in [0, 1] range
#         image.data.clamp_(0, 1)
        
#         # Print loss
#         if step % 10 == 0:
#             print(f"Step {step}, Loss: {loss.item()}")
    
#     return image

# # Run optimization
# optimized_image = optimize_image()

# # import torch
# # import torch.nn as nn
# # from PIL import Image

# # def save_tensorimage(tensor, min, max, filepath):
# #     # テンソルが2次元であることを確認
# #     if tensor.ndim != 2:
# #         raise ValueError("Dimension of the saved tensor must be 2.")
    
# #     # 値を0〜255に正規化
# #     tensor_min = min
# #     tensor_max = max
# #     normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * 255.0
    
# #     # テンソルをuint8型に変換
# #     tensor_uint8 = normalized_tensor.to(torch.uint8)
    
# #     # PIL.Imageに変換
# #     Image.fromarray(tensor_uint8.cpu().numpy()).save(filepath)

# # class SecondMomentLoss(nn.Module):
# #     def __init__(self, alpha=1.0, beta=1.0):
# #         super(SecondMomentLoss, self).__init__()
# #         self.alpha = alpha
# #         self.beta = beta

# #     def forward(self, x):
# #         # x: (B, 1, H, W) グレースケール画像 (バッチ, チャンネル, 高さ, 幅)
# #         H, W = x.size()
# #         device = x.device
        
# #         # 座標グリッドを作成
# #         y_coords, x_coords = torch.meshgrid(
# #             torch.arange(H, dtype=torch.float32, device=device),
# #             torch.arange(W, dtype=torch.float32, device=device),
# #             indexing='ij'
# #         )

# #         # 重心計算
# #         x_sum = torch.sum(x, dim=(0,1), keepdim=True)
# #         x_c = W/2
# #         y_c = H/2

# #         # 重心との差分計算
# #         x_diff = x_coords - x_c
# #         y_diff = y_coords - y_c

# #         # 二次モーメント計算
# #         M_xx = torch.sum(x_diff**2 * x, dim=(0,1))
# #         M_yy = torch.sum(y_diff**2 * x, dim=(0,1))
# #         M_xy = torch.sum(x_diff * y_diff * x, dim=(0,1))

# #         # ロス計算
# #         loss = -self.alpha * (M_xx + M_yy).mean() + self.beta * torch.abs(M_xy).mean()
# #         return loss

# # import matplotlib.pyplot as plt
# # import os

# # # 最適化後のマスクを保存するフォルダ
# # output_dir = "./moment/"
# # #os.makedirs(output_dir, exist_ok=True)

# # # 最適化ループを含むコードを実行
# # B, C, H, W = 1, 1, 64, 64  # バッチサイズ1, グレースケール, 64x64画像
# # mask = torch.full((H, W), 0.5, dtype = torch.float, requires_grad = True) # ランダム初期化
# # optimized_mask = mask.detach().squeeze().cpu().numpy()
# # # 画像として表示
# # plt.figure(figsize=(6, 6))
# # plt.imshow(optimized_mask, cmap="gray")
# # plt.colorbar()
# # plt.title("Optimized Mask")
# # plt.axis("off")

# # # 画像を保存
# # output_path = os.path.join(output_dir, f"optimized_mask_0.png")
# # plt.savefig(output_path)
# # plt.show()
# # optimizer = torch.optim.Adam([mask], lr=0.01)
# # loss_fn = SecondMomentLoss(alpha=1.0, beta=1.0)

# # # 最適化ループ
# # for step in range(10):
# #     optimizer.zero_grad()
# #     loss = loss_fn(mask)
# #     loss.backward()
# #     optimizer.step()
# #     mask.data.clamp_(0,1)

# #     # 途中結果の確認
# #     if step % 1 == 0:
# #         print(f"Step {step}, Loss: {loss.item():.4f}")
# #         # 最適化結果を取得し正規化
# #         save_tensorimage(mask, min = 0, max = 1,  filepath = f"./moment/optimized_mask_{step}.png")

# # output_path