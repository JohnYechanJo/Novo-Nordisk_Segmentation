import os
from PIL import Image
import numpy as np

def adjust_red_channel(image, factor):
    # 将图像转换为 numpy 数组
    img_array = np.array(image)
    # 对 R 通道进行调整
    img_array[:, :, 0] = img_array[:, :, 0] * factor
    # 确保值在 0-255 之间
    img_array = np.clip(img_array, 0, 255)
    # 将 numpy 数组转换回图像
    return Image.fromarray(img_array.astype('uint8'))

def process_images(src_folder, dest_folder, factor):
    # 创建目标文件夹，如果不存在
    os.makedirs(dest_folder, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_folder, filename)
            
            # 打开图像
            with Image.open(src_file) as img:
                # 调整 R 通道
                adjusted_img = adjust_red_channel(img, factor)
                # 保存调整后的图像
                adjusted_img.save(dest_file)
                print(f"Processed and saved: {dest_file}")

# 源文件夹
src_folder = r"D:\X\testfield\image"
# 目标文件夹
dest_folder = os.path.join(src_folder, "adjusted_images")

# 调整 R 通道的因子（40% 下降）
factor = 0.6

# 处理图像
process_images(src_folder, dest_folder, factor)