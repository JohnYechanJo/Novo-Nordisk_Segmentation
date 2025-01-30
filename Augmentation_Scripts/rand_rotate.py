import os
import random
from PIL import Image

def random_rotate_image(image):
    # 随机选择旋转角度（0 到 360 度之间的任意角度）
    angle = random.uniform(0, 360)
    return image.rotate(angle, expand=True)

def process_images(src_folder):
    # 遍历源文件夹及其子文件夹中的所有文件
    for root, dirs, files in os.walk(src_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png')):
                src_file = os.path.join(root, filename)
                
                # 打开图像
                with Image.open(src_file) as img:
                    # 随机旋转图像
                    rotated_img = random_rotate_image(img)
                    # 保存旋转后的图像（覆盖原文件）
                    rotated_img.save(src_file)
                    print(f"Processed and saved: {src_file}")

# 示例用法
src_folder = r"D:\X\testfield\image"  # 替换为你的源文件路径
process_images(src_folder)