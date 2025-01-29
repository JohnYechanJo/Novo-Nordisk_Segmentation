import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_red_channel_distribution(folder_path):
    red_values = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, filename)
                with Image.open(file_path) as img:
                    img_array = np.array(img)
                    red_channel = img_array[:, :, 0].flatten()
                    red_values.extend(red_channel)

    return red_values

def plot_red_channel_distribution(red_values):
    plt.hist(red_values, bins=256, color='red', alpha=0.7)
    plt.title('Red Channel Distribution')
    plt.xlabel('Red Value')
    plt.ylabel('Frequency')
    plt.show()

# 指定目录路径
folder_path = r"D:\X\testfield\adjusted_images"

# 获取R值分布
red_values = get_red_channel_distribution(folder_path)

# 绘制柱状图
plot_red_channel_distribution(red_values)