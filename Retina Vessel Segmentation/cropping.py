import os
from PIL import Image

def resize_images(folder_path, size=(512, 512)):
    # 遍历文件夹及其子文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 检查文件是否为图片（可以根据需要扩展文件类型）
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 构建文件的完整路径
                file_path = os.path.join(root, filename)
                try:
                    # 打开图片
                    with Image.open(file_path) as img:
                        # 调整图片大小
                        img = img.resize(size, Image.LANCZOS)
                        # 保存图片
                        img.save(file_path)
                        print(f"Resized: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# 示例用法
folder_path = r"D:\X\NovoNordisk\dataset\Retina Vessel Segmentation"
resize_images(folder_path)