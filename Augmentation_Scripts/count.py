import os

def count_image_files(folder_path):
    jpg_count = 0
    png_count = 0
    # 遍历文件夹及其子文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 检查文件是否为 .jpg 文件
            if filename.lower().endswith('.jpg'):
                jpg_count += 1
            # 检查文件是否为 .png 文件
            elif filename.lower().endswith('.png'):
                png_count += 1
    return jpg_count, png_count

# 示例用法
folder_path = r"D:\X\NovoNordisk\FinalData_Fine"
jpg_count, png_count = count_image_files(folder_path)
print(f"Total number of .jpg files in folder '{folder_path}': {jpg_count}")
print(f"Total number of .png files in folder '{folder_path}': {png_count}")