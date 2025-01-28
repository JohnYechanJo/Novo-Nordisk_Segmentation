import os

def delete_images_with_keyword(folder_path, keyword):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
            if keyword in filename:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

folder_path = r"D:\X\NovoNordisk\dataset\Retina Vessel Segmentation\Test\Masks"
keyword = "DRIVE"
delete_images_with_keyword(folder_path, keyword)