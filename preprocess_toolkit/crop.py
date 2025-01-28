import os
from PIL import Image

def resize_images(folder_path, size=(512, 512)):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.lower().endswith('.tif'):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:


                    with Image.open(file_path) as img:
                        width, height = img.size



                        if width > height:
                            left = (width - height) / 2
                            top = 0
                            right = (width + height) / 2
                            bottom = height
                        else:
                            left = 0
                            top = (height - width) / 2
                            right = width
                            bottom = (height + width) / 2
                        img = img.crop((left, top, right, bottom))
                        img = img.resize(size, Image.LANCZOS)
                        img.save(file_path)
                        print(f"Resized: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")





folder_path = r"D:\X\NovoNordisk\dataset\Retina Vessel Segmentation"
resize_images(folder_path)