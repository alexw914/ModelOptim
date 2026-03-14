import os, glob, random, shutil, argparse
from pathlib import Path

'''
用于已经标注好的数据划分训练集和验证集, 使用yolo格式
params:
    --data_folder   标注好的数据集文件夹, 包含images和labels, 分别存放图片和标签
    --save_folder   数据集存放路径
    --copy          拷贝模式，保留筛选图片在原来的文件夹
usage:
    python split_dataset.py --img_folders images1 images2 --merge_folder "merge"
'''

def _split_dataset(args):
    
    if not os.path.exists(os.path.join(args.data_folder, "images")) or not os.path.exists(os.path.join(args.data_folder, "images")):
        print("Data folder path error, not exists images and labels sub folder")
        return
    Path(args.save_folder).mkdir(exist_ok=True)
    for folder in ["images", "labels"]:
        Path(os.path.join(args.save_folder, folder)).mkdir(exist_ok=True)
        for set in ["train", "val"]:
            Path(os.path.join(args.save_folder, folder, set)).mkdir(exist_ok=True)

    img_files = os.listdir(os.path.join(args.data_folder, "images"))
    valid_elements = random.sample(img_files, int(len(img_files) * args.ratio))
    
    for element in valid_elements:
        if args.copy:
            shutil.copy(os.path.join(args.data_folder, "images", element), os.path.join(args.save_folder, "images/val"))
            shutil.copy(os.path.join(args.data_folder, "labels", element.replace("jpg", "txt")), os.path.join(args.save_folder, "labels/val"))
        else: 
            shutil.move(os.path.join(args.data_folder, "images", element), os.path.join(args.save_folder, "images/val"))
            shutil.move(os.path.join(args.data_folder, "labels", element.replace("jpg", "txt")), os.path.join(args.save_folder, "labels/val"))

    remain_imgs = glob.glob(os.path.join(args.data_folder, "images") + "/*.jpg")
    for remain_img in remain_imgs:
        element = os.path.basename(remain_img)
        if args.copy:  
            shutil.copy(os.path.join(args.data_folder, "images", element), os.path.join(args.save_folder, "images/train"))
            shutil.copy(os.path.join(args.data_folder, "labels", element.replace("jpg", "txt")), os.path.join(args.save_folder, "labels/train"))
        else:
            shutil.move(os.path.join(args.data_folder, "images", element), os.path.join(args.save_folder, "images/train"))
            shutil.move(os.path.join(args.data_folder, "labels", element.replace("jpg", "txt")), os.path.join(args.save_folder, "labels/train"))

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Compress image file.')
    parser.add_argument('--data_folder', type=str, default="./raw", help='path where store image files after select')
    parser.add_argument('--save_folder', type=str, default="./data", help='path where store image files after select')
    parser.add_argument('--ratio', type=float, default=0.2, help='ratio of valid set of all set')
    parser.add_argument('--copy', action="store_true", help='whether remove raw file')    
    args = parser.parse_args()
    _split_dataset(args)