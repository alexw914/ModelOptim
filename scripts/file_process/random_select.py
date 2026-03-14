# -*- coding: UTF-8 -*-
import cv2, os, shutil, argparse, random
from pathlib import Path

'''
随机筛选子文件夹下固定数量的图片
params:
    --img_folder    需要合并的子文件夹, 依次添加, 至少为1
    --select_folder 合并后的文件夹
    --copy          拷贝模式，保留筛选图片在原来的文件夹
usage:
    python merge_imgs_folder.py --img_folders images1 images2 --save_folder ./merge --nums 100 --copy
'''

def _select(args):
    
    Path(args.save_folder).mkdir(exist_ok=True) 
    files_all = []
    for folder in args.img_folders:
        img_files = os.listdir(folder)
        for img_file in img_files:
            files_all.append(os.path.join(folder, img_file))
    select_files = random.sample(files_all, args.nums)
    for img_path in select_files:
        if args.copy:
            shutil.copy(img_path, args.save_folder)
        else:
            shutil.move(img_path, args.save_folder)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Compress image file.')
    parser.add_argument('--img_folders', nargs='+', help='list of path where need merge', required=True)
    parser.add_argument('--save_folder', type=str, default="./select", help='path where store image files after select')
    parser.add_argument('--nums', type=int, default=100, help='nums of select image files')
    parser.add_argument('--copy', action="store_true", help='whether remove raw file')
    args = parser.parse_args()
    _select(args)
