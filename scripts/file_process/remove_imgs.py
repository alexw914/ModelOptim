import os, shutil, argparse
from pathlib import Path

'''
删除无人标签图片，或者标签数量不足的照片，min_num指定最小标签个数
params:
    --img_folder   存放源图片的路径
    --txt_folder   存放标签的路径,由yolo模型生成
    --remove       删除不满足条件的图片，否则移动到固定文件夹下
usage:
	python remove_imgs.py --img_folder data/images --txt_folder data/labels --min_num 3
'''

def _remove(args):
    
    if not args.remove:
        nolabel_path = os.path.join(args.img_folder, "nolabel")
        Path(nolabel_path).mkdir(exist_ok=True)
    if args.min_num > 1:
        less_path = os.path.join(args.img_folder,"less")
        Path(less_path).mkdir(exist_ok=True)
                
    img_files = os.listdir(os.path.join(args.img_folder))
    if len(img_files) == 0:
        return
    img_type = ".jpg"
    for img_file in img_files:
        txt_path = os.path.join(args.txt_folder, img_file.replace(img_type, ".txt"))
        if not os.path.exists(txt_path):
            if not args.remove:
                shutil.move(os.path.join(args.img_folder, img_file), nolabel_path)
            else:
                shutil.rmtree(os.path.join(args.img_folder, img_file))
        else:
            if args.min_num >= 1:
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                if len(lines) < args.min_num:
                    if not args.remove:
                        shutil.move(os.path.join(args.img_folder, img_file), less_path)
                    else:
                        shutil.rmtree(os.path.join(args.img_folder, img_file))            
            
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Compress image file.')
    parser.add_argument('--img_folder', type=str, default="images", required=True, help='path where store image files')
    parser.add_argument('--txt_folder', type=str, default="labels", required=True, help='path where store label files')
    parser.add_argument('--min_num', type=int, default=1, help='minum num of labels')
    parser.add_argument('--remove', action="store_true", help='whether remove raw file')    
    args = parser.parse_args()
    _remove(args)