# -*- coding: UTF-8 -*-
import cv2, os, argparse
from pathlib import Path

'''
用于对两个文件夹下的文件进行合并, 同时选择是否进行重命名, 6位数字, 前面补0。文件名后缀统一为jpg
params:
	--img_folders  需要合并的子文件夹, 依次添加, 至少为1
	--merge_folder 合并后的文件夹
	--rename       是否重命名
	--copy         拷贝模式, 保留源文件夹下内容
usage:
	python merge_folder.py --img_folders images1 images2 --merge_folder "merge" --rename
'''

def _merge(args):
    
	Path(args.merge_folder).mkdir(exist_ok=True)
	idx = 0
	for folder in args.img_folders:
		img_files = os.listdir(folder)
		for img_file in img_files:
			orig_img = cv2.imread(os.path.join(folder, img_file))
			if orig_img is None:
				continue
			idx = idx + 1
			img_file = (str(idx).rjust(6, "0") if args.rename else img_file.split(".")[0]) + ".jpg"
			cv2.imwrite(os.path.join(args.merge_folder, img_file), orig_img)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Merge image files in different folder.')
    parser.add_argument('--img_folders', nargs='+', help='list of path where need merge', required=True)
    parser.add_argument('--merge_folder', type=str, default="./merge", help='path where store image files after merge')
    parser.add_argument('--rename', action="store_true", help='whether rename the compressed img files')
    parser.add_argument('--copy', action="store_true", help='whether remove raw file')
    args = parser.parse_args()
    _merge(args)