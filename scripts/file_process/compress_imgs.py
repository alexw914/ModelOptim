# -*- coding: UTF-8 -*-
import os, argparse, cv2
from pathlib import Path

'''
用于压缩文件夹中的图片, 降低文件质量以减少存储空间。
params:
	img_folder:  存储图片文件夹
	new_folder:  存储新的压缩图片的文件夹
	img_quality: 压缩质量
	is_rename:   文件是否重命名
usage:
	python compress_imgs.py --img_folder ./images --new_folder=./compress_imgs --img_quality 90 --rename   
'''

def compress_imgs(args):
    
	Path(args.new_folder).mkdir(exist_ok=True)
	img_files = os.listdir(args.img_folder)
	if len(img_files) == 0:
		return

	idx = 0
	for img_file in img_files:
		orig_img  = cv2.imread(os.path.join(args.img_folder, img_file))
		idx = idx + 1
		img_file = (str(idx).rjust(6, "0") if args.rename else img_file.split(".")[0]) + ".jpg"
		cv2.imwrite(os.path.join(args.new_folder, img_file), orig_img, [int(cv2.IMWRITE_JPEG_QUALITY), args.img_quality])

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Compress image file.')
    parser.add_argument('--img_folder', type=str, default='./images', help='path where instore image files')
    parser.add_argument('--new_folder', type=str, default='./easy_data', help='path where instore easydata')
    parser.add_argument('--img_quality', type=int, default=90, help='quality of new image files')
    parser.add_argument('--rename', action="store_true", help='whether rename the compressed img files')
    args = parser.parse_args()
    compress_imgs(args)