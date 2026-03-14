# -*- coding: UTF-8 -*-
import os, shutil, json, argparse
from pathlib import Path

'''
用于对已经分类好的素材进行数据拆分, 数据来自easydata平台, 下载格式为coco格式。
params:
	--local_path:  easydata文件压缩包解压路径 例如: D:\\submit\\group1\\2024-2-2\\2019104_1706860631
	--save_folder: 拆分后的文件夹位置, 可任意指定。
	--remove       删除原来解压的文件夹
	注意应对新任务时, task_map需要重新定义
usage:
	python easydata_split.py --local_path D:\\submit\\group1\\2024-2-2\\2019104_1706860631
'''

def _split_data(args):
    
	json_path = os.path.join(args.local_path, "Annotations/coco_info.json")
	img_folder = os.path.join(args.local_path, "Images")
	Path(args.save_folder).mkdir(exist_ok=True)
	with open(json_path, "r", encoding="utf-8") as fp:
		coco_json = json.load(fp)
    
	cgs = coco_json["categories"]
	img_ids = coco_json["images"]
	annos = coco_json["annotations"]

	for anno in annos:
		img_id = img_ids[int(anno["image_id"])-1]['file_name']
		cg_id = args.task_map[cgs[int(anno["category_id"])]['name']]
		img_path = os.path.join(img_folder, img_id)
		cg_folder = os.path.join(args.save_folder, cg_id)
		Path(cg_folder).mkdir(exist_ok=True)
		shutil.copy(img_path, cg_folder)

	if args.remove:
		shutil.rmtree(args.local_path)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Compress image file.')
    parser.add_argument('--local_path', type=str, default='./2019104_1706860631', help='path where instore image files')
    parser.add_argument('--save_folder', type=str, default='./easydata', help='path where instore classified easydata imgs')
    parser.add_argument('--copy', action="store_true", help='whether remove raw file')
    args = parser.parse_args()
    args.task_map = {"安全帽": "WDAQM", "救生衣反光衣": "WCJSY", "玩手机": "WSJ", "吸烟": "XY"}  # 标注标签与文件夹映射
    _split_data(args)

