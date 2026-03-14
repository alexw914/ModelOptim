# -*- coding: UTF-8 -*-
import cv2, argparse, random, os

'''
用于验证转化的yolo格式标签是否正确, 按enter下一个, esc取消预览
params:
	--image_folder: easydata文件压缩包解压路径
	--label_folder: 拆分后的文件夹位置
usage:
	python label_view.py --image_folder  --label_folder ./data
'''

def _view(args):
		
	img_files = os.listdir(args.image_folder)
	random.shuffle(img_files)
	for img_file in img_files:
		img = cv2.imread(os.path.join(args.image_folder, img_file))
		img_h, img_w, _ = img.shape
		txt_file = img_file[:len(img_file) - 3] + "txt"
		with open(os.path.join(args.label_folder, txt_file), 'r') as f:
			objs = [l.strip() for l in f.readlines()]
		for obj in objs:
			cls, cx, cy, nw, nh = [float(item) for item in obj.split(' ')]
			color = (0, 0, 255) if cls == 0.0 else (0, 255, 0)
			x_min = int((cx - (nw / 2.0)) * img_w)
			y_min = int((cy - (nh / 2.0)) * img_h)
			x_max = int((cx + (nw / 2.0)) * img_w)
			y_max = int((cy + (nh / 2.0)) * img_h)
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
		cv2.imshow("verify_labels", img)
		if cv2.waitKey(0) == 27:
			break
	cv2.destroyAllWindows()	

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='Compress image file.')
	parser.add_argument('--image_folder', type=str, default='./images', help='path where instore image files')
	parser.add_argument('--label_folder', type=str, default='./labels', help='path where instore compressed image files')
	args = parser.parse_args()
	_view(args)
