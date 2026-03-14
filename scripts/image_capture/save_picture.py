import cv2, os, argparse
from pathlib import Path
#读取本地摄像头并设置视频格式。

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


def _get_img(args):
	capture = cv2.VideoCapture(args.url)

	Path(args.save_folder).mkdir(exist_ok=True)
	save_folder = os.path.join(args.save_folder + args.type_name)
	Path(save_folder).mkdir(exist_ok=True)

	save_pic = sorted(os.listdir(save_folder))
	pic_index = 0
	if save_pic != []:
		max_index = 0;
		for v in save_pic:
			index = int(v.split(".")[0].split("_")[-1])
			if index > max_index:
				max_index = index
		pic_index = max_index + 1
		print(pic_index)
	save_pic_folder = args.type_name + "_" +  str(pic_index)
	save_folder = os.path.join(save_folder, save_pic_folder)
	os.mkdir(save_folder)

	print(save_folder)


	fps = 25
	step = 4 * fps
	frame_idx = 0
	pic_idx = 0
	#逐帧读取视频并保存
	while(capture.isOpened()):
		retval,frame = capture.read()
		if retval == True:
			frame_idx = frame_idx + 1
			if frame_idx % step == 0:
				save_file = str(pic_idx).rjust(6, '0') + ".jpg"
				cv2.imwrite(os.path.join(save_folder, save_file), frame)
				pic_idx = pic_idx + 1
		key = cv2.waitKey(1)

	capture.release()
#释放摄像头
output.release()
# cv2.destroyALLWindows()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Save picture from rtsp url.')
    parser.add_argument('--url', type=str, help='rtsp url address')
    parser.add_argument('--save_folder', type=str, default='./pictures', help='folder where instore image files')
    parser.add_argument('--type_name', type=str, default='smoke',help='type of saved picture')
    args = parser.parse_args()
    # 默认下载到当前文件夹，若指定文件夹则自动创建
    Path(args.local_folder).mkdir(exist_ok=True)

