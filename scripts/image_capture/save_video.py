import cv2
import os
from pathlib import Path
#读取本地摄像头并设置视频格式。

capture = cv2.VideoCapture("rtsp://admin:hk123456@192.168.3.26:554/Streaming/Channels/301")
video_type = "fall"

save_folder = "./videos" + "/" + video_type
Path(save_folder).mkdir(exist_ok=True)
save_video = sorted(os.listdir(save_folder))
video_index = 0
if save_video != []:
	max_index = 0;
	for v in save_video:
		index = int(v.split(".")[0].split("_")[-1])
		if index > max_index:
			max_index = index
	video_index = max_index + 1
	print(video_index)
save_video_name = video_type + "_" +  str(video_index) + ".mp4"
save_path = os.path.join(save_folder, save_video_name)
print(save_path)


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
fps = 25
output = cv2.VideoWriter(save_path,fourcc,fps,(1920, 1080))
frame_Num = 10 * fps
#逐帧读取视频并保存
while(capture.isOpened() and frame_Num >0):
	retval,frame = capture.read()
	if retval == True:
		output.write(frame)
		# cv2.imshow('frame',frame)
	key = cv2.waitKey(1)
	frame_Num -=1
#释放摄像头
capture.release()
output.release()
# cv2.destroyALLWindows()
