import cv2

img_root = '/home/lhd/project/lab(1)/data/test/real_interlace/images/'  # 这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 25  # 保存视频的FPS，可以适当调整
size = (720, 576)
# 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('/home/lhd/project/lab(1)-without-Desnet/data/results/real_interlace/original.avi', fourcc, fps, size)  # 最后一个是保存图片的尺寸

# for(i=1;i<471;++i)
for i in range(0, 11996):
    frame = cv2.imread(img_root + str(i) + '.jpg')
    print(img_root + str(i) + '.jpg')
    videoWriter.write(frame)
videoWriter.release()
