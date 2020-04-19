import cv2

vidcap = cv2.VideoCapture('input.avi')
success, image = vidcap.read()
count = 0
while count<3:
    cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
