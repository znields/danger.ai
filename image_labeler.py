import csv
import cv2
import os
import time
import numpy as np


def label_image(in_path, out_name='labels', num_frames=548):

	cv2.namedWindow("stream")

	cap = cv2.VideoCapture(in_path)

	outputs = []



	num_frames = 123

	for i in range(num_frames):
	retval, frame = cap.read()
	if retval:
		cv2.imshow("stream", frame)
		# press 'q' to quit
		key = cv2.waitKey(-1) & 0xFF
		if key == ord('1'):
			outputs.append(1)
		if key == ord('0'):
			outputs.append(0)
			
	else:
		break
		
	cap.release()
	cv2.destroyAllWindows()        
		
	 

	np_outputs = np.asarray(outputs)

	np.save(out_name+".npy", np_outputs)


def get_input():
    i = int(input())
    if i == 1 or i == 0:
        return i
    else:
        return get_input()


def get_filename(folder, path):
    path = os.path.basename(path)
    name, _ = os.path.splitext(path)
    return folder + '/' + name + '.csv'


if __name__ == '__main__':
    label_image('data/Fight_RunAway1.mpg')
