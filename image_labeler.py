import cv2
import numpy as np


def label_image(in_path, num_frames=540):

    cap = cv2.VideoCapture('data/' + in_path)

    path, ext = in_path.split('.')

    outputs = []

    for i in range(num_frames):

        for _ in range(14):
            cap.read()

        ret, frame = cap.read()

        if ret:

            cv2.imshow("Video", frame)

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

    np.save('openpose/' + path + "-y.npy", np_outputs)

if __name__ == '__main__':
    label_image('gta-0.mp4')


