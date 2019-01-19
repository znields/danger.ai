import cv2
import numpy as np


def label_image(in_path, num_frames=540, starting_frame=0):

    cap = cv2.VideoCapture('data/' + in_path)
    print(cap.get(cv2.CAP_PROP_FPS))
    for _ in range(starting_frame):
        cap.read()

    path, ext = in_path.split('.')

    outputs = []

    for i in range(num_frames // 15):

        for _ in range(14):
            cap.read()

        ret, frame = cap.read()

        if ret:

            cv2.imshow("Video", frame)

            key = cv2.waitKey(-1) & 0xFF
            if key == ord('1'):
                outputs.append(1)

            elif key == ord('0'):
                outputs.append(0)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    np_outputs = np.asarray(outputs)

    np.save('openpose/' + path + '-' + str(int(starting_frame > 0)) + "-y.npy", np_outputs)


if __name__ == '__main__':
    # for i in range(1):
    #     label_image('gta-' + str(i) + '.mp4', starting_frame=0)
    #     label_image('gta-' + str(i) + '.mp4', starting_frame=540)

    for i in range(5, 7):
        label_image('swamphacks-' + str(i) + '.mp4', starting_frame=0)
        label_image('swamphacks-' + str(i) + '.mp4', starting_frame=540)


