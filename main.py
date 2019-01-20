import cv2
import imutils
from pixel.pixel_change import calculate_pixel_change
from open_pose import detect_humans


# TODO: add ability to save video; use out_path variable
def analyze_video(in_path, out_path='', width=600, max_frames=1000, show=True, open_pose=True, pixel_change=True, object_detection=True):

    # open the video for read
    cap = cv2.VideoCapture(in_path)

    # read the first frame and adjust its width
    _, curr = cap.read()
    curr = imutils.resize(curr, width=width)

    # while there are more frames in the video
    while max_frames:

        # set the current frame to the previous frame
        prev = curr
        ret, curr = cap.read()

        # if the read was successful
        if ret:

            # resize the current frame
            curr = imutils.resize(curr, width=width)

            if open_pose:
                # TODO: implement open pose algorithm
                # TODO: train machine learning algorithm on data and use here
                curr_humans, keypoints = detect_humans(curr)
                cv2.imshow("Humans", curr_humans)

                print(keypoints)

            if pixel_change:
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

                # TODO: use pixel change score
                pixel_change_score = calculate_pixel_change(curr_gray, prev_gray)

            if object_detection:
                # TODO: implement object detection with pre-trained tensorflow model
                # TODO: use word2vec to calculate an object detection score
                pass

            # display the current frame
            if show:
                cv2.imshow('Video', curr)

            # TODO: research how to read intervals (i.e. 0.5 seconds, 1 second, etc.)
            for _ in range(15):
                cap.read()

            # reduce the number of frames left by 1
                max_frames -= 1

        else:
            break

        # if the users presses q, break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    analyze_video('data/Fight_RunAway1.mpg')
