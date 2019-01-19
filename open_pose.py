import cv2
import numpy as np

protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee',
                    'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]

# index of pafs corresponding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
          [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
          [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
          [37, 38], [45, 46]]

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

# create the net
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def get_keypoints(prob_map, threshold=0.1):
    map_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)

    map_mask = np.uint8(map_smooth > threshold)
    keypoints = []

    # find the blobs
    contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
        masked_prob_mask = map_smooth * blob_mask
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_mask)
        keypoints.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def get_valid_pairs(output, width, height, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        paf_a = output[0, mapIdx[k][0], :, :]
        paf_b = output[0, mapIdx[k][1], :, :]
        paf_a = cv2.resize(paf_a, (width, height))
        paf_b = cv2.resize(paf_b, (width, height))

        # Find the keypoints for the first and second limb
        cand_a = detected_keypoints[POSE_PAIRS[k][0]]
        cand_b = detected_keypoints[POSE_PAIRS[k][1]]
        n_a = len(cand_a)
        n_b = len(cand_b)

        # If keypoints for the joint-pair is detected
        # check every joint in cand_a with every joint in cand_b
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if n_a != 0 and n_b != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(n_a):
                max_j = -1
                max_score = -1
                found = 0
                for j in range(n_b):
                    # Find d_ij
                    d_ij = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=n_interp_samples),
                                            np.linspace(cand_a[i][1], cand_b[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for l in range(len(interp_coord)):
                        paf_interp.append([paf_a[int(round(interp_coord[l][1])), int(round(interp_coord[l][0]))],
                                           paf_b[int(round(interp_coord[l][1])), int(round(interp_coord[l][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > max_score:
                            max_j = j
                            max_score = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[cand_a[i][3], cand_b[max_j][3], max_score]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else:  # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def get_personwise_keypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score
    personwise_keypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            part_as = valid_pairs[k][:, 0]
            part_bs = valid_pairs[k][:, 1]
            index_a, index_b = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwise_keypoints)):
                    if personwise_keypoints[j][index_a] == part_as[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwise_keypoints[person_idx][index_b] = part_bs[i]
                    personwise_keypoints[person_idx][-1] += keypoints_list[part_bs[i].astype(int), 2] + \
                        valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[index_a] = part_as[i]
                    row[index_b] = part_bs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwise_keypoints = np.vstack([personwise_keypoints, row])
    return personwise_keypoints


def detect_humans(image, in_height=328):
    width = image.shape[1]
    height = image.shape[0]

    # Fix the input Height and get the width according to the Aspect Ratio
    in_width = int((in_height / height) * width)

    in_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (in_width, in_height),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(in_blob)
    output = net.forward()
    # print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1

    res_keypoints = {}

    for part in range(nPoints):
        prob_map = output[0, part, :, :]
        prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))
        keypoints = get_keypoints(prob_map, threshold)
        # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        res_keypoints[keypointsMapping[part]] = keypoints
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    frame_clone = image.copy()

    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frame_clone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

    valid_pairs, invalid_pairs = get_valid_pairs(output, width, height, detected_keypoints)
    personwise_keypoints = get_personwise_keypoints(valid_pairs, invalid_pairs, keypoints_list)

    for i in range(17):
        for n in range(len(personwise_keypoints)):
            index = personwise_keypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            b = np.int32(keypoints_list[index.astype(int), 0])
            a = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frame_clone, (b[0], a[0]), (b[1], a[1]), colors[i], 3, cv2.LINE_AA)

    return frame_clone, res_keypoints


def extract_keypoints(keypoints):
    for i in keypoints:
        while len(keypoints[i]) != 3:
            keypoints[i].append((0, 0, 0))

    return keypoints

if __name__ == '__main__':

    import time
    filename = 'gta-0.mp4'

    json = {filename: []}

    cap = cv2.VideoCapture('data/' + filename)

    for i in range(30 * 30):

        ret, frame = cap.read()

        if ret:
            t0 = time.time()
            clone_frame, keypoints = detect_humans(frame)
            print(extract_keypoints(keypoints))

            t1 = time.time()

            print(t1 - t0)
