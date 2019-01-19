from google.cloud import videointelligence
import io
from os import listdir
from os.path import isfile, join
from word2vec.main import gen_vectors

def analyze_video(path):
    """Detect labels given a file path."""
    video_client = videointelligence.VideoIntelligenceServiceClient.from_service_account_json('SwampHacks2019-6ee04421862a.json')
    features = [videointelligence.enums.Feature.LABEL_DETECTION]

    with io.open(path, 'rb') as movie:
        input_content = movie.read()

    operation = video_client.annotate_video(
        features=features, input_content=input_content)
    print('\nProcessing video for label annotations:')

    result = operation.result(timeout=90)
    print('\nFinished processing.')

    objects = {}
    # Process shot level label annotations
    shot_labels = result.annotation_results[0].shot_label_annotations
    for i, shot_label in enumerate(shot_labels):

        objects[shot_label.entity.description] = []

        for i, shot in enumerate(shot_label.segments):
            start_time = (shot.segment.start_time_offset.seconds +
                          shot.segment.start_time_offset.nanos / 1e9)
            end_time = (shot.segment.end_time_offset.seconds +
                        shot.segment.end_time_offset.nanos / 1e9)
            objects[shot_label.entity.description].append((start_time, end_time))

    return objects


def create_frames(objects):
    print(objects)
    frames = [list() for i in range(540 // 15)]
    curr_t = 0.5
    idx = 0
    while curr_t <= 36.0:
        for obj in objects:
            if any(r[0] <= curr_t < r[1] for r in objects[obj]):
                frames[idx].append(obj)
        curr_t += 0.5

    return frames


if __name__ == '__main__':

    paths = [f for f in listdir('data') if isfile(join('data', f))]
    print(paths)
    for i in paths:
        annotations = analyze_video('data/' + i)
        fs = create_frames(annotations)
        for i in fs:
            if type(gen_vectors(i)) == str:



