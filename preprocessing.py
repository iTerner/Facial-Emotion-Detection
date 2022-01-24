import os
from PIL import Image
from tqdm import tqdm
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import dlib
import cv2
import pickle

# precentage of the train set
TRAIN = 0.8

# precentage of data to cut from the data set
CUT = 0.0


d = {
    "train": list(),
    "val": list(),
    "test": list(),
}

emotions = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

# the pixels for the detection of the mouth, eyebrows and eyes
MOUTH = (48, 68)
LEFT_EYEBROW = (22, 27)
RIGHT_EYEBROW = (17, 22)
LEFT_EYE = (42, 48)
RIGHT_EYE = (36, 42)

# create the detector and predictor objects
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "/content/eameo-faceswap-generator/shape_predictor_68_face_landmarks.dat")

# path to the original image folder
imgs_path = "/content/drive/MyDrive/deep_project_data/tmp/FERG_DB_256"

# path of where we want to save the data
path = "/content/drive/MyDrive/deep_project_data/"


# read all the images, and split them to train, validation and test sets
count = 0
persons = [f for f in os.listdir(
    imgs_path) if os.path.isdir(os.path.join(imgs_path, f))]
for person in tqdm(persons):
    for emotion in emotions.keys():
        temp_path = os.path.join(imgs_path, person, f"{person}_{emotion}/")
        pics_paths = [f for f in os.listdir(
            temp_path) if os.path.isfile(os.path.join(temp_path, f))]
        pics = []
        for pic in pics_paths:
            pic_path = os.path.join(temp_path, pic)
            image = cv2.imread(pic_path)
            image = imutils.resize(
                image, width=image.shape[0], height=image.shape[1])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pics.append(gray)
        count += len(pics)
        # cut
        n_cut = int(CUT * len(pics))
        inx = sample(range(len(pics)), n_cut)
        final_pics = []
        for i in range(len(pics)):
            if i not in inx:
                final_pics.append(pics[i])

        # train test split
        n = int(TRAIN * len(pics))
        inx = sample(range(len(pics)), n)
        for i in range(len(final_pics)):
            data = (final_pics[i], emotions[emotion])
            if i in inx:
                d["train"].append(data)
            else:
                inx2 = np.random.rand(1)
                if inx2[0] > 0.5:
                    d["test"].append(data)
                else:
                    d["val"].append(data)


print(f"total pics {count} before")
count = 0
for key, val in d.items():
    s = set()
    l = len(val)
    count += l
    for img, label in val:
        s.add(label)
    print(f"in {key}, there is total of {l} pics and labels {s}")
print(f"total images after {count}")

# print a sample image
plt.imshow(d["train"][0][0])


def get_critical_points(points):
    left_point = points[0]
    right_point = [0, 0]
    lower_point = points[0]
    upper_point = [0, 0]

    for p in points:
        if p[0] < left_point[0]:
            left_point = p
        if p[0] > right_point[0]:
            right_point = p
        if p[1] < lower_point[1]:
            lower_point = p
        if p[1] > upper_point[1]:
            upper_point = p
    return left_point, right_point, lower_point, upper_point


def to_bb(critical_points):
    left = (critical_points[0][0], critical_points[1][0])
    right = (critical_points[1][1], critical_points[3][1])

    return left, right


def get_detector_points(image):
    rects = detector(image, 1)
    if len(rects) == 0:
        # plt.imshow(image)
        return None
    shape = predictor(image, rects[0])
    face_points = []
    for part in shape.parts():
        face_points.append((part.x, part.y))
    np_face_points = np.array(face_points)
    mouse = np_face_points[MOUTH[0]:MOUTH[1]]
    upper_points = mouse
    left_eye = np_face_points[LEFT_EYE[0]:LEFT_EYE[1]]
    right_eye = np_face_points[RIGHT_EYE[0]:RIGHT_EYE[1]]
    left_eyebrow = np_face_points[LEFT_EYEBROW[0]:LEFT_EYEBROW[1]]
    right_eyebrow = np_face_points[RIGHT_EYEBROW[0]:RIGHT_EYEBROW[1]]
    lower_points = np.concatenate(
        (left_eye, right_eye, left_eyebrow, right_eyebrow))
    return lower_points, upper_points


def get_blocks(lst):
    """
      lst[(image, label)]
      lst[(lower_block, upper_block, label)]
    """
    X_WINDOW = 30
    Y_WINDOW = 30
    blocks = []
    for l in tqdm(lst):
        img = l[0]
        tmp = get_detector_points(img)
        if tmp != None:
            lower_points, upper_points = tmp
            critical_upper = get_critical_points(upper_points)
            upper = to_bb(critical_upper)
            upper_block_image = gray[upper[1][0]-X_WINDOW:upper[1]
                                     [1]+X_WINDOW, upper[0][0]-Y_WINDOW:upper[0][1]+Y_WINDOW]

            critical_lower = get_critical_points(lower_points)
            lower = to_bb(critical_lower)
            lower_block_image = gray[lower[1][0]-X_WINDOW:lower[1]
                                     [1]+X_WINDOW, lower[0][0]-Y_WINDOW:lower[0][1]+Y_WINDOW]

            blocks.append((upper_block_image, lower_block_image, l[1]))

    return blocks


# get the blocks for the train, validation and test sets
for key, val in d.items():
    print(f"getting blocks for {key}")
    d[key] = get_blocks(val)


# print a sample images
plt.imshow(d["train"][0][0])
plt.imshow(d["train"][0][1])

count = 0
for key, val in d.items():
    s = set()
    l = len(val)
    count += l
    for img1, img2, label in val:
        s.add(label)
    print(f"in {key}, there is total of {l} pics and labels {s}")
print(f"total images {count}")


def save_blocks(dictionary_data, filepath):
    """
    The function saves the data in the dictionary on a pickle file
    Args:
        dictionary_data - the data we want to save
        filepath - the path
    """
    for key, val in tqdm(dictionary_data.items()):
        a_file = open(f"{filepath}final{key}.pkl", "wb")
        pickle.dump(dictionary_data[key], a_file)
        a_file.close()


# save the data in a pickle file
save_blocks(d, path)
