import cv2
import numpy as np


def detect_objects(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        dilated,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    areas = []
    bb = []

    for cnt in contours:
        areas.append(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        bb.append([x, y, x+w, y+h])

    def get_sorted_indices(data):
        return [i[0] for i in sorted(enumerate(data), key=lambda x: x[1], reverse=True)]

    sorted_indices = get_sorted_indices(areas)

    indices = np.expand_dims(np.arange(hierarchy[0].shape[0]), 1)
    hierarchy_area = np.concatenate((indices, hierarchy[0], np.expand_dims(areas, 1), np.array(bb)), axis=1)

    return sorted_indices, contours, hierarchy_area


def calculate_border_score(img, contours, sorted_indices):
    width, height = img.shape[:2]
    image_area = width * height
    area_1 = cv2.contourArea(contours[sorted_indices[0]])
    area_2 = cv2.contourArea(contours[sorted_indices[1]])

    return 1 - ((((image_area - area_1) / image_area) + ((area_1 - area_2) / area_1)) / 2)
