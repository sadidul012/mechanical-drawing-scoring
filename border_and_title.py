import cv2
import numpy as np
import torch
from doctr.models import ocr_predictor


ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
ocr_model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


def detect_probable_title_sections(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    image_height, image_width = img_bin.shape

    scale = 40
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image_height // scale))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    vertical_lines = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 40)))

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image_width // scale, 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2)))

    mask = vertical_lines + horizontal_lines
    mask_comp = mask.copy()

    ocr_result = ocr_model([img])
    h, w, _ = img.shape

    for block in ocr_result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x_min, y_min), (x_max, y_max) = word.geometry
                x_min = int(x_min * w)
                y_min = int(y_min * h)
                x_max = int(x_max * w)
                y_max = int(y_max * h)

                cv2.rectangle(mask_comp, (x_min, y_min), (x_max, y_max), (255, 255, 255), cv2.FILLED)
                # cv2.rectangle(img_comp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    mask_comp = cv2.dilate(mask_comp, np.ones((5, 5), np.uint8), iterations=5)
    mask_comp = cv2.erode(mask_comp, np.ones((7, 7), np.uint8), iterations=5)
    mask_comp = cv2.dilate(mask_comp, np.ones((6, 6), np.uint8), iterations=3)

    _, binary = cv2.threshold(mask_comp, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def detect_borders(contours, sorted_indices):
    bb_1 = cv2.boundingRect(contours[sorted_indices[0]])
    bb_2 = cv2.boundingRect(contours[sorted_indices[1]])

    return bb_1, bb_2
