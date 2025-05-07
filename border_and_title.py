import cv2
import numpy as np
import pandas as pd
import torch
from doctr.models import ocr_predictor
from scoring import detect_objects


ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
ocr_model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def detect_probable_title_sections(img, return_states=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    image_height, image_width = img_bin.shape

    scale = 40
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image_height // scale))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    vertical_lines = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 40)))

    # lines_5 = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, 15, np.array([]), int((image_height * 5) / 100), 10)
    # if lines_5 is not None:
    #     for line in lines_5:
    #         for x1, y1, x2, y2 in line:
    #             if y1 > int((image_height * 80) / 100) and y2 > int((image_height * 80) / 100):
    #                 cv2.line(vertical_lines, (x1, y1), (x2, y2), (255, 255, 255), 3)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image_width // scale, 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2)))

    lines_70 = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180, 15, np.array([]), int((image_width * 95) / 100), 50)
    line_90 = None
    if lines_70 is not None:
        for line in lines_70:
            for x1, y1, x2, y2 in line:
                if (
                        y1 > int((image_height * 80) / 100) and y2 > int((image_height * 80) / 100)
                ) and (
                        y1 < int((image_height * 88) / 100) and y2 < int((image_height * 88) / 100)
                ):
                    cv2.line(horizontal_lines, (x1, y1), (x2, y2), (255, 255, 255), 5)
                    if line_90 is None:
                        line_90 = (x1, y1, x2, y2)

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

                # if not is_number(word.value):
                cv2.rectangle(mask_comp, (x_min, y_min), (x_max, y_max), (255, 255, 255), cv2.FILLED)
                # cv2.rectangle(img_comp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    mask_comp = cv2.dilate(mask_comp, np.ones((5, 5), np.uint8), iterations=5)
    mask_comp = cv2.erode(mask_comp, np.ones((7, 7), np.uint8), iterations=5)
    mask_comp = cv2.dilate(mask_comp, np.ones((6, 6), np.uint8), iterations=3)

    _, binary = cv2.threshold(mask_comp, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if return_states:
        return contours, line_90, ocr_result, mask_comp, mask

    return contours, line_90


def find_closest_lines(bx1, by1, bx2, by2, horizontal_lines, vertical_lines, t_dist=200):
    top = horizontal_lines[(horizontal_lines[:, 1] < by1) & (horizontal_lines[:, 1] > (by1 - t_dist)) & (horizontal_lines[:, 0] < bx1) & (horizontal_lines[:, 2] > bx2)]
    top = top[top[:, 0].argmin()]

    bottom = horizontal_lines[(horizontal_lines[:, 1] > by1) & (horizontal_lines[:, 1] < (by1 + t_dist)) & (horizontal_lines[:, 0] < bx1) & (horizontal_lines[:, 2] > bx2)]
    bottom = bottom[bottom[:, 0].argmax()]

    right = vertical_lines[(vertical_lines[:, 0] > bx1) & (vertical_lines[:, 0] < (bx1 + t_dist)) & (vertical_lines[:, 1] > by1) & (vertical_lines[:, 3] < by2)]
    right = right[right[:, 0].argmin()]

    left = vertical_lines[(vertical_lines[:, 0] < bx1) & (vertical_lines[:, 0] > (bx1 - t_dist)) & (vertical_lines[:, 1] > by1) & (vertical_lines[:, 3] < by2)]
    left = left[left[:, 0].argmax()]

    return np.array([top, bottom, right, left])


def process_text(ocr_result, im_h, im_w):
    df = pd.DataFrame.from_dict(ocr_result.export())
    pages = df.join(pd.json_normalize(df.pop('pages')))
    blocks = pages.explode("blocks")
    blocks['block_idx'] = np.arange(blocks.shape[0])
    blocks['index'] = blocks['block_idx']
    blocks = blocks.set_index('index')

    blocks = blocks.join(pd.json_normalize(blocks.pop('blocks')))
    blocks = blocks.rename(columns={'geometry': 'block_geometry'})
    lines = blocks.explode("lines")
    lines['line_idx'] = np.arange(lines.shape[0])
    lines['index'] = np.arange(lines.shape[0])
    lines = lines.set_index('index')
    lines = lines.join(pd.json_normalize(lines.pop('lines')), lsuffix='.lines')
    lines = lines.rename(columns={'geometry': 'line_geometry'})
    words = lines.explode("words")
    words['word_idx'] = np.arange(words.shape[0])
    words['index'] = np.arange(words.shape[0])
    words = words.set_index('index')

    words = words.join(pd.json_normalize(words.pop('words')), lsuffix='.words')
    words = words.rename(columns={'geometry': 'word_geometry'})

    words = words.dropna(subset=['word_geometry'])
    words["word_geometry"] = words.word_geometry.apply(
        lambda x: {"x1": x[0][0], "y1": x[0][1], "x2": x[1][0], "y2": x[1][1]}
    )
    words = words.join(pd.json_normalize(words.pop('word_geometry')))

    words["x1"] = words["x1"] * im_w
    words["x2"] = words["x2"] * im_w
    words["y1"] = words["y1"] * im_h
    words["y2"] = words["y2"] * im_h

    return words


from scipy.spatial import distance


def get_closest_line(row, ref_lines, threshold=50):
    try:
        distances = distance.cdist([row[["x1", "y1", "x2", "y2"]].astype(int)], ref_lines, "euclidean")[0]
        line_idx = distances.argmin()

        if np.sqrt(distances[line_idx]) < threshold:
            return line_idx
    except ValueError:
        pass

    return None


def detect_text_tables(img, ocr_result, mask):
    im_h, im_w, _ = img.shape
    img_comp = img.copy()

    straight_lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 15, np.array([]), 50, 10)
    straight_lines = straight_lines.squeeze(axis=1)

    horizontal_lines = []
    vertical_lines = []

    for (x1, y1, x2, y2) in straight_lines:
        if abs(y1 - y2) <= 5:
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(x1 - x2) <= 5:
            vertical_lines.append((x1, y1, x2, y2))

    horizontal_lines = np.array(horizontal_lines)
    vertical_lines = np.array(vertical_lines)

    new_mask = np.zeros_like(img_comp)
    words = process_text(ocr_result, im_h, im_w)

    for idx, row in words.iterrows():
        bx1, by1, bx2, by2 = row[["x1", "y1", "x2", "y2"]].astype(int)

        try:
            closest = find_closest_lines(bx1, by1, bx2, by2, horizontal_lines, vertical_lines, t_dist=200)
            for c in closest:
                x1, y1, x2, y2 = c
                cv2.line(img_comp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(new_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

            cv2.rectangle(img_comp, (bx1, by1), (bx2, by2), (0, 0, 255), 5)
            cv2.rectangle(new_mask, (bx1, by1), (bx2, by2), (255, 255, 255), cv2.FILLED)
        except ValueError:
            pass

    mask_comp = cv2.dilate(new_mask, np.ones((6, 6), np.uint8), iterations=5)
    mask_comp = cv2.erode(mask_comp, np.ones((7, 7), np.uint8), iterations=5)
    mask_comp = cv2.dilate(mask_comp, np.ones((5, 5), np.uint8), iterations=3)

    gray = cv2.cvtColor(mask_comp, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def detect_borders(contours, sorted_indices):
    bb_1 = cv2.boundingRect(contours[sorted_indices[0]])
    bb_2 = cv2.boundingRect(contours[sorted_indices[1]])

    return bb_1, bb_2
