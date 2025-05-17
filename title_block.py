from itertools import combinations

import cv2
import pandas as pd
import torch
from doctr.models import ocr_predictor
from shapely import box

from utils import *

ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
ocr_model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


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


def decide_inner(line_1, line_2):
    x1, y1, x2, y2 = abs(line_1 - line_2)
    if y1 > 100:
        x1, y1, x2, y2 = line_1
    else:
        x1, y1, x2, y2 = line_2

    return x1, y1, x2, y2


def detect_probable_title_sections(img, return_states=False):
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

    lines_90_h = cv2.HoughLinesP(horizontal_lines, 1, np.pi / 180, 15, np.array([]), int((image_width * 90) / 100), 50)
    lines_90_v = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, 15, np.array([]), int((image_height * 90) / 100), 50)

    inner_border_lines = []
    if lines_90_h is not None and len(lines_90_h) > 1:
        lines_90_h = remove_similar_lines(lines_90_h)
        inner_border_lines.extend([
            decide_inner(lines_90_h[0], lines_90_h[1]),
            decide_inner(lines_90_h[-1], lines_90_h[-2]),
        ])

    if  lines_90_v is not None and len(lines_90_v) > 1:
        lines_90_v = remove_similar_lines(lines_90_v)
        inner_border_lines.extend([
            decide_inner(lines_90_v[0], lines_90_v[1]),
            decide_inner(lines_90_v[-2], lines_90_v[-1])
        ])

    inner_border_lines = remove_similar_lines(np.array(inner_border_lines))

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
        return contours, line_90, dict(
            ocr_result=ocr_result,
            mask_comp=mask_comp,
            mask=mask,
            lines_90_h=lines_90_h,
            lines_90_v=lines_90_v,
            inner_border_lines=inner_border_lines
        )

    return contours, line_90


def detect_text_tables(img, words, mask):
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

    for idx, row in words.iterrows():
        bx1, by1, bx2, by2 = row[["x1", "y1", "x2", "y2"]].astype(int)

        try:
            closest = find_closest_lines(bx1, by1, bx2, by2, horizontal_lines, vertical_lines, t_dist=200)
            for c in closest:
                x1, y1, x2, y2 = c
                cv2.line(new_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

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


def table_content_score(bb, words):
    ratio = 1
    try:
        (cx1, cy1), (cx2, cy2) = bb
        words_in_bb = words.loc[(words.x1 > cx1) & (words.x2 < cx2) & (words.y1 > cy1) & (words.y2 < cy2)]
        sentence = "".join(words_in_bb.value.tolist())
        chars = sum(c.isalpha() for c in sentence)
        digits = sum(c.isdigit() for c in sentence)
        ratio = digits / chars

        if words_in_bb.shape[0] < 2 and ratio < 1:
            ratio += 0.2
        elif words_in_bb.shape[0] < 3 and ratio < 0.8:
            ratio += 0.2
        elif (len(sentence) / words_in_bb.shape[0]) < 2:
            ratio += 0.2
        elif digits == 0:
            ratio -= 0.5

        return ratio
    except ZeroDivisionError:
        ratio += 0.2

    return ratio


def get_title_boundary(boundary, line_90, title_contours, words, im_h):
    bb = []
    for cnt in title_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = x + w
        y1 = y + h
        aspect_ratio = w / float(h)
        area = w * h

        if area > 7000 and 1.3 < aspect_ratio < 10:
            ratio = table_content_score(((x, y), (x1, y1)), words)

            if ratio <= 1:
                if y > int((im_h * 70) / 100) and y + h > int((im_h * 93) / 100):
                    bb.append((x, y, x1, y1))

                elif y > int((im_h * 80) / 100) and y1 > int((im_h * 90) / 100):
                    bb.append((x, y, x1, y + h))

    title_boundary = [[0, 0], [0, 0]]

    try:
        bb = np.array(bb)

        if line_90 is not None:
            x1, y1, x2, y2 = line_90
            x1 = boundary[0][0]
            x2 = boundary[1][0]
        else:
            x2 = bb[:, 2].max()
            x1 = bb[:, 0].min()
            y1 = bb[:, 1].min()

            if abs(boundary[1][0] - bb[:, 2].max()) < 50:
                x2 = boundary[1][0]

            if abs(boundary[0][0] - bb[:, 0].min()) < 50:
                x1 = boundary[0][0]

        y2 = max(boundary[0][1], boundary[1][1])
        title_boundary = [[x1, y1], [x2, y2]]
    except IndexError:
        pass

    return title_boundary


def detect_table(img, mask, words):
    im_h, im_w, _ = img.shape

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

    table_lines = []

    for idx, row in words.iterrows():
        bx1, by1, bx2, by2 = row[["x1", "y1", "x2", "y2"]].astype(int)

        try:
            closest = find_closest_lines(bx1, by1, bx2, by2, horizontal_lines, vertical_lines, t_dist=200)
            for c in closest:
                table_lines.append(c)
        except ValueError:
            pass

    lines = np.array(table_lines)
    horizontal_lines = []
    vertical_lines = []

    for x1, y1, x2, y2 in lines:
        if y1 == y2:
            horizontal_lines.append((y1, min(x1, x2), max(x1, x2)))
        elif x1 == x2:
            vertical_lines.append((x1, min(y1, y2), max(y1, y2)))

    horizontal_dict = {}
    for y, x1, x2 in horizontal_lines:
        horizontal_dict.setdefault(y, []).append((x1, x2))

    vertical_dict = {}
    for x, y1, y2 in vertical_lines:
        vertical_dict.setdefault(x, []).append((y1, y2))

    ys = sorted(horizontal_dict.keys())
    xs = sorted(vertical_dict.keys())

    rectangles = set()

    for y1, y2 in combinations(ys, 2):
        for x1, x2 in combinations(xs, 2):
            top_valid = any(x1 >= h1 and x2 <= h2 for h1, h2 in horizontal_dict[y1])
            bottom_valid = any(x1 >= h1 and x2 <= h2 for h1, h2 in horizontal_dict[y2])

            left_valid = any(y1 >= v1 and y2 <= v2 for v1, v2 in vertical_dict[x1])
            right_valid = any(y1 >= v1 and y2 <= v2 for v1, v2 in vertical_dict[x2])

            if top_valid and bottom_valid and left_valid and right_valid:
                rectangles.add((x1, y1, x2, y2))

    rectangles = sorted(rectangles, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)

    boxes = [box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)) for x1, y1, x2, y2 in rectangles]

    filtered = []
    for i, b1 in enumerate(boxes):
        if not any(b2.contains(b1) for j, b2 in enumerate(boxes) if i != j):
            filtered.append(rectangles[i])

    return filtered