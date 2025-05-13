import cv2
import numpy as np
from skimage.metrics import structural_similarity

from utils import remove_similar_lines


def find_common_region(images, threshold_value=30):
    img_base = images[0]
    if len(img_base.shape) > 2:
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    else:
        img_base_gray = img_base
    common_mask = np.ones_like(img_base_gray, dtype=np.uint8) * 255

    for img in images[1:]:

        try:
            if len(img_base.shape) > 2:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            score = structural_similarity(img_base_gray, img_gray)
            print("Image Similarity: {:.4f}%".format(score * 100))

            diff = cv2.absdiff(img_base_gray, img_gray)

            _, diff_thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY_INV)
            common_mask = cv2.bitwise_and(common_mask, diff_thresh)
        except (cv2.error, ValueError):
            print("Image processing error")
            print(img.shape, img_base.shape)

            continue

    result = cv2.bitwise_and(img_base, img_base, mask=common_mask)

    return result, cv2.cvtColor(common_mask, cv2.COLOR_GRAY2BGR)


def select_bottom_top_lines(lines, threshold=10):
    lines = np.array(lines)
    selected_lines = []

    while len(lines) > 0:
        lines = sorted(lines, key=lambda x: x[1])

        picked_line = lines[0]
        selected_lines.append(picked_line)

        x1_pick, y1_pick, x2_pick, _ = picked_line
        min_x, max_x = min(x1_pick, x2_pick), max(x1_pick, x2_pick)

        new_lines = []
        for line in lines[1:]:
            x1, y1, x2, _ = line
            line_min_x, line_max_x = min(x1, x2), max(x1, x2)
            if not (y1 > y1_pick and line_min_x >= min_x - threshold and line_max_x <= max_x + threshold):
                new_lines.append(line)

        lines = new_lines

    return np.array(selected_lines)


def keep_top_below_line(lines, threshold=10):
    lines = np.array(lines)

    if len(lines) == 0:
        return np.array([])

    lines = sorted(lines, key=lambda x: -x[1])

    kept_lines = []
    picked_line = lines[0]
    kept_lines.append(picked_line)

    pick_x1, pick_y1, pick_x2, pick_y2 = picked_line
    pick_min_x, pick_max_x = min(pick_x1, pick_x2), max(pick_x1, pick_x2)

    for line in lines[1:]:
        x1, y1, x2, y2 = line
        min_x, max_x = min(x1, x2), max(x1, x2)

        if (min_x >= pick_min_x - threshold) and (max_x <= pick_max_x + threshold):
            continue
        else:
            kept_lines.append(line)

    return np.array(kept_lines)


def get_template_borders_from_structures(structures):
    template, common_mask = find_common_region(structures)
    template_height, template_width = template.shape[:2]
    straight_lines = cv2.HoughLinesP(template, 1, np.pi / 180, 15, np.array([]), 50, 10)
    straight_lines = straight_lines.squeeze(axis=1)

    bottom_horizontal_lines = straight_lines[(straight_lines[:, 1] > int((template_height * 70) / 100)) & (
                straight_lines[:, 3] > int((template_height * 70) / 100))]
    bottom_horizontal_lines = remove_similar_lines(
        bottom_horizontal_lines[abs(bottom_horizontal_lines[:, 1] - bottom_horizontal_lines[:, 3]) == 0])
    horizontal_bottom_borders = select_bottom_top_lines(bottom_horizontal_lines)

    top_horizontal_lines = straight_lines[(straight_lines[:, 1] < int((template_height * 30) / 100)) & (
                straight_lines[:, 3] < int((template_height * 30) / 100))]
    top_horizontal_lines = remove_similar_lines(
        top_horizontal_lines[abs(top_horizontal_lines[:, 1] - top_horizontal_lines[:, 3]) == 0])

    top_horizontal_borders = keep_top_below_line(top_horizontal_lines)

    borders = horizontal_bottom_borders

    if top_horizontal_borders.shape[0] > 0:
        borders = np.vstack([borders, top_horizontal_borders])

    return borders, template
