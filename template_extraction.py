import cv2
import numpy as np
from skimage.metrics import structural_similarity

from border_and_title import detect_intersection_with_boundary
from utils import remove_similar_lines, find_intersected_lines


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


def detect_intersection_with_template(img, boundary, borders, tolerance=10):
    im_h, im_w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )
    lines = remove_similar_lines(lines)

    template_boundaries = borders.copy()
    template_boundaries[(template_boundaries[:, 1] > int((im_h * 70) / 100)) & (
                template_boundaries[:, 3] > int((im_h * 70) / 100)), 3] = boundary[0][1]
    template_boundaries[(template_boundaries[:, 2] > int((im_h * 85) / 100)), 2] = boundary[1][0]
    template_boundaries[(template_boundaries[:, 0] < int((im_h * 15) / 100)), 0] = boundary[0][0]
    template_boundaries[(template_boundaries[:, 1] < int((im_h * 30) / 100)) & (
                template_boundaries[:, 3] < int((im_h * 30) / 100)), 1] = boundary[1][1]

    template_lines = []
    drawings = lines.copy()

    (x1, y1), (x2, y2) = boundary
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    drawings = drawings[
        ~((drawings[:, 1] <= top + tolerance) & (drawings[:, 3] <= top + tolerance))
        & ~((drawings[:, 1] >= bottom - tolerance) & (drawings[:, 3] >= bottom - tolerance))
        & ~((drawings[:, 0] <= left + tolerance) & (drawings[:, 2] <= left + tolerance))
        & ~((drawings[:, 0] > right - tolerance) & (drawings[:, 2] > right - tolerance))
        ]

    for b in template_boundaries:
        x1, y1, x2, y2 = b

        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        boundary_title_block_lines = [
            [left, top, right, top],  # Top
            [left, bottom, left, top],  # Left
            [right, bottom, right, top],  # Right
            [left, bottom, right, bottom],  # Bottom
        ]
        template_lines.extend(boundary_title_block_lines)

        drawings = drawings[
            ~(
                    (drawings[:, 1] >= top - tolerance)
                    & (drawings[:, 3] >= top - tolerance)
                    & (drawings[:, 0] >= left - tolerance)
                    & (drawings[:, 2] >= left - tolerance)
                    & (drawings[:, 3] <= bottom + tolerance)
                    & (drawings[:, 1] <= bottom + tolerance)
                    & (drawings[:, 2] <= right + tolerance)
                    & (drawings[:, 0] <= right + tolerance)
            )
        ]

    intersection_lines, intersection_points = detect_intersection_with_boundary(drawings, boundary, tolerance=10)

    for line in template_lines:
        intersected_lines, intersected_at = find_intersected_lines(line, drawings)
        intersection_points.extend(intersected_at)
        intersection_lines.extend(intersected_lines)

    return template_lines, intersection_lines, intersection_points
