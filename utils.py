import cv2
import numpy as np
from imutils import resize
from pdf2image import convert_from_path
from scipy.spatial import distance
import random

random.seed(42)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def random_rgb_color():
  """Generates a random RGB color as a tuple of integers (0-255)."""
  r = random.randint(0, 255)
  g = random.randint(0, 255)
  b = random.randint(0, 255)

  return r, g, b


def im_resize(img, size=1500):
    height, width = img.shape[:2]

    if height * width > 59478485:
        scale_percent = 50
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    longer = np.argmax(img.shape)

    if longer == 0:
        img = resize(img, size)
    else:
        img = resize(img, height=size)

    return img


def read_pdf(pdf_path):
    if pdf_path.endswith('.pdf'):
        pages = convert_from_path(pdf_path, dpi=300)
        img = np.array(pages[0])
    elif pdf_path.endswith('.png') or pdf_path.endswith('.jpg') or pdf_path.endswith('.jpeg') or pdf_path.endswith('.bmp'):
        img = cv2.imread(pdf_path)
    else:
        return None

    img = im_resize(img)
    return img


def remove_similar_lines(lines):
    lines = lines.reshape(-1, 4)
    threshold = 5

    keep = []
    visited = np.zeros(len(lines), dtype=bool)

    for i in range(len(lines)):
        if visited[i]:
            continue

        current_line = lines[i]
        visited[i] = True

        for j in range(i+1, len(lines)):
            if visited[j]:
                continue
            other_line = lines[j]

            if np.all(np.abs(current_line - other_line) <= threshold):
                visited[j] = True

        keep.append(current_line)

    return np.array(keep)


def calculate_area(bb):
    p1, p2 = bb

    h = p1[0] - p2[0]
    w = p1[1] - p2[1]

    return h * w


def find_closest_lines(bx1, by1, bx2, by2, horizontal_lines, vertical_lines, t_dist=200):
    horizontal_lines = np.stack([
        np.minimum(horizontal_lines[:, 0], horizontal_lines[:, 2]),
        np.minimum(horizontal_lines[:, 1], horizontal_lines[:, 3]),
        np.maximum(horizontal_lines[:, 0], horizontal_lines[:, 2]),
        np.maximum(horizontal_lines[:, 1], horizontal_lines[:, 3])
    ], axis=1)

    vertical_lines = np.stack([
        np.minimum(vertical_lines[:, 0], vertical_lines[:, 2]),
        np.minimum(vertical_lines[:, 1], vertical_lines[:, 3]),
        np.maximum(vertical_lines[:, 0], vertical_lines[:, 2]),
        np.maximum(vertical_lines[:, 1], vertical_lines[:, 3])
    ], axis=1)

    x = int((bx1 + bx2) / 2)
    y = int((by1 + by2) / 2)

    top = horizontal_lines[
        (horizontal_lines[:, 1] < y) & (horizontal_lines[:, 1] > (y - t_dist)) & (horizontal_lines[:, 0] < x) & (
                    horizontal_lines[:, 2] > x)]
    top = top[top[:, 1].argmax()]

    bottom = horizontal_lines[
        (horizontal_lines[:, 1] > y) & (horizontal_lines[:, 1] < (y + t_dist)) & (horizontal_lines[:, 0] < x) & (
                    horizontal_lines[:, 2] > x)]
    bottom = bottom[bottom[:, 0].argmax()]

    right = vertical_lines[
        (vertical_lines[:, 0] > x) & (vertical_lines[:, 0] < (x + t_dist)) & (vertical_lines[:, 1] < y) & (
                    vertical_lines[:, 3] > y)]
    right = right[right[:, 0].argmin()]

    left = vertical_lines[
        (vertical_lines[:, 0] < bx1) & (vertical_lines[:, 0] > (bx1 - t_dist)) & (vertical_lines[:, 1] < y) & (
                    vertical_lines[:, 3] > y)]
    left = left[left[:, 0].argmax()]

    return np.array([top, bottom, right, left])


def get_closest_line(row, ref_lines, threshold=50):
    try:
        distances = distance.cdist([row[["x1", "y1", "x2", "y2"]].astype(int)], ref_lines, "euclidean")[0]
        line_idx = distances.argmin()

        if np.sqrt(distances[line_idx]) < threshold:
            return line_idx
    except ValueError:
        pass

    return None


def calculate_line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_total_length(lines):
    lines = np.array(lines)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

    lengths = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    total_length = lengths.sum()
    return total_length


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def do_intersect(line1, line2):
    (x1, y1, x2, y2) = line1
    (x3, y3, x4, y4) = line2

    A = (x1, y1)
    B = (x2, y2)
    C = (x3, y3)
    D = (x4, y4)

    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def find_intersection_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None

    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    if (min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6 and
        min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6 and
        min(x3, x4) - 1e-6 <= x <= max(x3, x4) + 1e-6 and
        min(y3, y4) - 1e-6 <= y <= max(y3, y4) + 1e-6):
        return (x, y)
    else:
        return None


def find_intersected_lines(target_line, lines):
    intersected_indices = []
    intersection_points = []

    for idx, line in enumerate(lines):
        if do_intersect(target_line, line):
            intersected_indices.append(idx)
            intersection_points.append(find_intersection_point(target_line, line))

    return intersected_indices, intersection_points
