import numpy as np
from scipy.spatial import distance


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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
    top = horizontal_lines[(horizontal_lines[:, 1] < by1) & (horizontal_lines[:, 1] > (by1 - t_dist)) & (horizontal_lines[:, 0] < bx1) & (horizontal_lines[:, 2] > bx2)]
    top = top[top[:, 0].argmin()]

    bottom = horizontal_lines[(horizontal_lines[:, 1] > by1) & (horizontal_lines[:, 1] < (by1 + t_dist)) & (horizontal_lines[:, 0] < bx1) & (horizontal_lines[:, 2] > bx2)]
    bottom = bottom[bottom[:, 0].argmax()]

    right = vertical_lines[(vertical_lines[:, 0] > bx1) & (vertical_lines[:, 0] < (bx1 + t_dist)) & (vertical_lines[:, 1] > by1) & (vertical_lines[:, 3] < by2)]
    right = right[right[:, 0].argmin()]

    left = vertical_lines[(vertical_lines[:, 0] < bx1) & (vertical_lines[:, 0] > (bx1 - t_dist)) & (vertical_lines[:, 1] > by1) & (vertical_lines[:, 3] < by2)]
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
