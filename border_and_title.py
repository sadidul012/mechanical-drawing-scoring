from scoring import detect_objects
from title_block import *
from scipy.spatial import distance_matrix


y_index_titles = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
x_index_titles = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


def detect_points(points, threshold=10, close_n=3):
    try:
        dists = distance_matrix(points, points)
        close_counts = (dists < threshold).sum(axis=1) - 1
        point_means = points.mean(axis=1)
        point_mean = point_means.mean()

        point_1 = points.loc[(point_means < point_mean - (point_mean/2)) & (close_counts >= close_n)].values
        point_1 = (point_1[:, 0].min(), point_1[:, 1].max())

        point_2 = points.loc[(point_means > point_mean + (point_mean/2)) & (close_counts >= close_n)].values
        point_2 = (point_2[:, 0].min(), point_2[:, 1].max())
    except ValueError:
        point_1, point_2 = None, None

    return point_1, point_2


def detect_borders_with_contours(contours, sorted_indices):
    bb_1 = cv2.boundingRect(contours[sorted_indices[0]])
    bb_2 = cv2.boundingRect(contours[sorted_indices[1]])
    x, y, w, h = bb_1
    bb_1 = (x, y), (x + w, y + h)
    x, y, w, h = bb_2
    bb_2 = (x, y), (x + w, y + h)

    return bb_1, bb_2


def detect_borders(contours, sorted_indices, words):
    cbb_1, cbb_2 = detect_borders_with_contours(contours, sorted_indices)

    point_y_1, point_y_2 = detect_points(words.loc[words.value.isin(y_index_titles)][["y1", "y2"]].astype(int))
    point_x_1, point_x_2 = detect_points(words.loc[words.value.isin(x_index_titles)][["x1", "x2"]].astype(int))


    if point_x_1 is None or point_y_1 is None or point_x_2 is None or point_y_2 is None:
        bb_1 = cbb_1
        bb_2 = cbb_2
    else:
        bb_1 = (point_x_1[0], point_y_1[0]), (point_x_2[1], point_y_2[1])
        bb_2 = (point_x_1[1], point_y_1[1]), (point_x_2[0], point_y_2[0])

    a_bb_1 = calculate_area(bb_1)
    a_bb_2 = calculate_area(bb_2)

    if a_bb_1 * 0.85 > a_bb_2:
        bb_2 = (
            int(bb_1[0][0] + 30),
            int(bb_1[0][1] + 30)
        ), (
            int(bb_1[1][0] - 30),
            int(bb_1[1][1] - 30)
        )

    return bb_1, bb_2


def get_boundary(border_1, border_2, inner_border_lines):
    (x1, y1), (x2, y2) = border_1
    border_1_lines = [
        [x1, y2, x2, y2],  # Bottom
        [x1, y1, x2, y1],  # Top
        [x2, y2, x2, y1],  # Right
        [x1, y2, x1, y1],  # Left
    ]

    (x1, y1), (x2, y2) = border_2
    border_2_lines = [
        [x1, y2, x2, y2],  # Bottom
        [x1, y1, x2, y1],  # Top
        [x2, y2, x2, y1],  # Right
        [x1, y2, x1, y1],  # Left
    ]

    all_lines = [
        border_2_lines,
        border_1_lines
    ]
    if len(inner_border_lines) == 4:
        all_lines.append(inner_border_lines)

    boundary = [[0, 0], [0, 0]]
    for i, line in enumerate(zip(*all_lines)):
        line = np.array(line)

        if i == 0:
            x1, y1, x2, y2 = line[:, 0].min(), line[:, 1].min(), line[:, 2].max(), line[:, 3].min()
            boundary[0][1] = y1
        elif i == 1:
            x1, y1, x2, y2 = line[:, 0].min(), line[:, 1].max(), line[:, 2].max(), line[:, 3].max()
            boundary[1][1] = y2
        elif i == 2:
            x1, y1, x2, y2 = line[:, 0].min(), line[:, 1].max(), line[:, 2].min(), line[:, 3].min()
            boundary[1][0] = x2
        elif i == 3:
            x1, y1, x2, y2 = line[:, 0].max(), line[:, 1].max(), line[:, 2].max(), line[:, 3].min()
            boundary[0][0] = x1

    return boundary


def find_connected_lines_recursive(target_line, lines, tolerance=5):
    def is_connected(line_a, line_b):
        xa1, ya1, xa2, ya2 = line_a
        xb1, yb1, xb2, yb2 = line_b
        endpoints_a = [(xa1, ya1), (xa2, ya2)]
        endpoints_b = [(xb1, yb1), (xb2, yb2)]
        for (ax, ay) in endpoints_a:
            for (bx, by) in endpoints_b:
                if abs(ax - bx) <= tolerance and abs(ay - by) <= tolerance:
                    return True
        return False

    visited = set()
    to_visit = []

    for idx, line in enumerate(lines):
        if is_connected(target_line, line):
            to_visit.append(idx)
            visited.add(idx)

    while to_visit:
        current_idx = to_visit.pop()
        current_line = lines[current_idx]

        for idx, line in enumerate(lines):
            if idx not in visited and is_connected(current_line, line):
                to_visit.append(idx)
                visited.add(idx)

    return list(visited)


def detect_intersection_with_borders(img, boundary, title_boundary, return_states=False):
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

    tolerance = 10
    (x1, y1), (x2, y2) = boundary.copy()
    y2 += tolerance
    y1 -= tolerance
    x2 -= tolerance
    x1 += tolerance

    boundary_lines = [
        [x1, y2, x2, y2],  # Top
        [x1, y1, x2, y1],  # Bottom
        [x2, y2, x2, y1],  # Right
        [x1, y2, x1, y1],  # Left
    ]
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
    lines = drawings.copy()
    drawings = lines.copy()
    (x1, y1), (x2, y2) = title_boundary
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    boundary_title_block_lines = [
        [left, top, right, top],  # Top
        [left, bottom, left, top],  # Left
        [right, bottom, right, top],  # Right
        # [left, bottom, right, bottom],  # Bottom
    ]

    drawings = drawings[
        ~((drawings[:, 1] >= top - tolerance) & (drawings[:, 3] >= top - tolerance) & (drawings[:, 0] >= left - tolerance) & (drawings[:, 2] >= left - tolerance))
    ]
    lines = drawings.copy()

    intersected_lines = []
    intersected_points = []
    top_intersected, intersected_at = find_intersected_lines(boundary_lines[0], lines)
    top_intersected = lines[top_intersected]
    intersected_lines.extend(top_intersected)
    intersected_points.extend(intersected_at)

    bottom_intersected, intersected_at = find_intersected_lines(boundary_lines[1], lines)
    bottom_intersected = lines[bottom_intersected]
    intersected_lines.extend(bottom_intersected)
    intersected_points.extend(intersected_at)

    right_intersected, intersected_at = find_intersected_lines(boundary_lines[2], lines)
    right_intersected = lines[right_intersected]
    intersected_lines.extend(right_intersected)
    intersected_points.extend(intersected_at)

    left_intersected, intersected_at = find_intersected_lines(boundary_lines[3], lines)
    left_intersected = lines[left_intersected]
    intersected_lines.extend(left_intersected)
    intersected_points.extend(intersected_at)

    title_top_intersected, intersected_at = find_intersected_lines(boundary_title_block_lines[0], lines)
    title_top_intersected = lines[title_top_intersected]
    intersected_lines.extend(title_top_intersected)
    intersected_points.extend(intersected_at)

    title_left_intersected, intersected_at = find_intersected_lines(boundary_title_block_lines[1], lines)
    title_left_intersected = lines[title_left_intersected]
    intersected_lines.extend(title_left_intersected)
    intersected_points.extend(intersected_at)

    title_right_intersected, intersected_at = find_intersected_lines(boundary_title_block_lines[2], lines)
    title_right_intersected = lines[title_right_intersected]
    intersected_lines.extend(title_right_intersected)
    intersected_points.extend(intersected_at)

    intersected_lines = np.array(intersected_lines)
    intersected_points = np.array(intersected_points)
    connected_lines = []
    for intersected_line in intersected_lines:
        x1, y1, x2, y2 = intersected_line
        connected = find_connected_lines_recursive([x1, y1, x2, y2], lines)
        connected_lines.extend(connected)

    if return_states:
        return lines[list(set(connected_lines))], boundary_lines, boundary_title_block_lines, intersected_points

    return lines[list(set(connected_lines))]


def detect_intersected_texts(words, boundary_lines, boundary_title_block_lines):
    words_copy = words.copy()

    words_copy = words_copy.loc[
        ~words.value.isin(x_index_titles) &
        ~words.value.isin(y_index_titles) &
        (
            (words.x1 < boundary_title_block_lines[1][0])  # left
            | (words.y1 < boundary_title_block_lines[0][1])  # left
        )
    ]
    words_copy = words_copy.loc[
        (words.x1 < boundary_lines[3][0]) & (words.x2 > boundary_lines[3][2])  # left
        | (words.x1 < boundary_lines[2][0]) & (words.x2 > boundary_lines[2][2])  # right
        | (words.y1 < boundary_lines[1][1]) & (words.y2 > boundary_lines[1][3])  # top
        | (words.y1 < boundary_lines[0][1]) & (words.y2 > boundary_lines[0][3])  # bottom
    ]

    return words_copy
