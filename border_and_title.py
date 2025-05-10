from title_block import *
from scipy.spatial import distance_matrix


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

    point_y_1, point_y_2 = detect_points(words.loc[words.value.isin(["1", "2", "3", "4", "5", "6", "7", "8"])][["y1", "y2"]].astype(int))
    point_x_1, point_x_2 = detect_points(words.loc[words.value.isin(["A", "B", "C", "D", "F", "G", "H", "I"])][["x1", "x2"]].astype(int))


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
