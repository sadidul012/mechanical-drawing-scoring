import cv2
import numpy as np
from skimage.metrics import structural_similarity


def find_common_region(images, threshold_value=30):
    img_base = images[0]
    img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    common_mask = np.ones_like(img_base_gray, dtype=np.uint8) * 255

    for img in images[1:]:

        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
