from border_and_title import *


def generate_output(f):
    img = read_pdf(f)
    im_h, im_w, _ = img.shape
    sorted_indices, contours, hierarchy_area = detect_objects(img)
    title_contours, line_90, data = detect_probable_title_sections(img, return_states=True)
    ocr_result = data["ocr_result"]
    mask = data["mask"]
    inner_border_lines = data["inner_border_lines"]  # 0: Bottom, 1: Top, 2: Right, 3: Left
    words = process_text(ocr_result, im_h, im_w)

    border_1, border_2 = detect_borders(contours, sorted_indices, words)
    contours = detect_text_tables(img, words, mask)
    title_contours = title_contours + contours

    boundary = get_boundary(border_1, border_2, inner_border_lines)
    tables = detect_table(img, mask, words)
    tight_boxes = detect_text_area(words, tables, boundary, im_w, im_h)
    # title_boundary = get_title_boundary(boundary, line_90, title_contours, words, im_h)

    title_blocks, border_lines = detect_boundary_text_block(img, tight_boxes, words, boundary, title_contours, line_90)

    return dict(
        img=img,
        mask=mask,
        boundary=boundary,
        title_blocks=title_blocks,
        im_h=im_h,
        im_w=im_w,
        words=words,
        border_lines=border_lines,
    )


if __name__ == "__main__":
    data = generate_output("data/original/Good/300-014823.pdf")
    print(data)
