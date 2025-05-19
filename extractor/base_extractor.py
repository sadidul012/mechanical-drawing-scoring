from extractor.rn_vgg_ocr import OCR

class BaseExtractor(OCR):
    def __init__(self, debug=False):
        super().__init__(straighten_pages=False, debug=debug)
        self.debug = debug

