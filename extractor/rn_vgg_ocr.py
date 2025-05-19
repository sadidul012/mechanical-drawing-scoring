import torch
from doctr.models import detection_predictor, recognition_predictor
from doctr.models.predictor import OCRPredictor
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class OCR:
    def __init__(self, straighten_pages=False, debug=False):
        det_arch = 'db_resnet50'
        reco_arch = 'crnn_vgg16_bn'
        pretrained = True
        self.debug = debug

        det_predictor = detection_predictor(
            det_arch,
            pretrained=pretrained,
        )

        reco_predictor = recognition_predictor(
            reco_arch,
            pretrained=pretrained,
        )

        self.model = OCRPredictor(
            det_predictor,
            reco_predictor,
            straighten_pages=straighten_pages
        )

        if torch.cuda.is_available():
            if self.debug:
                print("Using GPU")
            device = torch.device("cuda:0")
            self.model.to(device)

        self.words = None
        self.lines = None

    def json_to_dataframe(self, result, im_w, im_h):
        df = pd.DataFrame.from_dict(result.export())
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
        self.lines = lines.rename(columns={'geometry': 'line_geometry'})
        words = self.lines.explode("words")
        words['word_idx'] = np.arange(words.shape[0])
        words['index'] = np.arange(words.shape[0])
        words = words.set_index('index')

        words = words.join(pd.json_normalize(words.pop('words')), lsuffix='.words')
        words = words.rename(columns={'geometry': 'word_geometry'})

        words = words.dropna(subset=['word_geometry'])
        words["word_geometry"] = words.word_geometry.apply(
            lambda x: {"x1": x[0][0], "y1": x[0][1], "x2": x[1][0], "y2": x[1][1]}
        )

        words["x1"] = words["x1"] * im_w
        words["x2"] = words["x2"] * im_w
        words["y1"] = words["y1"] * im_h
        words["y2"] = words["y2"] * im_h

        self.words = words.join(pd.json_normalize(words.pop('word_geometry')))

    def from_image(self, image):
        try:
            doc = [image]
            im_w, im_h = image.shape[1], image.shape[0]
            result = self.model(doc)
            self.json_to_dataframe(result, im_w, im_h)
        except Exception as e:
            if self.debug:
                print(e)
