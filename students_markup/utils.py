import os
import sys
import cv2
import numpy as np
from typing import List
from typing import Optional
from typing import Text
from typing import NoReturn
import logging
from insightface.app.common import Face


def save_image_with_faces(filename,
                          image: np.ndarray,
                          faces: List[Face],
                          labels=Optional[List[Text]],
                          landmarks=False,
                          color=(0, 255, 0),
                          miss_color=(0, 0, 255),
                          miss_label="") -> NoReturn:
    if labels is not None:
        assert len(faces) == len(labels)
        for (face, label) in zip(faces, labels):
            bbox = face.bbox.astype(np.int)
            landmarks = face.kps if landmarks else None

            color_copy = color
            if label is None:
                color_copy = miss_color
                label = miss_label
            _draw_single_face(image, bbox, label, landmarks, color_copy)
    else:
        for face in faces:
            bbox = face.bbox.astype(np.int)
            landmarks = face.kps if landmarks else None
            _draw_single_face(image, bbox, landmarks=landmarks, color=color)
    cv2.imwrite(filename, image)


def _draw_single_face(image,
                      bbox,
                      label=None,
                      landmarks=None,
                      color=(255, 0, 0)) -> NoReturn:
    cv2.rectangle(
        image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4
    )
    if label is not None:
        cv2.putText(
            image,
            str(label),
            (bbox[0], bbox[1] - 5),
            color=color,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=0.8,
            thickness=1,
        )
    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            cv2.circle(image, (landmarks[i][0], landmarks[i][1]), 10, color, -1)


def ensure_available(paths_to_check: List[str]) -> NoReturn:
    for path in paths_to_check:
        if not os.path.exists(path) and path is not None:
            raise RuntimeError("{} does not exists !".format(path))


def init_logger(name,
                level=logging.INFO,
                format='[%(name)s %(levelname)s] %(message)s'):
    logger = logging.Logger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
