import os
import sys
import threading
from typing import Any
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from insightface.app.common import Face
from insightface.model_zoo import SCRFD
from insightface.model_zoo import ArcFaceONNX
from insightface.model_zoo import Landmark
from insightface.model_zoo.model_zoo import PickableInferenceSession
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from interfaces import IMarkup
import utils

THRESH = 0.8


class FacesMarkup(IMarkup):
    def __init__(
        self,
        repo,
        detector_name,
        detector_weights,
        alignment_weights=None,
        rec_weights=None,
        insightface_root="~/insightface",
        providers=None,
        thresh=THRESH,
        distance_thresh=0.8,
        gpu_id=0,
        logger=None
    ):
        self.detector_name = detector_name
        self.detector_weights = detector_weights
        self.alignment_weights = alignment_weights
        self.rec_weights = rec_weights
        self.insightface_root = insightface_root
        self.providers = ["CUDAExecutionProvider"]
        if providers is not None:
            self.providers = providers
        self.thresh = thresh
        self.distance_thresh = distance_thresh
        self.gpu_id = gpu_id
        self.logger = logger
        if logger is None:
            self.logger = utils.init_logger("markup")
        self.mutex = threading.Lock()
        self.repo = repo
        self._init_models()

    def markup(
        self,
        image: np.ndarray,
        meta: Optional[Dict[Any, Any]] = None,
    ) -> List[Tuple[Any, List[float]]]:
        """
        Args:
            image: Image to mark
            meta: Meta information about the students to be marked,
                e.g. course, group etc.
            result: Path to output image file with markup.
        Returns:
            List[ ( primary key, [ x, y ,w, h ] ) ].
        """
        self.mutex.acquire()
        if meta is None:
            meta = {}

        markup = []
        faces = self._get_faces(image)
        self.logger.info("find {} faces.".format(len(faces)))

        if len(faces):
            if self.rec_weights:
                key2photos = self.repo.get_all_student_photos()
                key2faces = self._get_db_faces(key2photos)
                distance_matrix = self._compute_distance_matrix(
                    faces, key2faces.values()
                )
                print(distance_matrix)
                face_idxs, db_face_idxs = linear_sum_assignment(distance_matrix)
                print(face_idxs, db_face_idxs)
                num_of_not_found_faces = len(faces) - len(face_idxs)
                print(num_of_not_found_faces)

                keys = list(key2faces.keys())
                print(keys)
                for face_idx, db_face_idx in zip(face_idxs, db_face_idxs):
                    score = distance_matrix[face_idx, db_face_idx]
                    key = keys[db_face_idx]
                    if score > self.distance_thresh:
                        key = None
                    markup.append((key, faces[face_idx].bbox))

                if num_of_not_found_faces:
                    for i in range(len(faces)):
                        if i not in face_idxs:
                            markup.append((None, faces[i].bbox))

        print(markup)
        self.mutex.release()
        return markup

    @staticmethod
    def _make_cosine_distance(xface: Face, yfaces: List[Face]) -> float:
        mean_distance = 0
        for yface in yfaces:
            mean_distance += distance.cosine(xface.embedding, yface.embedding)
        mean_distance /= len(yfaces)
        return mean_distance

    def _compute_distance_matrix(self, image_faces, db_faces):
        distance_matrix = np.zeros((len(image_faces), len(db_faces)))
        for i, img_face in enumerate(image_faces):
            for j, db_face in enumerate(db_faces):
                distance_matrix[i, j] = self._make_cosine_distance(img_face, db_face)
        return distance_matrix

    def _get_db_faces(
        self, number2photos: Dict[int, List[str]]
    ) -> Dict[int, List[Face]]:
        number2faces = {}
        for number, photos in number2photos.items():
            for photo in photos:
                faces = self._get_faces(cv2.imread(photo))

                if len(faces) == 0:
                    self.logger.warning("photo {} does not contain any faces.".format(photo))
                    continue

                if len(faces) > 1:
                    self.logger.warning("photo {} contains multiple faces.".format(photo))
                    continue

                if number not in number2faces:
                    number2faces[number] = []
                number2faces[number].append(faces[0])

        return number2faces

    def _get_faces(self, image: np.ndarray) -> List[Face]:
        bboxes, kpss = self.det_model.detect(image, self.thresh)
        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            if hasattr(self, "alignment_model"):
                self.alignment_model.get(image, face)
            if hasattr(self, "rec_model"):
                self.rec_model.get(image, face)
            faces.append(face)
        return faces

    def _init_models(self) -> NoReturn:
        detector_name = self.detector_name.lower()
        if detector_name == "retinaface":
            det_dir = os.path.dirname(self.detector_weights)
        else:
            det_dir = self.detector_weights
        paths_to_check = [
            det_dir,
            self.alignment_weights,
            self.rec_weights,
            self.insightface_root,
        ]
        utils.ensure_available(paths_to_check)

        if detector_name == "retinaface":
            from detection.retinaface.retinaface import RetinaFace
            self.det_model = RetinaFace(
                self.detector_weights, 0, self.gpu_id, "net3"
            )
        elif detector_name == "scrfd":
            detection_session = PickableInferenceSession(
                self.detector_weights, providers=self.providers
            )
            self.det_model = SCRFD(
                self.detector_weights, session=detection_session
            )
            self.det_model.prepare(self.gpu_id)
        else:
            raise ValueError('"detector_name must be "retinaface" or "scfrd"')

        if self.alignment_weights is not None:
            alignment_session = PickableInferenceSession(
                self.alignment_weights, providers=self.providers
            )
            self.alignment_model = Landmark(
                self.alignment_weights, session=alignment_session
            )
            self.alignment_model.prepare(self.gpu_id)
        if self.rec_weights is not None:
            recognition_session = PickableInferenceSession(
                self.rec_weights, providers=self.providers
            )
            self.rec_model = ArcFaceONNX(
                self.rec_weights, session=recognition_session
            )
            self.rec_model.prepare(self.gpu_id)


if __name__ == "__main__":
    import argparse

    from repos.folder import FolderStudentsRepo

    INSIGHTFACE_ROOT = 'students-markup/face'
    sys.path.insert(0, INSIGHTFACE_ROOT)
    sys.path.append(os.path.join(INSIGHTFACE_ROOT, "detection", "retinaface"))

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("image", type=str, help="path to image with faces")
        args = parser.parse_args()
        return args

    args = parse_args()

    MODELS_DIR = "models"
    DATABASE_DIR = "database"

    RETINA_FACE_WEIGHTS = os.path.join("retinaface", "R50")
    SCRFD_WEIGHTS = "scrfd_10g_bnkps.onnx"
    ALIGNMENT_WEIGHTS = "2d106det.onnx"
    RECOGNITION_WEIGHTS = "glintr100.onnx"

    image_path = args.image
    image = cv2.imread(image_path)

    detection_path = os.path.join(MODELS_DIR, "detection", RETINA_FACE_WEIGHTS)
    alignment_path = os.path.join(MODELS_DIR, "alignment", ALIGNMENT_WEIGHTS)
    recognition_path = os.path.join(MODELS_DIR, "recognition", RECOGNITION_WEIGHTS)

    local_repo = FolderStudentsRepo(DATABASE_DIR)
    markup = FacesMarkup(
        local_repo,
        "retinaface",
        detection_path,
        alignment_path,
        recognition_path,
        INSIGHTFACE_ROOT,
        distance_thresh=0.8
    )
    rec_numbers = markup.markup(image)

    from utils import save_image_with_faces
    labels = [x[0] for x in rec_numbers]
    for i in range(len(labels)):
        if labels[i] is not None:
            labels[i] = " ".join(labels[i].split('_'))
    bboxes = [x[1] for x in rec_numbers]
    save_image_with_faces("makrup_result.jpg", image, [Face(bbox=x) for x in bboxes], labels=labels)
