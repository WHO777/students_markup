import numpy as np
from interfaces import IMarkup


class MockMarkup(IMarkup):
    def __init__(self, repo, seed=7, max_faces=100, unknown_prob=0.1):
        self._repo = repo
        self._max_faces = max_faces
        assert 0 <= unknown_prob <= 1
        self._unknown_prob = unknown_prob
        np.random.seed(seed)

    def markup(self, image, meta=None):
        keys = list(self._repo.get_all_student_photos().keys())
        num_of_faces = len(keys)
        face_idxs = list(range(num_of_faces))
        face_idxs.sort()

        w, h, _ = image.shape
        fake_faces = []
        for idx in face_idxs:
            key = keys[idx]
            if np.random.uniform() < self._unknown_prob:
                key = None
            x1, x2 = self._create_range(0, w)
            y1, y2 = self._create_range(0, h)
            bbox = np.array([x1, y1, x2, y2])
            fake_faces.append((key, bbox))
        return fake_faces

    @staticmethod
    def _create_range(min_, max_):
        x, y = 0, 0
        while x == y:
            x = np.random.randint(min_, max_)
            y = np.random.randint(min_, max_)
            if y < x:
                x, y = y, x
        return x, y