import json
import os
import os.path as osp
from glob import glob
from common import Student

from interfaces import IStudentsRepo


class FolderStudentsRepo(IStudentsRepo):
    def __init__(self, db_path: str):
        self.db_path = osp.abspath(osp.expanduser(db_path))
        assert osp.exists(self.db_path), "db path should exist"

    def get_all_student_photos(self):
        number2photos = {}
        names = os.listdir(self.db_path)
        for name in names:
            images_path = osp.join(self.db_path, name, "images")
            photos = glob(images_path + '/*')
            info_path = osp.join(self.db_path, name, "info.json")
            assert osp.exists(info_path), '"info.json" not found'
            with open(info_path, "r", encoding="UTF-8") as f:
                info = json.load(f)
            record_number = info["id"]
            number2photos[record_number] = photos
        return number2photos

    def get_student_by_id(self, id):
        if id is None:
            return id
        names = os.listdir(self.db_path)
        for name in names:
            info_path = osp.join(self.db_path, name, "info.json")
            assert osp.exists(info_path), '"info.json" not found'
            with open(info_path, "r", encoding="UTF-8") as f:
                info = json.load(f)
            if info['id'] == id:
                return Student.from_dict(info)
        return None
