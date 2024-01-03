from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Any
from common import Student
from typing import Optional


class IStudentsRepo(ABC):
    @abstractmethod
    def get_all_student_photos(self) -> Dict[int, List[str]]:
        """Returns Dict[record number, list of image paths]"""
        raise NotImplementedError()

    @abstractmethod
    def get_student_by_id(self, id: Optional[Any]) -> Optional[Student]:
        """Returns student by his id."""
        raise NotImplementedError()
