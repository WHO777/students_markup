from typing import Text
from typing import Optional
from typing import Tuple
from typing import NamedTuple
from enum import Enum


class DEGREES(Enum):
    BACHELOR = 'bachelor'
    MASTER = 'master'


class Student(NamedTuple):
    class StudentInfo(NamedTuple):
        first_name: Text
        last_name: Text
        middle_name: Optional[Text]
        group: Text
        course: int
        grade: DEGREES

        def to_dict(self):
            return dict(self._asdict())

    id: int
    first_name: Text
    last_name: Text
    middle_name: Optional[Text]
    group: Text
    course: int
    grade: DEGREES
    photo_links: Tuple[Text]

    @classmethod
    def from_dict(cls, dict):
        return cls.__new__(cls, **dict)

    # TODO: Deprecated
    def to_dict(self):
        return dict(self._asdict())

    def get_info(self):
        return Student.StudentInfo(first_name=self.first_name,
                                   last_name=self.last_name,
                                   middle_name=self.middle_name,
                                   group=self.group,
                                   course=self.course,
                                   grade=self.grade)
