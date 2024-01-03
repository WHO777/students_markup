from typing import Dict
from typing import List
from typing import Union

import psycopg2

from interfaces import IStudentsRepo


class PostgreStudentsRepo(IStudentsRepo):
    def __init__(
        self,
        database: str,
        user: str,
        password: str,
        host: str = "localhost",
        port: Union[str, int] = 5432,
    ):
        self.conn = psycopg2.connect(
            database=database,
            user=user,
            host=host,
            password=password,
            port=str(port),
        )

    def get_all_students(self) -> Dict[int, List[str]]:
        """Returns Dict[record number, list of image paths]"""
        cur = self.conn.cursor()
        cur.execute("SELECT students_id,photo_link FROM students_db")
        # REVIEW: use dictionary comprehension
        students = {}
        for row in cur:
            students[row[0]] = [row[1]]
        cur.fetchall()
        return students


if __name__ == "__main__":
    Repo = PostgreStudentsRepo(
        "postgres", "postgres", "localhost", "0", "5432"
    )
    students = Repo.get_all_students()
    print(students)

    # conn = psycopg2.connect(
    #     database="postgres",
    #     user="postgres",
    #     host="localhost",
    #     password="0",
    #     port="5432"
    # )
