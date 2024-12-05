import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    project: str
    data: str
    home: str = os.path.expanduser("~")

    def join(self, *args):
        return os.path.join(*args)
