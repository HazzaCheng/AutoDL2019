from enum import Enum, unique


def delete():
    pass


@unique
class Op(Enum):
    Delete = delete
    Terminate = None
