from abc import ABCMeta, abstractmethod
from typing import TypeVar

T = TypeVar('T')


class Registry(dict, metaclass=ABCMeta):
    def __init__(self):
        self.add_predefines()

    @abstractmethod
    def add_predefines(self):
        ...

    def add_item(self, item: T):
        name = getattr(item, '__name__')
        self[name] = item
