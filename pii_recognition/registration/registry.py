from typing import Dict, Generic, Optional, Type, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Registry(dict, Generic[T_co]):
    def register(self, item: Type[T_co], name: Optional[str] = None):
        if name:
            self[name] = item
        else:
            self[getattr(item, "__name__")] = item

    def create_instance(self, name: str, config: Optional[Dict] = None) -> T_co:
        if config is None:
            config = {}

        return self[name](**config)
