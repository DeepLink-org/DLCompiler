from dataclasses import dataclass
from typing import Any

@dataclass
class OpParams:
    name: str
    shapes: list

@dataclass
class OpImpl:
    name : str

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


def get_list_op_names() -> list[str]:
    return 

def get_args_from_op_name(name: str) -> OpParams:
    return


def get_op(name: str, params: OpParams) -> OpImpl:
    '''
    entrypoint of dlBLAS 
        based on name and params,
        return OpImpl
    '''
    return 
