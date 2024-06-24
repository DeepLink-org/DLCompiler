from dlblas.op_registry import OpParams, OpImpl, op_registry


def get_list_op_names() -> list[str]:
    return op_registry.get_list_op_names()


def get_args_from_op_name(name: str) -> list[OpParams]:
    return op_registry.get_args_from_op_name(name)


def get_op(name: str, params: OpParams) -> OpImpl:
    '''
    based on name and params,
    return OpImpl
    '''
    return op_registry.get_op(name, params)
