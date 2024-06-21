from dlblas.op_registry import OpParams, OpImpl


def get_list_op_names() -> list[str]:
    return ['matmul']


def get_args_from_op_name(name: str) -> OpParams:
    return


def get_op(name: str, params: OpParams) -> OpImpl:
    '''
    entrypoint of dlBLAS 
        based on name and params,
        return OpImpl
    '''
    return
