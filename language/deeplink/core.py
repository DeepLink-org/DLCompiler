class inline_lambda:
    """
    Inline a lambda function into the current block.
    """

    def __init__(self, node, closure_values, closure_names, arg_names):
        self.node = node
        self.closure_values = closure_values
        self.closure_names = closure_names
        self.arg_names = arg_names

    def __call__(self, *args, generator=None):
        if generator is None:
            raise RuntimeError("Generator must be provided for Lambda inlining")

        old_lscope = generator.lscope.copy()
        old_local_defs = generator.local_defs.copy()
        old_insert_block = generator.builder.get_insertion_block()
        try:
            closure_map = {}
            for name, value in zip(self.closure_names, self.closure_values):
                closure_map[name] = value

            param_map = {}
            for name, value in zip(self.arg_names, args):
                param_map[name] = value

            generator.lscope = {**old_lscope, **closure_map, **param_map}
            generator.local_defs = {**old_local_defs, **closure_map, **param_map}

            return generator.visit(self.node.body)
        finally:
            generator.lscope = old_lscope
            generator.local_defs = old_local_defs
            if old_insert_block:
                generator.builder.set_insertion_point_to_end(old_insert_block)
