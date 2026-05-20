"""
DICP NPU builder utilities for unified builder interface.
"""

__all__ = [
    "create_builder_method_wrapper",
    "attach_builder_methods",
    "setup_unified_builder",
]


def create_builder_method_wrapper(main_builder, delegate_builder, method_name):
    delegate_method = getattr(delegate_builder, method_name)

    def wrapper(*args, **kwargs):
        saved_ip = main_builder.get_insertion_point()
        saved_loc = main_builder.get_loc()
        delegate_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            delegate_builder.set_loc(saved_loc)
        result = delegate_method(*args, **kwargs)
        main_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            main_builder.set_loc(saved_loc)
        return result

    wrapper.__name__ = method_name
    wrapper.__doc__ = getattr(delegate_method, "__doc__", None)
    return wrapper


def attach_builder_methods(main_builder, delegate_builder, method_names):
    for method_name in method_names:
        wrapper = create_builder_method_wrapper(
            main_builder, delegate_builder, method_name
        )
        setattr(main_builder, method_name, wrapper)


def setup_unified_builder(main_builder, dicp_builder):
    main_builder._dicp_builder = dicp_builder
    dicp_methods = [
        "create_scope_op",
        "scope_return",
        "get_t_core_type_attr_name",
        "get_t_core_type_cube_attr",
        "get_t_core_type_vector_attr",
        "get_target_attribute",
        "create_get_sub_vec_id",
        "create_copy_buffer",
        "create_copy_tensor",
        "create_fixpipe",
        "create_bind_buffer",
        "create_debug_barrier",
        "is_910_95",
        "sync_block_set",
        "sync_block_wait",
        "create_convert_layout",
        "sync_block_all",
        "get_int_attr",
        "get_str_array_attr",
        "get_i64_array_attr",
        "get_core_type_attr",
        "get_pipe_attr",
        "get_vf_mode_attr",
        "get_iterator_types_attr",
        "parse_attr",
        "get_affine_map_attr",
        "get_affine_map_array_attr",
        "get_buffer_ty_with_affine_map",
        "create_custom_op",
    ]
    attach_builder_methods(main_builder, dicp_builder, dicp_methods)
