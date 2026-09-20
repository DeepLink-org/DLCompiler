from triton.backends.compiler import BaseBackend


class WaferExternalBackend(BaseBackend):
    binary_ext = "o"

    @classmethod
    def supports_target(cls, target):
        return False

    def hash(self):
        return "wafer-external"

    def parse_options(self, options):
        return options

    def add_stages(self, stages, options):
        raise RuntimeError(
            "The wafer runtime backend is installed by setup_on_wafer.py as "
            "triton.backends.dicp_triton."
        )

    def load_dialects(self, context):
        pass

    def get_module_map(self):
        return {}
