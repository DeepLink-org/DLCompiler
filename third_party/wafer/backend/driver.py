from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class WaferExternalDriver(DriverBase):
    @classmethod
    def is_active(cls):
        return False

    def get_current_target(self):
        return GPUTarget("wafer", 0, 32)

    def get_active_torch_device(self):
        return "cpu"

    def get_benchmarker(self):
        raise RuntimeError(
            "The wafer runtime backend is installed by setup_on_wafer.py as "
            "triton.backends.dicp_triton."
        )
