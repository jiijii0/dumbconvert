"""wgpu's OpenGL backend cannot share one device between render threads: it is not used."""

from unittest.mock import patch

from negpy.infrastructure.gpu import device as dev


class _Adapter:
    def __init__(self, backend):
        self.info = {"backend_type": backend, "adapter_type": "CPU"}
        self.summary = f"llvmpipe ({backend})"
        self.devices = 0

    def request_device_sync(self):
        self.devices += 1
        raise AssertionError("an OpenGL device must not be made")


def _fresh(adapter):
    with (
        patch.object(dev.wgpu.gpu, "request_adapter_sync", return_value=adapter),
        patch.object(dev, "warm_wgpu_cffi_types", lambda: None),
        patch.object(dev.GPUDevice, "_instance", None),
    ):
        return dev.GPUDevice()


def test_an_opengl_adapter_counts_as_no_gpu():
    d = _fresh(_Adapter("OpenGL"))
    assert not d.is_available
    assert d.adapter is None


def test_other_backends_are_still_used():
    assert not dev._is_opengl(_Adapter("Vulkan"))
    assert not dev._is_opengl(_Adapter("Metal"))
    assert dev._is_opengl(_Adapter("OpenGL"))
