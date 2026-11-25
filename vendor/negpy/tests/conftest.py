import os

import pytest

# Configure headless mode for CI/CD
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-runner"


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("metrics", "negpy performance metrics export")
    g.addoption(
        "--metrics-out",
        action="store",
        default=None,
        help="Write session metrics to this JSON path (overrides NEGPY_METRICS_OUT if set as non-empty).",
    )


@pytest.fixture(scope="session")
def qapp():
    """A Qt application, for the few tests that need one (NegPy's own app is not in this copy)."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    yield app
    app.quit()
    app.processEvents()


@pytest.fixture(autouse=True)
def _fresh_base_reads():
    """The base-read cache is process-wide and keyed by source hash, which tests reuse across images."""
    from negpy.services.rendering import base_read

    base_read.clear()
    yield
    base_read.clear()


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtestloop(session):
    """Stop background threads before pytest-cov generates its coverage report.

    trylast=True means this wrapper's post-yield runs *before* pytest-cov's,
    giving us a window to quit Qt threads and destroy the wgpu device before
    GC destroys the Qt thread wrappers — preventing the SIGABRT on CI.
    """
    yield
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.quit()
            app.processEvents()
    except Exception:
        pass
    try:
        from negpy.infrastructure.gpu.device import GPUDevice

        GPUDevice.destroy_singleton()
    except Exception:
        pass
