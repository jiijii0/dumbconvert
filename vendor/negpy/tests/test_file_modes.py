"""Files NegPy writes through a temporary name take the umask's mode, not tempfile's 0600."""

import os
import pathlib
import stat

import pytest

from negpy.kernel.system import paths
from negpy.kernel.system.paths import apply_umask_mode

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX modes")

EXPECTED = 0o666 & ~paths._UMASK


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_apply_umask_mode_opens_a_tempfile_file(tmp_path):
    import tempfile

    with tempfile.NamedTemporaryFile(dir=tmp_path, delete=False) as tmp:
        assert _mode(tmp.name) == 0o600
    apply_umask_mode(tmp.name)
    assert _mode(tmp.name) == EXPECTED


# App-internal files that stay private to their owner.
_PRIVATE = {"kernel/system/user_directory.py"}


def test_every_tempfile_rename_applies_the_umask_mode():
    """NamedTemporaryFile and mkstemp make 0600 files; mkdtemp and gettempdir make none."""
    root = pathlib.Path(paths.__file__).resolve().parents[2]
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if str(p.relative_to(root)) not in _PRIVATE
        and any(maker in (text := p.read_text(encoding="utf-8")) for maker in ("NamedTemporaryFile(", "mkstemp("))
        and any(renamer in text for renamer in ("os.replace(", "replace_file("))
        and "apply_umask_mode(" not in text
    ]
    assert offenders == [], "call apply_umask_mode before renaming a tempfile file into place"
