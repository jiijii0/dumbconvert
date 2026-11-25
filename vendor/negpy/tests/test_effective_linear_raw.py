"""The transfer path decodes without camera white balance, whatever the stored flag says.

Linear RAW is documented as inert on the E-6 as-captured path — the render applies the
camera matrix, which folds the as-shot multipliers back in itself. But the decode read the
stored flag regardless, so a hidden, stale toggle decided whether white balance was
applied, and two frames of one bracket could decode on different scales.
"""

from dataclasses import replace

import pytest

from negpy.features.process.logic import effective_linear_raw, linear_raw_token, should_fold_camera_wb
from negpy.features.process.models import ProcessConfig, ProcessMode


def cfg(mode=ProcessMode.C41, linear_raw=False, positive_source=False) -> ProcessConfig:
    return replace(ProcessConfig(), process_mode=mode, linear_raw=linear_raw, positive_source=positive_source)


class TestEffectiveLinearRaw:
    def test_transfer_path_decodes_neutral_even_with_the_flag_off(self):
        """The reported fault: the flag is hidden on this path but was still obeyed."""
        assert effective_linear_raw(cfg(ProcessMode.E6, linear_raw=False))

    def test_the_user_flag_still_wins_everywhere_else(self):
        assert effective_linear_raw(cfg(ProcessMode.C41, linear_raw=True))
        assert effective_linear_raw(cfg(ProcessMode.BW, linear_raw=True))
        assert effective_linear_raw(cfg(ProcessMode.E6, linear_raw=True, positive_source=True))

    @pytest.mark.parametrize("mode", [ProcessMode.C41, ProcessMode.BW])
    def test_nothing_else_is_forced(self, mode):
        """Only the transparency transfer is affected — every negative path keeps the
        decode it had."""
        assert not effective_linear_raw(cfg(mode, linear_raw=False))

    def test_the_token_follows_the_effective_value(self):
        """Caches key on this. Keying on the stored flag would serve a buffer decoded the
        other way round, which is invisible until the colors are wrong."""
        transfer = cfg(ProcessMode.E6, linear_raw=False)
        assert linear_raw_token(transfer) == linear_raw_token(cfg(ProcessMode.C41, linear_raw=True))
        assert linear_raw_token(transfer) != linear_raw_token(cfg(ProcessMode.C41, linear_raw=False))


class TestPositiveSource:
    """A Transparency source can be an already-finished positive (a scanned print, an
    exported TIFF) rather than a raw scanner/camera capture. Positive exempts it
    from the forced literal-linear read so its own embedded profile decides the decode.
    """

    def test_exempts_the_transfer_path(self):
        assert not effective_linear_raw(cfg(ProcessMode.E6, positive_source=True))

    def test_the_linear_raw_flag_still_wins(self):
        """An explicit request for literal linear data overrides Positive too."""
        assert effective_linear_raw(cfg(ProcessMode.E6, linear_raw=True, positive_source=True))

    def test_has_no_effect_off_the_transfer_path(self):
        """Nowhere else reads positive_source, so it must not change either negative path."""
        assert not effective_linear_raw(cfg(ProcessMode.C41, positive_source=True))
        assert not effective_linear_raw(cfg(ProcessMode.BW, positive_source=True))

    def test_default_is_off(self):
        """A frame with no saved value must keep today's forced-linear transfer decode."""
        assert not ProcessConfig().positive_source


class TestShouldFoldCameraWb:
    """The capture matrix folds the as-shot multipliers back in only where the decode left them out."""

    def test_folds_for_an_ordinary_transfer(self):
        assert should_fold_camera_wb(cfg(ProcessMode.E6, linear_raw=False))

    def test_folds_when_the_user_flag_is_on(self):
        assert should_fold_camera_wb(cfg(ProcessMode.C41, linear_raw=True))

    def test_never_folds_when_the_decode_already_carries_wb(self):
        assert not should_fold_camera_wb(cfg(ProcessMode.C41, linear_raw=False))

    def test_never_folds_when_reconstruction_bakes_white_balance_in(self):
        """An active reconstruction on the transfer path bakes real white balance into the
        decode itself (see highlight_reconstruction_bakes_wb); folding it again here would
        double-apply it, even though effective_linear_raw alone still reads True."""
        transfer_with_reconstruction = replace(cfg(ProcessMode.E6, linear_raw=False), highlight_reconstruction=5)
        assert effective_linear_raw(transfer_with_reconstruction)
        assert not should_fold_camera_wb(transfer_with_reconstruction)


class TestDecodeAndMatrixAgree:
    """The decode and the camera matrix must make the same choice.

    `camera_to_working_matrix` folds the as-shot multipliers in only for a buffer decoded
    without them. Apply white balance at one and not the other and the render is tinted by
    the raw green-to-red ratio, roughly 2:1.
    """

    def test_both_sides_read_one_helper(self):
        """The decode asks effective_linear_raw directly; the matrix asks should_fold_camera_wb,
        which wraps it, in density_stage, which the CPU matrix reads. Either way, both sides of
        the agreement trace back to the one flag."""
        import inspect

        from negpy.features.transparency import logic
        from negpy.features.transparency import processor as cpu
        from negpy.services.rendering import image_processor as ip

        src = inspect.getsource(ip)
        assert "effective_linear_raw" in src, "the decode no longer asks the shared helper"
        assert "should_fold_camera_wb" in inspect.getsource(logic), "the matrix no longer asks the shared helper"
        assert "density_stage(" in inspect.getsource(cpu), "the CPU matrix no longer reads density_stage"

    def test_the_gpu_matrix_agrees_with_the_cpu_one(self):
        import inspect

        from negpy.services.rendering import gpu_engine

        assert "density_stage(" in inspect.getsource(gpu_engine), "GPU matrix would drift from the CPU one"
