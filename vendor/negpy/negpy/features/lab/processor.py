import negpy_core
import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    chroma_denoise_radius,
    apply_glow_and_halation,
    apply_lab_sharpening,
    apply_exposure,
    apply_hsl,
    apply_tone,
    apply_wb,
    apply_saturation,
    auto_masking,
)
from negpy.features.lab.models import LabConfig, hsl_active


class PhotoLabProcessor:
    def __init__(self, config: LabConfig, mono: bool = False):
        self.config = config
        self.mono = mono

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        Apply effects from logic.py in sequence
        """
        img = image

        chroma_radius = chroma_denoise_radius(self.config.chroma_denoise, self.mono)
        if chroma_radius > 0:
            img = apply_chroma_denoise(img, chroma_radius, context.scale_factor)

        if self.config.saturation != 1.0 or self.config.skin_protection > 0:
            img = apply_saturation(img, self.config.saturation, self.config.skin_protection)


        if self.config.sharpen > 0:
            auto = auto_masking(img, self.config)
            if auto is not None:
                context.metrics["auto_masking"] = auto
            img = apply_lab_sharpening(img, self.config, auto)

        if self.config.glow_amount > 0 or self.config.halation_strength > 0:
            img = apply_glow_and_halation(
                img,
                self.config.glow_amount,
                self.config.halation_strength,
                bw=self.mono,
            )

        c = self.config                                # last, on the finished positive (lab.wgsl too)
        if not self.mono and (c.wb_temp or c.wb_tint):
            img = apply_wb(img, c.wb_temp, c.wb_tint)
        if c.exposure_ev != 0.0:
            img = apply_exposure(img, c.exposure_ev)
        if c.tone_highlights or c.tone_shadows or c.tone_whites or c.tone_blacks:
            img = apply_tone(img, c.tone_highlights, c.tone_shadows, c.tone_whites, c.tone_blacks)
        if not self.mono and hsl_active(c):            # after sharpening: lab.wgsl's sharpen state holds L*
            img = apply_hsl(img, c.hsl_hue, c.hsl_saturation, c.hsl_luminance)

        return negpy_core.clip01(np.ascontiguousarray(img, dtype=np.float32))
