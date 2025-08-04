from typing import Iterable
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
from gradio.themes import Color


class FVoiceTheme(Soft):  # Subclase personalizada del tema base
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = Color(
            c50="#29104A", c100="#FCEFEA", c200="#29104A", c300="#29104A",
            c400="#FFE3D8", c500="#E3B6B1", c600="#29104A", c700="#0E000F",
            c800="#150016", c900="#522C5D", c950="#0E000F"
        ),
        # secondary_hue: colors.Color | str = Color(
        #     c50="#FF0000", c100="#00FF00", c200="#0000FF", c300="#FF0000",
        #     c400="#00FF00", c500="#0000FF", c600="#FF0000", c700="#00FF00",
        #     c800="#0000FF", c900="#FF0000", c950="#00FF00"
        # ),
        neutral_hue: colors.Color | str = Color(
            c50="#29104A", c100="#FCEFEA", c200="#29104A", c300="#29104A",
            c400="#FFE3D8", c500="#E3B6B1", c600="#29104A", c700="#0E000F",
            c800="#150016", c900="#522C5D", c950="#0E000F"
        ),
        # spacing_size: sizes.Size | str = sizes.spacing_md,
        # radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            # secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            # spacing_size=spacing_size,
            # radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
