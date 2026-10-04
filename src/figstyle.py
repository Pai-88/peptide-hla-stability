"""One palette for every figure in the repo.

The figures were originally drawn on near-black because the deck was dark. The
site and the README are light, so a dark figure dropped into either reads as a
hole in the page. Everything is defined once here; the generators import it.

Set FIG_THEME=dark to get the original palette back.
"""

import os

_DARK = dict(
    BG="#0d0d0f", FG="#f2f2f2", MUTED="#8a8a93", LINE="#26262c",
    WIN="#4ad3a0", SHOT="#f2b544", PLAIN="#6f7380", CTRL="#5b6070",
    ACC="#4ad3a0", WARN="#e8704a", COOL="#5aa9e6",
    GOOD="#4ad3a0", BAD="#6f7380", WARM="#e8b04b",
)

# Light, matching the site: paper, near-black ink, one brick accent, and the
# same teal and slate used for the peptide and the groove in the 3D viewers.
_LIGHT = dict(
    BG="#FAFAF8", FG="#121212", MUTED="#8E8E88", LINE="#DCDCD5",
    WIN="#1F5E58", SHOT="#5B6B82", PLAIN="#9AA0A6", CTRL="#C6C6BE",
    ACC="#1F5E58", WARN="#B33A1A", COOL="#5B6B82",
    GOOD="#1F5E58", BAD="#9AA0A6", WARM="#A8781F",
)

P = _DARK if os.environ.get("FIG_THEME", "light").lower() == "dark" else _LIGHT
globals().update(P)
THEME = "dark" if P is _DARK else "light"
