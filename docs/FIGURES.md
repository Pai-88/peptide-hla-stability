# Structure figures — peptide/HLA-A*02:01, FRIDAYMERCHANTS

Two static PNGs. Pre-rendered, no viewer, no interactive widget, no web embed.
Rendered headlessly with open-source PyMOL (OSMesa software rendering); residue labels
are composited afterwards at controlled pixel positions, so no label sits on top of
another or on top of the highlighted residue. Neither image contains a burned-in
title — the title belongs on the slide.

## Structure used

**PDB 1DUZ** — human class I MHC **HLA-A\*02:01** (heavy chain + beta-2-microglobulin)
in complex with the 9-mer HTLV-1 Tax peptide **LLFGYPVYV**. X-ray, **1.80 A**.
Chain A = HLA heavy chain, chain C = peptide, numbered 1 to 9 in the file, so the
crystallographic numbering and our P1 to P9 numbering agree.

Why not 4MJI: 4MJI is an HLA-**B\*51:01** complex, a different allele with a different
groove, so it cannot illustrate an A\*02:01 claim. 1DUZ is a high-resolution A\*02:01
nonamer complex and is a standard A2 reference structure.

Only chain A residues 1 to 180 (the alpha-1/alpha-2 peptide-binding platform) and the
peptide are shown; beta-2-microglobulin and the alpha-3 domain are removed so the
groove is the only thing on screen. One highlight colour (orange) marks position 3;
everything else is neutral grey.

## structure_1_peptide_in_groove.png

Top-down view onto the A\*02:01 groove: two long helices as the jaws, the beta-sheet
floor between them, and the 9-mer peptide as sticks lying along the groove. All nine
positions are labelled, **in white gutters above and below the render** (odd positions
on top, even below), each joined to its alpha-carbon by a thin grey leader line ending
in a small ring marker. P3 is the only coloured residue and the only coloured label.

**Caption to say out loud:**
"This is a class I HLA molecule holding a nine-residue peptide: the peptide is clamped
along the groove, and position three, in orange, is the one our model wants to change."

## structure_2_p3_pocket.png

Same structure and the same camera as rendered here, zoomed onto the position-3 side
chain and the pocket that receives it, with the pocket-lining HLA residues within 5 A
shown as sticks. Layout (done by the coordinator on the clean unlabelled render): the
canvas is widened by 430 px on each side and the six residue names sit in the left and
right gutters, sorted by the vertical position of their anchor, each joined to its
residue by a thin grey leader line ending in a small ring marker. No burned-in title.

Pocket residues named in the figure: **His70, Arg97, Tyr99, Gln155, Leu156, Tyr159**.

The residue occupying that pocket in the image is the **phenylalanine of LLFGYPVYV, as
it appears in crystal structure 1DUZ** — it is *not* our proposed methionine and not a
model of our peptide. It is shown because it is a bulky hydrophobic aromatic, the same
chemical class as the model's top proposals (R3M, R3W, R3Y, R3F, R3I, R3L).

**Caption to say out loud:**
"Zoomed into position three, this is the pocket our substitution would sit in: it is
lined by tyrosines and a leucine, and in this experimental structure it is occupied by
a phenylalanine — the same hydrophobic, aromatic class the model picked out from data
alone."

## What these figures do NOT show

They do **not** test, support or confirm the 7.6x stability prediction for R3M on
FVRQCFNPM. They show a different peptide (LLFGYPVYV) in the same allele, and they show
only **where** position 3 sits and **what lines that pocket**. Any claim stronger than
"this is the environment the proposed residue would occupy" is not supported by these
images. Say "consistent with" at most; never "confirms" or "validates". Nothing in
either figure is a model of our peptide or of the proposed mutant.

## Cut, deliberately

A third figure (the proposed methionine, and an arginine, modelled into the same pocket
with the PyMOL mutagenesis wizard) was rendered and then **cut**. A single rotamer from
a rotamer library is not evidence of fit or misfit, and in the render the modelled
arginine happened to point away from the viewer, which made it look *smaller* than the
methionine — the opposite of the biology. On stage it would have been decorative at
best and misleading at worst.

## Label placement check

Label positions in both figures are asserted programmatically before the file is saved:
zero label-to-label bounding-box intersections, and zero labels overlapping the
bounding box of the orange position-3 residue. Both figures passed.

Rendering used the installed `pymol` skill (open-source PyMOL via uv, OSMesa).
Check PyMOL licensing terms at https://www.pymol.org/ .
