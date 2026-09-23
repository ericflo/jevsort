"""jevsort — rank anything with AI judges.

    import jevsort
    result = jevsort.sort(["idea one", "idea two", "idea three"], "Which idea has more impact?")
    result.sorted        # best first
    jevsort.compare("draft A", "draft B", "Which is clearer?")   # -> P(A is better)
    jevsort.sorter().by("Which is clearer?").judge("llm").budget(60).meta().sort(items)   # builder

PKPD pairwise sorting on Jev-style judges.

Sort anything by asking a calibrated judge many small pairwise questions
("which of A or B has stronger evidence?"), coupling the pairwise posteriors
into global posteriors (Price–Knerr–Personnaz–Dreyfus 1994, Eq. 7, or
Bradley–Terry), and blending several dimensions — optionally with the judge
itself as meta-judge and referee.
"""

from .api import FunctionJudge, Sorter, as_dimensions, as_items, as_judge, compare, sort, sorter
from .backends import Choice, JudgeBackend, make_backend
from .blend import LinearBlend
from .calibrate import Profile, apply_temperature, ece, fit_temperature
from .couple import Coupled, bradley_terry, couple, pkpd
from .pairwise import PairwiseMatrix, clip, symmetrize
from .sorter import PAPER_DIMENSIONS, Dimension, Item, JevSorter, SortResult

__version__ = "0.1.0"

__all__ = [
    "sort",
    "compare",
    "sorter",
    "Sorter",
    "FunctionJudge",
    "as_dimensions",
    "as_items",
    "as_judge",
    "Choice",
    "JudgeBackend",
    "make_backend",
    "LinearBlend",
    "Profile",
    "apply_temperature",
    "ece",
    "fit_temperature",
    "Coupled",
    "bradley_terry",
    "couple",
    "pkpd",
    "PairwiseMatrix",
    "clip",
    "symmetrize",
    "PAPER_DIMENSIONS",
    "Dimension",
    "Item",
    "JevSorter",
    "SortResult",
]
