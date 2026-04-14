from .detector import StructuralReferenceDetector
from .normalise import normalize_structural_reference_key
from .structural_reference_compiler import compile_structural_reference_patterns
from .types import StructuralReference, StructuralReferenceDetectorResult

__all__ = [
    "compile_structural_reference_patterns",
    "StructuralReferenceDetector",
    "normalize_structural_reference_key",
    "StructuralReference",
    "StructuralReferenceDetectorResult",
]
