from collections.abc import Sequence

from plainera_unacronym.orchestration.interface import PipelineKey


def normalise_targets(targets: Sequence[PipelineKey]) -> tuple[PipelineKey, ...]:
    seen: set[PipelineKey] = set()
    ordered: list[PipelineKey] = []

    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        ordered.append(target)

    return tuple(ordered)
