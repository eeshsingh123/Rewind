from collections import defaultdict
from collections.abc import Sequence

from rewind.contracts import Observation, Track


# Naive association for now: one track per segmenter object id and concept,
# split where the object is unseen for longer than max_gap_s, because SAM reuses ids.
def naive_tracks(
    footage_id: str, observations: Sequence[Observation], max_gap_s: float
) -> tuple[Track, ...]:
    by_object: defaultdict[tuple[str, int], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.object_id is not None:
            by_object[(observation.concept, observation.object_id)].append(observation)
    runs: list[tuple[str, list[Observation]]] = []
    for (concept, _), seen in by_object.items():
        seen.sort(key=lambda o: o.t)
        run = [seen[0]]
        for observation in seen[1:]:
            if observation.t - run[-1].t > max_gap_s:
                runs.append((concept, run))
                run = []
            run.append(observation)
        runs.append((concept, run))
    runs.sort(key=lambda r: (r[0], r[1][0].t))
    numbers: defaultdict[str, int] = defaultdict(int)
    tracks: list[Track] = []
    for concept, run in runs:
        numbers[concept] += 1
        tracks.append(
            Track(
                footage_id=footage_id,
                track_id=f"{concept}#{numbers[concept]}",
                concept=concept,
                observations=tuple(run),
            )
        )
    return tuple(tracks)
