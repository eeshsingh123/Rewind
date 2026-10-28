from collections.abc import Mapping

from rewind.core.perception.dense_track import DenseTrack
from rewind.core.perception.planner import SamplingStrategy

# Sampling strategies by name. A new strategy is a new module and one entry here.
STRATEGIES: Mapping[str, SamplingStrategy] = {
    "dense_track": DenseTrack(),
}
