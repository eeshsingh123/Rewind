from rewind.contracts import SamplingPlan, SamplingWindow
from rewind.core.perception.planner import PlanInput, rule_concepts


# Every concept the rules name, tracked on every active span at one fps.
class DenseTrack:
    def plan(self, request: PlanInput) -> SamplingPlan:
        concepts = rule_concepts(request.rules)
        windows = (
            tuple(
                SamplingWindow(span=span, fps=request.fps, concepts=concepts)
                for span in request.active_spans
            )
            if concepts
            else ()
        )
        return SamplingPlan(footage_id=request.footage.id, strategy="dense_track", windows=windows)
