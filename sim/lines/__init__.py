from sim.layout import Line
from sim.lines.carton_packing import CARTON_PACKING
from sim.lines.jar_line import JAR_LINE

LINES: dict[str, Line] = {"jar_line": JAR_LINE, "carton_packing": CARTON_PACKING}
