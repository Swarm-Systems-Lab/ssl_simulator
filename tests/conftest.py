import matplotlib

# Force the headless Agg backend for the whole test suite. Tests that render
# (e.g. LaTeX text rendering) shouldn't depend on an interactive GUI backend
# being available - TkAgg requires a working Tcl/Tk runtime, which uv's
# standalone Python builds don't consistently bundle, and CI runners have no
# display anyway. This must run before matplotlib.pyplot is imported by any
# test module, which pytest guarantees by loading conftest.py first.
matplotlib.use("Agg")
