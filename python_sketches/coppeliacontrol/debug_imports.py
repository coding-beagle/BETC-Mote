# debug_imports.py  — put this next to main.py and run it
import traceback

modules = [
    "utils.config",
    "utils.math_utils",
    "utils.pose_filter",
    "utils.calibrator",
    "utils.pose_utils",
    "utils.hand_gesture",
    "utils.camera_thread",
    "utils.hud",
    "utils.experiment_io",
    "utils",
]

for mod in modules:
    try:
        __import__(mod)
        print(f"OK   {mod}")
    except Exception:
        print(f"FAIL {mod}")
        traceback.print_exc()
        break
