import sys
import importlib
import traceback

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

with open("import_log.txt", "w", encoding="utf-8") as log:
    for mod in [
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
    ]:
        try:
            importlib.import_module(mod)
            log.write(f"OK   {mod}\n")
            print(f"OK   {mod}")
        except BaseException as e:
            log.write(f"\nFAIL {mod}\n")
            traceback.print_exc(file=log)
            log.write("\n")
            print(f"FAIL {mod} — check import_log.txt")

print("Done. Open import_log.txt for full details.")
