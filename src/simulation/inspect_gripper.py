import inspect
import torch

from pathlib import Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(0)

bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

init_file = (
    Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
)

states = torch.load(init_file, map_location="cpu", weights_only=False)

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl),
    camera_heights=128,
    camera_widths=128,
)

try:
    env.reset()
    env.set_init_state(states[0])

    robot = env.robots[0]
    gripper = robot.gripper
    controller = robot.controller

    print("\n========== GRIPPER ==========")

    print("Class:", type(gripper))
    print("DOF:", gripper.dof)
    print("Initial configuration:", gripper.init_qpos)

    print("\n========== GRIPPER ACTION MAPPING ==========")

    for cls in type(gripper).__mro__:
        if "format_action" in cls.__dict__:
            print(inspect.getsource(cls.__dict__["format_action"]))
            break
    else:
        print("format_action method not found")

    print("\n========== CONTROLLER ==========")

    print("Class:", type(controller))
    print("Input minimum:", controller.input_min)
    print("Input maximum:", controller.input_max)
    print("Output minimum:", controller.output_min)
    print("Output maximum:", controller.output_max)

finally:
    env.close()
