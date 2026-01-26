from isaaclab.app import AppLauncher
from isaacsim import SimulationApp

from typing import Any, Literal
import argparse


def launch_app(
    *runtime_args: dict[Literal["flag", "type", "default", "help"], str | type | Any],
    **static_args: Any
) -> tuple[SimulationApp, argparse.Namespace]:
    """
    Launch IsaacSim app with required flags
    
    Args:
        runtime_args (list[dict]): Arguments associated with execution of the given file
        static_args (Any): Potential arguments associated with the execution of the given file

    Returns:
        tuple[SimulationApp, argparse.Namespace]: A tuple containing:
            - simulation_app: Simulation app
            - args_cli: Arguments passed
    """
    # Define the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Object manipulation policy using the Franka Panda")
    # Handle runtime args
    for arg in runtime_args:
        parser.add_argument(
            arg.get('flag'),
            type=arg.get('type'),
            default=arg.get('default'),
            help=arg.get('help'),
        )
    # Append and parse args
    AppLauncher.add_app_launcher_args(parser)
    args_cli: argparse.Namespace = parser.parse_args()
    # Launch IsaacSim with args
    app_launcher: AppLauncher = AppLauncher(**static_args)
    simulation_app: SimulationApp = app_launcher.app
    
    return simulation_app, args_cli