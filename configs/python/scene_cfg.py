import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg

from isaaclab_assets import FRANKA_PANDA_CFG

from utils.config import config


@configclass
class SceneCfg(InteractiveSceneCfg):
    """
    Scene configuration
    """
    # Ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # Lighting
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=config["scene"]["light"]["intensity"],
            color=tuple(config["scene"]["light"]["color"]),
        ),
    )
    # Robot config
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0,),
        ),
    )
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True
    
    # Cube
    # Object to grasp
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=tuple(config["scene"]["cube"]["size"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(config["scene"]["cube"]["diffuse_color"])),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(config["scene"]["cube"]["pos"]),
        ),
    )
    
    '''    # Contact sensor
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
    )'''