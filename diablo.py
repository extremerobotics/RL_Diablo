from typing import Optional

import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema

class Diablo(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "franka",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            self._usd_path = "./diablo.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "/base_link/Rev1",
            "/base_link/Rev2",
            "/motor_left_link_1/Rev3",
            "/leg_left_link_1/Rev4", 
            "/leg2_left_link_1/Rev5",
            "/motor_right_link_1/Rev6",
            "/leg_right_link_1/Rev7",
            "/leg2_right_link_1/Rev8"
        ]

        drive_type = ["angular"] * 8
        default_dof_pos = [0.] * 8
        stiffness = [15000] * 8
        damping = [1] * 8
        max_force = [1000] * 8
        max_velocity = [1000] * 8

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )

    def set_diablo_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)

