# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math

import numpy as np
import torch
from collections import deque
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.exbot import Exbot


class ExbotTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        nb_observations = 15
        nb_past_observations = 10

        self._num_observations = nb_observations*nb_past_observations
        self._num_actions = 2

        RLTask.__init__(self, name, env)

        self.obs_que = deque([torch.zeros(self._num_envs, nb_observations, dtype=torch.float, device=self._device)]*nb_past_observations, maxlen=nb_past_observations)

        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_x"] = self._task_cfg["env"]["learn"]["linearVelocityXRewardScale"]
        self.rew_scales["lin_vel_y"] = self._task_cfg["env"]["learn"]["linearVelocityYRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]

        self.rew_scales["ang_vel_x"] = self._task_cfg["env"]["learn"]["angularVelocityXRewardScale"]
        self.rew_scales["ang_vel_y"] = self._task_cfg["env"]["learn"]["angularVelocityYRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]

        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["cosmetic"] = self._task_cfg["env"]["learn"]["cosmeticRewardScale"]
        
        self.rew_scales["orientation"] = self._task_cfg["env"]["learn"]["orientationRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.vx_lin_range = self._task_cfg["env"]["baseInitState"]["vxLinearRandom"]
        self.vz_ang_range = self._task_cfg["env"]["baseInitState"]["vzAngularRandom"]

        self.base_init_state = state

        # control
        self.action_space_mode = self._task_cfg["env"]["control"]["actionSpaceMode"]

        if self.action_space_mode == "variation":
            self.wheel_max_angular_velocity = self._task_cfg["env"]["control"]["wheelMaxAngularVelocity"]

        # default joint positions
        # self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        # self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        # self.Kd = self._task_cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._exbot_positions = torch.tensor([0.0, 0.0, 0.001])


    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:11] = 0.0  # commands
        # noise_vec[12:24] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[11:13] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[13:15] = 0.0  # previous actions
        return noise_vec


    def set_up_scene(self, scene) -> None:
        self.get_exbot()
        super().set_up_scene(scene)
        self._exbots = ArticulationView(
            prim_paths_expr="/World/envs/.*/Exbot/body", name="exbot_view", reset_xform_properties=False
        )
        scene.add(self._exbots)
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("exbot_view"):
            scene.remove_object("exbot_view", registry_only=True)
        self._exbots = ArticulationView(
            prim_paths_expr="/World/envs/.*/Exbot/body", name="exbot_view", reset_xform_properties=False
        )
        scene.add(self._exbots)

    def get_exbot(self):
        exbot = Exbot(
            prim_path=self.default_zero_env_path + "/Exbot", name="Exbot", translation=self._exbot_positions
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Exbot", get_prim_at_path(exbot.prim_path), self._sim_config.parse_actor_config("Exbot")
        )

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._exbots.get_world_poses(clone=False)
        root_velocities = self._exbots.get_velocities(clone=False)
        # dof_pos = self._exbots.get_joint_positions(clone=False)
        dof_vel = self._exbots.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * self.ang_vel_scale
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        # dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale

        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )

        obs = torch.cat(
            (
                base_lin_vel,                   # 3
                base_ang_vel,                   # 3
                projected_gravity,              # 3
                commands_scaled,                # 2
                # dof_pos_scaled,               # 2
                dof_vel * self.dof_vel_scale,   # 2
                self.actions,                   # 2
            ),
            dim=-1,
        )

        self.obs_que.appendleft(obs)


        # print("obs = ", obs)
        # print("dof_vel = ", dof_vel)
        # print("actions = ", self.actions* self.action_scale)

        self.obs_buf[:] = torch.cat(list(self.obs_que), dim=1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        observations = {self._exbots.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions[:] = actions.clone().to(self._device)

        if self.action_space_mode == "torque":
            self._exbots.set_joint_torque_targets(self.actions) * self.action_scale
            return
        elif self.action_space_mode == "normal":
            self.current_targets[:] = self.action_scale * self.actions[:, 0:2]

        elif self.action_space_mode == "differential":
            self.current_targets[:] = \
                self.action_scale * torch.mul( self.actions[:, 0].view(self._num_envs, 1),\
                                            torch.tensor([1.0, 1.0], device=self._device).repeat((self._num_envs, 1)) )+ \
                self.action_scale * torch.mul(self.actions[:, 1].view(self._num_envs, 1),\
                                            torch.tensor([1.0, -1.0], device=self._device).repeat((self._num_envs, 1)) )
            
        elif self.action_space_mode == "variation":
            self.current_targets[:] = torch.clamp(torch.mul(self.actions * self.action_scale, self.current_targets),\
                                                  -self.wheel_max_angular_velocity, self.wheel_max_angular_velocity )
            
        self._exbots.set_joint_velocity_targets(self.current_targets)
        

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        dof_pos = torch.zeros((num_resets, self._exbots.num_dof), device=self._device)
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._exbots.num_dof), device=self._device)

        self.current_targets[env_ids] = dof_vel[:]

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        root_vel[:, 0] = torch_rand_float(self.vx_lin_range[0], self.vx_lin_range[1], (1, num_resets), device=self._device)
        root_vel[:, 5] = torch_rand_float(self.vz_ang_range[0], self.vz_ang_range[1], (1, num_resets), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._exbots.set_joint_positions(dof_pos, indices)
        self._exbots.set_joint_velocities(dof_vel, indices)

        self._exbots.set_world_poses(
            self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices
        )
        self._exbots.set_velocities(root_vel, indices)

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :1], dim=1) > 0.25).unsqueeze(
            1
        )  # set small commands to zero

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0

    def post_reset(self):
        self.default_dof_pos = torch.zeros(
            (self.num_envs, 2), dtype=torch.float, device=self.device, requires_grad=False
        )

        self.initial_root_pos, self.initial_root_rot = self._exbots.get_world_poses()
        self.current_targets = self.default_dof_pos.clone()

        # dof_limits = self._exbots.get_dof_limits()
        # self.exbot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        # self.exbot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self.commands = torch.zeros(self._num_envs, 2, dtype=torch.float, device=self._device, requires_grad=False)

        # initialize some data used later on
        self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat((self._num_envs, 1))
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(
            (self._num_envs, 2), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )



        self.time_out_buf = torch.zeros_like(self.reset_buf)


        # randomize all envs
        indices = torch.arange(self._exbots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._exbots.get_world_poses(clone=False)
        root_velocities = self._exbots.get_velocities(clone=False)
        dof_pos = self._exbots.get_joint_positions(clone=False)
        dof_vel = self._exbots.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)

        # velocity tracking reward
        if self.rew_scales["lin_vel_x"] > 0:
            lin_vel_error = torch.square(self.commands[:, 0] - base_lin_vel[:, 0])
            rew_lin_vel_x = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_x"]
        else:
            rew_lin_vel_x = torch.square(base_lin_vel[:, 0]) * self.rew_scales["lin_vel_x"]
        rew_lin_vel_y = torch.square(base_lin_vel[:, 1]) * self.rew_scales["lin_vel_y"]
        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_x = torch.square(base_ang_vel[:, 0]) * self.rew_scales["ang_vel_x"]
        rew_ang_vel_y = torch.square(base_ang_vel[:, 1]) * self.rew_scales["ang_vel_y"]
        if self.rew_scales["ang_vel_z"] > 0:
            ang_vel_error = torch.square(self.commands[:, 1] - base_ang_vel[:, 2])
            rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]
        else:
            rew_ang_vel_z = torch.square(base_ang_vel[:, 2]) * self.rew_scales["ang_vel_z"]


        rew_joint_acc = \
            torch.sum(torch.square(self.last_dof_vel - dof_vel), dim=1) * self.rew_scales["joint_acc"]
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )
        rew_cosmetic = (
            torch.sum(torch.abs(dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1) * self.rew_scales["cosmetic"]
        )

        rew_orientation = torch.exp( 
            - torch.sum( torch.square(projected_gravity[:, :] - self.gravity_vec) , dim=1)/0.25 \
            ) * self.rew_scales["orientation"]


        total_reward = rew_lin_vel_x + rew_lin_vel_y + rew_lin_vel_z\
            + rew_ang_vel_x + rew_ang_vel_y + rew_ang_vel_z \
            + rew_joint_acc + rew_action_rate + rew_cosmetic + rew_orientation
        
        total_reward = torch.clip(total_reward, 0.0, None)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel[:]

        self.fallen_over = projected_gravity[:, 2] > -0.001
        total_reward[torch.nonzero(self.fallen_over)] = -1.0
        self.rew_buf[:] = total_reward.detach()


    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out | self.fallen_over
