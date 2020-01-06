import vrep.vrep as vrep
import vrep.vrepConst as const_v
import time
import sys
import numpy as np
import cv2
import math
import os
import gym
import subprocess, signal
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from threading import Thread


class rozum_sim:

    def __init__(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1024, 1024))

        self.DoF = 6
        # self.action_bound = [[-15,15],[-10,110],[-30,30],[-120,120],[-180,180],[-180,180]]
        self.action_bound = [[-180, 180], [-180, 180], [-180, 180], [-180, 180], [-180, 180], [-180, 180]]
        self.action_range = [-5, 5]
        self.action_dim = self.DoF

        self.action_space = gym.spaces.Box(shape=(self.DoF,), low=-5, high=5)
        self.observation_space = gym.spaces.Box(shape=(3 + self.DoF * 2,), low=-180, high=180)
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if b'vrep' in line:
                pid = int(line.split(None, -1)[0])
                os.kill(pid, signal.SIGKILL)

        self.vrep_root = "/home/ali/Downloads/VREP"
        self.scene_file = "/home/ali/RL_code/env/rozum_model.ttt"
        #
        os.chdir(self.vrep_root)
        os.system("./vrep.sh -s " + self.scene_file + " &")

        vrep.simxFinish(-1)
        time.sleep(1)

        # get the ID of the running simulation
        self.ID = vrep.simxStart('127.0.0.1', 19999, True, False, 5000, 5)
        # check the connection
        if self.ID != -1:
            print("Connected")
        else:
            sys.exit("Error")
        # get handles
        # for camera
        self.cam_handle = self.get_handle('Vision_sensor')
        (code, res, im) = vrep.simxGetVisionSensorImage(self.ID, self.cam_handle, 0, const_v.simx_opmode_streaming)

        self.render_handle = self.get_handle('render')
        (code, res, im) = vrep.simxGetVisionSensorImage(self.ID, self.render_handle, 0, const_v.simx_opmode_streaming)

        # joints
        self.joint_handles = []
        for i in range(self.DoF):
            tmp = self.get_handle("joint%d" % (i))
            self.joint_handles.append(tmp)
            code, angle = vrep.simxGetJointPosition(self.ID, tmp, const_v.simx_opmode_streaming)

        # gripper tip
        self.tip_handle = self.get_handle("Tip")
        (code, pose) = vrep.simxGetObjectPosition(self.ID, self.tip_handle, -1, const_v.simx_opmode_streaming)
        self.or_handle = self.get_handle("RG2_baseVisible")
        (code, pose) = vrep.simxGetObjectOrientation(self.ID, self.or_handle, -1, const_v.simx_opmode_streaming)
        # cube
        self.cube_handle = self.get_handle("Cube")
        (code, pose) = vrep.simxGetObjectPosition(self.ID, self.cube_handle, -1, const_v.simx_opmode_streaming)
        (code, pose) = vrep.simxGetObjectOrientation(self.ID, self.cube_handle, -1, const_v.simx_opmode_streaming)
        # get the goal handle
        self.goal_handle = self.get_handle("Goal")
        (code, pose) = vrep.simxGetObjectPosition(self.ID, self.goal_handle, -1, const_v.simx_opmode_streaming)

        # angles' array
        self.angles = self.get_angles()

        # gripper handles (used in closing and opening gripper)
        self.gripper_motor = self.get_handle('RG2_openCloseJoint')
        # task part
        self.task_part = 0

        self.init_angles = self.get_angles()
        self.init_orientation=self.get_orientation(self.or_handle)
        self.init_pose_cube = self.get_position(self.cube_handle)
        # print(self.init_pose_cube)
        self.init_goal_pose = self.get_position(self.goal_handle)
        # print(self.init_goal_pose)
        self.open_gripper()
        self.reset()
        self.tip_position = self.get_position(self.tip_handle)

        self.goal_l = (80, 0, 0)
        self.goal_u = (120, 255, 255)
        self.cube_l = (55, 50, 50)
        self.cube_u = (80, 255, 255)
        self.er_kernel = np.ones((2, 2), np.uint8)
        self.di_kernel = np.ones((2, 22), np.uint8)
        self.task_part = 0
        self.part_1_center = np.array([300.0, 335.0])
        self.part_2_center = np.array([320.0, 290.0])
        self.part_1_area = 0.25
        self.part_2_area = 0.75

    def get_handle(self, name):
        (check, handle) = vrep.simxGetObjectHandle(self.ID, name, const_v.simx_opmode_blocking)
        if check != 0:
            print("Couldn't find %s" % name)
        return handle

    def get_position(self, handle):
        (code, pose) = vrep.simxGetObjectPosition(self.ID, handle, -1, const_v.simx_opmode_buffer)
        # print(code)
        return np.array(pose)

    def get_orientation(self,handle):
        (code, ornt) = vrep.simxGetObjectOrientation(self.ID, handle, -1, const_v.simx_opmode_buffer)
        # print(code)
        return np.array([np.sin(ornt[0]),np.cos(ornt[0]),np.sin(ornt[1]),np.cos(ornt[1]),np.sin(ornt[2]),np.cos(ornt[2])])

    def close_gripper(self, render=False):
        code = vrep.simxSetJointForce(self.ID, self.gripper_motor, 20, const_v.simx_opmode_blocking)
        # print(code)
        code = vrep.simxSetJointTargetVelocity(self.ID, self.gripper_motor, -0.05, const_v.simx_opmode_blocking)
        if render:
            self.render()
        # print(code)
        # time.sleep(0.1)

    def open_gripper(self, render=False):
        code = vrep.simxSetJointForce(self.ID, self.gripper_motor, 20, const_v.simx_opmode_blocking)
        # print(code)
        code = vrep.simxSetJointTargetVelocity(self.ID, self.gripper_motor, 0.05, const_v.simx_opmode_blocking)
        if render:
            self.render()
        # print(code)
        # time.sleep(0.1)

    def get_image(self, cam_handle):
        (code, res, im) = vrep.simxGetVisionSensorImage(self.ID, cam_handle, 0, const_v.simx_opmode_buffer)
        # print(code)
        img = np.array(im, dtype=np.uint8)
        img.resize([res[0], res[1], 3])
        img = cv2.flip(img, 0)
        return img

    def move_joint(self, num, value, render=False):
        # in radian
        code = vrep.simxSetJointTargetPosition(self.ID, self.joint_handles[num], value * math.pi / 180,
                                               const_v.simx_opmode_blocking)
        if render:
            self.render()
        # print(code)
        # time.sleep(0.3)

    def get_angles(self):
        angles = []
        for i in range(self.DoF):
            code, angle = vrep.simxGetJointPosition(self.ID, self.joint_handles[i], const_v.simx_opmode_buffer)
            angles.append(angle * 180 / math.pi)
        return angles

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def step(self, action):
        self.angles = self.get_angles()
        for i in range(self.DoF):
            self.angles[i] += action[i]
            angle = np.clip(self.angles[i], *self.action_bound[i])
            self.move_joint(i, angle)
        time.sleep(0.3)
        angles = self.get_angles()
        pose = self.get_position(self.tip_handle)
        r = 0.0
        done = False
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        s = np.concatenate([pose, sin_cos], axis=0)
        d = np.linalg.norm(pose - self.init_pose_cube)
        r += (-d - 0.01 * np.square(action).sum())
        return s, r, done, None

    def reset(self):
        self.task_part = 0
        self.angles = self.init_angles.copy()
        for i in range(self.DoF):
            self.move_joint(i, self.angles[i])
        self.open_gripper()
        vrep.simxSetObjectPosition(self.ID, self.cube_handle, -1, self.init_pose_cube, const_v.simx_opmode_oneshot_wait)
        vrep.simxSetObjectPosition(self.ID, self.goal_handle, -1, self.init_goal_pose, const_v.simx_opmode_oneshot_wait)
        time.sleep(2)
        angles = self.get_angles()
        pose = self.get_position(self.tip_handle)
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        s = np.concatenate([pose, sin_cos], axis=0)
        return s

    def state_cost(self, state):
        # [8000,10]
        target = self.init_pose_cube.copy()
        target[2] += 0.1
        target = torch.from_numpy(target).to(device).float()
        dis = state[:, :3] - target
        # dis = state - self.env.target
        cost = (dis ** 2).sum(dim=-1)
        # target = np.array([a*10 for a in target])
        cost = -torch.exp(-cost)
        return cost

    @staticmethod
    def action_cost(action):
        return 0.001 * (action ** 2).sum(dim=1)

    def render(self):
        img = self.get_image(self.render_handle)
        self.out.write(img)
