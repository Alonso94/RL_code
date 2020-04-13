import vrep.sim as vrep
import vrep.simConst as const_v
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

    def __init__(self,render):
        self.render=render
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1024, 1024))

        self.DoF = 7
        # self.action_bound = [[-15,15],[-10,110],[-30,30],[-120,120],[-180,180],[-180,180]]
        self.action_bound = [[-180, 180], [-180, 180], [-180, 180], [-180, 180], [-180, 180], [-180, 180],[-180, 180]]
        self.action_range = [-5, 5]
        self.action_dim = self.DoF

        self.action_space = gym.spaces.Box(shape=(self.DoF,), low=-5, high=5)
        self.observation_space = gym.spaces.Box(shape=(9 + self.DoF * 2,), low=-180, high=180)
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if b'coppeliaSim' in line:
                pid = int(line.split(None, -1)[0])
                os.kill(pid, signal.SIGKILL)

        self.vrep_root = "/home/ali/Downloads/VREP"
        self.scene_file = "/home/ali/RL_code/env/kuka_2.ttt"
        #
        os.chdir(self.vrep_root)
        os.system("./coppeliaSim.sh -s " + self.scene_file + " &")

        vrep.simxFinish(-1)
        time.sleep(1)

        # get the ID of the running simulation
        self.ID = vrep.simxStart('127.0.0.1', 19999, True, False, 5000, 5)
        # check the connection
        if self.ID != -1:
            print("Connected")
        else:
            sys.exit("Error")

        time.sleep(0.5)
        # get handles
        # for cameras
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

        # usb stick handle
        self.usb_handle = self.get_handle("USB")
        (code, pose) = vrep.simxGetObjectPosition(self.ID, self.usb_handle, -1, const_v.simx_opmode_streaming)
        # get the socket handle
        self.socket_handle = self.get_handle("Socket")
        (code, pose) = vrep.simxGetObjectPosition(self.ID, self.socket_handle, -1, const_v.simx_opmode_streaming)

        # angles' array
        self.angles = self.get_angles()
        self.init_angles = self.angles.copy()
        self.init_socket_pose = [-0.2284,0.0,0.025]

        self.reset()


    def get_handle(self, name):
        (check, handle) = vrep.simxGetObjectHandle(self.ID, name, const_v.simx_opmode_blocking)
        if check != 0:
            print("Couldn't find %s" % name)
        return handle

    def get_position(self, handle):
        (code, pose) = vrep.simxGetObjectPosition(self.ID, handle, -1, const_v.simx_opmode_buffer)
        return np.array(pose)

    def get_orientation(self,handle):
        (code, ornt) = vrep.simxGetObjectOrientation(self.ID, handle, -1, const_v.simx_opmode_buffer)
        # print(code)
        return np.array([np.sin(ornt[0]),np.cos(ornt[0]),np.sin(ornt[1]),np.cos(ornt[1]),np.sin(ornt[2]),np.cos(ornt[2])])

    def get_image(self, cam_handle):
        (code, res, im) = vrep.simxGetVisionSensorImage(self.ID, cam_handle, 0, const_v.simx_opmode_buffer)
        img = np.array(im, dtype=np.uint8)
        img.resize([res[0], res[1], 3])
        img = cv2.flip(img, 0)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def move_joint(self, num, value):
        # in radian
        code = vrep.simxSetJointTargetPosition(self.ID, self.joint_handles[num], value * math.pi / 180,
                                               const_v.simx_opmode_blocking)
        if self.render:
            self.add_to_video()

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
        pose = self.get_position(self.usb_handle)
        orientation = self.get_orientation(self.usb_handle)
        r = 0.0
        done = False
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        s = np.concatenate([pose,orientation, sin_cos], axis=0)
        target=self.get_position(self.socket_handle)
        d = np.linalg.norm(pose - target)
        r += (-d - 0.01 * np.square(action).sum())
        return s, r, done, None

    def reset(self):
        self.task_part = 0
        self.angles = self.init_angles.copy()
        for i in range(self.DoF):
            self.move_joint(i, self.angles[i])
        vrep.simxSetObjectPosition(self.ID, self.socket_handle, -1, self.init_socket_pose, const_v.simx_opmode_oneshot_wait)
        # time.sleep(2)
        angles = self.get_angles()
        pose = self.get_position(self.usb_handle)
        orientation=self.get_orientation(self.usb_handle)
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        s = np.concatenate([pose, orientation, sin_cos], axis=0)
        # print(s)
        return s

    def state_cost(self, state):
        # [8000,10]
        pose = self.get_position(self.socket_handle)
        orientation = self.get_orientation(self.socket_handle)
        target_pose = torch.from_numpy(pose).to(device).float()
        target_orientation = torch.from_numpy(orientation).to(device).float()
        dis_pose = state[:, :3] - target_pose
        dis_orientation=state[:,3:9] - target_orientation
        # dis = state - self.env.target
        cost= (dis_pose ** 2).sum(dim=-1) + torch.mul((dis_orientation ** 2).sum(dim=-1),0.07)
        # cost = (dis ** 2).sum(dim=-1)
        # target = np.array([a*10 for a in target])
        cost = -torch.exp(-cost)
        return cost

    @staticmethod
    def action_cost(action):
        return 0.01 * (action ** 2).sum(dim=1)

    def add_to_video(self):
        img = self.get_image(self.render_handle)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # cv2.imshow("1",img)
        # cv2.waitKey(25)
        self.out.write(img)

# env=rozum_sim()

