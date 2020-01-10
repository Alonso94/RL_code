import requests
import json
import cv2
import numpy as np
from threading import Thread, Semaphore
import queue
import time
import math
import cv2
import gym
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Rozum:
    def __init__(self):
        self.host = "http://10.10.10.20:8081"
        self.joint_angles = self.get_joint_angles()
        self.position, self.orientation = self.get_position()

    def get_joint_angles(self):
        # in degrees
        response = requests.get(self.host + '/pose')
        return response.json()['angles']

    def send_joint_angles(self):
        # speed 10
        requests.put(self.host + '/pose?speed=10', data=json.dumps({
            "angles": self.joint_angles
        }))
        url = self.host + '/status/motion'
        response = requests.get(url)
        while (response.content != b'"IDLE"'):
            response = requests.get(url)

    def get_joints_current(self):
        response = requests.get(self.host + '/status/motors')
        currents = []
        motor_info = response.json()
        for motor in motor_info:
            currents.append(motor["rmsCurrent"])
        return currents

    def get_position(self):
        response = requests.get(self.host + '/position')
        pose_info = response.json()
        point = pose_info["point"]
        position = [point["x"], point["y"], point["z"]]
        rot = pose_info["rotation"]
        orientation = [rot["roll"], rot["pitch"], rot["yaw"]]
        sin_cos = []
        for a in orientation:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        return np.array(position),np.array(sin_cos)

    def send_position(self):
        # speed 10
        res = requests.put(self.host + '/position?speed=10', data=json.dumps({
            "point": {
                "x": self.position[0],
                "y": self.position[1],
                "z": self.position[2]
            },
            "rotation": {
                "roll": self.orientation[0],
                "pitch": self.orientation[1],
                "yaw": self.orientation[2]
            }
        }))
        url = self.host + '/status/motion'
        response = requests.get(url)
        while (response.content != b'"IDLE"'):
            response = requests.get(url)

    def open_gripper(self):
        requests.put(self.host + '/gripper/open')

    def close_gripper(self):
        requests.put(self.host + '/gripper/close')

    def recover(self):
        requests.put(self.host + '/recover')

    def update_joint_angles(self, values):
        for i in range(len(self.joint_angles)):
            self.joint_angles[i] = values[i]

    def update_position(self, position, orientation):
        for i in range(3):
            self.position[i] = position[i]
            self.orientation[i] = orientation[i]


class rozum_real:

    def __init__(self):
        self.robot = Rozum()
        self.DoF = 6
        # self.action_bound = [[-15,15],[-10,110],[-30,30],[-120,120],[-180,180],[-180,180]]
        self.action_bound = [[-240, -180], [-180, 180], [-180, 180], [-220, -100], [-180, 180], [-180, 180]]
        self.action_range = [-5, 5]

        self.action_space = gym.spaces.Box(shape=(self.DoF,), low=-5, high=5)
        self.observation_space = gym.spaces.Box(shape=(3 + 6+self.DoF * 2,), low=-180, high=180)
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        self.goal_l = (80, 40, 0)
        self.goal_u = (110, 255, 255)
        self.cube_l = (55, 50, 0)
        self.cube_u = (80, 255, 255)
        self.er_kernel = np.ones((7, 7), np.uint8)
        self.di_kernel = np.ones((10, 10), np.uint8)
        self.task_part = 0
        self.part_1_center = np.array([300.0 / 640, 335.0 / 480])
        self.part_2_center = np.array([320.0 / 640, 290.0 / 480])
        self.part_1_area = 0.25
        self.part_2_area = 0.75
        self.target = np.array([300.0 / 640, 335.0 / 480, 0.25, 0.0])
        # self.target=np.array([-0.375, 0.441, 0.357])
        # self.count=0


        self.robot.open_gripper()
        self.init_pose, self.init_orientation = self.robot.get_position()
        # self.init_angles = [-200,-90,-90,-90,90,0]
        self.init_angles = [-210.0, -110.0, 0.0, -160.0, 90.0, -35.0]
        self.s = self.reset()
        self.angles = self.init_angles.copy()


        # task part
        self.task_part = 0

        self.init_pose_cube = np.array([-0.348, 0.202, 0.204])
        self.cube_orient=np.array([0.0,-1.0,0.0,1.0,1.0,0.0])
        # print(self.init_pose_cube)
        self.init_goal_pose =np.array([-0.278, 0.413, 0.409])
        # print(self.init_goal_pose)
        self.robot.open_gripper()
        self.tip_position,_ = self.robot.get_position()


    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def step(self, action):
        self.angles = self.robot.get_joint_angles()
        for i in range(self.DoF):
            angle=self.angles[i] + action[i]
            angle = np.clip(angle, *self.action_bound[i])
            self.angles[i]=angle.copy()
        self.robot.update_joint_angles(self.angles)
        self.robot.send_joint_angles()
        angles = self.robot.get_joint_angles()
        pose ,orientation= self.robot.get_position()
        r = 0.0
        done = False
        if self.task_part == 0:
            target = self.init_pose_cube.copy()
            r = 0.0
        else:
            target = self.init_goal_pose.copy()
            target[2] += 0.1
            r += 10
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        s = np.concatenate([pose,orientation, sin_cos], axis=0)
        d = np.linalg.norm(pose - target)
        d_or=np.linalg.norm(orientation-self.cube_orient)
        r += (-d - 0.01 * np.square(action).sum())
        if d < 0.02 and  d_or < 0.02 and self.task_part == 0:
            self.task_part = 1
            self.robot.close_gripper()
            time.sleep(1)
            self.angles = self.init_angles.copy()
            self.robot.update_joint_angles(self.angles)
            self.robot.send_joint_angles()
            time.sleep(3)
            return s, r, done, None
        if self.task_part == 1 and abs(target[2] - pose[2]) < 0.05:
            self.robot.open_gripper()
            time.sleep(2)
            r += 10
            done = True
        return s, r, done, None

    def reset(self):
        self.task_part = 0
        self.angles = self.init_angles.copy()
        self.robot.update_joint_angles(self.angles)
        self.robot.send_joint_angles()
        self.robot.open_gripper()
        angles = self.robot.get_joint_angles()
        pose ,orientation= self.robot.get_position()
        # pose = [a *10 for a in pose]
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        s = np.concatenate([pose,orientation, sin_cos], axis=0)
        return s

    def state_cost(self, state):
        # [8000,10]
        if self.task_part == 0:
            target = self.init_pose_cube.copy()
            target = torch.from_numpy(target).to(device).float()
            target_or=self.cube_orient.copy()
            target_or = torch.from_numpy(target_or).to(device).float()
            dis = state[:, :3] - target
            or_diff = state[:, 3:9] - target_or
            # dis = state - self.env.target
            cost = (dis ** 2).sum(dim=-1) + torch.mul((or_diff ** 2).sum(dim=-1),0.03)
        else:
            target = self.init_goal_pose
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
        return 0.01 * (action ** 2).sum(dim=1)
