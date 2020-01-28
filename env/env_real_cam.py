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
        return np.array(position), np.array(sin_cos)

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


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()

        self.t = Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class rozum_real:

    def __init__(self):
        self.robot = Rozum()
        self.DoF = 6
        # self.action_bound = [[-15,15],[-10,110],[-30,30],[-120,120],[-180,180],[-180,180]]
        self.action_bound = [[-240, -180], [-180, 180], [-180, 180], [-220, -100], [-180, 180], [-180, 180]]
        self.action_range = [-5, 5]
        self.cam = VideoCapture(0)
        self.w = self.cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.action_space = gym.spaces.Box(shape=(self.DoF,), low=-5, high=5)
        self.observation_space = gym.spaces.Box(shape=(5 + self.DoF * 2,), low=-1, high=1)
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
        self.part_1_area = 0.2
        self.part_2_area = 0.75
        self.task_1_target = np.array([self.part_1_center[0], self.part_1_center[1], self.part_1_area, 0.0, 1.0])
        self.task_2_target = np.array([self.part_2_center[0], self.part_2_center[1], self.part_2_area, 0.0, 1.0])
        # self.target=np.array([-0.375, 0.441, 0.357])
        # self.count=0

        self.currents_thread = Thread(target=self.current_reader)
        self.currents_thread.daemon = True
        self.currents_thread.start()

        self.robot.open_gripper()
        self.init_pose, self.init_orientation = self.robot.get_position()
        # self.init_angles = [-200,-90,-90,-90,90,0]
        self.init_angles = [-210.0, -110.0, 0.0, -160.0, 90.0, -35.0]
        self.s = self.reset()
        self.angles = self.init_angles.copy()
        self.saved_angles = self.init_angles.copy()

        # task part
        self.task_part = 0
        self.robot.open_gripper()

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def current_reader(self):
        while True:
            self.currents = self.robot.get_joints_current()

    def get_inclination(self, box):
        th1 = np.arctan2(box[0, 1] - box[1, 1], box[0, 0] - box[1, 0])
        th2 = np.arctan2(box[1, 1] - box[2, 1], box[1, 0] - box[2, 0])
        th3 = np.arctan2(box[2, 1] - box[3, 1], box[2, 0] - box[3, 0])
        th4 = np.arctan2(box[3, 1] - box[0, 1], box[3, 0] - box[0, 0])
        th=th1+th2+th3+th4-np.pi
        return th

    def image_processing(self, img, lower, upper, num_iter):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        binary = cv2.inRange(hsv, lower, upper)
        binary = cv2.erode(binary, self.er_kernel, iterations=num_iter[0])
        binary = cv2.dilate(binary, self.di_kernel, iterations=num_iter[1])
        cv2.imshow("1", binary)
        cv2.waitKey(25)
        cnt, _ = cv2.findContours(binary, 1, 1)
        cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
        center = np.array([0.0, 0.0])
        area_percentage = 0
        angle = 0
        if len(cnt) > 0:
            rect = cv2.minAreaRect(cnt[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            angle = self.get_inclination(box)
            center = np.average(box, axis=0)
            area = cv2.contourArea(cnt[0])
            area_percentage = area / (640 * 480)
        features = np.array([center[0] / 640, center[1] / 480, area_percentage, np.sin(angle), np.cos(angle)])
        return features

    def step(self, action):
        self.angles = self.robot.get_joint_angles()
        for i in range(self.DoF):
            angle = self.angles[i] + action[i]
            angle = np.clip(angle, *self.action_bound[i])
            self.angles[i] = angle.copy()
        self.robot.update_joint_angles(self.angles)
        self.robot.send_joint_angles()
        angles = self.robot.get_joint_angles()
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        img = self.cam.read()
        r = 0.0
        done = False
        if self.task_part == 0:
            features = self.image_processing(img, self.goal_l, self.goal_u, [2, 2])
            target = self.part_1_center.copy()
            r = 0.0
        else:
            features = self.image_processing(img, self.cube_l, self.cube_u, [2, 2])
            target = self.part_2_center.copy()
            r += 10
        s = np.concatenate([features, sin_cos], axis=0)
        if features[2] < 0.05:
            done = True
            return s, r, done, None
        d = np.linalg.norm(features[:2] - target)
        # print("d",d)
        r += (-d - 0.01 * np.square(action).sum())
        if d < 0.02 and self.task_part == 0:
            d_full = np.linalg.norm(features - self.task_1_target)
            # print("inside:",d_full)
            if d_full < 0.1:
                self.saved_angles = angles.copy()
                self.angles = self.init_angles.copy()
                self.robot.update_joint_angles(self.angles)
                self.robot.send_joint_angles()
                time.sleep(3)
                r += 10
                self.task_part = 1
                return s, r, done, None
        if d < 0.05 and self.task_part == 1:
            d_full = np.linalg.norm(features - self.task_1_target)
            if d_full < 0.1:
                self.robot.close_gripper()
                time.sleep(1)
                self.angles = self.init_angles.copy()
                self.robot.update_joint_angles(self.angles)
                self.robot.send_joint_angles()
                time.sleep(3)
                self.angles = self.saved_angles.copy()
                self.robot.update_joint_angles(self.angles)
                self.robot.send_joint_angles()
                time.sleep(3)
                self.robot.open_gripper()
                done = True
                return s, r, done, None
        return s, r, done, None

    def reset(self):
        self.task_part = 0
        self.angles = self.init_angles.copy()
        self.robot.update_joint_angles(self.angles)
        self.robot.send_joint_angles()
        self.robot.open_gripper()
        angles = self.robot.get_joint_angles()
        sin_cos = []
        for a in angles:
            sin_cos.append(np.sin(a))
            sin_cos.append(np.cos(a))
        img = self.cam.read()
        features = self.image_processing(img, self.goal_l, self.goal_u, [2, 2])
        s = np.concatenate([features, sin_cos], axis=0)
        return s

    def state_cost(self, state):
        # [8000,10]
        if self.task_part == 0:
            # target = self.part_1_center.copy()
            target_full = self.task_1_target.copy()
            target=target_full.copy()[:2]
        else:
            # target = self.part_2_center.copy()
            target_full = self.task_2_target.copy()
            target = target_full.copy()[:2]
        target = torch.from_numpy(target).to(device).float()
        dis = state[:, :2] - target
        # if torch.max(dis) < 0.02:
        target_full = torch.from_numpy(target_full).to(device).float()
        diff = state[:, :5] - target_full
        cost = (dis ** 2).sum(dim=-1) + torch.mul((diff ** 2).sum(dim=-1), 0.3)
        # else:
        #     cost = (dis ** 2).sum(dim=-1)
        cost = -torch.exp(-cost)
        cost[state[:, 2] < 0.1] = 1e6
        return cost

    @staticmethod
    def action_cost(action):
        return 0.01 * (action ** 2).sum(dim=1)
