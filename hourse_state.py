import cv2
import numpy as np
import simulator

class hourse_state:
    def __init__(self, y, name, type, time, time_3F, num):
        self.x = 20
        self.y = y
        self.v_temp = 0
        self.v = 0
        self.d = 0
        self.dy = 0
        self.name = name
        self.type = type
        self.time = time
        self.time_3F = time_3F
        self.num = num
        self.L = 0
        self.F = 0
        self.FR = 0
        self.R = 0
        self.direction = "st"
        self.simulator = simulator.simulator()
        self.height = self.simulator.height
        self.PIX = self.height * 10 - 20 # PIX = 600m で対応づけ
        if self.time != 0:
            self.coef_a = 2 * self.PIX / (3 * (self.time ** 2))
            self.coef_b = (4 * self.PIX) / (3 * (self.time - self.time_3F)) * (1 / (self.time_3F - self.time) - 1 / self.time)
            self.coef_c = 2 * self.PIX / (3 * self.time) + self.coef_b * self.time
        else:
            self.coef_a = 0
            self.coef_b = 0
            self.coef_c = 0
        assert (self.time > 0 and self.coef_a > 0) or self.time == 0
        assert (self.time > 0 and self.coef_c > 0) or self.time == 0
        if self.type == "FR":
            self.coef_in = 2
        else:
            self.coef_in = 1

    def velocity(self, t):
        if self.time == 0:
            self.v_temp = 0
        elif t <= self.time:
            self.v_temp = self.coef_a * 0.1
        else:
            self.v_temp = - self.coef_b * 0.1 + self.coef_c

    def distance(self, t):
        if self.time == 0:
            self.d = 0
        elif t <= 0.1:
            if t <= self.time:
                self.d = self.coef_a * (t ** 2) / 2
            else:
                self.d = self.coef_a * (self.time ** 2) / 2 - self.coef_b * (t ** 2 - self.time ** 2) / 2 + self.coef_c * (t - self.time)
        else:
            if t <= self.time:
                # self.d = self.coef_a * (t ** 2) / 2
                self.d = self.coef_a * (0.2 * t - 0.01) / 2
            else:
                # self.d = self.coef_a * (self.time ** 2) / 2 - self.coef_b * (t ** 2 - self.time ** 2) / 2 + self.coef_c * (t - self.time)
                self.d = -self.coef_b * (0.2 * t - 0.01) / 2 + self.coef_c * 0.1
            assert self.d >= 0

    def go(self):
        self.x += self.d
        self.y += self.dy
        self.v += self.v_temp
