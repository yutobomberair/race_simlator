import os
import shutil
import cv2
import numpy as np
import pandas as pd
import glob
from moviepy.editor import ImageClip, concatenate_videoclips

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QCheckBox, QComboBox,
                             QSlider, QLineEdit, QCalendarWidget, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui, QtCore

import simulator
import hourse_state
 
class gui_app(QWidget):
    items = ["FR", "IN", "OUT", "LA"]

    def __init__(self, parent=None):
        super(gui_app, self).__init__(parent)
        self.initUI()

    def initUI(self):
        def dec_combbox(self):
            vtypebox = QComboBox(self)
            for i in self.items:
                vtypebox.addItem(i)
            return vtypebox

        self.setGeometry(500, 100, 1600, 520) # left:500, right:1100
        self.setWindowTitle('race simulator')
        self.setWindowIcon(QIcon('./gui_parts/icon.png'))

         # フォーム(form)レイアウトの作成とメインウィンドウへの設定
        window = qtw.QHBoxLayout() # main window
        layout = qtw.QGridLayout() # param form
        view_layout = qtw.QVBoxLayout()
        botton_layout = qtw.QHBoxLayout()
        # layout.setGeometry(500, 100, 500, 520)
        
        self.graphic_view = qtw.QGraphicsView() # show movie
        self.scene = qtw.QGraphicsScene()
        image = QtGui.QImage('./gui_parts/waiting.png')
        pixmap = QtGui.QPixmap.fromImage(image)
        self.scene.addPixmap(pixmap)
        self.graphic_view.setScene(self.scene)

        view_layout.addWidget(self.graphic_view)
        view_layout.addLayout(botton_layout)
        
        window.addLayout(layout)
        window.addLayout(view_layout)
        self.setLayout(window)

        # ウィジェットを用意
        self.label1 = qtw.QLabel("馬番", self)
        self.label2 = qtw.QLabel("馬名", self)
        self.label3 = qtw.QLabel("テン1F", self)
        self.label4 = qtw.QLabel("テン3F", self)
        self.label5 = qtw.QLabel("テンtype", self)

        self.umaban1 = qtw.QLabel(" 1", self)
        self.umaban2 = qtw.QLabel(" 2", self)
        self.umaban3 = qtw.QLabel(" 3", self)
        self.umaban4 = qtw.QLabel(" 4", self)
        self.umaban5 = qtw.QLabel(" 5", self)
        self.umaban6 = qtw.QLabel(" 6", self)
        self.umaban7 = qtw.QLabel(" 7", self)
        self.umaban8 = qtw.QLabel(" 8", self)
        self.umaban9 = qtw.QLabel(" 9", self)
        self.umaban10 = qtw.QLabel("10", self)
        self.umaban11 = qtw.QLabel("11", self)
        self.umaban12 = qtw.QLabel("12", self)
        self.umaban13 = qtw.QLabel("13", self)
        self.umaban14 = qtw.QLabel("14", self)
        self.umaban15 = qtw.QLabel("15", self)
        self.umaban16 = qtw.QLabel("16", self)
        self.umaban17 = qtw.QLabel("17", self)
        self.umaban18 = qtw.QLabel("18", self)

        self.namebox1 = qtw.QLineEdit(self)
        self.namebox2 = qtw.QLineEdit(self)
        self.namebox3 = qtw.QLineEdit(self)
        self.namebox4 = qtw.QLineEdit(self)
        self.namebox5 = qtw.QLineEdit(self)
        self.namebox6 = qtw.QLineEdit(self)
        self.namebox7 = qtw.QLineEdit(self)
        self.namebox8 = qtw.QLineEdit(self)
        self.namebox9 = qtw.QLineEdit(self)
        self.namebox10 = qtw.QLineEdit(self)
        self.namebox11 = qtw.QLineEdit(self)
        self.namebox12 = qtw.QLineEdit(self)
        self.namebox13 = qtw.QLineEdit(self)
        self.namebox14 = qtw.QLineEdit(self)
        self.namebox15 = qtw.QLineEdit(self)
        self.namebox16 = qtw.QLineEdit(self)
        self.namebox17 = qtw.QLineEdit(self)
        self.namebox18 = qtw.QLineEdit(self)

        self.timebox1 = qtw.QLineEdit(self)
        self.timebox2 = qtw.QLineEdit(self)
        self.timebox3 = qtw.QLineEdit(self)
        self.timebox4 = qtw.QLineEdit(self)
        self.timebox5 = qtw.QLineEdit(self)
        self.timebox6 = qtw.QLineEdit(self)
        self.timebox7 = qtw.QLineEdit(self)
        self.timebox8 = qtw.QLineEdit(self)
        self.timebox9 = qtw.QLineEdit(self)
        self.timebox10 = qtw.QLineEdit(self)
        self.timebox11 = qtw.QLineEdit(self)
        self.timebox12 = qtw.QLineEdit(self)
        self.timebox13 = qtw.QLineEdit(self)
        self.timebox14 = qtw.QLineEdit(self)
        self.timebox15 = qtw.QLineEdit(self)
        self.timebox16 = qtw.QLineEdit(self)
        self.timebox17 = qtw.QLineEdit(self)
        self.timebox18 = qtw.QLineEdit(self)

        self.timebox1_3F = qtw.QLineEdit(self)
        self.timebox2_3F = qtw.QLineEdit(self)
        self.timebox3_3F = qtw.QLineEdit(self)
        self.timebox4_3F = qtw.QLineEdit(self)
        self.timebox5_3F = qtw.QLineEdit(self)
        self.timebox6_3F = qtw.QLineEdit(self)
        self.timebox7_3F = qtw.QLineEdit(self)
        self.timebox8_3F = qtw.QLineEdit(self)
        self.timebox9_3F = qtw.QLineEdit(self)
        self.timebox10_3F = qtw.QLineEdit(self)
        self.timebox11_3F = qtw.QLineEdit(self)
        self.timebox12_3F = qtw.QLineEdit(self)
        self.timebox13_3F = qtw.QLineEdit(self)
        self.timebox14_3F = qtw.QLineEdit(self)
        self.timebox15_3F = qtw.QLineEdit(self)
        self.timebox16_3F = qtw.QLineEdit(self)
        self.timebox17_3F = qtw.QLineEdit(self)
        self.timebox18_3F = qtw.QLineEdit(self)

        self.vtypebox1 = dec_combbox(self)
        self.vtypebox2 = dec_combbox(self)
        self.vtypebox3 = dec_combbox(self)
        self.vtypebox4 = dec_combbox(self)
        self.vtypebox5 = dec_combbox(self)
        self.vtypebox6 = dec_combbox(self)
        self.vtypebox7 = dec_combbox(self)
        self.vtypebox8 = dec_combbox(self)
        self.vtypebox9 = dec_combbox(self)
        self.vtypebox10 = dec_combbox(self)
        self.vtypebox11 = dec_combbox(self)
        self.vtypebox12 = dec_combbox(self)
        self.vtypebox13 = dec_combbox(self)
        self.vtypebox14 = dec_combbox(self)
        self.vtypebox15 = dec_combbox(self)
        self.vtypebox16 = dec_combbox(self)
        self.vtypebox17 = dec_combbox(self)
        self.vtypebox18 = dec_combbox(self)
        
        # レイアウトにウィジェットを追加
        layout.addWidget(self.label1, 0, 0)
        layout.addWidget(self.label2, 0, 1)
        layout.addWidget(self.label3, 0, 2)
        layout.addWidget(self.label4, 0, 3)
        layout.addWidget(self.label5, 0, 4)

        layout.addWidget(self.umaban1, 1, 0)
        layout.addWidget(self.umaban2, 2, 0)
        layout.addWidget(self.umaban3, 3, 0)
        layout.addWidget(self.umaban4, 4, 0)
        layout.addWidget(self.umaban5, 5, 0)
        layout.addWidget(self.umaban6, 6, 0)
        layout.addWidget(self.umaban7, 7, 0)
        layout.addWidget(self.umaban8, 8, 0)
        layout.addWidget(self.umaban9, 9, 0)
        layout.addWidget(self.umaban10, 10, 0)
        layout.addWidget(self.umaban11, 11, 0)
        layout.addWidget(self.umaban12, 12, 0)
        layout.addWidget(self.umaban13, 13, 0)
        layout.addWidget(self.umaban14, 14, 0)
        layout.addWidget(self.umaban15, 15, 0)
        layout.addWidget(self.umaban16, 16, 0)
        layout.addWidget(self.umaban17, 17, 0)
        layout.addWidget(self.umaban18, 18, 0)

        layout.addWidget(self.namebox1, 1, 1)
        layout.addWidget(self.namebox2, 2, 1)
        layout.addWidget(self.namebox3, 3, 1)
        layout.addWidget(self.namebox4, 4, 1)
        layout.addWidget(self.namebox5, 5, 1)
        layout.addWidget(self.namebox6, 6, 1)
        layout.addWidget(self.namebox7, 7, 1)
        layout.addWidget(self.namebox8, 8, 1)
        layout.addWidget(self.namebox9, 9, 1)
        layout.addWidget(self.namebox10, 10, 1)
        layout.addWidget(self.namebox11, 11, 1)
        layout.addWidget(self.namebox12, 12, 1)
        layout.addWidget(self.namebox13, 13, 1)
        layout.addWidget(self.namebox14, 14, 1)
        layout.addWidget(self.namebox15, 15, 1)
        layout.addWidget(self.namebox16, 16, 1)
        layout.addWidget(self.namebox17, 17, 1)
        layout.addWidget(self.namebox18, 18, 1)

        layout.addWidget(self.timebox1, 1, 2)
        layout.addWidget(self.timebox2, 2, 2)
        layout.addWidget(self.timebox3, 3, 2)
        layout.addWidget(self.timebox4, 4, 2)
        layout.addWidget(self.timebox5, 5, 2)
        layout.addWidget(self.timebox6, 6, 2)
        layout.addWidget(self.timebox7, 7, 2)
        layout.addWidget(self.timebox8, 8, 2)
        layout.addWidget(self.timebox9, 9, 2)
        layout.addWidget(self.timebox10, 10, 2)
        layout.addWidget(self.timebox11, 11, 2)
        layout.addWidget(self.timebox12, 12, 2)
        layout.addWidget(self.timebox13, 13, 2)
        layout.addWidget(self.timebox14, 14, 2)
        layout.addWidget(self.timebox15, 15, 2)
        layout.addWidget(self.timebox16, 16, 2)
        layout.addWidget(self.timebox17, 17, 2)
        layout.addWidget(self.timebox18, 18, 2)

        layout.addWidget(self.timebox1_3F, 1, 3)
        layout.addWidget(self.timebox2_3F, 2, 3)
        layout.addWidget(self.timebox3_3F, 3, 3)
        layout.addWidget(self.timebox4_3F, 4, 3)
        layout.addWidget(self.timebox5_3F, 5, 3)
        layout.addWidget(self.timebox6_3F, 6, 3)
        layout.addWidget(self.timebox7_3F, 7, 3)
        layout.addWidget(self.timebox8_3F, 8, 3)
        layout.addWidget(self.timebox9_3F, 9, 3)
        layout.addWidget(self.timebox10_3F, 10, 3)
        layout.addWidget(self.timebox11_3F, 11, 3)
        layout.addWidget(self.timebox12_3F, 12, 3)
        layout.addWidget(self.timebox13_3F, 13, 3)
        layout.addWidget(self.timebox14_3F, 14, 3)
        layout.addWidget(self.timebox15_3F, 15, 3)
        layout.addWidget(self.timebox16_3F, 16, 3)
        layout.addWidget(self.timebox17_3F, 17, 3)
        layout.addWidget(self.timebox18_3F, 18, 3)

        layout.addWidget(self.vtypebox1, 1, 4)
        layout.addWidget(self.vtypebox2, 2, 4)
        layout.addWidget(self.vtypebox3, 3, 4)
        layout.addWidget(self.vtypebox4, 4, 4)
        layout.addWidget(self.vtypebox5, 5, 4)
        layout.addWidget(self.vtypebox6, 6, 4)
        layout.addWidget(self.vtypebox7, 7, 4)
        layout.addWidget(self.vtypebox8, 8, 4)
        layout.addWidget(self.vtypebox9, 9, 4)
        layout.addWidget(self.vtypebox10, 10, 4)
        layout.addWidget(self.vtypebox11, 11, 4)
        layout.addWidget(self.vtypebox12, 12, 4)
        layout.addWidget(self.vtypebox13, 13, 4)
        layout.addWidget(self.vtypebox14, 14, 4)
        layout.addWidget(self.vtypebox15, 15, 4)
        layout.addWidget(self.vtypebox16, 16, 4)
        layout.addWidget(self.vtypebox17, 17, 4)
        layout.addWidget(self.vtypebox18, 18, 4)

        # ボタン
        self.btn0 = QPushButton("mode", self)
        self.btn1 = QPushButton("set param", self)
        self.btn2 = QPushButton("start simulation", self)
        self.btn3 = QPushButton("read", self)
        self.btn4 = QPushButton("restart", self)

        botton_layout.addWidget(self.btn0)
        botton_layout.addWidget(self.btn3)
        botton_layout.addWidget(self.btn1)
        botton_layout.addWidget(self.btn2)
        botton_layout.addWidget(self.btn4)

    def stop_video(self): #動画表示用
        self.video.release()

    def start_video(self): #動画表示用
        #動画表示用のwidget
        self.centralwidget = qtw.QWidget(self)
        self.graphic_view = qtw.QGraphicsView(self.centralwidget)
        self.horizontalSlider = qtw.QSlider(self.centralwidget)
        self.horizontalSlider.setOrientation(Qt.Horizontal)

        self.time_line = QtCore.QTimeLine()
        self.time_line.valueChanged.connect(self.draw_frame)
        self.time_line.finished.connect(self.stop_video)
        self.video = cv2.VideoCapture('./mov/sim.mp4')

        frame_rate = self.video.get(cv2.CAP_PROP_FPS) * 4 # FPSの取得
        frame_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数の取得
        frame_time = 1000 / frame_rate
        video_time = frame_count / frame_rate * 1000
        self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.horizontalSlider.setMaximum(int(frame_count))
        self.graphic_view.setScene( 
            qtw.QGraphicsScene(0, 0, self.video_width, self.video_height, self.graphic_view) 
        )
        
        self.time_line.setDuration(int(video_time))
        self.time_line.setUpdateInterval(int(frame_time))
        self.time_line.start()

    def draw_frame(self): #動画表示用
        ret, frame = self.video.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data
            image = QtGui.QImage(rgb, self.video_width, self.video_height, self.video_width*3, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.graphic_view.scene().clear()
            self.graphic_view.scene().addPixmap(pixmap)
        self.horizontalSlider.setValue(self.horizontalSlider.value()+1)

    def set_param(self):
        self.name1 = str(self.namebox1.text())
        self.name2 = str(self.namebox2.text())
        self.name3 = str(self.namebox3.text())
        self.name4 = str(self.namebox4.text())
        self.name5 = str(self.namebox5.text())
        self.name6 = str(self.namebox6.text())
        self.name7 = str(self.namebox7.text())
        self.name8 = str(self.namebox8.text())
        self.name9 = str(self.namebox9.text())
        self.name10 = str(self.namebox10.text())
        self.name11 = str(self.namebox11.text())
        self.name12 = str(self.namebox12.text())
        self.name13 = str(self.namebox13.text())
        self.name14 = str(self.namebox14.text())
        self.name15 = str(self.namebox15.text())
        self.name16 = str(self.namebox16.text())
        self.name17 = str(self.namebox17.text())
        self.name18 = str(self.namebox18.text())

        self.time1 = str(self.timebox1.text())
        self.time2 = str(self.timebox2.text())
        self.time3 = str(self.timebox3.text())
        self.time4 = str(self.timebox4.text())
        self.time5 = str(self.timebox5.text())
        self.time6 = str(self.timebox6.text())
        self.time7 = str(self.timebox7.text())
        self.time8 = str(self.timebox8.text())
        self.time9 = str(self.timebox9.text())
        self.time10 = str(self.timebox10.text())
        self.time11 = str(self.timebox11.text())
        self.time12 = str(self.timebox12.text())
        self.time13 = str(self.timebox13.text())
        self.time14 = str(self.timebox14.text())
        self.time15 = str(self.timebox15.text())
        self.time16 = str(self.timebox16.text())
        self.time17 = str(self.timebox17.text())
        self.time18 = str(self.timebox18.text())

        self.time1_3F = str(self.timebox1_3F.text())
        self.time2_3F = str(self.timebox2_3F.text())
        self.time3_3F = str(self.timebox3_3F.text())
        self.time4_3F = str(self.timebox4_3F.text())
        self.time5_3F = str(self.timebox5_3F.text())
        self.time6_3F = str(self.timebox6_3F.text())
        self.time7_3F = str(self.timebox7_3F.text())
        self.time8_3F = str(self.timebox8_3F.text())
        self.time9_3F = str(self.timebox9_3F.text())
        self.time10_3F = str(self.timebox10_3F.text())
        self.time11_3F = str(self.timebox11_3F.text())
        self.time12_3F = str(self.timebox12_3F.text())
        self.time13_3F = str(self.timebox13_3F.text())
        self.time14_3F = str(self.timebox14_3F.text())
        self.time15_3F = str(self.timebox15_3F.text())
        self.time16_3F = str(self.timebox16_3F.text())
        self.time17_3F = str(self.timebox17_3F.text())
        self.time18_3F = str(self.timebox18_3F.text())

        self.vtype1 = str(self.vtypebox1.currentText())
        self.vtype2 = str(self.vtypebox2.currentText())
        self.vtype3 = str(self.vtypebox3.currentText())
        self.vtype4 = str(self.vtypebox4.currentText())
        self.vtype5 = str(self.vtypebox5.currentText())
        self.vtype6 = str(self.vtypebox6.currentText())
        self.vtype7 = str(self.vtypebox7.currentText())
        self.vtype8 = str(self.vtypebox8.currentText())
        self.vtype9 = str(self.vtypebox9.currentText())
        self.vtype10 = str(self.vtypebox10.currentText())
        self.vtype11 = str(self.vtypebox11.currentText())
        self.vtype12 = str(self.vtypebox12.currentText())
        self.vtype13 = str(self.vtypebox13.currentText())
        self.vtype14 = str(self.vtypebox14.currentText())
        self.vtype15 = str(self.vtypebox15.currentText())
        self.vtype16 = str(self.vtypebox16.currentText())
        self.vtype17 = str(self.vtypebox17.currentText())
        self.vtype18 = str(self.vtypebox18.currentText())

        print("set paramerters")

    def start_sim(self):
        print("start simulation")
        hname = [self.name1, self.name2, self.name3, self.name4, self.name5, self.name6
                , self.name7, self.name8, self.name9, self.name10, self.name11, self.name12
                , self.name13, self.name14, self.name15, self.name16, self.name17, self.name18]
        vtype = [self.vtype1, self.vtype2, self.vtype3, self.vtype4, self.vtype5, self.vtype6
                , self.vtype7, self.vtype8, self.vtype9, self.vtype10, self.vtype11, self.vtype12
                , self.vtype13, self.vtype14, self.vtype15, self.vtype16, self.vtype17, self.vtype18]
        time = [self.time1, self.time2, self.time3, self.time4, self.time5, self.time6
                , self.time7, self.time8, self.time9, self.time10, self.time11, self.time12
                , self.time13, self.time14, self.time15, self.time16, self.time17, self.time18]
        time_3F = [self.time1_3F, self.time2_3F, self.time3_3F, self.time4_3F, self.time5_3F, self.time6_3F
                , self.time7_3F, self.time8_3F, self.time9_3F, self.time10_3F, self.time11_3F, self.time12_3F
                , self.time13_3F, self.time14_3F, self.time15_3F, self.time16_3F, self.time17_3F, self.time18_3F]
        
        self.simulator = simulator.simulator()
        self.simulator.mkdir()
        self.simulator.pic_format()
        hname = [i for i in hname if i is not None]
        time = [float(i) for i in time[0:len(hname)]]
        time_3F = [float(i) for i in time_3F[0:len(hname)]]
        vtype = vtype[0:len(hname)]

        self.simulator.hnum = self.simulator.hnum[0:len(hname)]
        self.simulator.y_coord = self.simulator.y_coord[18-len(hname):]
        instance = [hourse_state.hourse_state(y, h, v, t, t3, num) for y, h, v, t, t3, num in zip(reversed(self.simulator.y_coord), hname, vtype, time, time_3F, self.simulator.hnum)]
        for t in range(500):
            copied_img = self.simulator.start.copy()
            instance = self.simulator.behavior(instance, t)
            copied_img = self.simulator.make_picture(instance, copied_img)
            top = max([i.x for i in instance])
            self.simulator.angle_of_view(top, copied_img, t)
            if top >= self.simulator.PIX:
                break

        self.simulator.make_movie()
        # self.simulator.show_movie()
    
    def screen_transition(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.scene.addPixmap(pixmap)
        self.graphic_view.setScene(self.scene)
        self.show()

    def sim_transition(self):
        self.start_sim()
        self.start_video()
        self.screen_transition()
        # image = QtGui.QImage('./gui_parts/waiting.png')
        # self.screen_transition(image)
    
    def restart(self):
        self.start_video()
        self.screen_transition()
        # image = QtGui.QImage('./gui_parts/waiting.png')
        # self.screen_transition(image)

    def read_paramfile(self):
        namebox = [self.namebox1, self.namebox2, self.namebox3, self.namebox4, self.namebox5, self.namebox6
                , self.namebox7, self.namebox8, self.namebox9, self.namebox10, self.namebox11, self.namebox12
                , self.namebox13, self.namebox14, self.namebox15, self.namebox16, self.namebox17, self.namebox18]
        timebox = [self.timebox1, self.timebox2, self.timebox3, self.timebox4, self.timebox5, self.timebox6
                , self.timebox7, self.timebox8, self.timebox9, self.timebox10, self.timebox11, self.timebox12
                , self.timebox13, self.timebox14, self.timebox15, self.timebox16, self.timebox17, self.timebox18]
        timebox_3F = [self.timebox1_3F, self.timebox2_3F, self.timebox3_3F, self.timebox4_3F, self.timebox5_3F, self.timebox6_3F
                , self.timebox7_3F, self.timebox8_3F, self.timebox9_3F, self.timebox10_3F, self.timebox11_3F, self.timebox12_3F
                , self.timebox13_3F, self.timebox14_3F, self.timebox15_3F, self.timebox16_3F, self.timebox17_3F, self.timebox18_3F]
        vtypebox = [self.vtypebox1, self.vtypebox2, self.vtypebox3, self.vtypebox4, self.vtypebox5, self.vtypebox6
                , self.vtypebox7, self.vtypebox8, self.vtypebox9, self.vtypebox10, self.vtypebox11, self.vtypebox12
                , self.vtypebox13, self.vtypebox14, self.vtypebox15, self.vtypebox16, self.vtypebox17, self.vtypebox18]
        
        df = pd.read_csv("./param/takaraduka.csv")
        name = df["name"].tolist()
        time = df["time"].tolist()
        time_3F = df["time3F"].tolist()
        vtype = df["type"].tolist()

        for i in range(len(name)):
            namebox[i].setText(str(name[i]))
            timebox[i].setText(str(time[i]))
            timebox_3F[i].setText(str(time_3F[i]))
            vtypebox[i].setCurrentIndex(self.items.index(str(vtype[i])))


if __name__ == '__main__':
    app = QApplication(sys.argv) #PyQtで必ず呼び出す必要のあるオブジェクト
    main_window = gui_app() #ウィンドウクラスのオブジェクト生成
    main_window.btn1.clicked.connect(main_window.set_param)
    main_window.btn2.clicked.connect(main_window.sim_transition)   
    main_window.btn3.clicked.connect(main_window.read_paramfile) 
    main_window.btn3.clicked.connect(main_window.start_video)   
    
    main_window.show() #ウィンドウの表示
    sys.exit(app.exec_()) #ループ表示
