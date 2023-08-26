import os
import shutil
import cv2
import numpy as np
import glob
from moviepy.editor import ImageClip, concatenate_videoclips
from operator import attrgetter

import hourse_state
import color_variation

###############################################params######################################################
hname = ["", "アフリカンゴールド", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
vtype = ["dash", "dash", "linear", "linear", "linear", "dash", "linear", "dash", "linear", "dash", "dash", "dash", "linear", "linear", "dash", "linear", "dash", "linear"]
vtype = ["linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear", "linear"]
time = [0, 128, 136, 133, 139, 125, 134, 129, 132, 130, 125, 128, 134, 139, 126, 138, 130, 136]
time = [0, 12.8, 13.6, 13.3, 13.9, 12.5, 13.4, 12.9, 13.2, 13.0, 12.5, 12.8, 13.4, 13.9, 12.6, 13.8, 13.0, 13.6]
####################################################################################################################################
class simulator():
    def __init__(self):
        # input画像の加工
        self.start = cv2.imread("./mov/start.png")
        self.PIX = self.start.shape[0] * 10
        self.width = self.start.shape[1]
        self.height = self.start.shape[0]
        self.y_coord = [12 + 25 * i for i in range(19)]
        del self.y_coord[-13]
        self.hnum = [i + 1 for i in range(18)]
        self.color_variation = color_variation.color_variation()
        
    def pic_format(self):
        real_width = self.height * 10 # 表示の関係で横幅は200m分のpixを600mとして
        necessary = real_width - (self.width) # x=20から発馬するため

        rem = necessary % 100
        block = (necessary - rem) / 100

        rem_img = self.start[:, self.width-rem:self.width-1]
        block_img = self.start[:, self.width-100:self.width-1]

        self.start = cv2.hconcat([self.start, rem_img])
        for i in range(int(block)):
            self.start = cv2.hconcat([self.start, block_img])
        self.PIX = self.start.shape[1] - 20

        x_1_3 = int(20 + (self.PIX - 20) / 3) - 5
        x_2_3 = int(20 + (self.PIX - 20) * 2 / 3) - 5
        x_3_3 = 20 + self.PIX - 5
        cv2.line(self.start, (x_1_3, 0), (x_1_3, self.height-1), (0, 0, 0), thickness=5)
        cv2.line(self.start, (x_2_3, 0), (x_2_3, self.height-1), (0, 0, 0), thickness=5)
        cv2.line(self.start, (x_3_3, 0), (x_3_3, self.height-1), (0, 0, 0), thickness=5)
        cv2.putText(self.start,
                            text="200m",
                            org=(x_1_3-200, 60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=3,
                            lineType=cv2.LINE_4)
        cv2.putText(self.start,
                            text="400m",
                            org=(x_2_3-200, 60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=3,
                            lineType=cv2.LINE_4)
        cv2.putText(self.start,
                            text="600m",
                            org=(x_3_3-200, 60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=3,
                            lineType=cv2.LINE_4)

    def mkdir(self):
        if(os.path.isdir("./mov/img")):
            shutil.rmtree("./mov/img")
        os.mkdir("./mov/img")

    def angle_of_view(self, top, copied_img, t):
        cp_start = self.start.copy()
        div = 5.5
        if top <= (self.PIX / div) * 0.9:
            left = 0
            right = self.PIX / div
        elif top + (self.PIX / div) * 0.1 >= self.PIX:
            left = self.PIX * (1 - 1 / div)
            right = self.PIX
        else:
            left = top - (self.PIX / div) * 0.9
            right = top + (self.PIX / div) * 0.1
        cut_angle = copied_img[:, int(left):int(right)]
        cp_start = cv2.rectangle(cp_start, (int(left), 0), (int(right), int(self.height-1)), (0, 0, 255), thickness=25)
        cp_start = cv2.resize(cp_start, (int(self.start.shape[1]/div), int(self.start.shape[0]/div)))
        black_img = np.stack(np.zeros((10, int(self.PIX / div)),np.uint8)*3, -1)
        black_img = np.repeat(black_img, 3).reshape(black_img.shape[1],black_img.shape[0],3)
        black_img = cv2.resize(black_img, (cp_start.shape[1], int(cp_start.shape[0]/3)))
        output = cv2.vconcat([black_img, cp_start])
        if output.shape[1] != cut_angle.shape[1]:
            cut_angle = cv2.resize(cut_angle, (output.shape[1], cut_angle.shape[0]))
        output = cv2.vconcat([cut_angle, output])
        cv2.putText(output,
                    text=str("t=")+str(int(0.1*t)),
                    org=(60, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_4)
        cv2.imwrite("./mov/img/img_"+f"{t:04}"+".png", output)

    def behavior(self, instance, t):
        refine_ins = []
        for N, I in enumerate(instance):
            new_ins = [i for n, i in enumerate(instance) if n != N]
            I = self.set_around(I, new_ins)
            I = self.decide_direction(I, instance)
            refine_ins.append(I)
        refine_ins = self.move(refine_ins, t)
        return refine_ins

    def move(self, instance, t):
        for i in instance:
            if i.direction == "bk":
                assert instance[i.F - 1].num == i.F
                instance[i.F - 1].velocity(0.1 * t)
                instance[i.F - 1].distance(0.1 * t)
                i.d = instance[i.F - 1].d * 0.9
                i.v_temp = instance[i.F - 1].v_temp
                i.dy = 0
            elif i.direction == "st":
                i.velocity(0.1 * t)
                i.distance(0.1 * t)
                i.dy = 0
            elif i.direction == "in":
                i.velocity(0.1 * t)
                i.distance(0.1 * t)
                if i.y < 462:
                    i.dy = 1.2 * i.coef_in
                else:
                    i.dy = 0
            elif i.direction == "out":
                i.velocity(0.1 * t)
                i.distance(0.1 * t)
                if i.y > 10:
                    i.dy = - 1.5
                else:
                    i.dy = 0
            elif i.direction == "stop":
                i.d = 0
                i.dy = 0
            i.go()
        return instance

    def set_around(self, I, new_ins):
        center_L = (I.x-10, I.y-10)
        center_R = (I.x+10, I.y+10)
        Front_L = (center_L[0]+20, center_L[1])
        Front_R = (center_R[0]+20, center_R[1])
        Right_L = (center_L[0], center_L[1]+20)
        Right_R = (center_R[0], center_R[1]+20)
        FR_L = (Right_L[0]+20, Right_L[1])
        FR_R = (Right_R[0]+20, Right_R[1])
        Left_L = (center_L[0], center_L[1]-20)
        Left_R = (center_L[0]+40, center_L[1])
        I.L = 0
        I.F = 0
        I.R = 0
        I.FR = 0
        for i in new_ins:
            if Front_L[0] <= i.x and i.x <= Front_R[0] and Front_L[1] <= i.y and i.y <= Front_R[1]:
                I.F = i.num
            if Right_L[0] <= i.x and i.x <= Right_R[0] and Right_L[1] <= i.y and i.y <= Right_R[1]:
                I.R = i.num
            if FR_L[0] <= i.x and i.x <= FR_R[0] and FR_L[1] <= i.y and i.y <= FR_R[1]:
                I.FR = i.num
            if Left_L[0] <= i.x and i.x <= Left_R[0] and Left_L[1] <= i.y and i.y <= Left_R[1]:
                I.L = i.num
        return I
    
    def decide_direction(self, I, instance):
        if I.time == 0:
            I.direction = "stop"
        elif I.x <= 40:
            I.direction = "st"
        elif I.F != 0:
            if I.L == 0:
                I.direction = "out"
            elif I.v >= instance[I.F - 1].v:
                I.direction = "bk"
            else:
                if I.x + 15 <= instance[I.F - 1].x:
                    I.direction = "st"
                else:
                    I.direction = "bk" 
        elif I.R != 0 or I.FR != 0:
            if I.L == 0:
                I.direction = "out"
            else:
                I.direction = "st"
        elif I.FR == 0 and I.R == 0:
            I.direction = "in"
        return I

    def make_picture(self, instance, copied_img):
        self.select_color(instance)
        for i in instance:
            copied_img = cv2.circle(copied_img, (int(i.x), int(i.y)), 10, color=self.COLOR[i.num], thickness=-1)
            color = (255, 255, 255)
            if self.COLOR[i.num] == (255, 255, 255) or self.COLOR[i.num] == (0, 255, 255):
                color = (0, 0, 0)
            if len(str(i.num)) != 1:
                x = i.x-9
            else:
                x = i.x-5
            cv2.putText(copied_img,
                        text=str(i.num),
                        org=(int(x), int(i.y+3)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.45,
                        color=color,
                        thickness=1,
                        lineType=cv2.LINE_4)
        return copied_img

    def select_color(self, instance):
        length = len([i for i in instance if i.name != "nan"])
        if length == 18:
            self.COLOR = self.color_variation.COLOR18
        elif length == 17:
            self.COLOR = self.color_variation.COLOR17
        elif length == 16:
            self.COLOR = self.color_variation.COLOR16
        elif length == 15:
            self.COLOR = self.color_variation.COLOR15
        elif length == 14:
            self.COLOR = self.color_variation.COLOR14
        elif length == 13:
            self.COLOR = self.color_variation.COLOR13
        elif length == 12:
            self.COLOR = self.color_variation.COLOR12
        elif length == 11:
            self.COLOR = self.color_variation.COLOR11
        elif length == 10:
            self.COLOR = self.color_variation.COLOR10
        elif length == 9:
            self.COLOR = self.color_variation.COLOR9
        elif length == 8:
            self.COLOR = self.color_variation.COLOR8
        elif length == 7:
            self.COLOR = self.color_variation.COLOR7
        elif length == 6:
            self.COLOR = self.color_variation.COLOR6
        elif length == 5:
            self.COLOR = self.color_variation.COLOR5
        else:
            self.COLOR = self.color_variation.COLOR18

    def make_movie(self):
        # inputディレクトリ以下の拡張子が.jpgのファイル名リストを一括取得
        file_list = glob.glob("./mov/img/*")
        # ファイル名リストを昇順にソート
        file_list.sort()

        # スライドショーを作る元となる静止画情報を格納する処理
        clips = [] 
        for m in file_list:
            clip = ImageClip(m).set_duration('00:00:00.60')
            # clip = clip.resize(newsize=(size[1], size[0]))
            clips.append(clip)

        # スライドショーの動画像を作成する処理
        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(r"./mov/sim.mp4", 
                                    fps=5,
                                    write_logfile=False,
                                    codec='libx264', 
                                    audio_codec='aac', 
                                    temp_audiofile='temp-audio.m4a', 
                                    remove_temp=True)

if __name__ == '__main__':
    simulator = simulator()
    simulator.mkdir()
    simulator.pic_format()
    instance = [hourse_state.hourse_state(y, h, v, t, num) for y, h, v, t, num in zip(reversed(simulator.y_coord), hname, vtype, time, simulator.hnum)]
    for t in range(100):
        copied_img = simulator.start.copy()
        for i in instance:
            i.velocity(0.1 * t)
            i.distance(0.1 * t)
            i.go()
            copied_img = cv2.circle(copied_img, (int(i.x), int(i.y)), 7, color=simulator.COLOR[i.num], thickness=-1)
        top = max([i.x for i in instance])
        simulator.angle_of_view(top, copied_img, t)
        if instance[1].x >= simulator.PIX:
            break

    simulator.make_movie()
    simulator.show_movie()
    
    