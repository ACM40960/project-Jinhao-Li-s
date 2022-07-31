import os
import cv2 as cv
import numpy as np
from math import *
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9,
              "苏": 10, "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19,
              "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29,
              "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39,
              "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49,
              "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59,
              "V": 60, "W": 61, "X": 62, "Y": 63, "Z": 64}

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
              "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
              "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
              "新", "0", "1", "2", "3", "4", "5", "6", "7", "8",
              "9", "A", "B", "C", "D", "E", "F", "G", "H", "J",
              "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
              "V", "W", "X", "Y", "Z"]
def SPD(img, angel, shape, max_angel):
    """
    Sing perspective distortion
    """
    size_o = [shape[1], shape[0]]
    size = (shape[1]+ int(shape[0] * cos((float(max_angel ) / 180) * 3.14)), shape[0])
    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))
    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
    PT = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, PT, size)
    return dst


def SPDRandrom(img, factor, size):
    """
    Add radiation distortion
    :param img: The input image
    :param factor: Parameter of distortion
    :param size: Image target size
    :return: images after add radiation distortion
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    PT = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, PT, size)
    return dst


def sln(img):
    """
    Add saturation light noise
    """
    hv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    hv[:, :, 0] = hv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hv[:, :, 1] = hv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hv[:, :, 2] = hv[:, :, 2] * (0.2 + np.random.random() * 0.8)
    img = cv.cvtColor(hv, cv.COLOR_HSV2BGR)
    return img

def enoise(img, noplate_bg):
    """
    Add the noise of the natural environment, noplate_bg is the background image without the license plate
    """
    bg_index = r(len(noplate_bg))
    env = cv.imread(noplate_bg[bg_index])
    env = cv.resize(env, (img.shape[1], img.shape[0]))
    bak = (img == 0)
    bak = bak.astype(np.uint8) * 255
    inv = cv.bitwise_and(bak, env)
    img = cv.bitwise_or(inv, img)
    return img

def GenCh(f, val):
    """
    Generating Chinese characters
    """
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, (0, 0, 0), font=f)
    img =  img.resize((23, 70))
    A = np.array(img)
    return A

def GenEn(f, val):
    """
    Generating English characters
    """
    img =Image.new("RGB", (23, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2), val, (0, 0, 0), font=f)    # val.decode('utf-8')
    A = np.array(img)
    return A


def AG(img, level):
    """
    Add Gaussian blur
    """ 
    return cv.blur(img, (level * 2 + 1, level * 2 + 1))

def r(val):
    return int(np.random.random() * val)

def ADN(single):
    """
    Add Gaussian noise
    """
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + r(6), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise *= diff
    dst = single + noise
    return dst

def AD(img): 
    img[:, :, 0] = ADN(img[:, :, 0])
    img[:, :, 1] = ADN(img[:, :, 1])
    img[:, :, 2] = ADN(img[:, :, 2])
    return img
 
class GenPlate:
    def __init__(self, fontCh, fontEng, NoPlates):
        self.fontC = ImageFont.truetype(fontCh, 43, 0)
        self.fontE = ImageFont.truetype(fontEng, 60, 0)
        self.img = np.array(Image.new("RGB", (226, 70),(255, 255, 255)))
        # template.bmp:Background of license plate
        self.bg  = cv.resize(cv.imread("./data/images/template.bmp"), (226, 70))
        # smu2.jpg:Blurred image
        self.smu = cv.imread("./data/images/smu2.jpg")   
        self.noplates_path = []
        for parent, parent_folder, filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent + "/" + filename
                self.noplates_path.append(path)
 #Generate license plate numbers
    def draw(self, val):
        offset = 2
        #GenCh() Generating Chinese characters
        self.img[0:70, offset+8:offset+8+23] = GenCh(self.fontC, val[0])
        #GenEn Generating English characters
        self.img[0:70, offset+8+23+6:offset+8+23+6+23] = GenEn(self.fontE, val[1])
        #GenEn Generate the remaining 5 digits
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base:base+23] = GenEn(self.fontE, val[i+2])
        return self.img    
    def generate(self, text):
        if len(text) == 7:
            #Perform binary operations on each pixel of the image
            fg = self.draw(text)   
            fg = cv.bitwise_not(fg)
            com = cv.bitwise_or(fg, self.bg)
            #Adding perspective distortion
            com = SPD(com, r(60)-30, com.shape,30)   
            #Add radiation distortion
            com = SPDRandrom(com, 10, (com.shape[1], com.shape[0]))  
            # Add saturation light noise
            com = sln(com)
            #Add noise from the natural environment
            com = enoise(com, self.noplates_path)
            # Add Gaussian blur
            com = AG(com, 1+r(4))
            #Add Gaussian noise
            com = AD(com)
            return com
    @staticmethod
    def genPlateString(pos, val):
        """
	    Generate a license plate string and save it as an 'image'
        Generate license plate list and save it as 'label'
        """
        plateStr = ""  
        plateList=[]
        box = [0, 0, 0, 0, 0, 0, 0]
        if pos != -1:
            box[pos] = 1
        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plateStr += val
                plateList.append(val)
            else:
                if cpos == 0:
                    plateStr += chars[r(31)]
                    plateList.append(plateStr)
                elif cpos == 1:
                    plateStr += chars[41 + r(24)]
                    plateList.append(plateStr)
                else:
                    plateStr += chars[31 + r(34)]
                    plateList.append(plateStr)
        plate = [plateList[0]]
        b = [plateList[i][-1] for i in range(len(plateList))]
        plate.extend(b[1:7])
        return plateStr, plate

    @staticmethod
    def genBatch(batchsize, outputPath, size):
        """
        Write the generated license plate picture into the folder and the corresponding label into label.txt
        :param batchsize:  the size of batch
        :param outputPath: Save path of output image
        :param size: Size of the output image
        :return: None
        """
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        outfile = open('./data/plate/label.txt', 'w', encoding='utf-8')
        for i in range(batchsize):
            plateStr, plate = G.genPlateString(-1, -1)
            # print(plateStr, plate)
            img = G.generate(plateStr)
            img = cv.resize(img, size)
            #imwrite()Use to save the image to the specified file
            #zfill() Method returns a string of the specified length 00-100
            cv.imwrite(outputPath + "/" + str(i).zfill(2) + ".jpg", img)
            outfile.write(str(plate) + "\n")

if __name__ == '__main__':
    G = GenPlate("./data/font/platech.ttf", './data/font/platechar.ttf', "./data/NoPlates")
    G.genBatch(101, './data/plate', (272, 72))



