<!-- # CNN China license plate recognition model based on Tensorflow -->

<h1 align="center">
  <img alt="cover" src="./readme photo/cover.png"/><br/>
</h1>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<p align="center">
本模型采用的是<b>同时</b>训练车牌中七个字符，七个字符使用<b>单独的loss函数</b>进行训练。<br/>
对于一些非高清摄像头所拍摄的车牌图片，由于图片中<b>环境噪声和图片畸变</b>的影响，分割车牌的字符变得很难。<br/>
本模型通过随机生成的车牌号码图片来训练CNN，使模型对模糊车牌号图片具有较高的识别精度。<br/>

</p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<p align="center">    
  <img src="https://img.shields.io/badge/python-%203.9.13-blue" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-6.4.11-critical" alt="Jupytor" />
  <img src="https://img.shields.io/badge/tensorflow--macos-%20%202.9.2-orange" alt="Tensorflow" />
  <img src="https://img.shields.io/badge/numpy%20%20-%20%201.23.1-green" alt="Numpy" />
  <img src="https://img.shields.io/badge/openCV-4.6.0-lightgrey" alt="OpenCV" />
 </p>   
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<p align="center">   
  <img src="https://media.giphy.com/media/jNMrGhfwlGGIdUTcji/giphy.gif" alt="GIF" />
</p>




# Table of contents

- [Table of contents](#table-of-contents)
- [Introduction of Chinese license plates](#Introduction-of-Chinese-license-plates)
- [Installation](#installation)
- [Project File Directory](#Project-File-Directory)
- [Development](#development)
- [Contribute](#contribute)
- [License](#license)
- [Footer](#footer)


# Introduction of Chinese license plates
<h1 align="center">
  <img alt="The license plate images" src="./readme photo/20191009090652976.jpg"/><br/>
</h1>

The Chinese license plate consists of seven characters, the first character is Chinese, which represents the abbreviation of 31 provinces in China.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

Province abbreviation:("皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新")

The second character represents the city code of a province in China, which is one of the 24 English letters.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

City-code:("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X", "Y", "Z")

The last five characters are composed of English letters and numbers, the letters are 24 uppercase letters (excluding I and O), and the numbers are represented by numbers between "0-9".
[(Back to top)](#table-of-contents)

# Installation
![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## python 3.9.13
1.首先，从Python网站下载安装包。https://www.python.org/downloads/ </br>
它会自动检测您的操作系统并显示一个大按钮，用于在您的 Mac 上下载最新版本的 Python 安装程序,本项目使用Python3.9.13。
<h1 align="center">
  <img alt="python" src="./readme photo/python.png" width="60%" height="60%"/><br/>
</h1>

2. 下载完成后，双击包开始安装Python。安装程序将引导您完成安装，在大多数情况下，默认设置运行良好，因此请像 macOS 上的其他应用程序一样安装它。你可能还需要输入你的 Mac 密码，让它知道你同意安装 Python。
<h1 align="center">
  <img alt="python" src="./readme photo/inspython.png" width="40%" height="40%"/><br/>
</h1>

3. 在终端中检查Python。
<h1 align="center">
  <img alt="python" src="./readme photo/verpython.png" width="60%" height="60%"/><br/>
</h1>

## Anaconda
1.Download the graphical macOS installer for your version of Python.https://www.anaconda.com/products/distribution#macos


 <h1 align="center">
      <img src="./readme photo/pageanaconda.png" alt="pageAnaconda" width="60%" height="80%">
 </h1>  


2.Double-click the downloaded file and click Continue to start the installation.

3.安装成功界面
<h1 align="center">
  <img alt="insanaconda" src="./readme photo/insanaconda.png" width="40%" height="40%"/><br/>
</h1>
4.打开Anaconda-Navigator，点击install安装Jupytor notebook.
<h1 align="center">
  <img alt="screenshot" src="./readme photo/screenshot.png" width="30%" height="30%"/><br/>
</h1>

## Jupytor notebook
安装完成后，选择你所需的环境，点击launch即可进入项目列表。
<h1 align="center">
  <img alt="showanaconda" src="./readme photo/showanaconda.jpg" width="65%" height="65%"/><br/>
</h1>
<h1 align="center">
  <img alt="showpage" src="./readme photo/showpage.png" width="65%" height="65%"/><br/>
</h1>


## Mac os M1 安装Tensorflow

解决Macos M1芯片无法使用Tensorflow的问题：</br>
XCODE:在终端中执行以下命令
```
catchzeng@m1 ~ % xcode-select --install
```
Homebrew:在终端中执行以下命令
```
catchzeng@m1 ~ % /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
安装Miniforge：Anaconda 无法在 M1 上运行, Miniforge 是用来替代它的。
从 https://github.com/conda-forge/miniforge 下载 Miniforge3-MacOSX-arm64。
<h1 align="center">
  <img alt="Miniforge" src="./readme photo/Miniforge.png" width="60%" height="60%"/><br/>
</h1>

如果你使用的是 bash，执行以下命令，安装 Miniforge
```
catchzeng@m1 ~ % bash Miniforge3-MacOSX-arm64.sh
```
```
catchzeng@m1 ~ % zsh Miniforge3-MacOSX-arm64.sh
```
重启终端并检查 Python 安装情况。
```(base) catchzeng@m1 ~ % which python
/Users/catchzeng/miniforge3/bin/python
(base) catchzeng@m1 ~ % which pip
/Users/catchzeng/miniforge3/bin/pip
```
创建虚拟环境:创建一个 conda 创建虚拟环境 (TensorFlow 需要)
```
(base) catchzeng@m1 ~ % conda create -n tensorflow python=3.9.13
(base) catchzeng@m1 ~ % conda activate tensorflow
(tensorflow) catchzeng@m1 ~ %
```
安装 Tensorflow dependencies:
```
(tensorflow) catchzeng@m1 ~ % conda install -c apple tensorflow-deps
```
安装 Tensorflow:
```
(tensorflow) catchzeng@m1 ~ % python -m pip install tensorflow-macos
```
安装 metal plugin:
```
(tensorflow) catchzeng@m1 ~ % python -m pip install tensorflow-metal
```
安装必须的包:
```
(tensorflow) catchzeng@m1 ~ % brew install libjpeg
(tensorflow) catchzeng@m1 ~ % conda install -y matplotlib jupyterlab
```
[(Back to top)](#table-of-contents)



# Project File Directory:
```
└─project
    ├─ data 
    |	├─font         ---->字体文件
    |	├─images       ---->噪声背景图片
    |	├─NoPlates     ---->车牌背景图片
    |	└─plate        ---->生成车牌图片
    |        └─label.txt
    ├─input_data.py      ---->输入车牌数据集
    ├─genplate.py        ---->生成车牌数据集
    ├─model.py		 ---->构建CNN模型
    ├─Testmodel.ipynb	 ---->测试单个车牌
    ├─runmodel.ipynb     ---->训练模型
    ├─logs		 ---->日志文件
    ├─copy		 ---->备份
    ├─readme photo	
    └─saved_model	 ---->模型文件保存
```


[(Back to top)](#table-of-contents)

# Development

## Important dependencies to install before running the code.

	+ matplotlib (3.5.1)
	+ numpy (1.23.1)
	+ opencv-python (4.6.0.66)
	+ pandas (0.24.0)
	+ tensorboard (2.9.1)
	+ tensorflow-macos (2.9.2)
	+ tensorflow-estimator (2.9.0)
  
## 1.生成车牌数据集   genplate.py
<p >
生成车牌所需文件：<br/>
<b>Font file:</b> Chinese 'Platech.ttf', English and digital 'Platechar.ttf'. <br/>
<b>Background:</b> file 'NoPlates'. Which from a cropped image of a vehicle without a license plate. <br/>
<b>License plate (blue background) :</b> template.bmp. <br/>
<b>The noise of image：</b>smu2.jpg. <br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

Through the Python third-party library PIL Image, ImageFont, ImageDraw module, use the font file in the font file to randomly generate the license plate number, and add it to the license plate background in the images folder.<br/>

加载字体，车牌背景和噪声图片
```python	
class GenPlate:
    def __init__(self, fontCh, fontEng, NoPlates):
        #.truetype() Load the TrueType or OpenType font file and create a font object
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
```
随机生成车牌号（包括第一位汉字位，第二位字母位，和剩余5位）
```python
	#Generate license plate numbers
    def draw(self, val):
        offset = 2
        #GenCh() Generating Chinese characters
        self.img[0:70, offset+8:offset+8+23] = GenCh(self.fontC, val[0])
        #GenCh1 Generating English characters
        self.img[0:70, offset+8+23+6:offset+8+23+6+23] = GenCh1(self.fontE, val[1])
        #GenCh1 Generate the remaining 5 digits
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base:base+23] = GenCh1(self.fontE, val[i+2])
        return self.img
```
对生成的车牌号图片添加噪声（包括畸变和各种噪声）来模拟实际情况中的车牌截图。
```python	
def generate(self, text):
        if len(text) == 7:
            #Perform binary operations on each pixel of the image
            fg = self.draw(text)   
            fg = cv.bitwise_not(fg)
            com = cv.bitwise_or(fg, self.bg)
            #Adding perspective distortion
            com = rot(com, r(60)-30, com.shape,30)   
            #Add radiation distortion
            com = rotRandrom(com, 10, (com.shape[1], com.shape[0]))  
            # Add saturation light noise
            com = tfactor(com)
            #Add noise from the natural environment
            com = random_envirment(com, self.noplates_path)
            # Add Gaussian blur
            com = AddGauss(com, 1+r(4))
            #Add Gaussian noise
            com = addNoise(com)
            return com
```	

生成的车牌图像尺寸选取：272 * 72<br/>

<b>After the license plate is generated, save it to the 'plate' folder as shown in the following example:</b>
<h1 align="center">
  <img alt="plate" src="./readme photo/plate.png" width="60%" height="60%"/><br/>
</h1>
</p>



## 2.数据集导入      input_data.py
测试产生的车牌图片和标签
```python
O = OCRIter(2, 72, 272)
img, lbl = O.iter()
for im in img:
    plt.imshow(im, cmap='gray')
    plt.show()
print(lbl)
```
<h1 align="center">
  <img alt="inputtest" src="./readme photo/inputtest.png" width="60%" height="60%"/><br/>
</h1>

## 3.构建CNN模型     model.py

CNN网络结构：
1. 输入层：72x272
2. 第一层卷积：卷积核大小：3x3，卷积核个数：32，Stride 步长：1，VALID卷积
3. 第二层卷积：卷积核大下：3x3，卷积核个数：32，Stride 步长：1，VALID卷积
4. 第一层池化：池化大小：2x2，max pool,Stride 步长：2
5. 第三层卷积：卷积核大小：3x3，卷积核个数：64，Stride 步长：1，VALID卷积
6. 第四层卷积：卷积核大小：3x3，卷积核个数：64，Stride 步长：1，VALID卷积
7. 第二层池化：池化大小：2x2，max pool,Stride 步长：2
8. 第五层卷积：卷积核大小：3x3，卷积核个数：128，Stride 步长：1，VALID卷积
9. 第六层卷积：卷积核大小：3x3，卷积核个数：128，Stride 步长：1，VALID卷积
10. 第三层池化：池化大小：2x2，max pooling,Stride :2
11. 第一层全连接：65个神经元，激活函数为softmax。



### 卷积层：
本模型采用六层卷积层。每层所对应的filter是由去掉过大偏离点的正态分布随机数生成，标准差为0.1（也代表卷积核个数）。每一层有一个对应的filter。
```python
	'conv1': tf.Variable(tf.random.truncated_normal([3, 3, 3, 32],
                                                        stddev=0.1)),
        'conv2': tf.Variable(tf.random.truncated_normal([3, 3, 32, 32],
                                                        stddev=0.1)),
        'conv3': tf.Variable(tf.random.truncated_normal([3, 3, 32, 64],
                                                        stddev=0.1)),
        'conv4': tf.Variable(tf.random.truncated_normal([3, 3, 64, 64],
                                                        stddev=0.1)),
        'conv5': tf.Variable(tf.random.truncated_normal([3, 3, 64, 128],
                                                        stddev=0.1)),
        'conv6': tf.Variable(tf.random.truncated_normal([3, 3, 128, 128],
```
### 池化层：
本模型采用三层池化层，采用Max pooling从校正后的特征图中取最大元素。池化层的存在是为了减少参数个数，避免过拟合。
```python
# Layer 1 pooling layer
    pool1 = tf.nn.max_pool2d(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# Layer 2 pooling layer
    pool2 = tf.nn.max_pool2d(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# Layer 3 pooling layer
    pool3 = tf.nn.max_pool2d(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
```
### 全连接层：
本模型采用1层全连接层，其中一层分为七个子连接层来分别预测车牌的七位号码。
```python
    reshape = tf.reshape(pool3, [-1, 5 * 30 * 128])
    fc1 = tf.nn.dropout(reshape, keep_prob)
    fc1_1 = tf.add(tf.matmul(fc1, W_conv['fc1_1']), b_conv['fc1_1'])
    fc1_2 = tf.add(tf.matmul(fc1, W_conv['fc1_2']), b_conv['fc1_2'])
    fc1_3 = tf.add(tf.matmul(fc1, W_conv['fc1_3']), b_conv['fc1_3'])
    fc1_4 = tf.add(tf.matmul(fc1, W_conv['fc1_4']), b_conv['fc1_4'])
    fc1_5 = tf.add(tf.matmul(fc1, W_conv['fc1_5']), b_conv['fc1_5'])
    fc1_6 = tf.add(tf.matmul(fc1, W_conv['fc1_6']), b_conv['fc1_6'])
    fc1_7 = tf.add(tf.matmul(fc1, W_conv['fc1_7']), b_conv['fc1_7'])
    return fc1_1, fc1_2, fc1_3, fc1_4, fc1_5, fc1_6, fc1_7

```
## 4.模型训练        runmodel.ipynb
模型训练参数介绍：
```python
img_h = 72 #height of img
img_w = 272 #weight of img 
num_label = 7 #size of label
batch_size = 32
epoch = 10000
learning_rate = 0.0001
logs_path = 'logs/1005'
model_path = 'saved_model/1005'
```
模型训练完整代码：![runmodel.py](https://github.com/ACM40960/project-Jinhao-Li-s/edit/main/)

## 5.识别单张车牌     Testmodel.ipynb

Randomly get a single license plate image:
```python
def get_one_image(test):
    n = len(test)
    #np.random.randint（0，n）生成0-n之间的随机整数
    rand_num =np.random.randint(0,n)
    img_dir = test[rand_num]
    image_show = Image.open(img_dir)
    plt.imshow(image_show)    # display license plate image
    image = cv.imread(img_dir)
    #image = array(img).reshape(1, 64, 64, 3)
    image = image.reshape(1,72, 272, 3)
    image = np.multiply(image, 1 / 255.0)
    return image
```
检查预测车牌号是否与选取的车牌号相同：
```python
with tf.compat.v1.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print ("Reading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(model_path)
    print(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')

    pre1, pre2, pre3, pre4, pre5, pre6, pre7 = sess.run(
        [logit1, logit2, logit3, logit4, logit5, logit6, logit7],
        feed_dict={x:image_array, keep_prob:1.0})
    prediction = np.reshape(np.array([pre1, pre2, pre3, pre4, pre5, pre6, pre7]), [-1, 65])

    max_index = np.argmax(prediction, axis=1)
    print(max_index)
    line = ''
    result = np.array([])
    for i in range(prediction.shape[0]):
        if i == 0:
            result = np.argmax(prediction[i][0:31])
        if i == 1:
            #第二位为英文，序号41之后
            result = np.argmax(prediction[i][41:65]) + 41
        if i > 1:
            #第三位为数字或英文，序号31之后
            result = np.argmax(prediction[i][31:65]) + 31
        line += chars[result]+" "
    print ('predicted: ' + line)
plt.show()
```
<h1 align="center">
  <img alt="testone" src="./readme photo/testone.png" width="60%" height="60%"/><br/>
</h1>
在tensorboard中查看训练过程，accuracy曲线在epoch = 10000左右时达到收敛，最终精确度在94%左右，loss1为汉字字符，识别相对字母数字较难，loss1=0.03左右
<h1 align="center">
  <img alt="accuracy" src="./readme photo/accuracy.png" width="60%" height="60%"/><br/>
</h1>
<h1 align="center">
  <img alt="loss1.png" src="./readme photo/loss1.png" width="60%" height="60%"/><br/>
</h1>

[(Back to top)](#table-of-contents)


# Contribute

[(Back to top)](#table-of-contents)

<!-- This is where you can let people know how they can **contribute** to your project. Some of the ways are given below.
Also this shows how you can add subsections within a section. -->




# License
This Application is currently not licensed and is free to use by everyone.

[(Back to top)](#table-of-contents)

<!-- Adding the license to README is a good practice so that people can easily refer to it.
Make sure you have added a LICENSE file in your project folder. **Shortcut:** Click add new file in your root of your repo in GitHub > Set file name to LICENSE > GitHub shows LICENSE templates > Choose the one that best suits your project!
I personally add the name of the license and provide a link to it like below. -->

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)

# Footer
[(Back to top)](#table-of-contents)
![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) 
