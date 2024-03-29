<!-- # CNN China license plate recognition model based on Tensorflow -->

<h1 align="center">
  <img alt="cover" src="./readme photo/cover.png"/><br/>
</h1>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<p align="center">
This model uses the <b>simultaneous training</b> of seven characters in the license plate, and the seven characters are trained using a <b>separate</b> loss function.<br/>
For some license plate pictures captured by non-HD cameras, it becomes difficult to segment the characters of the license plate due to the influence of <b>ambient noise</b> and <b>image distortion</b> in the pictures.<br/>
This model trains CNN through randomly generated license plate number pictures, so that the model has high recognition accuracy for fuzzy license plate number pictures.<br/>
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
- [Conclusion](#Conclusion)
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
1. First, download the installation package from the Python website. https://www.python.org/downloads/ </br>
It will automatically detect your OS and display a big button to download the latest version of the Python installer on your Mac, this project uses Python 3.9.13.
<h1 align="center">
  <img alt="python" src="./readme photo/python.png" width="60%" height="60%"/><br/>
</h1>

2. Once the download is complete, double-click the package to start installing Python. The installer will walk you through the installation, and in most cases the default settings work fine, so install it like any other application on macOS. You may also need to enter your Mac password to let it know that you agree to install Python.
<h1 align="center">
  <img alt="python" src="./readme photo/inspython.png" width="40%" height="40%"/><br/>
</h1>

3. View Python in a terminal.
<h1 align="center">
  <img alt="python" src="./readme photo/verpython.png" width="60%" height="60%"/><br/>
</h1>

## Anaconda
1.Download the graphical macOS installer for your version of Python.https://www.anaconda.com/products/distribution#macos


 <h1 align="center">
      <img src="./readme photo/pageanaconda.png" alt="pageAnaconda" width="60%" height="80%">
 </h1>  


2.Double-click the downloaded file and click Continue to start the installation.

3.Installation success interface
<h1 align="center">
  <img alt="insanaconda" src="./readme photo/insanaconda.png" width="40%" height="40%"/><br/>
</h1>

4.Open Anaconda-Navigator and click ’install‘ to install Jupyter notebook.
<h1 align="center">
  <img alt="screenshot" src="./readme photo/screenshot.png" width="30%" height="30%"/><br/>
</h1>

## Jupytor notebook
After the installation is complete, select the environment you need and click launch to enter the project list.
<h1 align="center">
  <img alt="showanaconda" src="./readme photo/showanaconda.jpg" width="65%" height="65%"/><br/>
</h1>
<h1 align="center">
  <img alt="showpage" src="./readme photo/showpage.png" width="65%" height="65%"/><br/>
</h1>


## Mac os M1 install Tensorflow

Solve the problem that the Macos M1 cannot use Tensorflow:</br>
XCODE: Execute the following command in terminal
```
catchzeng@m1 ~ % xcode-select --install
```
Homebrew: Execute the following command in the terminal
```
catchzeng@m1 ~ % /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Miniforge: Anaconda cannot run on M1, Miniforge is used to replace it.Download Miniforge3-MacOSX-arm64 from https://github.com/conda-forge/miniforge.
<h1 align="center">
  <img alt="Miniforge" src="./readme photo/Miniforge.png" width="60%" height="60%"/><br/>
</h1>

If you are using bash, execute the following command to install Miniforge
```
catchzeng@m1 ~ % bash Miniforge3-MacOSX-arm64.sh
```
If you are using zsh, execute the following command to install Miniforge
```
catchzeng@m1 ~ % zsh Miniforge3-MacOSX-arm64.sh
```
Restart the terminal and check the Python installation.
```(base) catchzeng@m1 ~ % which python
/Users/catchzeng/miniforge3/bin/python
(base) catchzeng@m1 ~ % which pip
/Users/catchzeng/miniforge3/bin/pip
```
Create a virtual environment: create a conda to create a virtual environment (required by TensorFlow)
```
(base) catchzeng@m1 ~ % conda create -n tensorflow python=3.9.13
(base) catchzeng@m1 ~ % conda activate tensorflow
(tensorflow) catchzeng@m1 ~ %
```
Install Tensorflow dependencies:
```
(tensorflow) catchzeng@m1 ~ % conda install -c apple tensorflow-deps
```
Install Tensorflow:
```
(tensorflow) catchzeng@m1 ~ % python -m pip install tensorflow-macos
```
Install the metal plugin:
```
(tensorflow) catchzeng@m1 ~ % python -m pip install tensorflow-metal
```
Install the required packages:
```
(tensorflow) catchzeng@m1 ~ % brew install libjpeg
(tensorflow) catchzeng@m1 ~ % conda install -y matplotlib jupyterlab
```
[(Back to top)](#table-of-contents)



# Project File Directory:
```
└─project
    ├─ data 
    |	├─font         ---->font file
    |	├─images       ---->background image
    |	├─NoPlates     ---->plate background image
    |	└─plate        ---->plate image
    |        └─label.txt
    ├─input_data.py      ---->input license plate data
    ├─genplate.py        ---->Generate a license plate dataset
    ├─model.py		 ---->Building CNN model
    ├─Testmodel.ipynb	 ---->Test one license plates
    ├─runmodel.ipynb     ---->Training model
    ├─logs		 ---->The log file
    ├─readme photo	
    └─saved_model	 ---->Save the model file (too large to upload, automatically generated when running runModel.ipynb)
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
  
## 1.Generate license plate dataset   genplate.py
<p >
Documents required to generate license plates：<br/>
<b>Font file:</b> Chinese 'Platech.ttf', English and digital 'Platechar.ttf'. <br/>
<b>Background:</b> file 'NoPlates'. Which from a cropped image of a vehicle without a license plate. <br/>
<b>License plate (blue background) :</b> template.bmp. <br/>
<b>The noise of image：</b>smu2.jpg. <br/>
<b>data file link(link only for ucdconnect.ie):</b>https://drive.google.com/drive/folders/12YlRszYvpdxKXZirxzgT0AKGzBse6KGK?usp=sharing<br/>
<b>Model traning time might be around 190 minutes</b></br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

Through the Python third-party library PIL Image, ImageFont, ImageDraw module, use the font file in the font file to randomly generate the license plate number, and add it to the license plate background in the images folder.<br/>

Load fonts, license plate backgrounds and noise images.
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
Randomly generate the license plate number (including the first Chinese character, the second letter character, and the remaining 5 characters).
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
Add noise (including distortion and various noises) to the generated license plate number image to simulate the license plate screenshot in the actual situation.
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

The size of the generated license plate image is selected: 272 * 72.<br/>

<b>After the license plate is generated, save it to the 'plate' folder as shown in the following example:</b>
<h1 align="center">
  <img alt="plate" src="./readme photo/plate.png" width="60%" height="60%"/><br/>
</h1>
</p>



## 2.Dataset import      input_data.py
Test the resulting license plate images and labels.
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

## 3.Build a CNN model    model.py

CNN network structure：</br>
1.Input layer: 72x272 </br>
2.The first layer of convolution: convolution kernel size: 3x3, number of convolution kernels: 32, Stride step size: 1, VALID convolution</br>
3.The second layer of convolution: convolution kernel: 3x3, number of convolution kernels: 32, Stride step size: 1, VALID convolution</br>
4.The first layer of pooling: pooling size: 2x2, max pool, Stride step size: 2</br>
5.The third layer of convolution: convolution kernel size: 3x3, number of convolution kernels: 64, Stride step size: 1, VALID convolution</br>
6.Fourth layer convolution: convolution kernel size: 3x3, number of convolution kernels: 64, Stride step size: 1, VALID convolution</br>
7.Second layer pooling: pooling size: 2x2, max pool, Stride step size: 2</br>
8.The fifth layer of convolution: convolution kernel size: 3x3, number of convolution kernels: 128, Stride step size: 1, VALID convolution</br>
9.The sixth layer of convolution: convolution kernel size: 3x3, number of convolution kernels: 128, Stride step size: 1, VALID convolution</br>
10.The third layer of pooling: pooling size: 2x2, max pooling, Stride: 2</br>
11.The first layer is fully connected: 65 neurons, the activation function is softmax.</br>

### Convolutional layers：
This model uses six convolutional layers. The filter corresponding to each layer is generated by a normal distribution random number that removes excessive deviation points, and the standard deviation is 0.1. Each layer has a corresponding filter.
```python
	conv1 = tf.nn.conv2d(images, W_conv['conv1'], strides=[1,1,1,1], padding='VALID')
	conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
    	conv1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(conv1, W_conv['conv2'], strides=[1,1,1,1], padding='VALID')
	conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
    	conv2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(pool1, W_conv['conv3'], strides=[1,1,1,1], padding='VALID')
	conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
    	conv3 = tf.nn.relu(conv3)
        conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=[1,1,1,1], padding='VALID')
	conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
    	conv4 = tf.nn.relu(conv4)
        conv5 = tf.nn.conv2d(pool2, W_conv['conv5'], strides=[1,1,1,1], padding='VALID')
	conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
    	conv5 = tf.nn.relu(conv5)
        conv6 = tf.nn.conv2d(conv5, W_conv['conv6'], strides=[1,1,1,1], padding='VALID')
	conv6 = tf.nn.bias_add(conv6, b_conv['conv6'])
   	conv6 = tf.nn.relu(conv6)
```
### Pooling layers：
This model uses three pooling layers, and uses Max pooling to take the largest element from the corrected feature map. The existence of the pooling layer is to reduce the number of parameters and avoid overfitting.
```python
# Layer 1 pooling layer
    pool1 = tf.nn.max_pool2d(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# Layer 2 pooling layer
    pool2 = tf.nn.max_pool2d(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# Layer 3 pooling layer
    pool3 = tf.nn.max_pool2d(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
```
### Fully connected layer：
This model adopts 1 fully connected layer, one of which is divided into seven sub-connected layers to predict the seven-digit number of the license plate respectively.
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
### Loss
Use the multi-class cross entropy function to calculate the Loss, which is suitable for the case that each category is independent and exclusive. In this model, in order to predict the seven digits of the license plate number at the same time, I set seven separate Loss for training. Among them, Loss1 corresponds to the first Chinese of the license plate number. Compared with English and numbers, Chinese is more difficult to identify.
```python
def calc_loss(logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):
    labels = tf.convert_to_tensor(labels, tf.int32)
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit1, labels=labels[:, 0]))
    #Save all summaries to disk for Tensorboard to display
    tf.compat.v1.summary.scalar('loss1', loss1)

    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit2, labels=labels[:, 1]))
    tf.compat.v1.summary.scalar('loss2', loss2)

    loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit3, labels=labels[:, 2]))
    tf.compat.v1.summary.scalar('loss3', loss3)

    loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit4, labels=labels[:, 3]))
    tf.compat.v1.summary.scalar('loss4', loss4)

    loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit5, labels=labels[:, 4]))
    tf.compat.v1.summary.scalar('loss5', loss5)

    loss6 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit6, labels=labels[:, 5]))
    tf.compat.v1.summary.scalar('loss6', loss6)

    loss7 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logit7, labels=labels[:, 6]))
    tf.compat.v1.summary.scalar('loss7', loss7)

    return loss1, loss2, loss3, loss4, loss5, loss6, loss7
```
## 4.model training       runmodel.ipynb
Model training parameters introduction:
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
Set placeholders for the image data, label data and dropout scale to be input:
```python
image_holder = tf.compat.v1.placeholder(tf.float32, [batch_size, img_h, img_w, 3])
label_holder = tf.compat.v1.placeholder(tf.int32, [batch_size, 7])
keep_prob    = tf.compat.v1.placeholder(tf.float32)
```
```python
logit1, logit2, logit3, logit4, logit5, logit6, logit7 = model.cnn_inference(image_holder, keep_prob)

loss1, loss2, loss3, loss4, loss5, loss6, loss7 = model.calc_loss(logit1, logit2, logit3, logit4, logit5, logit6, logit7, label_holder)

train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7 = model.train_step(
    loss1, loss2, loss3, loss4, loss5, loss6, loss7, learning_rate)

accuracy = model.pred_model(logit1, logit2, logit3, logit4, logit5, logit6, logit7, label_holder)

input_image=tf.compat.v1.summary.image('input', image_holder)

summary_op = tf.compat.v1.summary.merge(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    
    train_writer = tf.compat.v1.summary.FileWriter(logs_path, sess.graph)
    saver = tf.compat.v1.train.Saver()

    start_time1 = time.time()
    for step in range(epoch):
        # Generate license plate image and label data
        img_batch, lbl_batch = get_batch()

        start_time2 = time.time()
        time_str = datetime.datetime.now().isoformat()

        feed_dict = {image_holder:img_batch, label_holder:lbl_batch, keep_prob:0.6}
        _1, _2, _3, _4, _5, _6, _7, ls1, ls2, ls3, ls4, ls5, ls6, ls7, acc = sess.run(
            [train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7, 
             loss1, loss2, loss3, loss4, loss5, loss6, loss7, accuracy], feed_dict)
        summary_str = sess.run(summary_op, feed_dict)
        train_writer.add_summary(summary_str,step)
        duration = time.time() - start_time2
        loss_total = ls1 + ls2 + ls3 + ls4 + ls5 + ls6 + ls7
        if step % 10 == 0:
            sec_per_batch = float(duration)
            print('%s: Step %d, loss_total = %.2f, acc = %.2f%%, sec/batch = %.2f' %
                (time_str, step, loss_total, acc * 100, sec_per_batch))
        if step % 5000 == 0 or (step + 1) == epoch:
            checkpoint_path = os.path.join(model_path,'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    end_time = time.time()
    print("Training over. It costs {:.2f} minutes".format((end_time - start_time1) / 60))
```

## 5.Identify a single license plate     Testmodel.ipynb

Randomly get a single license plate image:
```python
def get_one_image(test):
    n = len(test)
    #np.random.randint（0，n）Generate random integers between 0-n
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
Check if the predicted license plate number is the same as the selected license plate number:
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
            #The second digit is in English, after the serial number 41
            result = np.argmax(prediction[i][41:65]) + 41
        if i > 1:
            #The third digit is a number or English, after the serial number 31
            result = np.argmax(prediction[i][31:65]) + 31
        line += chars[result]+" "
    print ('predicted: ' + line)
plt.show()
```
<h1 align="center">
  <img alt="testone" src="./readme photo/testone.png" width="60%" height="60%"/><br/>
</h1>
Looking at the training process in tensorboard, the accuracy curve reaches convergence at about epoch = 10000, the final accuracy is about 94%, loss1 is Chinese characters, and it is difficult to identify relative alphanumerics,around loss1=0.03.
<h1 align="center">
  <img alt="accuracy" src="./readme photo/accuracy.png" width="60%" height="60%"/><br/>
</h1>
<h1 align="center">
  <img alt="loss1.png" src="./readme photo/loss1.png" width="60%" height="60%"/><br/>
</h1>

[(Back to top)](#table-of-contents)


# Conclusion

The system first randomly generates the license plate number and combines it with the license plate background to form a license plate image. Then add ambient noise and image distortion to each license plate image to interfere with the clarity of the image, and store it in a specific folder. Save the license plate pictures in the folder into the list, and create a label for each picture. Build a CNN model with 6 convolutional layers, three pooling layers and a fully connected layer to train the pictures in the list. After 190 minutes of training, store the trained model in the model folder. Finally, a license plate photo is randomly selected to test the recognition accuracy of the model.
I have learned a lot of knowledge through the course of image processing, and I deeply feel that image recognition plays an irreplaceable role in various fields, ranging from license plate recognition to image guidance for cruise missiles. In the traditional image recognition, not only the complex processing of the image is required, but the recognition accuracy is also deeply disturbed by external factors. In recent years, deep learning has made breakthroughs in the field of image recognition, making image recognition models represented by CNNs widely used. Although CNN is an end-to-end image recognition model, the accuracy of image recognition will also be increased by performing certain processing on the image by traditional methods in the early stage of transmitting the original image to the CNN input. Therefore, dealing with a problem may sometimes require drawing on the strengths of various methods to solve the problem perfectly.

## Future Work：
1. Continue to debug the model, such as changing the values of eqoch and learning rate to reduce the training time of the model and improve the accuracy of the model.
2. By using openCV, adding a license plate interception module, you can locate the license plate from the front photo of the vehicle and intercept it as a 72X272 size picture. The required techniques include image noise reduction, grayscale stretching, image difference, and binarization. processing, edge detection, and more. The system realizes the functions of locating the license plate, intercepting the license plate and recognizing the license plate from a picture.

[(Back to top)](#table-of-contents)

# License
This Application is currently not licensed and is free to use by everyone.

[(Back to top)](#table-of-contents)


# Footer
[(Back to top)](#table-of-contents)
![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) 
