<!-- # CNN China license plate recognition model based on Tensorflow -->

<h1 align="center">
  <img alt="cover" src="./readme photo/cover.png"/><br/>
</h1>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<p align="center">
使用随机生成的车牌图片来训练CNN模型，使模型对车牌号图片具有较高的识别精度。<br/>
对于一些非高清摄像头所拍摄的车牌图片，由于图片中<b>噪声</b>的影响，分割车牌字符变得很难。<br/>
本模型采用的是<b>同时</b>训练车牌中七个字符，七个字符使用<b>单独的loss函数</b>进行训练。<br/>

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

- [Project Title](#project-title)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Introduction of Chinese license plates](#Introduction-of-Chinese-license-plates)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contribute](#contribute)
    - [Sponsor](#sponsor)
    - [Adding new features or fixing bugs](#adding-new-features-or-fixing-bugs)
- [License](#license)
- [Footer](#footer)


# Introduction of Chinese license plates
<h1 align="center">
  <img alt="The license plate images" src="./readme photo/20191009090652976.jpg"/><br/>
</h1>

中国车牌由七个字符组成，第一个字符为中文，代表中国31个省份简称。
第二个字符代表中国某省的城市代号，为24个英文字母中的一个。
后五个字符由英文字母和数字组合而成，字母是24个大写字母（除去 I 和 O）的组合，数字用＂０-９＂之间的数字表示。

[(Back to top)](#table-of-contents)

# Installation
![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## python 3.9.13
1.首先，从Python网站下载安装包。https://www.python.org/downloads/
它会自动检测您的操作系统并显示一个大按钮，用于在您的 Mac 上下载最新版本的 Python 安装程序,本项目使用Python3.9.13。
<h1 align="center">
  <img alt="python" src="./readme photo/python.png"/><br/>
</h1>

2. 下载完成后，双击包开始安装Python。安装程序将引导您完成安装，在大多数情况下，默认设置运行良好，因此请像 macOS 上的其他应用程序一样安装它。你可能还需要输入你的 Mac 密码，让它知道你同意安装 Python。
<h1 align="center">
  <img alt="python" src="./readme photo/inspython.png"/><br/>
</h1>

3. 在终端中检查Python。
<h1 align="center">
  <img alt="python" src="./readme photo/verpython.png"/><br/>
</h1>

## Anaconda
## Jupytor notebook
## Mac os M1 安装Tensorflow
[(Back to top)](#table-of-contents)



# Usage
[(Back to top)](#table-of-contents)

<!-- This is optional and it is used to give the user info on how to use the project after installation. This could be added in the Installation section also. -->

# Development
[(Back to top)](#table-of-contents)

<!-- This is the place where you give instructions to developers on how to modify the code.
You could give **instructions in depth** of **how the code works** and how everything is put together.
You could also give specific instructions to how they can setup their development environment.
Ideally, you should keep the README simple. If you need to add more complex explanations, use a wiki. Check out [this wiki](https://github.com/navendu-pottekkat/nsfw-filter/wiki) for inspiration. -->

# Contribute
[(Back to top)](#table-of-contents)

<!-- This is where you can let people know how they can **contribute** to your project. Some of the ways are given below.
Also this shows how you can add subsections within a section. -->

### Sponsor
[(Back to top)](#table-of-contents)

<!-- Your project is gaining traction and it is being used by thousands of people(***with this README there will be even more***). Now it would be a good time to look for people or organisations to sponsor your project. This could be because you are not generating any revenue from your project and you require money for keeping the project alive.
You could add how people can sponsor your project in this section. Add your patreon or GitHub sponsor link here for easy access.
A good idea is to also display the sponsors with their organisation logos or badges to show them your love!(*Someday I will get a sponsor and I can show my love*) -->

### Adding new features or fixing bugs
[(Back to top)](#table-of-contents)

<!-- This is to give people an idea how they can raise issues or feature requests in your projects. 
You could also give guidelines for submitting and issue or a pull request to your project.
Personally and by standard, you should use a [issue template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/ISSUE_TEMPLATE.md) and a [pull request template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/PULL_REQ_TEMPLATE.md)(click for examples) so that when a user opens a new issue they could easily format it as per your project guidelines.
You could also add contact details for people to get in touch with you regarding your project. -->

# License
[(Back to top)](#table-of-contents)

<!-- Adding the license to README is a good practice so that people can easily refer to it.
Make sure you have added a LICENSE file in your project folder. **Shortcut:** Click add new file in your root of your repo in GitHub > Set file name to LICENSE > GitHub shows LICENSE templates > Choose the one that best suits your project!
I personally add the name of the license and provide a link to it like below. -->

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)

# Footer
[(Back to top)](#table-of-contents)

<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.
Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->

Leave a star in GitHub, give a clap in Medium and share this guide if you found this helpful.

<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->
