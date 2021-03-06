### 申松 | 工具开发工程师
mrshensong@163.com | 15754323012

---  
**语言：***英语 CET-4*

**教育经历：***长春大学-自动化-本科(2014.09-2018.06)*  
 
**相关技能：***Python | PyQt | OpenCV | UIautomator | Linux | Shell | Git | Arduino | C(硬件)*

---
### 2019.07--至今 (理想汽车) 
##### 项目：基于图像识别的性能响应时间测试界面化工具 
**背景:** 车机性能响应测试耗时费力，容易使测试人员产生厌烦，并且手工测试一致性不强，
造成误差会稍大，故而有了自动化解决的必要。

**简介:** 此工具使用python通过qt和opencv编写，利用高帧率摄像头将车机屏幕内容
变化实时展示在界面，使用鼠标点击、滑动、框选图片模板等等方式来通过
控制机械手进行case的图形化编写，保存之后形成xml标签文件case，
测试时执行case，同时拍摄视频并保存，执行完case后对视频自动进行图像处理，
计算出性能响应时间，最后自动将报告展示出来在界面。界面工具支持离线
视频的播放、分帧、进度条拖动播放、视频帧roi；图片展示、放大、缩小、
roi、旋转；文本文件的编辑、修改、保存；html文件的渲染展示等等。

视频demo: https://www.jianshu.com/p/7e9fc15d55f6
#### 项目：定制化xml编辑器
**背景:** 现有xml编辑器在使用中不能导入自定义的函数，以及不能调用自定义的
方法等等，在脚本编写过程中使得效率无法提升，故而有此需求。

**简介:** 此工具使用python和qt编写，可以直接打开项目，展开项目文件树，文件树支持
常用操作(新建文件/文件夹、删除、重命名、复制、复制绝对路径、粘贴、
双击打开等待)；编辑区支持语法高亮、自动补全、换行缩进、括号匹配、快捷注释、
函数名跳转、自定义代码块导入、文本搜索、文本替换等等。编写完xml脚本后，
可以直接在工具中一键运行。 
#### 项目：视频图片处理工具
**背景:** 在日常进行的功能自动化测试中，会需要对大量的视频和图片进行
编辑操作，之后写入xml脚本中，故而针对此需求定制开发。

**简介:** 工具使用python、qt和opencv开发，支持视频导入、图片导入、视频逐帧播放、
视频帧roi裁剪、roi顶点显示，图片旋转，放大缩小，图片roi裁剪，图片灰度化
等等，通过此工具可以方便快捷的处理视频和图片。提高了测试过程中的效率。

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
### 2018.07--2019.07(小米移动软件)
#### 项目：基于图像识别的机械臂自动化测试工具
**背景:** 为了解决测试过程中繁琐的重复性工作、使得测试操作更加拟人化、
提高测试一致性，使得测试结果更加准确，开发此工具。

**简介:** 工具使用python、qt和opencv、以及高速摄像头和机械臂开发。
使用摄像头将手机画面实时展示在工具界面，通过鼠标在界面点击滑动
操作控制机械臂操作手机，可以支持脚本录制、脚本编辑、逻辑/循环语句，adb
命令、也可以通过roi实现实时目标点击等等。此项目在手机功耗模型测试中，显示
了其高效性，测试一致性高，测试数据准确，节省时间、人力。

#### 项目：基于图像识别的手机camera性能测试工具
**背景:** 手机camera的性能至关重要，但在日常测试中对环境要求较高,
又费时费力，并且准确度不高，故而有了camera性能自动化的需求。

**简介:** 工具使用python、qt、uiautomator和opencv、以及高速摄像头和
Arduino控制灯光智能化开发。使用摄像头将手机画面实时展示在工具界面，
通过鼠标在界面点击滑动操作通过adb控制手机。支持动作录制、预期界面匹配、
视频录制，脚本编辑等等。界面端可以直接执行生成的脚本，执行过程中会保存
摄像头录制的视频，之后使用opencv对视频自动处理，得出camera性能时间。


