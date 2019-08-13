# nodule_object_detection_system
This is a pulmonary nodule detection system based on a deep learning neural network developed using the Flask framework.

基于神经网络的肺结节检测和反馈系统 安装说明：
【1】requirements.txt文件，包含了当前环境中所有包及各自的版本的列表
安装导入环境依赖包
pip install -r requirements.txt

【2】Flask项目运行：
  环境要求：python3 ,flask、matplotlib等
  python app.py启动本地服务
  本地浏览器访问：
  127.0.0.1:5000/   用户系统
  127.0.0.1:5000/admin   后台管理员系统

 【3】其他可独立运行的py
 mhd_transform_jpg.py mhd文件转换

 【4】127.0.0.1:5000/admin   后台管理员系统使用
  1.用户名：admin  密码：admin123 登录
  2.管理非法用户 可删除
 【5】127.0.0.1:5000/   用户系统使用
 http://127.0.0.1:5000/   用户登录    默认用户：root  密码：root
 http://127.0.0.1:5000/register  用户注册
 http://127.0.0.1:5000/index/  用户首页
 http://127.0.0.1:5000/mhd/   mhd文件上传


基于神经网络的肺结节检测和反馈系统  flask项目说明：
templates目录下文件说明：
【1】admin.html为后台管理员系统设计
【2】user.html为管理员系统设计
【3】login1.html为用户登录界面设计
【4】register1.html为用户注册界面设计
【5】index.html为用户系统首页设计
【6】predict.html为深度学习模型调用设计
【7】set.html为用户图片重置设计
【8】final.html为图片重置细节设计
【9】upload_mhd.html为mhd文件上传设计
static目录下文件说明：
【1】css、js、LoginSpecial、images及img下page1_0~4图片 为前端界面样式设计依样式设计
【2】mhd文件上传目录是static/data
    解析处理的路径是static/data/object_detection_img
【3】用户登录后检测的图片保存在static/img(为了保持文件的唯一性，文件后台根据时间戳重命名)
【4】predictImg为经过深度学习模型预测后保存结果
【5】tmp_Img为经过用户图片重置后的结果
user目录说明：
【1】user模拟数据库后端用户数据存储（系统内置测试用户root及test）
【2】admin模拟后台管理员数据存储（管理员默认用户名：admin；密码：admin123）
utils目录说明：深度学习调用所依赖的一些模型数据
model目录说明：深度学习调用所依赖的一些模型数据

Flask说明：
【1】app.py：flask项目路由建立及逻辑设计
【2】mhd_transform_jpg.py: mhd转换测试（可以单独运行）
【3】pascal_label_map.pbtxt：deepLearning 模型需要
【4】utils.py：整体设计过程中自定义的utils池，用来存储函数定义等

