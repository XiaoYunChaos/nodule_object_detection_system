from flask import Flask,render_template,request,jsonify,session,redirect,url_for
import os
from datetime import datetime
import numpy as np
from PIL import ImageDraw,ImageFont
import math
import sys
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'

@app.route('/index/')
def hello_world():
    user = session.get('username')
    return render_template('index.html',user=user)

@app.route('/admin/',methods=['GET','POST'])
def admin():
    if request.method == 'GET':
        return render_template('admin.html')
    else:
        user = request.form.get('login')
        password = request.form.get('pwd')
        yanzheng = request.form.get('code')

        print(user,password,yanzheng)

        u_file = open('user/user', encoding='utf-8', mode='r')
        u_str = u_file.read()
        u_list = u_str.split('\n')
        u_list = u_list[:]

        user_result = []
        for item in u_list:
            index = u_list.index(item)
            user = item.split('  ')[0]
            pwd = item.split('  ')[1]
            user_result.append([index,user,pwd])

        user_file = open('user/admin', encoding='utf-8', mode='r')
        user_str = user_file.read()
        user_list = user_str.split('\n')

        for item in user_list:
            user = item.split('  ')[0]
            pwd = item.split('  ')[1]
            if (user == user and password == pwd):
                # session['username'] =  user
                return render_template('user.html',user_result=user_result)
            else:
                continue
        return redirect(url_for('admin'))

@app.route('/delete/<name>')
def delete(name):
    u_file = open('user/user', encoding='utf-8', mode='r')
    u_str = u_file.read()
    print(u_str)
    u_list = u_str.split('\n')
    u_list = u_list[1:]
    # print(u_list)
    u_list.remove(name)
    # print(u_list)

    user_result = []
    user_file = open('user/user', encoding='utf-8', mode='w')
    user_file.write('root' + '  ' + 'root')
    user_file.write('\n')
    for item in u_list:
        index = u_list.index(item)
        # print(item)
        user = item.split('  ')[0]
        pwd = item.split('  ')[1]
        # user_result.append([index, user, pwd])
        user_file.write(user + '  ' + pwd)
        if(index == len(u_list)-1):
            break
        user_file.write('\n')
    # user_file.write('***' + '  ' + '***')
    user_file.close()

    u_file = open('user/user', encoding='utf-8', mode='r')
    u_str = u_file.read()
    u_list = u_str.split('\n')
    u_list = u_list[:]

    result = []
    print('u_list::', u_list)
    for item in u_list:
        index = u_list.index(item)
        user = item.split('  ')[0]
        print('item',item)
        print('user',user)
        # print(pwd)
        pwd = item.split('  ')[1]

        result.append([index, user, pwd])

    return render_template('user.html',user_result=result)

@app.route('/',methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
                    ##render_template('login1.html') 改成 render_template('login.html')
    else:
        user = request.form.get('user')
        password = request.form.get('pwd')   #get('password')  改成get('pwd')
        yanzheng = request.form.get('yanzheng')
        if yanzheng == "HETI":
            flag = True

        user_file = open('user/user', encoding='utf-8', mode='r')
        user_str = user_file.read()
        user_list = user_str.split('\n')

        for item in user_list:
            user = item.split('  ')[0]
            pwd = item.split('  ')[1]
            if (user == user and password == pwd):
                session['username'] =  user
                return redirect(url_for('hello_world'))
            else:
                continue
        return redirect(url_for('regist'))



@app.route('/register',methods=['GET','POST'])
def regist():
    if request.method == 'GET':
        return render_template('register1.html')
    else:
        user = request.form.get('user')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        if(password1==password2 and user!=None):
            user_file = open('user/user', encoding='utf-8', mode='a+')
            user_file.write('\n')
            user_file.write(user+'  '+password1)
            return redirect(url_for('login'))
        else:
            return redirect(url_for('regist'))

@app.route('/logout/')
def logout():
    session.pop('username')
    session.clear()
    return redirect(url_for('login'))


@app.route('/mhd/')
def mhd():
    return render_template('upload_mhd.html')

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image_file']
        if file:
            # filename = secure_filename(file.filename)
            newName = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+ '.' + file.filename.split('.')[-1]
            session['Picname'] = newName
            file.save(os.path.join(app.static_folder, 'img', newName))      #static/
            org_path = '../static/img/%s' % newName
            predict_path = None
            return render_template('predict.html', org_path=org_path,predict_path=predict_path)
    else:
        print('上传失败')
        return '上传失败'

def solve(length_,path):
    from skimage import io
    img_data = io.imread(path)  # 图片路径
    io.imshow(img_data)
    """
    中心裁剪任意尺寸的图片（以中心为原点）
    """
    slice_width, slice_height, _ = img_data.shape
    width_crop = (slice_width - length_) // 2
    height_crop = (slice_height - length_) // 2
    if width_crop > 0:
        img_data = img_data[width_crop:-width_crop, :, :]
    if height_crop > 0:
        img_data = img_data[:, height_crop:-height_crop, :]
    io.imshow(img_data)
    io.imsave(path,img_data)

def mhd_transform_jpg(path,outpath):
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    # path = 'data/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    # outpath = 'data/object_detection_img'
    index = -1
    for img_item in image:
        index = index + 1
        print('index', index)
        plt.imshow(image[index, :, :], cmap='gray')
        plt.savefig("%s/%d.jpg" % (outpath, index))
        # plt.show()
        plt.axis('off')
        path = "%s/%d.jpg" % (outpath, index)
        solve(480, path)
    print('success!')


@app.route('/upload_file/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['uploadfile']
        print('file',file)
        if file:
            #给上传的文件起新名字
            newName = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+ '.' + file.filename.split('.')[-1]
            #保存文件
            # file.save(os.path.join(app.static_folder, 'data', file.filename))
            # file.save(os.path.join( 'data','object_detection-img', file.filename))
            file.save(os.path.join('data',file.filename))

            #mhd转换的输入输出路径
            # path = os.path.join(app.static_folder, 'data', file.filename)
            # path = os.path.join('data', 'object_detection-img', file.filename)
            path = os.path.join('data', file.filename)
            outpath = os.path.join(app.static_folder, 'data', 'object_detection_img')
            print('\n'+path+'\n'+outpath)

            # path = os.path.join('data', file.filename)

            #mhd转换
            mhd_transform_jpg(path,outpath)
            return 'success!'
    else:
        print('上传失败')
        return '上传失败'

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks', 'detection_characteristics'
                ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            output_dict['detection_characteristics'] = output_dict['detection_characteristics'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def demo(newName,PATH_TO_CKPT):    # PATH_TO_CKPT为导入模型的名字
    image_path = '../nodule/static/img/%s' % newName
    print(image_path)
    # This is needed since the notebook is stored in the object_detection_img folder.
    sys.path.append("..")
    from object_detection.utils import ops as utils_ops

    # if tf.__version__ < '1.4.0':
    #   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
    # PATH_TO_CKPT = 'model//frozen_inference_graph.pb'
    PATH_TO_LABELS = 'pascal_label_map.pbtxt'
    NUM_CLASSES = 1
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # print('image_np_expanded', image_np_expanded)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=1)
    char = str(
        output_dict['detection_characteristics'][0][0]) + " " + str(
        output_dict['detection_characteristics'][0][1]) + " " + str(
        output_dict['detection_characteristics'][0][2]) + " " + str(
        output_dict['detection_characteristics'][0][3]) + " " + str(
        output_dict['detection_characteristics'][0][4]) + " " + str(
        output_dict['detection_characteristics'][0][5]) + " " + str(
        output_dict['detection_characteristics'][0][6]) + " " + str(
        output_dict['detection_characteristics'][0][7])
    print(output_dict['detection_boxes'][0])
    print(output_dict['detection_scores'][0])
    print(char)

    # from PIL import Image
    im = Image.fromarray(image_np)
    im.save('../nodule/static/predictImg/%s'% newName)

def draw(filename, result,nodule_str):
    path = 'static/img/%s' % filename
    img = Image.open(path)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    result = np.array(result)
    x = result[0][0]
    y = result[0][1]
    angle = result[0][2]
    height = result[0][3]
    width = result[0][4]

    anglePi = -angle * math.pi / 180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    x1 = x
    y1 = y

    x0 = x + width
    y0 = y1

    x2 = x1
    y2 = y +  height

    x3 = x0
    y3 = y2

    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    draw.line([(x0n, y0n), (x1n, y1n)], fill='yellow',width=4)
    draw.line([(x1n, y1n), (x2n, y2n)], fill='yellow',width=4)
    draw.line([(x2n, y2n), (x3n, y3n)], fill='yellow',width=4)
    draw.line([(x0n, y0n), (x3n, y3n)], fill='yellow',width=4)
    print([[(x0n, y0n), (x1n, y1n)], [(x1n, y1n), (x2n, y2n)], [(x2n, y2n), (x3n, y3n)], [(x0n, y0n), (x3n, y3n)]])
    font = ImageFont.truetype('simsun.ttc', 26)
    draw.text([x1n, y1n-26], nodule_str, fill='black', font=font)
    img.save('static/tmp_img/' + filename)
    # draw.text([x2n,y2n],'nodule:97%',fill='black',font=font)
    '''
    thickness: 1 color Chartreuse
    [0.2698766  0.14902619 0.9060464  0.80354726]
    0.67177457
    4.4416866 1.3917235 3.2794006 2.4602408 4.151168 0.8174313 0.6488167 4.1170287
    '''

@app.route('/predict/',methods=['POST'])
def predict():
    # import object_detection_img.Demo
    newName = session.get('Picname')
    org_path  = '../static/img/%s' % newName
    print(newName)
    PATH_TO_CKPT = 'model//frozen_inference_graph.pb'
    demo(newName,PATH_TO_CKPT)
    predict_path = '../static/predictImg/%s'% newName
    return render_template('predict.html', org_path=org_path, predict_path=predict_path)

@app.route('/predict2/',methods=['POST'])
def predict2():
    # import object_detection_img.Demo
    newName = session.get('Picname')
    org_path  = '../static/img/%s' % newName
    print(newName)
    PATH_TO_CKPT = 'model//lungModel.pb'
    demo(newName,PATH_TO_CKPT)
    predict_path = '../static/predictImg/%s'% newName
    return render_template('predict2.html', org_path=org_path, predict_path=predict_path)

@app.route('/reset/')
def reset():
    newName = session.get('Picname')
    org_path  = '../static/img/%s' % newName
    flag = 0
    predict_path = None
    return render_template('predict.html', org_path=org_path, predict_path=predict_path,flag=flag)
    # return 'Hello World!'

@app.route('/set/',methods=['GET'])
def set():
    newName = session.get('Picname')
    possi = request.args.get('possi')
    print('possi',possi)
    org_path  = '../static/img/%s' % newName
    return render_template('set.html',org_path=org_path,possi=possi)

pos = None
@app.route('/getjson/', methods = ['GET', 'POST'])
def getjson():
    a = request.json
    print('a',a,a['mydata'])
    pos = str(a['mydata']).split(',')
    print(pos)
    if pos:
        print(pos)
        filename = session.get('Picname')
        print(filename)
        tmp_path = filename
        result = [[int(pos[0]), int(pos[1]), 0, 60, 60]]
        nodule_str = ''
        newPic = draw(filename, result,nodule_str)
        return 'hello'

def draw_possi(filename,nodule_str):
    path = 'static/tmp_img/%s' % filename
    img = Image.open(path)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('simsun.ttc', 26)
    draw.text([0, 0], nodule_str, fill='yellow', font=font)
    img.save('static/tmp_img/' + filename)

@app.route('/show/<possi>')
def show(possi):
    newName = session.get('Picname')
    path= '../static/tmp_img/%s' % newName
    possi = 'nodule:'+possi
    draw_possi(newName,possi)
    org_path  = '../static/img/%s' % newName
    return render_template('set.html',org_path = org_path, path=path)


@app.route('/pred/')
def pred():
    # import object_detection_img.Demo
    newName = session.get('Picname')
    org_path  = '../static/img/%s' % newName

    predict_path = None
    return render_template('predict.html', org_path=org_path, predict_path=predict_path)

@app.route('/pred2/')
def pred2():
    # import object_detection_img.Demo
    newName = session.get('Picname')
    org_path  = '../static/img/%s' % newName
    predict_path = None
    return render_template('predict2.html', org_path=org_path, predict_path=predict_path)

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)
