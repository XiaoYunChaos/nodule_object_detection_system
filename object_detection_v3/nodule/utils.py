import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw,ImageFont
import math

def draw(filename, result):
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

    x1 = x - 0.5 * width
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

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
    print([[(x0n, y0n), (x1n, y1n)],[(x1n, y1n), (x2n, y2n)],[(x2n, y2n), (x3n, y3n)],[(x0n, y0n), (x3n, y3n)]])
    font = ImageFont.truetype('simsun.ttc', 26)
    draw.text([x1n, y1n-26], 'nodule:69%', fill='black', font=font)
    # draw.text([x2n,y2n],'nodule:97%',fill='black',font=font)
    '''
    thickness: 1 color Chartreuse
    [0.2698766  0.14902619 0.9060464  0.80354726]
    0.67177457
    4.4416866 1.3917235 3.2794006 2.4602408 4.151168 0.8174313 0.6488167 4.1170287
    '''
    plt.figure(figsize=(18, 18))
    plt.imshow(img)
    plt.savefig('static/tmp_img/' + filename)
    plt.show()

def Demo(newName):
    image_path = '../nodule/static/img/%s' % newName
    import numpy as np
    import sys
    import tensorflow as tf

    from matplotlib import pyplot as plt
    from PIL import Image

    # This is needed since the notebook is stored in the object_detection_img folder.
    sys.path.append("..")
    from object_detection.utils import ops as utils_ops

    # if tf.__version__ < '1.4.0':
    #   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
    PATH_TO_CKPT = 'model//frozen_inference_graph.pb'
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

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

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
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
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

    # 输入图片
    # image_path = 'static/img/1.jpg'
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
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
    plt.figure(figsize=(18, 18))
    plt.imshow(image_np)
    plt.savefig('../nodule/static/predictImg/%s'% newName)
    plt.show()


# Demo('1.jpg')