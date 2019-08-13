import SimpleITK as sitk
import matplotlib.pyplot as plt
def solve(length_,path):
    """
    中心裁剪任意尺寸的图片（以中心为原点）
    """
    from skimage import io
    img_data = io.imread(path)  # 图片路径
    io.imshow(img_data)

    slice_width, slice_height, _ = img_data.shape
    width_crop = (slice_width - length_) // 2
    height_crop = (slice_height - length_) // 2
    if width_crop > 0:
        img_data = img_data[width_crop:-width_crop, :, :]
    if height_crop > 0:
        img_data = img_data[:, height_crop:-height_crop, :]
    io.imshow(img_data)
    io.imsave(path,img_data)
# path = 'static/data/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
path ='data/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
image =sitk.ReadImage(path)
image = sitk.GetArrayFromImage(image)
outpath = 'static/data/object_detection_img'
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

