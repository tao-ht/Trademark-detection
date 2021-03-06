import os
import numpy as np
import random
#提取目录下所有图片,更改尺寸后保存到另一目录
from PIL import Image
import os.path

# 训练datasets图片的存放路径
TRAIN_IMAGE_PATH = "./data_sets/train_norl"
# 用于模型测试的datasets图片的存放路径
TEST_IMAGE_PATH = "./data_sets/test_norl"
# 图片名存放路径
train_file_dir = './data_sets/train.txt'
test_file_dir = './data_sets/test_pro.txt'
TEST_SAVE_PATH = './data_sets/test_pre_save/'

class_num = 100
image_wight = 200
image_hight = 100
img_channels = 3
#图片前期尺寸归一化处理
def pic_normalize(file_dir,out_dir,width=image_wight,height=image_hight):
    for root,dirs,file_names in os.walk(file_dir):
        for file_name in file_names:
            file_path = os.path.join(file_dir,file_name)
            file_out_path = os.path.join(out_dir,file_name)
            img = Image.open(file_path)
            try:
                #resize
                img2 = img.resize((width, height), Image.BILINEAR)
                img2.save(file_out_path)
            except Exception as error:
                print(error)
# pic_normalize(file_dir = "./data_sets/train_ori_jq/",out_dir ="./data_sets/train_ori_jq_norl/")

#onehot_encoding,输入输出要为整数型
def onehot_encoding(Y_data,classes=100):
    Y_label = np.zeros((Y_data.shape[0],classes),dtype = int)
    for i in range(Y_data.shape[0]):
        Y_label[i,Y_data[i,]] = 1
    return Y_label

"""load train or test pictures data,get name and labels"""
def pic_name_label_load(filename,data_path = 'train'):
    X_name = []
    Y_label = []
    with open(filename, "r") as file_in:
        pics_name = file_in.readlines()
        # 如果是训练数据集，打乱顺序
        if data_path == 'train':
            random.shuffle(pics_name)
        for i in pics_name:
            X_name.append(i.split(" ", 1)[0])
            Y_label.append(int((i.split(" ",1)[1])[:-1]))  #切片提取label，剔除换行符
    Y_label = np.array(Y_label).reshape(-1,1)
    return X_name, Y_label


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

#颜色通道，预处理
def color_preprocessing(x_train):#, x_test
    x_train = x_train.astype('float32')
    # x = x-[x（均值）/x（标准差）]
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    return x_train

#数据增强
def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [200, 100], 4)
    return batch

def prepare_data():
    X_name_train, Y_label_train = pic_name_label_load(train_file_dir,'train')
    # Y_label_train -= 1 # 使用原始训练数据时减1
    X_name_test, Y_label_test = pic_name_label_load(test_file_dir,'test')
    Y_label_test -= 1  # 使用原始训练数据时减1
    return X_name_train, Y_label_train,X_name_test, Y_label_test
