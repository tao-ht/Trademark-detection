import os
import numpy as np
import random
#提取目录下所有图片,更改尺寸后保存到另一目录
from PIL import Image
import os.path

# 训练datasets图片的存放路径
TRAIN_IMAGE_PATH = "./datasets/train_classification_ex/train_100_50"
# 用于模型测试的datasets图片的存放路径
TEST_IMAGE_PATH = "./datasets/train_classification_ex/test_norl"
# 图片名存放路径
train_file_dir = './datasets/train_classification_ex/train.txt'
test_file_dir = './datasets/train_classification_ex/test.txt'

IMAGE_WIDHT = 100
IMAGE_HEIGHT = 50

#图片前期尺寸归一化处理
def pic_normalize(file_dir,out_dir,width=IMAGE_WIDHT,height=IMAGE_HEIGHT):
    for root,dirs,file_names in os.walk(file_dir):
        for file_name in file_names:
            file_path = os.path.join(file_dir,file_name)
            file_out_path = os.path.join(out_dir,file_name)
            img = Image.open(file_path)
            try:
                #resize
                img2 = img.resize((width, height), Image.BILINEAR)
                #转化为灰度图
                # img2 = img.convert("L")
                # #对比度增强
                # enh_con = ImageEnhance.Contrast(img)
                # contrast = 1.5
                # image_contrasted = enh_con.enhance(contrast)
                # 深度边缘增强滤波or中值滤波 or else
                # img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)#可选用不同的滤波器
                img2.save(file_out_path)
            except Exception as error:
                print(error)
# pic_normalize(file_dir = "./datasets/train2_norl/",out_dir ="./datasets/train2_norl_L/")

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
            # X_name.append(i[:-4])
            if data_path == 'train':
                X_name.append(i.split(" ", 1)[0])
                Y_label.append(int((i.split(" ",1)[1])[:-1])+1)  #切片提取label，剔除换行符
            else:
                X_name.append(i[:-1])
                Y_label.append(0)
    Y_label = np.array(Y_label)
    return X_name, Y_label
X_name_train_all, Y_label_train_all = pic_name_label_load(train_file_dir)
X_name_train, Y_label_train = X_name_train_all[400:], Y_label_train_all[400:]
X_name_verify, Y_label_verify =  X_name_train_all[0:400], Y_label_train_all[0:400]
# x,y = pic_name_label_load("./datasets/train_classification_ex/test.txt",'test')
# print('x',x[0:6],'\n','y',y[0:150])

def pic_data_label_load(data_path = TRAIN_IMAGE_PATH,data_type = 'train',batchSize=50,step=0):# train or test
    file_dir = './datasets/'+data_type+'.txt'
    batch_X_name = []
    batch_Y_label = np.zeros([batchSize])
    if data_type == 'test':
        X_name, Y_label = pic_name_label_load(test_file_dir, 'test')
        totalNumber = len(X_name)
        for i in range(batchSize):
            indexStart = step * batchSize
            index = (i+indexStart)%totalNumber
            batch_X_name.append(X_name[index])
            batch_Y_label[i] = Y_label[index]
    elif data_type == 'train':
        totalNumber = len(X_name_train)
        for i in range(batchSize):
            indexStart = step * batchSize
            index = (i+indexStart)%totalNumber
            batch_X_name.append(X_name_train[index])
            batch_Y_label[i] = Y_label_train[index]
    elif data_type == 'verify':
        for i in range(batchSize):
            batch_X_name.append(X_name_train[i])
            batch_Y_label[i] = Y_label_train[i]
    pic_data = np.zeros([len(batch_X_name), 3 * IMAGE_WIDHT * IMAGE_HEIGHT])#(-1,5000)
    pic_label = np.array(batch_Y_label,dtype = int).reshape(-1,1)#(-1,1)
    for img_index in range(len(batch_X_name)):
        img_dir = os.path.join(data_path, batch_X_name[img_index])
        img = Image.open(img_dir)
        image_array = np.array(img).transpose(2,1,0).reshape(1,-1)#3通道时，transpose(2,1,0)
        pic_data[img_index,] = image_array.flatten()/255
        #转化X_name
        # pic_name.append(X_name[img_index])
    return pic_data, pic_label, batch_X_name
# data, label, pic_name = pic_data_label_load(TEST_IMAGE_PATH, 'test')
# print('data.shape =',data.shape,'label.shape =',label.shape,'name =',pic_name[0:5])

#onehot_encoding,输入输出要为整数型
def onehot_encoding(Y_data,classes=100):
    Y_label = np.zeros((Y_data.shape[0],classes),dtype = int)
    for i in range(Y_data.shape[0]):
        Y_label[i,Y_data[i,]-1] = 1
    return Y_label

# 生成一个训练batch
def get_next_batch(batchSize=50, trainOrTest='train', step=0):
    # batch_data = np.zeros([batchSize, 3 * IMAGE_WIDHT * IMAGE_HEIGHT])
    # batch_label = np.zeros([batchSize,100])
    if trainOrTest == 'train':
        data, label, pic_name = pic_data_label_load(TRAIN_IMAGE_PATH, 'train', batchSize, step)
    elif trainOrTest == 'verify':
        data, label, pic_name = pic_data_label_load(TRAIN_IMAGE_PATH, 'verify', batchSize)
    else:
        data, label, pic_name = pic_data_label_load(TEST_IMAGE_PATH, 'test', batchSize, step)
    label = onehot_encoding(label,classes = 100)
    # totalNumber = data.shape[0]
    # batch_name = list(range(0, batchSize))
    # indexStart = step * batchSize
    # for i in range(batchSize):
    #     index = (i + indexStart) % totalNumber
    #     # name = fileNameList[index]
    #     # img_data, img_label = get_data_and_label(name)
    #     batch_data[i, :] = data[index]
    #     batch_label[i, :] = label[index,:]
    #     batch_name[i] = pic_name[index]
    # print('batch_data.shape =',batch_data.shape,'batch_label =',batch_label.shape)
    return data, label, pic_name
# batch_data,batch_label,name3 = get_next_batch(step=0)
# print('aa',batch_data[0:5,0:10],batch_label.shape,'\n')
# batch_data1,batch_label1,_ = get_next_batch(step=2)
# print('bb',batch_data1[0:10],batch_label1.shape,name3)
