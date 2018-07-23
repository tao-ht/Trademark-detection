import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from data_deal import *
import numpy as np

class_num = 100
image_wight = 96
image_hight = 48
img_channels = 3

weight_decay = 0.0005#0.0005
momentum = 0.9

init_learning_rate = 0.0006 #0.1-》0.05->0.005->0.002
reduction_ratio = 4 #减速比


#56773
batch_size = 100
iteration = 572 #568*100=56800>56773—+400
# 128 * 391 ~ 50,000

#test 数据集循环的次数
verify_iteration = 10
#整个图片数据集，循环的周期数
total_epochs = 100  #100*4=400

#test predict数据集循环的次数
test_iteration = 10
#整个图片数据集，循环的周期数
test_total_epochs = 100  #100*10=1000

#预测结果由字典写入test.csv文件
def writeToCsv(test_name,predict_label,epoch,pre_index, file_path=TEST_SAVE_PATH):
    predict = predict_label.reshape(-1, 1)
    file_path1 = file_path + 'test_9_'+'_epoch_' +str(epoch)+'_'+ str(pre_index) + '.csv'
    try:
        with open(file_path1, 'a', newline='')as f:
            for i in range(predict.shape[0]):
                # if predict[i] == 99:
                #     f.write(test_name[i] + ' ' + str(1) + '\n')
                # else:
                f.write(test_name[i] + ' ' + str(predict[i, :][0] + 1) + '\n')
    except Exception as error:
        print(error)

def data_label_load(data_name,data_label,data_type = 'train'):# train or test
    X_name, Y_label = data_name, data_label
    if data_type == 'test':
        data_path = TEST_IMAGE_PATH
        Y_label -= 1
    elif data_type == 'verify':
        data_path = TEST_IMAGE_PATH
        Y_label -= 1
    else:
        data_path = TRAIN_IMAGE_PATH
    pic_label = onehot_encoding(Y_label, class_num)
    pic_data = np.zeros([len(X_name), image_wight,image_hight,3])#(-1,96,48,3)
    for img_index in range(len(X_name)):
        img_dir = os.path.join(data_path, X_name[img_index])
        img = Image.open(img_dir)#(48*96*3)
        image_array = np.array(img).transpose(1,0,2)#.reshape(1,-1)##(96*48*3)
        pic_data[img_index,:,:,:] = image_array
    # pic_data.reshape([-1, image_wight, image_hight, 3])
    # print(np.shape(pic_data),pic_data[0,20:30,0:10,1])
    return pic_data, pic_label, X_name

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        network = Relu(network)
        return network

def Fully_connected(x, units=class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Avg_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    #arg_scope()为batch_norm填充一些共有的默认参数
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        #tf.cond(),类似，if else
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Evaluate(sess,epoch, pre_index, Evaluate_count):
    Evaluate_count += 1
    verify_acc = 0.0
    verify_loss = 0.0
    verify_pre_index = 0
    add = 100   #100*10=1000
    #循环去test数据
    for it in range(verify_iteration):
        verify_batch_name = verify_name[verify_pre_index: verify_pre_index + add]
        verify_batch_label = verify_label[verify_pre_index: verify_pre_index + add]
        verify_batch_x, verify_batch_y, _ = data_label_load(verify_batch_name, verify_batch_label, data_type='verify')
        verify_batch_x = color_preprocessing(verify_batch_x)
        verify_batch_x = data_augmentation(verify_batch_x)
        verify_pre_index = verify_pre_index + add
        # #循环移Evaluate_count位
        # print('----',Evaluate_count)
        # print('++++***',np.argmax(verify_batch_y, 1),'\n')
        # tem = (np.argmax(verify_batch_y, 1)+Evaluate_count+1)%(class_num-1)
        # verify_batch_y = onehot_encoding(tem,class_num)
        # print('***',np.argmax(verify_batch_y, 1),'\n')
        verify_feed_dict = {
            x: verify_batch_x,
            label: verify_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        # loss_, acc_ = sess.run([cost, accuracy], feed_dict=verify_feed_dict)

        loss_, pre_labels_, acc_ = sess.run([cost, labels_max_idx, accuracy], feed_dict=verify_feed_dict)

        writeToCsv(verify_name, pre_labels_, epoch, pre_index)
        verify_loss += loss_
        verify_acc += acc_

    verify_loss /= verify_iteration # average loss
    verify_acc /= verify_iteration # average accuracy

    # if verify_acc >0.98:
    #     writeToCsv(test_name, labels_pre, verify_acc)

    summary = tf.Summary(value=[tf.Summary.Value(tag='verify_loss', simple_value=verify_loss),
                                tf.Summary.Value(tag='verify_accuracy', simple_value=verify_acc)])

    return verify_acc, verify_loss, summary
#predict_TEST_LABELS
"""
def predict_test_labels(sess,epoch):
    test_loss = 0.0
    acc = 0.0
    test_pre_index = 0
    add = 100   #100*10=1000
    #循环去test数据
    for it in range(test_iteration):
        test_batch_name = test_name[test_pre_index: test_pre_index + add]
        test_batch_label = test_label[test_pre_index: test_pre_index + add]
        test_batch_x, test_batch_y, test_batch_name = data_label_load(test_batch_name, test_batch_label, data_type='test')
        test_batch_x = color_preprocessing(test_batch_x)
        test_batch_x = data_augmentation(test_batch_x)
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, pre_labels_ ,acc= sess.run([cost, labels_max_idx, accuracy], feed_dict=test_feed_dict)
        # print('loss=',loss_)
        writeToCsv(test_name, pre_labels_, epoch)
        test_loss += loss_
        acc += acc
    test_loss /= test_iteration # average loss
    acc /= test_iteration
    # summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
    #                             tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_loss, acc
"""
class SE_Inception_v4():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def Stem(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_conv1')
            x = conv_layer(x, filter=32, kernel=[3,3], padding='VALID', layer_name=scope+'_conv2')
            block_1 = conv_layer(x, filter=64, kernel=[3,3], layer_name=scope+'_conv3')

            split_max_x = Max_pooling(block_1)
            split_conv_x = conv_layer(block_1, filter=96, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')
            x = Concatenation([split_max_x,split_conv_x])

            split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv3')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7,1], layer_name=scope+'_split_conv5')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1,7], layer_name=scope+'_split_conv6')
            split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_conv_x1,split_conv_x2])

            split_conv_x = conv_layer(x, filter=192, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv8')
            split_max_x = Max_pooling(x)

            x = Concatenation([split_conv_x, split_max_x])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_A(self, x, scope):
        with tf.name_scope(scope) :
            split_conv_x1 = Avg_pooling(x)
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=96, kernel=[1,1], layer_name=scope+'_split_conv2')

            split_conv_x3 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(split_conv_x3, filter=96, kernel=[3,3], layer_name=scope+'_split_conv4')

            split_conv_x4 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv5')
            split_conv_x4 = conv_layer(split_conv_x4, filter=96, kernel=[3,3], layer_name=scope+'_split_conv6')
            split_conv_x4 = conv_layer(split_conv_x4, filter=96, kernel=[3,3], layer_name=scope+'_split_conv7')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_B(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = Avg_pooling(x)
            split_conv_x1 = conv_layer(split_conv_x1, filter=128, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_split_conv2')

            split_conv_x3 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(split_conv_x3, filter=224, kernel=[1,7], layer_name=scope+'_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=256, kernel=[1,7], layer_name=scope+'_split_conv5')

            split_conv_x4 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv6')
            split_conv_x4 = conv_layer(split_conv_x4, filter=192, kernel=[1,7], layer_name=scope+'_split_conv7')
            split_conv_x4 = conv_layer(split_conv_x4, filter=224, kernel=[7,1], layer_name=scope+'_split_conv8')
            split_conv_x4 = conv_layer(split_conv_x4, filter=224, kernel=[1,7], layer_name=scope+'_split_conv9')
            split_conv_x4 = conv_layer(split_conv_x4, filter=256, kernel=[7,1], layer_name=scope+'_split_connv10')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_C(self, x, scope):
        with tf.name_scope(scope) :
            split_conv_x1 = Avg_pooling(x)
            split_conv_x1 = conv_layer(split_conv_x1, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv2')

            split_conv_x3 = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3_1 = conv_layer(split_conv_x3, filter=256, kernel=[1,3], layer_name=scope+'_split_conv4')
            split_conv_x3_2 = conv_layer(split_conv_x3, filter=256, kernel=[3,1], layer_name=scope+'_split_conv5')

            split_conv_x4 = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_split_conv6')
            split_conv_x4 = conv_layer(split_conv_x4, filter=448, kernel=[1,3], layer_name=scope+'_split_conv7')
            split_conv_x4 = conv_layer(split_conv_x4, filter=512, kernel=[3,1], layer_name=scope+'_split_conv8')
            split_conv_x4_1 = conv_layer(split_conv_x4, filter=256, kernel=[3,1], layer_name=scope+'_split_conv9')
            split_conv_x4_2 = conv_layer(split_conv_x4, filter=256, kernel=[1,3], layer_name=scope+'_split_conv10')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3_1, split_conv_x3_2, split_conv_x4_1, split_conv_x4_2])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_A(self, x, scope):
        with tf.name_scope(scope) :
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=n, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=k, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_B(self, x, scope):
        with tf.name_scope(scope) :
            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            split_conv_x3 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3,3], layer_name=scope+'_split_conv6')
            split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])

            scale = input_x * excitation

            return scale

    def Build_SEnet(self, input_x):
        # [0,0],[0,0],[24,24],[0,0],(96*48)->(96*96),baidu architecture
        input_x = tf.pad(input_x, [[0, 0], [0, 0], [24, 24], [0, 0]])

        x = self.Stem(input_x, scope='stem')

        for i in range(4) :
            x = self.Inception_A(x, scope='Inception_A'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A'+str(i))

        x = self.Reduction_A(x, scope='Reduction_A')

        for i in range(7)  :
            x = self.Inception_B(x, scope='Inception_B'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B'+str(i))

        x = self.Reduction_B(x, scope='Reduction_B')

        for i in range(3) :
            x = self.Inception_C(x, scope='Inception_C'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C'+str(i))

        x = Global_Average_Pooling(x)
        x = Dropout(x, rate=0.2, training=self.training)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x

"""开始的地方，lode数据，开始训练"""
train_name, train_label, verify_name, verify_label, test_name, test_label = prepare_data()

x = tf.placeholder(tf.float32, shape=[None, image_wight, image_hight, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_v4(x, training=training_flag).model
labels_max_idx = tf.argmax(logits, axis=1, name='labels_max_idx')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)#######

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

# with tf.Session() as sess:
with tf.Session() as sess :
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    #调用次数，即结果循环移位的次数
    Evaluate_count = 0

    #可变的学习速率
    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        if epoch % 10 == 0 :#整个数据集循环几个周期后，学习速率下降10倍
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0
        """
        #进行预测输出
        loss_test = predict_test_labels(sess, epoch)
        print('test_loss: %.4f'%loss_test)#无意义，因为test的label为0，其损失无意义
        """
        # 循环训练，
        for step in range(1, iteration + 1):
            #50000总的样本数量，周期循环
            """ 这里batch循环读取"""
            #这里batch循环读取
            if pre_index + batch_size < 57173:#56773
                batch_train_name = train_name[pre_index: pre_index + batch_size]
                batch_train_label = train_label[pre_index: pre_index + batch_size]
            else:
                batch_train_name = train_name[pre_index:]
                batch_train_label = train_label[pre_index:]
            batch_x, batch_y, _ = data_label_load(batch_train_name, batch_train_label, data_type='train')
            # print(len(batch_x),len(batch_x[0]))
            batch_x = color_preprocessing(batch_x)
            batch_x = data_augmentation(batch_x)
            # print('new', len(batch_x[0]), len(batch_x[0][0]), len(batch_x[0][0][0]))
            # print(batch_x[0][0])
            #训练数据字典
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }
            print('pre_index  =',pre_index)
            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            print('batch_loss =',batch_loss)

            pre_labels = labels_max_idx.eval(feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)
            print('batch_acc  =', batch_acc,'\n')

            train_loss += batch_loss
            train_acc += batch_acc
            if pre_index %5000 == 0:
                saver.save(sess=sess, save_path='./model/Inception_v4.ckpt')

            if pre_index %10000 == 0:
                test_acc, test_loss, _ = Evaluate(sess, epoch, pre_index, Evaluate_count)
                print("test_sets_loss: %.4f, test_sets_acc: %.4f \n" % (test_loss, test_acc))

            #循环steps递增
            pre_index += batch_size


        train_loss /= iteration # average loss
        train_acc /= iteration # average accuracy
        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

        test_acc, test_loss, test_summary = Evaluate(sess, epoch, pre_index, Evaluate_count)

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
            epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)

        # line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f \n" % (
        #          epoch, total_epochs, train_loss, train_acc)
        print(line)
        #增加训练log日志
        with open('logs.txt', 'a') as f:
            f.write(line)
        #保存训练好的模型，这里每训练一次都保存
        saver.save(sess=sess, save_path='./model/Inception_v4.ckpt')