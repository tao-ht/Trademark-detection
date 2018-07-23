import tensorflow as tf
import data_deal

# # 训练datasets图片的存放路径
# TRAIN_IMAGE_PATH = "./datasets/train2_norl/"
# # 用于模型测试的datasets图片的存放路径
# TEST_IMAGE_PATH = "./datasets/test/"

TEST_PREDICTS_PATH = "./datasets/"
# 存放训练好的模型的路径
MODEL_SAVE_PATH = 'D:/A_Program_My_b/B_Baidu_Trademark_Rec/models/'
TEST_SAVE_PATH = './datasets/train_classification_ex/test_addsets_3_100_50_5_29_00_30.csv'

IMAGE_WIDHT = 100
IMAGE_HEIGHT = 50

#预测结果由字典写入test.csv文件
def writeToCsv(test_name,predict_label,file_path=TEST_SAVE_PATH):
    predict = predict_label.reshape(-1, 1)
    try:
        with open(file_path, 'a', newline='')as f:
            for i in range(predict.shape[0]):
                f.write(test_name[i]+' '+str(predict[i,:][0]+1)+'\n')
    except Exception as error:
        print(error)

# 构建卷积神经网络并训练
def train_data_with_CNN():
        # 初始化权值
    def weight_variable(shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

        # 初始化偏置
    def bias_variable(shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(init, name=name)
        return var

        # 卷积
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True,name=name)

        # 池化
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # 输入层
    X = tf.placeholder(tf.float32, [None, IMAGE_WIDHT * IMAGE_HEIGHT * 3], name='data-input')
    Y = tf.placeholder(tf.float32, [None, 100], name='label-input')
    x_input = tf.reshape(X, [-1, IMAGE_WIDHT, IMAGE_HEIGHT, 3], name='x-input')
    # dropout,防止过拟合
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 3, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # 第三层卷积
    W_conv3 = weight_variable([5, 5, 64, 96], 'W_conv3')
    B_conv3 = bias_variable([96], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # 第四层卷积
    W_conv4 = weight_variable([5, 5, 96, 96], 'W_conv4')
    B_conv4 = bias_variable([96], 'B_conv4')
    conv4 = tf.nn.relu(conv2d(conv3, W_conv4, 'conv4') + B_conv4)
    # conv4 = max_pool_2X2(conv4, 'conv4-pool')
    conv4 = tf.nn.dropout(conv4, keep_prob)
    # 全链接层
    # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([13 * 7 * 96, 2048], 'W_fc1')
    B_fc1 = bias_variable([2048], 'B_fc1')
    fc1 = tf.reshape(conv4, [-1, 13 * 7 * 96])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # 输出层
    W_fc2 = weight_variable([2048, 100], 'W_fc2')
    B_fc2 = bias_variable([100], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    predict = tf.reshape(output, [-1, 100], name='predict')
    labels = tf.reshape(Y, [-1, 100], name='labels')

    # 预测结果
    predict_max_idx = tf.argmax(predict, axis=1, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=1, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        for epoch in range(21000):#循环训练周期
            train_data, train_label, _ = data_deal.get_next_batch(batchSize=200,trainOrTest = 'train',step=steps)
            sess.run(optimizer, feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})

            if steps % 100 == 0:
                test_data1, test_label1, _ = data_deal.get_next_batch(batchSize=300,trainOrTest = 'train',step=steps)
                acc1 = sess.run(accuracy, feed_dict={X: test_data1, Y: test_label1, keep_prob: 1.0})
                print("steps=%d, train_sets accuracy1=%f" % (steps, acc1))
                test_data, test_label, _ = data_deal.get_next_batch(batchSize=400,trainOrTest = 'verify')
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("steps=%d, verify_sets accuracy=%f\n" % (steps, acc))
                test_steps = 0
                #如果精确度足够高，保存模型，生成test的预测结果文件
                if acc > 0.9999 or epoch == 20000:
                    saver.save(sess, MODEL_SAVE_PATH + "crack_captcha.model66", global_step=steps)
                    for test_batch in range(10):# test数据提取batch
                        if test_steps ==10:
                            break
                        test_data, test_label, test_name = data_deal.get_next_batch(batchSize=100, trainOrTest='test',step=test_steps)
                        predict_max_idx1 = sess.run(predict_max_idx, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                        writeToCsv(test_name,predict_max_idx1,TEST_SAVE_PATH)
                        test_steps +=1
                    break
            steps += 1


if __name__ == '__main__':
    train_data_with_CNN()
    print('Training finished')



































