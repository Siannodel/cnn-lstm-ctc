import dataset
import tensorflow as tf
import sys
sys.path.append('../../MyLibrary/')
import siannodel.ml.tf_estimator as myestimator
import siannodel.ml.tf_extend as mytf
from easydict import EasyDict
import siannodel.mytime as mytime
import os

#构建模型
class Net(myestimator.BaseNet):
    def __init__(self, config):
        super(Net,self).__init__(config)
        self.kernel_regularizer =\
            tf.contrib.layers.l2_regularizer(self.config.regularaztion_rate)
    def __call__(self,input_dict,training):
        '''
        @x:输入的tensor,形状需要满足期望的条件，函数直接进入推理
        @training:为True时为训练模式
        @return: 前向推理结果，输出大小与分类数无关，需要再接输出层
        '''
        images = input_dict['image']
        images = tf.reshape(images,self.config.input_tensor_shape)
        #此处定义主要网络结构
        #block1
        x = tf.layers.conv2d(inputs=images,filters=64,kernel_size=[3,3],
                             padding='same',
                             name='conv1_1',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x,filters=64,kernel_size=[3,3],
                             padding='same',
                             name='conv1_2',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, name='pool1')
        #block2
        x = tf.layers.conv2d(inputs=x,filters=128,kernel_size=[3,3],
                             padding='same',
                             name='conv2_1',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x,filters=128,kernel_size=[3,3],
                             padding='same',
                             name='conv2_2',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, name='pool2')
        #block3
        x = tf.layers.conv2d(inputs=x,filters=256,kernel_size=[3,3],
                             padding='same',
                             name='conv3_1',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x,filters=256,kernel_size=[3,3],
                             padding='same',
                             name='conv3_2',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x,filters=256,kernel_size=[3,3],
                             padding='same',
                             name='conv3_3',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, name='pool3')
        #block4
        x = tf.layers.conv2d(inputs=x,filters=512,kernel_size=[3,3],
                             padding='same',
                             name='conv4_1',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x,filters=512,kernel_size=[3,3],
                             padding='same',
                             name='conv4_2',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x,filters=512,kernel_size=[3,3],
                             padding='same',
                             name='conv4_3',
                             kernel_regularizer=self.kernel_regularizer)
        x = tf.layers.batch_normalization(x,training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, name='pool4')
        
        # LSTM
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_hidden)
        # 图片的宽度代表着序列，将其调整到第二维度
        # x.shape = (b,w,h,c)
        x = tf.transpose(x,[0,2,1,3])
        #这个地方如何自动
        x = tf.reshape(x,[-1,16,4*512])
        x,_ = tf.nn.dynamic_rnn(lstm_cell,x,time_major=False, 
                              dtype=tf.float32)
        # x经过LSTM后，shape = (batch_size,times(16),num(128))
        
        x = tf.reshape(x,[-1,self.config.num_hidden])
        x = tf.layers.dense(x,self.config.num_classes+1,name='fn1',
                            kernel_regularizer=self.kernel_regularizer)
        x = tf.reshape(x,[-1,16,
                          self.config.num_classes+1])

        x = tf.transpose(x, (1, 0, 2))
        output_dict = {
            'logits': x,
        }
        return output_dict

class Model(myestimator.BaseModel):
    def __init__(self,config,net):      
        super(Model,self).__init__(config)
        self.net = net
        self.seq_len_size = self.config.batch_size
    def __call__(self,features,labels,mode):
        
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        global_step = tf.Variable(0, trainable=False)
        
        output_dict = self.net(features,training)
        
        logits = output_dict['logits']
        #seq_lens = features['seq_len']

        ema = tf.train.ExponentialMovingAverage(
            self.config.moving_average_decay,
            global_step)
            
        if training:
            ema_op = ema.apply(tf.trainable_variables())
        else:
            ema_restore = ema.variables_to_restore()

        seq_lens = [16] * self.seq_len_size
        #预测
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_lens, merge_repeated=False)
        pred = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        prob = tf.exp(log_prob[0])
        if mode == tf.estimator.ModeKeys.PREDICT:
            return pred, prob, ema_restore

        #损失
        loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=labels,inputs=logits,
            sequence_length=seq_lens))

        loss += tf.losses.get_regularization_loss()
        tf.summary.scalar('loss',loss)
        
        # 训练
        if mode == tf.estimator.ModeKeys.TRAIN:
             #加入bn参数
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #学习率衰减
            learning_rate = tf.train.exponential_decay(
                self.config.learning_rate,
                global_step,
                5000,0.5,staircase=True)
            
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
            
            optim_op = optimizer.minimize(
                    loss=loss,
                    global_step = global_step
                )
            # 保证先优化前执行extra_update_ops
            # train_op 会执行optim_op,ema_op这两个操作
            with tf.control_dependencies(extra_update_ops):
                with tf.control_dependencies([optim_op,ema_op]):
                    train_op = tf.no_op(name='train_op')
            return loss,train_op,global_step,learning_rate
        #评估
        distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        return loss,distance,ema_restore

class MyEstimator(object):
    def __init__(self,model_fn,config):
        assert isinstance(config,EasyDict)
        #print(config)
        self.config = config
        self.model_fn = model_fn
        if hasattr(config,'model_path') \
            and not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        if hasattr(config,'log_path') \
            and not os.path.exists(config.log_path):
            os.makedirs(config.log_path)
        self.inference_sess = None

    def __del__(self):
        if self.inference_sess != None:
            self.inference_sess.close()
    
    def compute_accuracy(preds,labels):
        '''
        @destription：计算准确率，预测结果与标签均为稀疏矩阵表示，
            不定长多标签分类，一个验证码全部预测对算对
        '''
        #首先对比输出的序列数是否相等
        if preds[2][0] !=  labels[2][0]:
            #抛出错误
            print('预测结果数与标签数不相等！')
            return -1
        right_num = 0
        length = labels[2][0]
        for i in range(length):
            tmp_pred = preds[1][np.where(preds[0][:,0]==i)]
            tmp_label = labels[1][np.where(labels[0][:,0]==i)]
            if len(tmp_pred) == len(tmp_label)\
                and (tmp_pred == tmp_label).all():
                right_num += 1
        acc = right_num/length
        return acc

    def train(self,input_fn):
        
        tf.reset_default_graph()
        self.model_fn.seq_len_size = self.config.batch_size
        #features, labels 
        batch = input_fn()
        features = {
            'image' : batch['image'],
            'seq_len': batch['seq_len'],
        }
        labels = mytf.dense2sparse(batch['label'])
        self.model_fn.seq_len_size = self.config.batch_size

        loss,train_op,global_step,learning_rate = self.model_fn(
            features,labels,tf.estimator.ModeKeys.TRAIN)
        
        saver = tf.train.Saver(max_to_keep=10)
        init_op = tf.group(tf.local_variables_initializer(),
                           tf.global_variables_initializer())
        
        with tf.Session() as sess:
            sess.run(init_op)
            model_file = tf.train.latest_checkpoint(self.config.model_path)
            if model_file is None:
                print('Not find model_file!')
            else:
                print('Find model_file successfully！')
                saver.restore(sess,model_file)
            summary_writer = tf.summary.FileWriter(
                self.config.log_path,graph=tf.get_default_graph())
            
            start_step = sess.run(global_step)
            print('Start training, step:',start_step,
                  ' current time：',mytime.current_time())
            
            sum_loss = 0
            for i in range(start_step,self.config.max_steps):
                _,loss_value,step,lr = sess.run([train_op,loss,global_step,learning_rate])
                sum_loss += loss_value
                if (i+1)%self.config.display_step == 0:
                    print("Iter"+str(step)+",Training Loss={:.6f}"\
                          .format(loss_value),
                          'lr='+str(lr),
                          mytime.current_time())

                if (i+1)%self.config.save_step == 0:
                    saver.save(sess,
                               os.path.join(self.config.model_path,
                                                 'model.ckpt'),
                               global_step=step)
                    print('Save model!')
                
            summary_writer.close()
            print("Optimization Finished! current time：",
                  mytime.current_time())
    
    def evaluate(self, input_fn):
        tf.reset_default_graph()
        
        features, labels = input_fn()
        loss,train_op,global_step = self.model_fn(
            features,labels,tf.estimator.ModeKeys.EVAL)
    
    def prepare_inference(self, batch_size = 1):
        tf.reset_default_graph()
        shape = [None, self.config.image_shape[0], self.config.image_shape[1], 3]
        self.input_dict = {
            'image': tf.placeholder(tf.float32, shape=shape, name='image'),
        }
        self.model_fn.seq_len_size = batch_size
        pred, prob, ema_restore = self.model_fn(
            self.input_dict, None, tf.estimator.ModeKeys.PREDICT
        )
        saver = tf.train.Saver(ema_restore)
        init_op = tf.group(tf.local_variables_initializer(),
                           tf.global_variables_initializer())
        self.op_dict = {
            'pred': pred,
            'prob': prob,
        }
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.inference_sess = tf.Session(config = config)
        self.inference_sess.run(init_op)
        file = tf.train.latest_checkpoint(self.config.model_path)
        if file is None:
            print('Load model failed!')
            self.inference_sess.close()
            return False
        saver.restore(self.inference_sess, file)
        print('Find model_file successfully！')
        return True
    
    def inference(self, input_data):
        '''
        input_data:可以直接输入网络的数据
        '''
        pred, prob = self.inference_sess.run([self.op_dict['pred'],
                                                     self.op_dict['prob']],
                                                    feed_dict={self.input_dict['image']:input_data})
        
        return pred, prob
        
        

        
