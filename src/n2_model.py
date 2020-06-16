from glob import glob
from conv3 import *
import numpy as np
import nrrd
from data_loader import *

#**************************************************************

# the codes are adapted from 
# https://link.springer.com/chapter/10.1007/978-3-319-75541-0_23
# the network architecture/data loader/loss function is adapted

#**************************************************************


class auto_encoder(object):
    def __init__(self, sess):
        self.sess           = sess
        self.phase          = 'train'
        self.batch_size     = 1
        self.inputI_chn     = 1
        self.output_chn     = 2
        self.lr             = 0.0001
        self.beta1          = 0.3
        self.epoch          = 10000
        self.model_name     = 'n2.model'
        self.save_intval    = 100
        self.build_model()

        # directory where the checkpoint can be saved/loaded
        self.chkpoint_dir   = "../ckpt"
        # directory containing the 100 training defective skulls
        self.train_data_dir = "../training_defective_skull"
        # ground truth (implants) for the training data
        self.train_label_dir = "/implants"
        # test data directory
        self.test_data_dir = "../testing_defective_skulls"
        # directory where the predicted implants from model n1 is stored 
        self.bbox_dir = "../predictions_n1"
        # where to save the predicted implants
        self.save_dir = "../predictions_n2/"



     # 3D dice loss function 
     # credits to (https://link.springer.com/chapter/10.1007/978-3-319-75541-0_23)
    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, 2)
        dice = 0
        for i in range(2):
            inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2*inse/(l+r)
        return -dice


    def build_model(self):
        print('building patch based model...')       
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,256,256,128, self.inputI_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int64, shape=[self.batch_size,256,256,128,1], name='target')
        self.soft_prob , self.task0_label = self.encoder_decoder(self.input_I)
        #3D voxel-wise dice loss
        self.main_dice_loss = self.dice_loss_fun(self.soft_prob, self.input_gt[:,:,:,:,0])
        #self.main_softmax_loss=self.softmax_crossentropy_loss(self.soft_prob, self.input_gt[:,:,:,:,0])
        # final total loss
        self.dice_loss=200000000*self.main_dice_loss
        self.Loss = self.dice_loss
        # create model saver
        self.saver = tf.train.Saver()



    def encoder_decoder(self, inputI):
        phase_flag = (self.phase=='train')
        conv1_1 = conv3d(input=inputI, output_chn=16, kernel_size=3, stride=2, use_bias=True, name='conv1')
        conv1_bn = tf.contrib.layers.batch_norm(conv1_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv1_batch_norm")
        conv1_relu = tf.nn.relu(conv1_bn, name='conv1_relu')
        conv2_1 = conv3d(input=conv1_relu, output_chn=16, kernel_size=3, stride=2, use_bias=True, name='conv2')
        conv2_bn = tf.contrib.layers.batch_norm(conv2_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv2_batch_norm")
        conv2_relu = tf.nn.relu(conv2_bn, name='conv2_relu')
        conv3_1 = conv3d(input=conv2_relu, output_chn=16, kernel_size=3, stride=2, use_bias=True, name='conv3a')
        conv3_bn = tf.contrib.layers.batch_norm(conv3_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv3_1_batch_norm")
        conv3_relu = tf.nn.relu(conv3_bn, name='conv3_1_relu')
        conv4_1 = conv3d(input=conv3_relu, output_chn=16, kernel_size=3, stride=2, use_bias=True, name='conv4a')
        conv4_bn = tf.contrib.layers.batch_norm(conv4_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv4_1_batch_norm")
        conv4_relu = tf.nn.relu(conv4_bn, name='conv4_1_relu')
        conv5_1 = conv3d(input=conv4_relu, output_chn=16, kernel_size=3, stride=2, use_bias=True, name='conv5a')
        conv5_bn = tf.contrib.layers.batch_norm(conv5_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv5_1_batch_norm")
        conv5_relu = tf.nn.relu(conv5_bn, name='conv5_1_relu')
        conv6_1 = conv3d(input=conv4_relu, output_chn=16, kernel_size=3, stride=2, use_bias=True, name='conv6a')
        conv6_bn = tf.contrib.layers.batch_norm(conv6_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv6_1_batch_norm")
        conv6_relu = tf.nn.relu(conv6_bn, name='conv6_1_relu')
        conv5_1 = conv3d(input=conv6_relu, output_chn=64, kernel_size=3, stride=1, use_bias=True, name='conv55a')
        conv5_bn = tf.contrib.layers.batch_norm(conv5_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv55_1_batch_norm")
        conv5_relu = tf.nn.relu(conv5_bn, name='conv55_1_relu')
        feature= conv_bn_relu(input=conv5_relu, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv6_1')
        deconv1_1 = deconv_bn_relu(input=feature, output_chn=32, is_training=phase_flag, name='deconv1_1')
        deconv1_2 = conv_bn_relu(input=deconv1_1, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_2')
        deconv2_1 = deconv_bn_relu(input=deconv1_2, output_chn=32, is_training=phase_flag, name='deconv2_1')
        deconv2_2 = conv_bn_relu(input=deconv2_1, output_chn=32, kernel_size=3,stride=1, use_bias=True, is_training=phase_flag, name='deconv2_2')
        deconv3_1 = deconv_bn_relu(input=deconv2_2, output_chn=32, is_training=phase_flag, name='deconv3_1')
        deconv3_2 = conv_bn_relu(input=deconv3_1, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv3_2')
        deconv4_1 = deconv_bn_relu(input=deconv3_2, output_chn=32, is_training=phase_flag, name='deconv4_1')
        deconv4_2 = conv_bn_relu(input=deconv4_1, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv4_2')
        deconv5_1 = deconv_bn_relu(input=deconv4_2, output_chn=16, is_training=phase_flag, name='deconv5_1')
        deconv5_2 = conv_bn_relu(input=deconv5_1, output_chn=16, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv5_2')
        pred_prob1 = conv_bn_relu(input=deconv5_2, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='pred_prob1')
        pred_prob = conv3d(input=pred_prob1, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob')
        pred_prob2 = conv3d(input=pred_prob, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob2')
        pred_prob3 = conv3d(input=pred_prob2, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob3')
        soft_prob=tf.nn.softmax(pred_prob3,name='task_0')
        task0_label=tf.argmax(soft_prob,axis=4,name='argmax0')
        return soft_prob,task0_label




    def train(self):
        print('training the n2 model')
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.Loss)
        init_op = tf.global_variables_initializer()
        loss_summary_0 =tf.summary.scalar('dice loss',self.Loss)
        self.sess.run(init_op)
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter=1
        data_list =glob('{}/*.nrrd'.format(self.train_data_dir))
        label_list=glob('{}/*.nrrd'.format(self.train_label_dir))
        bbox_list=glob('{}/*.nrrd'.format(self.bbox_dir))
        i=0
        for epoch in np.arange(self.epoch):
            i=i+1
            print('creating batches for training epoch :',i)
            batch_img1, batch_label1,hd,hl= load_bbox_pair(bbox_list,data_list,label_list)
            print('epoch:',i )
            _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: batch_img1, self.input_gt: batch_label1})
            train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: batch_img1})
            print('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(batch_label1),np.sum(train_output0)))
            summary_0=self.sess.run(loss_summary_0,feed_dict={self.input_I: batch_img1,self.input_gt: batch_label1})
            self.log_writer.add_summary(summary_0, counter)           
            print('current training loss:',cur_train_loss)
            counter+=1
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)





    def test(self):
        print('testing patch based model...')  
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load_chkpoint(self.chkpoint_dir):
            print(" *****Successfully load the checkpoint**********")
        else:
            print("*******Fail to load the checkpoint***************")
        data_list =glob('{}/*.nrrd'.format(self.test_data_dir))
        bbox_list=glob('{}/*.nrrd'.format(self.bbox_dir))
        k=1
        for i in range(len(data_list)):
            print('generating result for test sample',k)
            test_input,header=load_bbox_pair_test(bbox_list,data_list,i)
            test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input})
            #implants_post_processed=post_processing(test_output[0,:,:,:])
            #filename=self.save_dir+"implants%d.nrrd"%i
            filename=self.save_dir+bbox_list[i][-15:-5]+'.nrrd'
            nrrd.write(filename,test_output[0,:,:,:].astype('float32'),header)
            k+=1
  

    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s" % ('n2_ckpt')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s" % ('n2_ckpt')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False









