import os
import tensorflow as tf
from n1_model import auto_encoder
#from n2_model import auto_encoder
#from skull_completion_model import auto_encoder

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess1 = tf.Session(config=config)
with sess1.as_default():
    with sess1.graph.as_default():
        model = auto_encoder(sess1)
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('trainable params:',total_parameters)
        # to train the selected model
        model.train()
        # to generate implants using the trained model
        #model.test()


 






