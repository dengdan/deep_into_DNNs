
# coding: utf-8

# In[4]:

# load data
import cPickle as pkl;
import numpy as np;
import mxnet as mx;
import time;
import os;
import matplotlib.pyplot as plt;
from pylab import *
get_ipython().magic(u'matplotlib inline')
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class network:
    def __init__(self):
        MAX = "max"
        AVE = "avg"
        kernel5x5 = (5, 5);
        kernel3x3 = (3, 3);
        stride2x2 = (2, 2);
        stride1x1 = (1, 1);
        pad2x2 = (2, 2);
        pad2x2 = (2, 2);
        def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
            conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='%s_conv_%s' %(name, suffix))
            #bn = mx.symbol.BatchNorm(data=conv);
            act = mx.sym.Activation(data = conv, act_type = 'relu', name='%s_relu_%s' %(name, suffix));
            return act

            
        # network architecture
        def cnn():
            data = mx.sym.Variable('data');
            conv1 = ConvFactory(data = data,  num_filter = 3, kernel = kernel5x5, stride = stride1x1, pad = pad2x2, name = "conv1");
            pool1 = mx.symbol.Pooling(data = conv1, kernel = kernel3x3, stride = stride2x2 , pool_type = MAX, name='pooling1_max');
            conv2 = ConvFactory(data = pool1, num_filter = 32, kernel = kernel5x5, stride = stride1x1, pad = pad2x2, name = 'conv2');
            pool2 = mx.symbol.Pooling(data = conv2, kernel = kernel3x3, stride = stride2x2, pool_type = AVE, name='pooling2_avg');
            conv3 = ConvFactory(data = pool2, num_filter = 64, kernel = kernel5x5, stride = stride1x1, pad = pad2x2, name = 'conv3');
            pool3 = mx.symbol.Pooling(data = conv3, kernel = kernel3x3, stride = stride2x2, pool_type = AVE, name='pooling3_avg');
            flatten = mx.symbol.Flatten(data=pool3);
            fc1 = mx.sym.FullyConnected(data = flatten, num_hidden = 64);
            #relu1 = mx.sym.Activation(data = fc1, act_type = 'relu');
            fc2 = mx.sym.FullyConnected(data = fc1, num_hidden = 10);
            output = mx.sym.SoftmaxOutput(data = fc2, name = 'softmax');# name = 'softmax'不可少
            return output;

        self.model = cnn();        
    
    def dump_result(self, dump_path):
        self.dump_path = dump_path;
        timestamp = time.strftime('%Y-%m-%d__%H_%M_%S',time.localtime(time.time()));
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path);
        dump_file = self.dump_path +'/mxnet_cifar10-' + timestamp ;
        self.dump_file = dump_file;
        with open(dump_file+'results.pkl', 'w') as f:
            pkl.dump(self, f);            
        
        
    def plot_result(self):
        # plot
        ax1 = subplot();
        x = arange(len(self.test_acc)) * self.test_interval;
        x[-1] = self.niter;
        ax1.plot(x, self.train_acc,'red',label='Training Accuracy');
        ax1.plot(x, self.test_acc, 'green', label = 'Test Accuracy');
        ax1.set_xlabel('Iteration');
        ax1.set_ylabel('Accuracy');
        ax1.legend(bbox_to_anchor=(1.5, 0.5))
        ax1.set_title('training accuracy ={0:.2f}, test accuracy = {1:.2f}'.format(self.train_acc[-1], self.test_acc[-1]));
        
    
    def train(self, gpu = 1, niter = 5000, lr= 0.1, test_interval = 100, data_path = '../data/mxnet/', do_print = True):
        self.niter = niter;
        self.test_interval = test_interval;
        batch_size = 100;
        image_size = 32;
        train_iter = mx.io.ImageRecordIter(
            shuffle=True,
            path_imgrec= data_path + "train.rec",
            #mean_img = data_path + "mean32.bin",
            rand_crop = False,
            rand_mirror = False,
            data_shape = (3, image_size, image_size),
            batch_size = batch_size,
            prefetch_buffer=4,
            preprocess_threads=1)

        test_iter = mx.io.ImageRecordIter(
            path_imgrec = data_path + "test.rec",
            #mean_img = data_path + "mean32.bin",
            rand_crop = False,
            rand_mirror = False,
            data_shape = (3, image_size, image_size),
            batch_size = batch_size,
            prefetch_buffer = 4,
            preprocess_threads = 1,
            round_batch = False);


        
        train_acc = [];
        test_acc = [];

        
        # bind the input to the model
        input_shapes = dict(train_iter.provide_data + train_iter.provide_label);
        cnn = self.model;
        exe = cnn.simple_bind(mx.gpu(gpu), **input_shapes);

        # get the handles of input data and label
        arg_arrays = dict(zip(cnn.list_arguments(), exe.arg_arrays));
        data = arg_arrays[train_iter.provide_data[0][0]];
        label = arg_arrays[train_iter.provide_label[0][0]];

        #initialize weights
        #init = mx.init.Uniform(0.01);
        init = mx.init.Normal(0.01);
        for name, arr in arg_arrays.items():
            if name not in input_shapes:
                init(name, arr);

        # create an updater
        opt = mx.optimizer.SGD(
            learning_rate = lr,
            #momentum = 0.9,
            #wd = 0.00001,
            rescale_grad = 1.0 / train_iter.batch_size
        );
        updater = mx.optimizer.get_updater(opt);

        # create a metric
        metric = mx.metric.Accuracy();
        
        # train
        start = time.time();
        epochs = niter/(50000/batch_size);
        it = 0;
        if epochs == 0:
            epochs = 1;
        for epoch in xrange(epochs):
            train_iter.reset();
            for batch in train_iter:
                data[:] = batch.data[0];
                label[:] = batch.label[0];
                exe.forward(is_train=True);
                exe.backward();
                
                for i, pair in enumerate(zip(exe.arg_arrays, exe.grad_arrays)):
                    weight, grad = pair;
                    updater(i, grad, weight);
                
               
                if it % test_interval == 0 or it == niter:
                    metric.update(batch.label, exe.outputs);
                    train_acc.append(metric.get()[1]);
                    corrects = 0;
                    test_iter.reset();
                    for test_batch in test_iter:
                        data[:] = test_batch.data[0];
                        exe.forward();
                        corrects += sum(np.argmax(exe.outputs[0].asnumpy(), axis=1) ==test_batch.label[0].asnumpy());
                    test_acc.append(corrects * 1.0 / 10000);
                    if do_print:     
                        print "Epoch: ", epoch,", Iteration ", it, "\n\tTrain Accuracy:\t", train_acc[-1], "\n\tTest Accuracy:\t\t", test_acc[-1]
                
                it += 1;
        end = time.time();
        time_cost_in_min = (end - start)/60;
        print "Time consumed: " , time_cost_in_min, " minutes."
        self.test_acc = test_acc;
        self.train_acc = train_acc;
        #self.exe = exe;
