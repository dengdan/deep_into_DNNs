#coding:utf-8
import time
import logging
import os
import sys
import mxnet as mx



def train(symbol = None, epochs = 1000, learning_rate = 0.00001, dump_dir = './dump', training_data = "../data/mxnet/train.rec", test_data = "../data/mxnet/test.rec", image_size = 32, batch_size = 100, test_interval = 100, gpu = 0):
        if not os.path.exists(dump_dir):
                os.mkdir(dump_dir);
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()));
        
        # logging
        logging_file = '%s/%s.log'%(dump_dir, timestamp);
        handler = logging.FileHandler(logging_file);
        head = '%(asctime)-15s: %(message)s'
        formatter = logging.Formatter(head);
        handler.setFormatter(formatter);

        logger = logging.getLogger();
        logger.addHandler(handler);
        logger.setLevel(logging.DEBUG);
        logging.info(timestamp)        
             
        # data                
        train_iter = mx.io.ImageRecordIter(
            shuffle=True,
            path_imgrec= training_data,
            #mean_img = data_path + "mean32.bin",
            rand_crop = False,
            rand_mirror = False,
            data_shape = (3, image_size, image_size),
            batch_size = batch_size,
            prefetch_buffer=4,
            preprocess_threads=3)

        test_iter = mx.io.ImageRecordIter(
            path_imgrec = test_data,
            #mean_img = data_path + "mean32.bin",
            rand_crop = False,
            rand_mirror = False,
            data_shape = (3, image_size, image_size),
            batch_size = batch_size,
            prefetch_buffer = 4,
            preprocess_threads = 1,
            round_batch = False);       
        
        # dump model                
        checkpoint = mx.callback.do_checkpoint('model')            
        
        # train
        epoch_size = 50000 / batch_size
        model = mx.model.FeedForward(
                ctx                = mx.gpu(gpu),
                symbol             = symbol,
                num_epoch          = epochs,
                learning_rate      = learning_rate,
                momentum           = 0.9,
                wd                 = 0.00001,
                initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34)
        )

        eval_metrics = ['accuracy']
        ## TopKAccuracy only allows top_k > 1
        #for top_k in [1]:
        #        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))

        batch_end_callback = []
        batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

        model.fit(
                X                  = train_iter,
                eval_data          = test_iter,
                eval_metric        = eval_metrics,
                batch_end_callback = batch_end_callback,
                epoch_end_callback = checkpoint)
