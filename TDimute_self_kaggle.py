import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os

def get_next_batch(dataset1, batch_size_1, step, ind): # dataset, 10, 0, [0,1,2,3,4,5,6,7,8,9]
    start = step * batch_size_1 # 0, 1
    end = ((step + 1) * batch_size_1) # 10, 20
    sel_ind = ind[start:end] # [0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18,19]
    newdataset1 = dataset1[sel_ind, :] # dataset[[0,1,2,3,4,5,6,7,8,9], :], dataset[[10,11,12,13,14,15,16,17,18,19], :]
    return newdataset1


# Build a tf1 graph for a 2-layer (inputs--> 4000 unit ---> 19k units---> output) fully connected autoencoder model
def train(drop_prob, dataset_train, dataset_test, sav=True, checkpoint_file='default.ckpt'):
    input_image = tf.placeholder(tf.float32, batch_shape_input, name='input_image')
    is_training = tf.placeholder(tf.bool) # to control dropout
    scale = 0. # for the L2 regularization
    with tf.variable_scope('autoencoder') as scope:
        fc_1 = tf.layers.dense(inputs=input_image, units=4000, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        fc_1_out = tf.nn.sigmoid(fc_1)
        fc_1_dropout = tf.layers.dropout(inputs=fc_1_out, rate=drop_prob, training=is_training)
        fc_2_dropout = tf.layers.dense(inputs=fc_1_dropout, units=19027)  # 46744 (the original prediction)
        fc_2_out = tf.nn.sigmoid(fc_2_dropout)
        reconstructed_image = fc_2_out  # fc_2_dropout (the predictored gene expression values after normalization)

    original = tf.placeholder(tf.float32, batch_shape_output, name='original')
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(reconstructed_image, original))))
    l2_loss = tf.losses.get_regularization_loss()
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss + l2_loss) # runs backpropagation and updates the weights

    # set up the training variables and environment
    init = tf.global_variables_initializer() # initialize all variables in the graph (weights and biases)
    saver = tf.train.Saver()
    start = time.time()
    loss_val_list_train = 0
    loss_val_list_test = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session: # open the session to run the computation graph
        session.run(init) # give random initial values to all variables

        dataset_size_train = dataset_train.shape[0]
        dataset_size_test = dataset_test.shape[0]
        print("Dataset size for training:", dataset_size_train)
        print("Dataset size for test:", dataset_size_test)
        num_iters = (num_epochs * dataset_size_train) // batch_size
        print("Num iters:", num_iters)
        ind_train = []

        # Generate for every epoch a random training samples,
        for i in range(num_epochs):
            ind_train = np.append(ind_train, np.random.permutation(np.arange(dataset_size_train)))
        ind_train = np.asarray(ind_train).astype("int32")

        total_cost_train = 0.
        num_batchs = dataset_size_train // batch_size  # "//" => int()
        for step in range(num_iters): # num of iteration cover all batches and epochs
            ############### train section  ###########################
            temp = get_next_batch(dataset_train, batch_size, step, ind_train)
            train_batch = np.asarray(temp).astype("float32")

            train_loss_val, _ = session.run([loss, optimizer],  # compute loss and do the optimization (backpropagation) for one batch
                                            feed_dict={input_image: train_batch[:, 19027:],
                                                       original: train_batch[:, :19027],
                                                       is_training: True})
            loss_val_list_train = np.append(loss_val_list_train, train_loss_val) # append the training loss value for this batch
            total_cost_train += train_loss_val # accumulate the training loss value

            print_epochs = 10
            if step % (num_batchs * print_epochs) == 0:  # by epoch, num_batchs * batch_size = dataset_train_size
                ############### test section  ###########################
                dataset_test = np.asarray(dataset_test).astype("float32")
                test_loss_val = session.run(loss, feed_dict={input_image: dataset_test[:, 19027:],
                                                             original: dataset_test[:, :19027],
                                                             is_training: False})
                loss_val_list_test = np.append(loss_val_list_test, test_loss_val)

                reconstruct = session.run(reconstructed_image,
                                          feed_dict={input_image: dataset_test[:, 19027:], is_training: False}) # get the predicted gene expression values for test set
                nz = dataset_test[:, :19027].shape[0] * dataset_test[:, :19027].shape[1] # total number of GE values in test set (no of samples * 19K )
                diff_mat = ((reconstruct  - dataset_test[:, :19027] ) * 16.5) ** 2
                loss_test = np.sqrt(np.sum(diff_mat) / nz) # compute the RMSE 

                print('RMSE loss by train_data_size: ', step, "/", num_iters,
                      total_cost_train / (num_batchs * print_epochs),
                      test_loss_val, loss_test)
                # print('RMSE loss by train_data_size: ', step, "/", num_iters, total_cost_train / num_batchs,
                #       total_cost_validation / num_batchs)
                #                 print('new loss: ', step, "/", num_iters, train_loss_val, valid_loss_val)
                total_cost_train = 0.
                total_cost_validation = 0.

        ############### test section  ###########################
        dataset_test = np.asarray(dataset_test).astype("float32")
        test_loss_val = session.run(loss, feed_dict={input_image: dataset_test[:, 19027:],
                                                     original: dataset_test[:, :19027],
                                                     is_training: False})
        loss_val_list_test = np.append(loss_val_list_test, test_loss_val)

        reconstruct = session.run(reconstructed_image,
                                  feed_dict={input_image: dataset_test[:, 19027:], is_training: False})
        nz = dataset_test[:, :19027].shape[0] * dataset_test[:, :19027].shape[1]
        diff_mat = ((reconstruct  - dataset_test[:, :19027] ) * 16.5) ** 2
        loss_test = np.sqrt(np.sum(diff_mat) / nz)

        print('RMSE loss by train_data_size: ', step, "/", num_iters, test_loss_val, loss_test)
        # print('RMSE loss by train_data_size: ', step, "/", num_iters, total_cost_train / num_batchs,
        #       total_cost_validation / num_batchs)
        #                 print('new loss: ', step, "/", num_iters, train_loss_val, valid_loss_val)
    #             save_path = saver.save(session, checkpoint_file)  # " checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    #             print(("Model saved in file: %s" % save_path))

    end = time.time()
    el = end - start
    print(("Time elapsed %f" % el))
    return (loss_val_list_train, loss_val_list_test, loss_test, reconstruct)
	
#############################################################################################################
import getopt
import sys
print(sys.argv)

# datadir = sys.argv[1] #e.g.,'/data0/zhoux'
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] # set the GPU id
cancer_names = [sys.argv[2]] #smaple size greater than 200 , take a cancer name as input (dont know what is its usage yet)
# cancer_names = ['LUSC', 'KIRC', 'CESC', 'STAD', 'SARC', 'COAD','KIRP', 'LUAD', 'BLCA', 'BRCA','HNSC','LGG_','PRAD','THCA','SKCM', 'LIHC']
full_dataset_path = sys.argv[3]
imputed_dataset_path = sys.argv[4]
datadir = os.path.dirname(os.path.abspath(full_dataset_path))


sample_size = 1 # was 5 
loss_list = np.zeros([16, 5, sample_size]) # 16 cancers, 5 missing rates, sample_size repeats
loss_summary = np.zeros([16, 5]) # 16 cancers, 5 missing rates
cancer_c = 0
for cancertype in cancer_names:
    perc = 0
    for missing_perc in [0.5]:   #  [0.1,0.3,0.5,0.7,0.9] # the percentage control the split between train and test data
        for sample_count in range(1,sample_size+1): # What is sample_count used for?
            # shuffle_cancer = pd.read_csv(datadir+'/full_datasets/' + cancertype + str(int(missing_perc * 100)) + '_' + str(sample_count) + '.csv',
            #     delimiter=',', index_col=0, header=0)
            shuffle_cancer = pd.read_csv(full_dataset_path, delimiter=',', index_col=0, header=0, nrows=1000)

            # loop around 4 times getting chunk of 2k rows and do normalization and concateate
            
            print('name:', cancertype, ' missing rate:', missing_perc, ' data size:',shuffle_cancer.shape)
            ########################Create set for training and testing
            ## 16.5 is for gene expression normalization // linearly scaled to [0, 1], the same as the DNA methylation data (beta values)
            aa = np.concatenate((shuffle_cancer.values[:, :19027] / 16.5, shuffle_cancer.values[:, 19027:]), axis=1)
            shuffle_cancer = pd.DataFrame(aa, index=shuffle_cancer.index, columns=shuffle_cancer.columns)
            RDNA = shuffle_cancer.values # RDNA is now a matrix contain just numeric value the value of feature j for sample i.
            test_data = RDNA[0:int(RDNA.shape[0] * missing_perc), :]
            train_data = RDNA[int(RDNA.shape[0] * missing_perc):, :]
            print('train datasize:', train_data.shape[0], ' test datasize: ', test_data.shape[0])

            num_epochs = 100
            batch_size = 16
            lr = 0.0001  # learning_rate = 0.1
            feature_size = RDNA.shape[1]  # 10595 #17176 #(8856, 109995)
            drop_prob = 0.
            batch_shape_input = (None, 27717)  # (128, 11853)
            batch_shape_output = (None, 19027)
            tf.reset_default_graph()
            loss_val_list_train, loss_val_list_test, loss_test, reconstruct = train(drop_prob, train_data, test_data,
                                                                                    sav=True, checkpoint_file=datadir +"/checkpoints/"+ cancertype+str(missing_perc*100)+'_'+str(sample_count)+ '_imputationmodel.ckpt')

            imputed_data = np.concatenate([reconstruct*16.5, train_data[:, :19027]*16.5], axis=0)
            RNA_txt = pd.DataFrame(imputed_data[:, :19027], index=shuffle_cancer.index,
                                   columns=shuffle_cancer.columns[:19027])
            # RNA_txt.to_csv(datadir+'/imputed_data/TDimpute_without_tf_'+cancertype+str(missing_perc*100)+'_'+str(sample_count)+'.csv')
            RNA_txt.to_csv(imputed_dataset_path)

            loss_list[cancer_c, perc, sample_count-1] = loss_test #loss_val_list_test[-1]

        perc = perc + 1
        np.set_printoptions(precision=3)
        print(np.array([np.mean(loss_list[cancer_c,i,:]) for i in range(0,5)] ))
    loss_summary[cancer_c,:] = np.array( [np.mean(loss_list[cancer_c,i,:]) for i in range(0,5)] )
    cancer_c = cancer_c + 1

# print(loss_list)
print('RMSE by cancer (averaged by sampling times):')
print(loss_summary)


