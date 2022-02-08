import numpy as np
import tensorflow as tf
import pandas as pd
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from yellowfin import YFOptimizer
from numpy.random import seed
seed(2020)
from tensorflow import set_random_seed
set_random_seed(2020) 

class DeepFM(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 feature_size, 
                 field_size,
                 embedding_size=32, 
                 dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], 
                 dropout_deep=[1.0,1.0,1.0,],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10, 
                 batch_size=50,
                 learning_rate=0.0001, 
                 optimizer="adam",
                 batch_norm=0.6, 
                 batch_norm_decay=0.90,
                 verbose=False, 
                 random_seed=2016,
                 use_fm=True, 
                 use_deep=True,
                 loss_type="logloss", 
                 eval_metric=roc_auc_score,
                 l2_reg=0.001, 
                 greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result,self.valid_result,self.test_result  = [],[],[]

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            #input data
            with tf.name_scope('inputs'):

                self.feat_index = tf.placeholder(tf.int32,shape=[None,None],name='feat_index')
                self.feat_value = tf.placeholder(tf.float32,shape=[None,None], name='feat_value')
                self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
            self.dropout_keep_fm = tf.placeholder(tf.float32,shape=[None],name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool,name='train_phase')
            
            #weight
            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
            self.embeddings = tf.multiply(self.embeddings,feat_value)


            # first order term
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'],self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order,feat_value),2)
            self.y_first_order = tf.nn.dropout(self.y_first_order,self.dropout_keep_fm[0])

            # second order term
            # sum-square-part
            self.summed_features_emb = tf.reduce_sum(self.embeddings,1) # None * k
            self.summed_features_emb_square = tf.square(self.summed_features_emb) # None * K

            # squre-sum-part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            #second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,self.squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order,self.dropout_keep_fm[1])


            # Deep component
            self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[0])
            
            
            #deep_layers_activation
            for i in range(0,len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["layer_%d" %i]), self.weights["bias_%d"%i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                    #print("norm yes")
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])


            #----DeepFM---------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep

            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])

            # loss
            with tf.name_scope('loss'):
                if self.loss_type == "logloss":
                    self.out = tf.nn.sigmoid(self.out)
                    self.loss = tf.losses.log_loss(self.label, self.out)
                elif self.loss_type == "mse":
                    self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
                # l2 regularization on weights
                if self.l2_reg > 0:
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["concat_projection"])
                    if self.use_deep:
                        for i in range(len(self.deep_layers)):
                            self.loss += tf.contrib.layers.l2_regularizer(
                                self.l2_reg)(self.weights["layer_%d" % i])
                tf.summary.scalar('loss', self.loss)
                            
            with tf.name_scope('train_optimizer'):
                if self.optimizer_type == "adam":
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.loss)
                elif self.optimizer_type == "adagrad":
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
                elif self.optimizer_type == "gd":
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                elif self.optimizer_type == "momentum":
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)
                elif self.optimizer_type == "yellowfin":
                    self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(self.loss)


            #init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter("logs/20210506_lasso_xgboost",self.sess.graph)
            self.test_writer = tf.summary.FileWriter("logs/20210506_lasso_xgboost", self.sess.graph)
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)





    def _initialize_weights(self):
        weights = dict()

        
        with tf.name_scope('layer'):
            
            #embeddings
            with tf.name_scope('feature_embeddings'):
                weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size,self.embedding_size],0.0,0.01),name='feature_embeddings')
                #weights['feature_embeddings'] 存放的每一个值其实就是FM中的vik，所以它是F * K的。其中，F代表feture的大小(将离散特征转换成one-hot之后的特征总量),K代表dense vector的大小
                tf.summary.histogram('feature_embeddings', weights['feature_embeddings']) 
            
            with tf.name_scope('_feature_bias'):       
                weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size,1],0.0,1.0),name='feature_bias')
                #weights['feature_bias']是FM中的一次项的权重
                tf.summary.histogram('feature_bias', weights['feature_bias']) 


            #deep layers
            num_layer = len(self.deep_layers)
            input_size = self.field_size * self.embedding_size
            glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))
            
            with tf.name_scope('layer_0'):    
                weights['layer_0'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,self.deep_layers[0])),dtype=np.float32,name="weights_layer_0")
                tf.summary.histogram('layer_0', weights['layer_0']) 
            with tf.name_scope('bias_0'):
                weights['bias_0'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32,name="weights_bias_0")
                tf.summary.histogram('bias_0', weights['bias_0']) 
    
    
            for i in range(1,num_layer):
                layer_names="layer_%d" % i
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                with tf.name_scope("layer_%d" % i):
                    weights["layer_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),name="layer_",dtype=np.float32)  # layers[i-1] * layers[i]
                    tf.summary.histogram(layer_names+'layer' , weights["layer_%d" % i]) 
                with tf.name_scope("bias_%d" % i):
                    weights["bias_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),name="bias_",dtype=np.float32)  # 1 * layer[i]
                    tf.summary.histogram(layer_names+'bias' , weights["bias_%d" % i]) 
    
    
            # final concat projection layer
    
            if self.use_fm and self.use_deep:
                input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
            elif self.use_fm:
                input_size = self.field_size + self.embedding_size
            elif self.use_deep:
                input_size = self.deep_layers[-1]
    
            glorot = np.sqrt(2.0/(input_size + 1))
            with tf.name_scope("weights_concat_projection"):
                weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32,name="concat_projection")
                tf.summary.histogram('concat_projection' , weights['concat_projection']) 
            with tf.name_scope("weights_concat_bias"):
                weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32,name="concat_bias")
                tf.summary.histogram('concat_bias' , weights['concat_bias']) 


        return weights
        
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self,Xi,Xv,y,batch_size,index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end],Xv[start:end],[[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)#roc

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         #self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),#dropout是多少，是这个数字决定
                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),#dropout
                         self.train_phase: False}
            #做预测，所以这是是false
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def fit_on_batch(self,Xi,Xv,y):
        feed_dict = {self.feat_index:Xi,
                     self.feat_value:Xv,
                     self.label:y,
                     self.dropout_keep_fm:  self.dropout_fm,#dropout是多少有这个数字决定
                     self.dropout_keep_deep:  self.dropout_dep,
                     self.train_phase:True}#做拟合，所以这是是true

        loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
        rs = self.sess.run(self.merged,feed_dict = feed_dict)

        return loss,rs

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            Xi_test=None, Xv_test=None,y_test=None,
            early_stopping=True, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        loss_batch = []
        has_valid = Xv_valid is not None
        #nonorm时候可以使用SMOTE，否则使用RandomOverSampler
# =============================================================================
#         Xv_resampled, y_resampled1 = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(Xv_train), pd.DataFrame(y_train))
#         Xi_resampled, y_resampled2 = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(Xi_train), pd.DataFrame(y_train))
#         if all(y_resampled1 == y_resampled2) :
#             print(Xv_resampled.shape)
#             print(pd.DataFrame(Xv_train).shape)
#             Xv_train = Xv_resampled.tolist()
#             Xi_train = Xi_resampled.tolist()
#             y_train = y_resampled1.tolist()
# =============================================================================
            
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            #这里y_trains是整体训练集y中的cv中的train部分，相当于2/3个y
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                loss,rs = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                loss_batch.append(loss)
#                print('loss = %.4f' % loss)
                self.train_writer.add_summary(rs,epoch) 

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            
#                test_result = self.evaluate(Xi_test, Xv_test, y_test)
#                self.test_result.append(test_result)
                
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
#                    print("[%d] train-result=%.4f, valid-result=%.4f ,test-result=%.4f [%.1f s]"
#                        % (epoch + 1, train_result, valid_result, test_result,time() - t1))
                    continue
                else:
#                    print("[%d] train-result=%.4f [%.1f s]"
#                        % (epoch + 1, train_result, time() - t1))
                    continue
            #设置early_stopping
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        #在train + valid上再适应几个epoch，直到结果达到best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,self.batch_size, i)
                    loss,rs = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                    
#                    print('loss2 = %.4f' % loss)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break
        return loss_batch

#评价指标如果是gini，那么最后一个小于倒数第二个，小于倒数第三个，小于倒数第四个，小于第一个，这时候就可以停了！！！！
    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False













