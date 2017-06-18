import tensorflow as tf
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GRN():
    def __init__(self,
                 placeholders,# transMatrix, labels, labels_mask
                 feature,
                 trans_matrix,
                 cell,
                 input_size,
                 classnum,
                 hidden_size,
                 learning_rate,
                 dropout_keep_proba,
                 max_grad_norm,
                 trainable,
                 featureType,
                 scope=None):

        self.input_size = input_size
        self.classnum = classnum
        self.hidden_size = hidden_size
        self.cell = cell
        self.max_grad_norm=max_grad_norm
        self.placeholders = placeholders
        self.dropout_keep_proba = dropout_keep_proba

        with tf.variable_scope(scope or 'gnn') as scope:

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.is_training = placeholders['is_training']
            # self.output_dim = placeholders['labels'].get_shape().as_list()[1]
            self.inputs = placeholders['inputs']
            print self.inputs.shape
            # self.inputs = tf.placeholder(tf.int32, [None, None])
            self.labels = placeholders['labels']
            self.test_mask = placeholders['test_mask']
            self.val_mask = placeholders['val_mask']
            self.train_mask = placeholders['train_mask']
            # self.transMatrix = placeholders['graph']
            self.transMatrix = trans_matrix
            # self.batch, self.sqlen = tf.shape(self.inputs)

            if featureType == 'bow' or featureType == 'embedding':
                if trainable:
                    self.feature_matrix = tf.Variable(initial_value=feature, name="feature_matrix", trainable=True)
                else:
                    # self.feature_matrix = tf.Variable(initial_value=feature, name="feature_matrix", trainable=False)
                    self.feature_matrix = placeholders['features']
            else:
                self.feature_matrix = tf.get_variable(initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                                      shape=[self.labels.shape[0], input_size], dtype=tf.float32, name="feature_matrix", trainable=True)

            self.get_embedding(scope=scope)
            self.build(scope)

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

            #self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            self.loss = masked_loss(self.cross_entropy, self.train_mask)
            tf.summary.scalar('loss', self.loss)
            self.loss_val = masked_loss(self.cross_entropy, self.val_mask)
            # self.loss_test = masked_loss(self.cross_entropy, self.test_mask)

            # self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
            self.boolres = tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32)
            self.accuracy = masked_accuracy(self.boolres, self.train_mask)
            tf.summary.scalar('accuracy', self.accuracy)
            self.accuracy_val = masked_accuracy(self.boolres, self.val_mask)
            self.accuracy_test = masked_accuracy(self.boolres, self.test_mask)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(learning_rate)
            #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

            self.train_op = opt.apply_gradients(zip(grads, tvars), name='train_op',global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    # def _loss(self):
    #     # Weight decay loss
    #     for var in self.layers[0].vars.values():
    #         self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
    #
    #     # Cross entropy error
    #     self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
    #                                               self.placeholders['labels_mask'])

    # def _accuracy(self):
    #     self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
    #                                     self.placeholders['labels_mask'])

    def build(self, scope):
        with tf.variable_scope('graphrnn') as scope:
            graphrnn = GraphRNN(
                            inputs=self.inputs_embedding,
                            cell_fn=self.cell,
                            # cell_init_args={'state_is_tuple':True},
                            n_hidden=self.hidden_size,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1),
                            # sequence_length=self.sqlen,
                            initial_state=None,
                            transfer_matrix=self.transMatrix,
                            dropout=None,
                            return_last=True,
                            scope=None)

            gnnoutput = graphrnn.outputs

            with tf.variable_scope('dropout'):
                gnnoutput = layers.dropout(gnnoutput, keep_prob=self.dropout_keep_proba,
                    is_training=self.is_training,)
            self.output = gnnoutput
            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(self.output, self.classnum, activation_fn=None)
                self.prediction = tf.argmax(self.logits, axis=-1)

    def get_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.inputs_embedding = tf.nn.embedding_lookup(self.feature_matrix, self.inputs)
        print self.inputs_embedding.shape

    # def predict(self):
    #     return tf.nn.softmax(self.outputs)