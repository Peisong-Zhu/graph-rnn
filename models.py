import tensorflow as tf
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

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
            self.inputs = placeholders['inputs']
            self.labels = placeholders['labels']
            self.test_mask = placeholders['test_mask']
            self.val_mask = placeholders['val_mask']
            self.train_mask = placeholders['train_mask']
            self.transMatrix = trans_matrix

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
            self.build(trainable, scope)

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

    def build(self, trainable, scope):
        with tf.variable_scope('graphrnn') as scope:
            graphrnn = GraphRNN(
                            inputs=self.inputs_embedding,
                            cell_fn=self.cell,
                            n_hidden=self.hidden_size,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1),
                            initial_state=None,
                            transfer_matrix=self.transMatrix,
                            dropout=None,
                            return_last=False,
                            scope=None)

            gnnoutput = graphrnn.outputs

            if trainable:
                with tf.variable_scope('dropout'):
                    gnnoutput = layers.dropout(gnnoutput, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training)
            self.output = gnnoutput
            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(self.output, self.classnum, activation_fn=None)
                self.prediction = tf.argmax(self.logits, axis=-1)

    def get_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.inputs_embedding = tf.nn.embedding_lookup(self.feature_matrix, self.inputs)
        print self.inputs_embedding.shape

class Pointer_GRN():
    def __init__(self,
                 placeholders,# transMatrix, labels, labels_mask
                 feature,
                 trans_matrix,
                 labeldistribution,
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
            self.inputs = placeholders['inputs']
            self.labels = placeholders['labels']
            self.test_mask = placeholders['test_mask']
            self.val_mask = placeholders['val_mask']
            self.train_mask = placeholders['train_mask']
            self.transMatrix = trans_matrix
            self.labeldis = labeldistribution
            self.tensor_labeldis = tf.convert_to_tensor(self.labeldis)

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
            self.build(trainable, scope)

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

            self.train_op = opt.apply_gradients(zip(grads, tvars), name='train_op',global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    def build(self, trainable, scope):
        with tf.variable_scope('graphrnn') as scope:
            graphrnn = GraphRNN(
                            inputs=self.inputs_embedding,
                            cell_fn=self.cell,
                            n_hidden=self.hidden_size,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1),
                            initial_state=None,
                            transfer_matrix=self.transMatrix,
                            dropout=None,
                            return_last=True,
                            scope=None)

            gnnoutput = graphrnn.outputs

            if trainable:
                with tf.variable_scope('dropout'):
                    gnnoutput = layers.dropout(gnnoutput, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training)

            self.output = gnnoutput
            with tf.variable_scope('pointer'):
                p_gen = linear([self.feature_matrix, self.output], 1, True)
                p_gen = tf.sigmoid(p_gen)
            # with tf.variable_scope('reset'):
            #     r_gen = linear([self.tensor_labeldis, self.output], 1, True)
            #     r_gen = tf.sigmoid(r_gen)
            # with tf.variable_scope('candidate'):
            #     n_gen = linear([self.tensor_labeldis, r_gen * self.output], 1, True)
            #     n_gen = tf.sigmoid(n_gen)
            with tf.variable_scope('final_distribution'):
                # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
                self.gendis = layers.fully_connected(self.output, self.classnum, activation_fn=None)
                self.logits = p_gen * self.gendis + (1 - p_gen) * self.labeldis
                # self.logits = p_gen * self.gendis + (1 - p_gen) * n_gen
                self.prediction = tf.argmax(self.logits, axis=-1)

    def get_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.inputs_embedding = tf.nn.embedding_lookup(self.feature_matrix, self.inputs)
        print self.inputs_embedding.shape

def linear(args, output_size, bias, bias_start=0.0, scope=None):
      """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

      Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
      """
      if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
      if not isinstance(args, (list, tuple)):
        args = [args]

      # Calculate the total size of arguments on dimension 1.
      total_arg_size = 0
      shapes = [a.get_shape().as_list() for a in args]
      for shape in shapes:
        if len(shape) != 2:
          raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
          raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
          total_arg_size += shape[1]

      # Now the computation.
      with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
          res = tf.matmul(args[0], matrix)
        else:
          res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
          return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
      return res + bias_term
