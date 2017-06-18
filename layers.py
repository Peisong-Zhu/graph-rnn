import tensorflow as tf
# from tf.contrib.rnn.python.ops.core_rnn_cell_impl import BasicRNNCell as BasicRNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicRNNCell as BasicRNNCell
import tensorflow.contrib.layers as layers

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

class GraphBasicRNNCell(BasicRNNCell):

    def __init__(self, num_units, transfer_matrix, input_size=None):
        super(GraphBasicRNNCell, self).__init__(num_units)
        self.transfer_matrix = transfer_matrix

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        statetrans = tf.matmul(self.transfer_matrix, state)
        output, output = super(GraphBasicRNNCell, self).__call__(inputs=inputs, state=statetrans, scope=scope)
        # with tf.variable_scope(scope or "graph_basic_rnn_cell"):
        #     output = self._activation(
        #         corernncellimpl._linear([inputs, state], self._num_units, True, scope=scope))
        return output, output

class GraphBasicRNNAttentionCell(BasicRNNCell):

    def __init__(self, num_units, transfer_matrix, input_size=None):
        super(GraphBasicRNNAttentionCell, self).__init__(num_units)
        self.transfer_matrix = transfer_matrix

    def __call__(self, inputs, state, output_size=64, activation_fn=tf.tanh, scope=None):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        with tf.variable_scope('attention') as scope:
            attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                       shape=[output_size, 1],
                                                       initializer=layers.xavier_initializer(),
                                                       dtype=tf.float32)
            input_projection = layers.fully_connected(state, output_size, activation_fn=activation_fn, scope=scope)
            exp = tf.exp(tf.matmul(input_projection, attention_context_vector))
            one = tf.zeros(shape=[1, 1], dtype=tf.float32)
            expnew = tf.concat([one, exp], 0)
            with tf.variable_scope('expembedding') as scope:
                trans = tf.nn.embedding_lookup(expnew, self.transfer_matrix)
            (n1, n2, n3) = tf.unstack(tf.shape(trans))
            trans = tf.reshape(trans, [n1, n2])
            print trans.shape
            expsum = tf.reduce_sum(trans, 1, keep_dims=True)
            print expsum.shape
            trans = trans / expsum
            print trans.shape
            # expsum = tf.matmul(self.transfer_matrix, exp)
            # transVariable = tf.convert_to_tensor(self.transfer_matrix)
            # trans = self.transfer_matrix

            # for row_idx, row in enumerate(trans):
            #     for col_idx, element in enumerate(row):
            #         if element != 0.0:
            #             print element, trans[row_idx, col_idx]
            #              transVariable[row_idx, col_idx] = tf.multiply(tf.divide(exp[col_idx, 0], expsum[row_idx, 0]), element)
        statetrans = tf.matmul(trans, state)
        output, output = super(GraphBasicRNNAttentionCell, self).__call__(inputs=inputs, state=statetrans, scope=scope)
        return output, output


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.histogram_summary(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.histogram_summary(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.histogram_summary(self.name + '/vars/' + var, self.vars[var])

class GraphRNN(Layer):

    def __init__(
        self,
        layer = None,
        inputs = None,
        cell_fn = None,#tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple':True},
        #cell_init_args=None,
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        sequence_length=None,
        initial_state=None,
        transfer_matrix=None,
        dropout = None,
        n_layer = 1,
        return_last = False,
        return_seq_2d = False,
        dynamic_rnn_init_args={},
        scope=None,
        name='graph_rnn_layer',
    ):
        Layer.__init__(self, name=name)
        if cell_fn is None:
            raise Exception("Please put in cell_fn")
        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except:
                pass

        self.inputs = inputs

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("       batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("       non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        with tf.variable_scope(name, initializer=initializer) as vs:
            # Creats the cell function
            cell_instance_fn=lambda: cell_fn(num_units=n_hidden, transfer_matrix=transfer_matrix)
            # cell_instance_fn = lambda: cell_fn(num_units=, **cell_init_args)

            self.cell=cell_instance_fn()
            # Initial state of RNN
            if initial_state is None:
                self.initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.initial_state = initial_state

            outputs, final_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=self.inputs,
                sequence_length=sequence_length,
                initial_state=self.initial_state,
                scope=scope,
                **dynamic_rnn_init_args
            )
            if return_last:
                # [batch_size, 2 * n_hidden]
                self.outputs = final_state

