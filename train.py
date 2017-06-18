from __future__ import division
from __future__ import print_function

import time
import os
import shutil
import tensorflow as tf

from utils import *
from models import GRN
from layers import GraphBasicRNNCell
from layers import GraphBasicRNNAttentionCell


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora2708', 'Dataset string.')  
flags.DEFINE_integer('classnum', 7, 'Number of class')
flags.DEFINE_string('model', 'gcn', 'Model string.') 
flags.DEFINE_integer('percent', 80, ' ')
flags.DEFINE_integer('k_level', 5, ' ')
flags.DEFINE_integer('trainable', 0, ' ')
flags.DEFINE_string('feature', 'bow', ' ') # 'bow' 'embedding' 'random'
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('input_size', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('attention', 0, 'whether use attention mechanism')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_grad_norm', 5.0, 'max-grad-norm')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')


def construct_feed_dict(inputs, labels, train_mask, val_mask, test_mask, protransfer, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['inputs']: inputs})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['train_mask']: train_mask})
    feed_dict.update({placeholders['val_mask']: val_mask})
    feed_dict.update({placeholders['test_mask']: test_mask})
    # feed_dict.update({placeholders['graph']: protransfer})
    feed_dict.update({placeholders['inputs']: inputs})
    feed_dict.update({placeholders['is_training']: True})
    return feed_dict

# Define model evaluation function
def evaluate(sess, model, feed_dict):
    loss, accuracy_val, accuracy_test = sess.run([
        model.loss_val,
        model.accuracy_val,
        model.accuracy_test], feed_dict)
    return loss, accuracy_val, accuracy_test

def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    fres = open('data/' + FLAGS.dataset + '/res_attention.txt', 'a')
    fres.write('percent:' + str(FLAGS.percent) + ' attention' + str(FLAGS.attention) + ' k_level:' + str(FLAGS.k_level) + ' trainabel:' + str(FLAGS.trainable) +
               ' featureType:' + FLAGS.feature + ' lr:' + str(FLAGS.learning_rate) + ' input_size:' + str(FLAGS.input_size) + ' hidden_size:' + str(FLAGS.hidden1) + '\n')
    print('k_level:' + str(FLAGS.k_level) + ' trainabel:' + str(FLAGS.trainable) + ' featureType:' + FLAGS.feature +
               ' lr:' + str(FLAGS.learning_rate) + ' input_size:' + str(FLAGS.input_size) + ' hidden_size:' + str(FLAGS.hidden1))
    maxValAuc_test_allfold = 0.0
    minValLoss_test_allfold = 0.0
    for fold in range(10):
        f = str(fold)
        fres.write('fold ' + f + ':')
        print('fold' + f)
        base = 'data/' + FLAGS.dataset + '/' + f
        tflog_dir = os.path.join(base + '/tflog')
        # Load data
        id, labels, train_mask, val_mask, test_mask, protransfer, inputs = load_data(FLAGS.dataset, FLAGS.percent, FLAGS.k_level, f, FLAGS.attention)
        idnum = len(id)
        # Some preprocessing
        # features = preprocess_features(features)

        if FLAGS.feature == 'bow':
            features = np.zeros((idnum, idnum), np.float32)
            for i in range(idnum):
                features[i][i] = 1.0
        elif FLAGS.feature == 'embedding':
            features = getEmbedding(base + '/embedding.txt', idnum, FLAGS.input_size)
        else:
            features = np.zeros((1, 1), np.int16)

        tf.reset_default_graph()
        # Define placeholders
        placeholders = {
            'inputs': tf.placeholder(tf.int32, shape=(idnum, FLAGS.k_level)),
            'labels': tf.placeholder(tf.int32, shape=(idnum)),
            'train_mask': tf.placeholder(tf.int32, shape=(idnum)),
            'test_mask': tf.placeholder(tf.int32, shape=(idnum)),
            'val_mask': tf.placeholder(tf.int32, shape=(idnum)),
            # 'graph': tf.placeholder(tf.float32, shape=(None, None)),
            'is_training': tf.placeholder(dtype=tf.bool, name='is_training')
            # 'dropout': tf.placeholder_with_default(0., shape=()),
        }
        if FLAGS.feature == 'bow':
            placeholders['features'] = tf.placeholder(tf.float32, shape=(idnum, idnum))
        if FLAGS.feature == 'embedding':
            placeholders['features'] = tf.placeholder(tf.float32, shape=(idnum, FLAGS.input_size))


        # Set random seed
        seed = 123
        np.random.seed(seed)
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            # Create model
            if FLAGS.attention == 0:
                cell_type = GraphBasicRNNCell
            else:
                cell_type = GraphBasicRNNAttentionCell
            model = GRN(
                placeholders=placeholders,  # transMatrix, labels, labels_mask
                feature=features,
                trans_matrix=protransfer,
                cell=cell_type,
                input_size=FLAGS.input_size,
                classnum=FLAGS.classnum,
                hidden_size=FLAGS.hidden1,
                learning_rate=FLAGS.learning_rate,
                dropout_keep_proba=0.5,
                max_grad_norm=FLAGS.max_grad_norm,
                trainable=FLAGS.trainable,
                featureType=FLAGS.feature,
                scope=None)

            # Init variables
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_maxacc)
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            summary_writer = tf.summary.FileWriter(tflog_dir, graph=tf.get_default_graph())

            # Construct feed dictionary
            feed_dict = construct_feed_dict(inputs, labels, train_mask, val_mask, test_mask, protransfer, placeholders)
            if FLAGS.feature == 'bow' or FLAGS.feature == 'embedding':
                feed_dict.update({placeholders['features']: features})
            # feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            cost_val = []
            maxValAuc = 0.0
            minValLoss = 10.0
            maxValAuc_test = 0.0
            minValLoss_test = 0.0
            # f = open(base + '/pre.txt', 'w')
            # Train model
            for epoch in range(FLAGS.epochs):
                feed_dict.update({placeholders['is_training']: True})
                t = time.time()
                # step, summaries, loss, accuracy, loss_val, accuracy_val, accuracy_test, _ = sess.run([
                #     model.global_step,
                #     model.summary_op,
                #     model.loss,
                #     model.accuracy,
                #     model.loss_val,
                #     model.accuracy_val,
                #     model.accuracy_test,
                #     model.train_op], feed_dict)
                step, summaries, loss, accuracy, _ = sess.run([
                    model.global_step,
                    model.summary_op,
                    model.loss,
                    model.accuracy,
                    model.train_op], feed_dict)

                step, summaries, summary_writer.add_summary(summaries, global_step=step)
                feed_dict.update({placeholders['is_training']: False})
                loss_val, accuracy_val, accuracy_test = evaluate(sess, model, feed_dict)
                cost_val.append(loss_val)
                if accuracy_val >= maxValAuc:
                    maxValAuc = accuracy_val
                    maxValAuc_test = accuracy_test
                if loss_val <= minValLoss:
                    minValLoss = loss_val
                    minValLoss_test = accuracy_test

                # Print results
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss),
                      "train_acc=", "{:.5f}".format(accuracy), "val_loss=", "{:.5f}".format(loss_val),
                      "val_acc=", "{:.5f}".format(accuracy_val), "time=", "{:.5f}".format(time.time() - t))

                if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                    # lastepoch_test = accuracy_test
                    print("Early stopping...")
                    break

            print("Optimization Finished!")

            # Testing
            # test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
            fres.write("  maxValAuc_accuracy = " + "{:.5f}".format(maxValAuc_test) + "  minValLoss_accuracy = " + "{:.5f}".format(minValLoss_test) + "\n")
            print("Test set results:", "maxValAuc_accuracy=", "{:.5f}".format(maxValAuc_test),
                  "minValLoss_accuracy=", "{:.5f}".format(minValLoss_test))
            maxValAuc_test_allfold += maxValAuc_test
            minValLoss_test_allfold += minValLoss_test

            shutil.rmtree(tflog_dir)

    maxValAuc_test_allfold = maxValAuc_test_allfold/10
    minValLoss_test_allfold = minValLoss_test_allfold/10
    fres.write("allfold_meanResult:  maxValAuc_accuracy = " + "{:.5f}".format(maxValAuc_test_allfold) +
               "  minValLoss_accuracy = " + "{:.5f}".format(minValLoss_test_allfold) + "\n")
    fres.write("\n")
    print("allfold mean results:", "maxValAuc_accuracy=", "{:.5f}".format(maxValAuc_test_allfold), "minValLoss_accuracy=", "{:.5f}".format(minValLoss_test_allfold))



if __name__ == '__main__':
    tf.app.run()
    # main()
