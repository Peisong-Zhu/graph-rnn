import tensorflow as tf


def masked_softmax_cross_entropy(labels, logits, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_loss(loss, mask):
    """Softmax cross-entropy loss with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(res, mask):
    """Accuracy with masking."""
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    mask = tf.cast(mask, dtype=tf.float32)
    res *= mask
    auc = tf.reduce_sum(res)/tf.reduce_sum(mask)
    return tf.reduce_mean(auc)