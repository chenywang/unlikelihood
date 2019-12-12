# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import tensorflow as tf


def sequence_unlikelihood_loss(logits,
                               targets,
                               weights,
                               dtype=tf.float32,
                               special_token_index=list(),
                               average_across_timesteps=True,
                               average_across_batch=True,
                               name=None):
    """
    an implement of `NEURAL TEXT DEGENERATION WITH UNLIKELIHOOD TRAINING` by tf
    paper: https://arxiv.org/abs/1908.04319?context=stat
    logits: A Tensor of shape
      `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
      int. The target represents the true class at each timestep.
    weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
      float. `weights` constitutes the weighting of each prediction in the
      sequence. When using `weights` as masking, set all valid timesteps to 1
      and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
    special_token_index: the special token index which won't be punished if
      they occur repeatedly such as UNK, SOS, EOS.
    average_across_timesteps: If set, sum the cost across the sequence
      dimension and divide the cost by the total label weight across timesteps.
    average_across_batch: If set, sum the cost across the batch dimension and
      divide the returned cost by the batch size.
    name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
        A float Tensor of rank 0, 1, or 2 depending on the
        `average_across_timesteps` and `average_across_batch` arguments. By default,
        it has rank 0 (scalar) and is the weighted average cross-entropy
        (log-perplexity) per symbol.

    Raises:
        ValueError: logits does not have 3 dimensions or targets does not have 2
                    dimensions or weights does not have 2 dimensions.
    """
    if len(logits.get_shape()) != 3:
        raise ValueError("Logits must be a "
                         "[batch_size x sequence_length x logits] tensor")
    if len(targets.get_shape()) != 2:
        raise ValueError("Targets must be a [batch_size x sequence_length] "
                         "tensor")
    if len(weights.get_shape()) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] "
                         "tensor")
    with tf.name_scope(name, "candidate_penalty_sequence_loss", [logits, targets, weights]):
        sequence_length = tf.shape(targets)[1]
        batch_size = tf.shape(targets)[0]
        num_decoder_symbols = tf.shape(logits)[2]

        # get the candidate word index that should be the loss ([batch_size, sequence_len, num_decoder_symbols])
        mask = tf.sequence_mask(tf.range(sequence_length), sequence_length, dtype=tf.int32)
        candidate_matrix = tf.reshape(tf.tile(targets, [1, sequence_length]),
                                      (batch_size, sequence_length, sequence_length))
        candidate_matrix = candidate_matrix + 1
        candidate_matrix *= tf.cast(mask, dtype=candidate_matrix.dtype)
        candidate_matrix = tf.reduce_sum(
            tf.one_hot(tf.cast(candidate_matrix, dtype=tf.int32), num_decoder_symbols + 1)[:, :, :, 1:], axis=2)
        candidate_matrix = tf.cast(tf.cast(candidate_matrix, dtype=tf.bool), dtype=tf.float32)  # 只要0或者1

        # mask the candidate word index that should be masked because it's the word that we want to predict
        current_word_mask = tf.cast(tf.math.logical_not(
            tf.cast(tf.one_hot(tf.cast(targets, dtype=tf.int32), num_decoder_symbols), dtype=tf.bool)), dtype=tf.int32)

        # mask the candidate word index that out of the weights
        word_len_mask = tf.transpose(tf.tile(tf.expand_dims(weights, 1), [1, num_decoder_symbols, 1]), [0, 2, 1])

        # TODO mask the candidate word index that belong to the special word

        # cast to float32
        candidate_matrix = tf.cast(candidate_matrix, tf.float32)
        current_word_mask = tf.cast(current_word_mask, tf.float32)
        word_len_mask = tf.cast(word_len_mask, tf.float32)

        # get the final candidate word index
        candidate_matrix_after_mask = candidate_matrix * current_word_mask * word_len_mask

        # get the probability based on logits
        probs = tf.nn.softmax(logits, axis=-1)
        unlikelihood_logits = tf.log(1 - probs)

        # get the loss matrix
        loss_matrix = candidate_matrix_after_mask * unlikelihood_logits

        if average_across_timesteps and average_across_batch:
            total_step = 1e-12 + tf.cast(tf.reduce_sum(weights), dtype=tf.float32)
            loss = -tf.reduce_sum(loss_matrix) / total_step
        elif average_across_timesteps:
            total_step = 1e-12 + tf.cast(tf.reduce_sum(weights, axis=1), dtype=tf.float32)
            loss = -tf.reduce_sum(loss_matrix, axis=[1, 2]) / total_step
        elif average_across_batch:
            total_step = 1e-12 + tf.cast(tf.reduce_sum(weights, axis=0), dtype=tf.float32)
            loss = -tf.reduce_sum(loss_matrix, axis=[0, 2]) / total_step
        else:
            total_step = 1e-12 + tf.cast(weights, dtype=tf.float32)
            loss = -tf.reduce_sum(loss_matrix, axis=2) / total_step
    return loss
