# -*- coding:utf-8 -*-
# @Author : Michael-Wang
import numpy as np
import tensorflow as tf

from unlikelihood_loss import sequence_unlikelihood_loss

if __name__ == '__main__':
    """
    calculate unlikelihood manually
    targets = [[0, 1, 1, 4, 0]]
    weights = [[1, 1, 1, 1, 0]]
    logits array([[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]],
          dtype=float32)
    step 0：
    loss = 0

    step 1: should not occur 0
    loss = -log(1 - 0.1)

    step 2：should not occur 0
    loss = -log(1 - 0.1)

    step 3：should not occur 0 and 1
    loss = -log(1 - 0.1) + -log(1 - 0.1)

    step 4：out of length
    loss = 0

    total_loss = 0.5
    total_step = 4

    candidate_loss = -log(1 - 0.1) * 4 / 4 = -log(1 - 0.1) = 0.10536051565782628
    """

    """
    function result:
    0.10536051565782628
    """
    vocab_size = 10
    sequence_len = 5
    batch_size = 1
    dtype = tf.float32
    targets = tf.constant(
        [[0, 1, 1, 4, 0]],
        dtype=dtype
    )
    weights = tf.constant(
        [[1, 1, 1, 1, 0]],
        dtype=dtype
    )
    logits = tf.constant(
        np.ones((batch_size, sequence_len, vocab_size,)) * (1.0 / vocab_size),
        dtype=dtype
    )
    g = sequence_unlikelihood_loss(logits, targets, weights)
    print(tf.Session().run(g))

