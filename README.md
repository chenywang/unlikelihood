## Neural Text deGeneration with Unlikelihood Training
tensorflow implementation of the paper:
    [Neural Text Generation with Unlikelihood Training](https://arxiv.org/pdf/1908.04319.pdf)\
    Sean Welleck\*, Ilia Kulikov\*, Stephen Roller, Emily Dinan, Kyunghyun Cho, Jason Weston\
    \*Equal contribution. The order was decided by a coin flip.

## notice
add it to sequence_loss:
```
loss = seq2seq.sequence

loss = seq2seq.sequence_loss(
            logits=self.decoder_logits_train,
            targets=self.decoder_targets_train,
            weights=masks,
            average_across_timesteps=True,
            average_across_batch=True
        ) +
        alpha * sequence_unlikelihood_loss(logits, targets, weights)

```
I set alpha as 1.0

## requirement
tensorflow==1.12.0
python 3.6