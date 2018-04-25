import tensorflow as tf
import tensorflow.contrib as tc
import os
from collections import namedtuple

BatchedInput = namedtuple('BatchedInput',
                          ['iterator', 'texts', 'text_lens', 'labels'])


def build_dataset(config, mode, char_vocab, label_vocab):
    char_table = tc.lookup.index_table_from_tensor(char_vocab.id2token,
                                                   default_value=char_vocab.token2id[char_vocab.unk_token])
    label_table = tc.lookup.index_table_from_tensor(label_vocab.id2token)

    text_set = tf.data.TextLineDataset(os.path.join(config.data_dir, mode, 'texts.txt'))
    label_set = tf.data.TextLineDataset(os.path.join(config.data_dir, mode, 'labels.txt'))

    dataset = tf.data.Dataset.zip((text_set, label_set))
    dataset = dataset.map(
        lambda text, label: (tf.string_split([text]).values, label),
        num_parallel_calls=config.num_parallel_calls
    )
    dataset = dataset.filter(lambda text, label: tf.size(text) < 1000)
    dataset = dataset.map(
        lambda text, label: (char_table.lookup(text), label_table.lookup(label)),
        num_parallel_calls=config.num_parallel_calls
    )

    # Use a dict instead of a tuple for each element in dataset, because the new bucket_by_sequence_length() API
    # has some issues for nested Dataset element: https://github.com/tensorflow/tensorflow/issues/17932. Hope this could
    # be fixed soon, cause this is freakin dumb
    dataset = dataset.map(lambda text, label: {'text': text, 'text_len': tf.size(text), 'label': label})
    padded_shapes = {
        'text': tf.TensorShape([None]),
        'text_len': tf.TensorShape([]),
        'label': tf.TensorShape([])
    }
    padding_values = {
        'text': char_table.lookup(tf.constant(char_vocab.pad_token, dtype=tf.string)),
        'text_len': tf.constant(0, dtype=tf.int32),
        'label': tf.constant(0, dtype=tf.int64)
    }
    dataset = dataset.apply(
        tc.data.bucket_by_sequence_length(element_length_func=lambda d: tf.size(d['text']),
                                          bucket_boundaries=[10, 30, 60, 100, 200],
                                          bucket_batch_sizes=[config.batch_size] * 5 + [config.batch_size // 5],
                                          padded_shapes=padded_shapes,
                                          padding_values=padding_values)
    )
    return dataset


def build_predict_dataset(text, char_vocab):
    char_table = tc.lookup.index_table_from_tensor(char_vocab.id2token,
                                                   default_value=char_vocab.token2id[char_vocab.unk_token])
    text_set = tf.data.Dataset.from_tensor_slices([text])
    label_set = tf.data.Dataset.from_tensor_slices([tf.constant(0, dtype=tf.int64)])

    dataset = tf.data.Dataset.zip((text_set, label_set))
    dataset = dataset.map(
        lambda text, label: (tf.string_split([text]).values, label),
    )
    dataset = dataset.map(
        lambda text, label: (char_table.lookup(text), label),
    )
    dataset = dataset.map(lambda text, label: (text, tf.size(text), label))
    dataset = dataset.batch(1)
    dataset = dataset.map(
        lambda text, text_len, label: {'text': text, 'text_len': text_len, 'label': label}
    )

    return dataset


def build_inputs(output_types, output_shapes):
    # build an iterator with specific output types and shapes that can switch the underline dataset
    iterator = tf.data.Iterator.from_structure(output_types, output_shapes)
    d = iterator.get_next()
    texts = d['text']
    text_lens = d['text_len']
    labels = d['label']
    return BatchedInput(iterator, texts, text_lens, labels)

