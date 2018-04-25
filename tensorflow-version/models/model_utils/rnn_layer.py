"""
This module implements a wrapper for customizing RNNs in Tensorflow
"""
import tensorflow as tf
from functools import partial


def rnn(mode, type, inputs, length, hidden_size, num_layers=1, dropout_keep_prob=1.0):
    if mode == 'uni':
        cell = get_cell(type, hidden_size, num_layers, dropout_keep_prob)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, length, dtype=tf.float32)

        if num_layers == 1:
            if type == 'lstm':
                states = states[1]
            final_state = states
        else:
            if type == 'lstm':
                states = tuple([h for c, h in states])
            final_state = tf.concat(states, 1)

        return outputs, final_state

    elif mode == 'bi':
        cell_fw = get_cell(type, hidden_size, num_layers, dropout_keep_prob)
        cell_bw = get_cell(type, hidden_size, num_layers, dropout_keep_prob)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, length, dtype=tf.float32)
        states_fw, states_bw = states

        if num_layers == 1:
            if type == 'lstm':
                states_fw = states_fw[1]
                states_bw = states_bw[1]
            final_state = tf.concat([states_fw, states_bw], 1)
        else:
            if type == 'lstm':
                states_fw = tuple([h for c, h in states_fw])
                states_bw = tuple([h for c, h in states_bw])
            # alternatively concat forward and backward states
            final_states = []
            for n in range(num_layers):
                final_states.append(states_fw[n])
                final_states.append(states_bw[n])
            final_state = tf.concat(final_states, 1)

        return tf.concat(outputs, 2), final_state

    else:
        raise NotImplementedError('Only "uni" and "bi" modes are currently supported')


def get_cell(type, hidden_size, num_layers=1, dropout_keep_prob=1.0):
    if type == 'rnn':
        cell_fn = partial(tf.nn.rnn_cell.BasicRNNCell, num_units=hidden_size)
    elif type == 'lstm':
        cell_fn = partial(tf.nn.rnn_cell.LSTMCell, num_units=hidden_size)
    elif type == 'gru':
        cell_fn = partial(tf.nn.rnn_cell.GRUCell, num_units=hidden_size)
    else:
        raise NotImplementedError('Unsupported rnn type: {}'.format(type))

    if num_layers == 1:
        return tf.nn.rnn_cell.DropoutWrapper(cell_fn(),
                                             input_keep_prob=dropout_keep_prob,
                                             output_keep_prob=dropout_keep_prob)
    else:
        cells = []
        for n in range(num_layers):
            cell = tf.nn.rnn_cell.DropoutWrapper(cell_fn(),
                                                 input_keep_prob=dropout_keep_prob,
                                                 output_keep_prob=dropout_keep_prob)
            cells.append(cell)
        return tf.nn.rnn_cell.MultiRNNCell(cells)
