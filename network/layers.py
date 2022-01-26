import tensorflow as tf
import numpy as np


"""
Build commonly-used neural network layers for
SIPOMDPLite-net.
"""


def conv_layer(
        input_tensor,
        kernel_size,
        num_filter,
        name,
        w_mean=0.0,
        w_std=None,
        add_bias=True,
        strides=(1, 1, 1, 1),
        padding='SAME'):
    """
    Create variables and operator for a convolutional layer
    :param input_tensor: input tensor
    :param kernel_size: size of kernel
    :param num_filter: number of convolutional filters
    :param name: variable name for convolutional kernel and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights.
     Use 1/sqrt(input_param_count) if None.
    :param add_bias: add bias if True
    :param strides: convolutional strides, match TF
    :param padding: padding, match TF
    :return: output tensor
    """
    dtype = tf.float32

    input_size = int(input_tensor.get_shape()[3])
    if w_std is None:
        w_std = 1.0 / np.sqrt(float(input_size * kernel_size * kernel_size))

    kernel = tf.compat.v1.get_variable(
        name='w_' + name,
        shape=(kernel_size, kernel_size, input_size, num_filter),
        initializer=tf.compat.v1.truncated_normal_initializer(
            mean=w_mean, stddev=w_std, dtype=dtype),
        dtype=dtype)
    output_tensor = tf.nn.conv2d(
        input_tensor, kernel, strides=strides, padding=padding)

    if add_bias:
        biases = tf.compat.v1.get_variable(
            name='b_' + name,
            shape=[num_filter],
            initializer=tf.constant_initializer(0.0))
        output_tensor = tf.nn.bias_add(output_tensor, biases)
    return output_tensor


def conv_layers(
        tensor,
        conv_params,
        names,
        **kwargs):
    """
    Build convolution layers from a list of descriptions.
    Each descriptor is a list: [kernel, hidden filters, activation]
    :param tensor:
    :param conv_params:
    :param names:
    """
    for layer_i in range(conv_params.shape[0]):
        kernel_size = int(conv_params[layer_i][0])
        output_size = int(conv_params[layer_i][1])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names + '_%d' % layer_i
        tensor = conv_layer(
            tensor,
            kernel_size=kernel_size,
            num_filter=output_size,
            name=name,
            **kwargs)
        tensor = activation(
            tensor,
            activation_func=conv_params[layer_i][2])

    return tensor


def linear_layer(
        input_tensor,
        output_size,
        name,
        w_mean=0.0,
        w_std=None):
    """
    Create variables and operator for a linear layer
    :param input_tensor: input tensor
    :param output_size: output size, number of hidden units
    :param name: variable name for linear weights and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for
    kernel weights. Use 1/sqrt(input_param_count) if None.
    :return: output tensor
    """
    assert input_tensor.get_shape().ndims == 2, \
        "The input to a fully-connected layer has to be " \
        "2-dimensional (including the dimension of batch size)"
    if w_std is None:
        w_std = 1.0 / np.sqrt(float(input_tensor.get_shape().as_list()[1]))

    weights = tf.compat.v1.get_variable(
        name='w_' + name,
        shape=(input_tensor.get_shape()[1], output_size),
        initializer=tf.compat.v1.truncated_normal_initializer(
            mean=w_mean, stddev=w_std, dtype=tf.float32),
        dtype=tf.float32)

    bias = tf.compat.v1.get_variable(
        name="b_" + name,
        shape=[output_size],
        initializer=tf.constant_initializer(0.0))

    output_tensor = tf.matmul(input_tensor, weights) + bias

    return output_tensor


def fc_layers(
        tensor,
        fc_params,
        names,
        **kwargs):
    """
    Build convolution layers from a list of descriptions.
    Each descriptor is a list: [size, _, activation]
    :param tensor:
    :param fc_params:
    :param names:
    """
    for layer_i in range(fc_params.shape[0]):
        size = int(fc_params[layer_i][0])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names + '_%d' % layer_i
        tensor = linear_layer(tensor, size, name, **kwargs)
        tensor = activation(tensor, fc_params[layer_i][-1])

    return tensor


def activation(
        tensor,
        activation_func):
    """
    Apply activation function to tensor
    :param tensor: input tensor
    :param activation_func: string that defines activation [lin, relu, tanh, sig]
    :return: output tensor
    """
    if activation_func.lower() in ['l', 'lin', 'linear'] or not activation_func:
        pass
    elif activation_func.lower() in ['r', 'relu']:
        tensor = tf.nn.relu(tensor)
    elif activation_func.lower() in ['t', 'tanh']:
        tensor = tf.nn.tanh(tensor)
    elif activation_func.lower() in ['s', 'sig', 'sigmoid']:
        tensor = tf.nn.sigmoid(tensor)
    elif activation_func.lower() in ['sm', 'smax', 'softmax']:
        tensor = tf.nn.softmax(tensor, dim=-1)
    else:
        raise NotImplementedError

    return tensor


def conv4d(
        input,
        filters,
        kernel_size,
        strides=(1, 1, 1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
    """
    Performs a 4D convolution of the ``(t, z, y, x)`` dimensions of a tensor
    with shape ``(b, c, l, d, h, w)`` with ``k`` filters. The output tensor
    will be of shape ``(b, k, l', d', h', w')``. ``(l', d', h', w')`` will be
    smaller than ``(l, d, h, w)`` if a ``valid`` padding was chosen.

    This operator realizes a 4D convolution by performing several 3D
    convolutions. The following example demonstrates how this works for a 2D
    convolution as a sequence of 1D convolutions::

        I.shape == (h, w)
        k.shape == (U, V) and U%2 = V%2 = 1

        # we assume kernel is indexed as follows:
        u in [-U/2,...,U/2]
        v in [-V/2,...,V/2]

        (k*I)[i,j] = Σ_u Σ_v k[u,v] I[i+u,j+v]
                   = Σ_u (k[u]*I[i+u])[j]
        (k*I)[i]   = Σ_u k[u]*I[i+u]
        (k*I)      = Σ_u k[u]*I_u, with I_u[i] = I[i+u] shifted I by u

        Example:

            I = [
                [0,0,0],
                [1,1,1],
                [1,1,0],
                [1,0,0],
                [0,0,1]
            ]

            k = [
                [1,1,1],
                [1,2,1],
                [1,1,3]
            ]

            # convolve every row in I with every row in k, comments show output
            # row the convolution contributes to
            (I*k[0]) = [
                [0,0,0], # I[0] with k[0] ⇒ (k*I)[ 1] ✔
                [2,3,2], # I[1] with k[0] ⇒ (k*I)[ 2] ✔
                [2,2,1], # I[2] with k[0] ⇒ (k*I)[ 3] ✔
                [1,1,0], # I[3] with k[0] ⇒ (k*I)[ 4] ✔
                [0,1,1]  # I[4] with k[0] ⇒ (k*I)[ 5]
            ]
            (I*k[1]) = [
                [0,0,0], # I[0] with k[1] ⇒ (k*I)[ 0] ✔
                [3,4,3], # I[1] with k[1] ⇒ (k*I)[ 1] ✔
                [3,3,1], # I[2] with k[1] ⇒ (k*I)[ 2] ✔
                [2,1,0], # I[3] with k[1] ⇒ (k*I)[ 3] ✔
                [0,1,2]  # I[4] with k[1] ⇒ (k*I)[ 4] ✔
            ]
            (I*k[2]) = [
                [0,0,0], # I[0] with k[2] ⇒ (k*I)[-1]
                [4,5,2], # I[1] with k[2] ⇒ (k*I)[ 0] ✔
                [4,2,1], # I[2] with k[2] ⇒ (k*I)[ 1] ✔
                [1,1,0], # I[3] with k[2] ⇒ (k*I)[ 2] ✔
                [0,3,1]  # I[4] with k[2] ⇒ (k*I)[ 3] ✔
            ]

            # the sum of all valid output rows gives k*I (here shown for row 2)
            (k*I)[2] = (
                [2,3,2] +
                [3,3,1] +
                [1,1,0] +
            ) = [6,7,3]
    """

    # check arguments
    assert len(input.get_shape().as_list()) == 6, (
        "Tensor of shape (b, c, l, d, h, w) expected")
    assert len(kernel_size) == 4, "4D kernel size expected"
    assert strides == (1, 1, 1, 1), (
        "Strides other than 1 not yet implemented")
    assert data_format == 'channels_first', (
        "Data format other than 'channels_first' not yet implemented")
    assert dilation_rate == (1, 1, 1, 1), (
        "Dilation rate other than 1 not yet implemented")

    if not name:
        name = 'conv4d'

    # input, kernel, and output sizes
    (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.get_shape().as_list())
    (l_k, d_k, h_k, w_k) = kernel_size

    # output size for 'valid' convolution
    if padding == 'valid':
        (l_o, d_o, h_o, w_o) = (
            l_i - l_k + 1,
            d_i - d_k + 1,
            h_i - h_k + 1,
            w_i - w_k + 1
        )
    else:
        (l_o, d_o, h_o, w_o) = (l_i, d_i, h_i, w_i)

    # output tensors for each 3D frame
    frame_results = [ None ]*l_o

    # convolve each kernel frame i with each input frame j
    for i in range(l_k):

        # reuse variables of previous 3D convolutions for the same kernel
        # frame (or if the user indicated to have all variables reused)
        reuse_kernel = reuse

        for j in range(l_i):

            # add results to this output frame
            out_frame = j - (i - l_k//2) - (l_i - l_o)//2
            if out_frame < 0 or out_frame >= l_o:
                continue

            # convolve input frame j with kernel frame i
            frame_conv3d = tf.compat.v1.layers.conv3d(
                tf.reshape(input[:,:,j,:], (b, c_i, d_i, h_i, w_i)),
                filters,
                kernel_size=(d_k, h_k, w_k),
                padding=padding,
                data_format='channels_first',
                activation=None,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                trainable=trainable,
                name=name + '_3dchan%d'%i,
                reuse=reuse_kernel)

            # subsequent frame convolutions should use the same kernel
            reuse_kernel = True

            if frame_results[out_frame] is None:
                frame_results[out_frame] = frame_conv3d
            else:
                frame_results[out_frame] += frame_conv3d

    output = tf.stack(frame_results, axis=2)

    if activation:
        output = activation(output)

    return output


def conv4d_layer(
        input_tensor,
        kernel_size,
        num_filter,
        name,
        w_mean=0.0,
        w_std=None,
        use_bias=True,
        strides=(1, 1, 1, 1),
        padding='SAME',
        data_format='channels_first'):
    """
    Create variables and operator for a convolutional layer
    :param input_tensor: input tensor
    :param kernel_size: size of kernel
    :param num_filter: number of convolutional filters
    :param name: variable name for convolutional kernel and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights.
     Use 1/sqrt(input_param_count) if None.
    :param use_bias: add bias if True
    :param strides: convolutional strides, match TF
    :param padding: padding, match TF
    :param data_format
    :return: output tensor
    """
    input_size = input_tensor.get_shape().as_list()[1]
    if w_std is None:
        w_std = 1.0 / np.sqrt(float(input_size * (kernel_size ** 4)))

    bias_initializer = tf.compat.v1.get_variable(
        name='b_' + name,
        shape=[num_filter],
        initializer=tf.constant_initializer(0.0)) if use_bias else None

    output_tensor = conv4d(
        input_tensor,
        filters=num_filter,
        kernel_size=kernel_size,
        kernel_initializer=tf.compat.v1.truncated_normal_initializer(
            mean=w_mean,
            stddev=w_std,
            dtype=tf.float32),
        strides=strides,
        padding=padding,
        data_format=data_format,
        use_bias=use_bias,
        bias_initializer=bias_initializer,
        name=name)

    return output_tensor


def conv4d_layers(tensor, conv_params, names, **kwargs):
    """
    Build convolution layers from a list of descriptions.
    Each descriptor is a list: [kernel, hidden filters, activation]
    :param tensor:
    :param conv_params:
    :param names:
    """
    for layer_i in range(conv_params.shape[0]):
        kernel_size = int(conv_params[layer_i][0])
        output_size = int(conv_params[layer_i][1])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names + '_%d' % layer_i
        tensor = conv4d_layer(
            tensor,
            kernel_size=kernel_size,
            num_filter=output_size,
            name=name,
            **kwargs)
        tensor = activation(
            tensor,
            activation_func=conv_params[layer_i][2])

    return tensor


