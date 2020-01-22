import math
from typing import List

from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda, Add
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.models import Sequential

# noinspection PyPackageRequirements
class ResidualBlock(Layer):

    def __init__(self,
                 dilation_rate,
                 nb_filters,
                 kernel_size,
                 padding,
                 conv_activation=('relu', 'relu'),
                 residual_block_activation=None,
                 dropout_rate=0,
                 kernel_initializer='he_normal',
                 normalization=('weight', 'weight'),
                 last_block=True,
                 name='ResidualBlock',
                 **kwargs):

        # type: (int, int, int, str, str, str, float, str, str, bool, bool, dict) -> None
        """Defines the residual block for the WaveNet TCN

        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            conv_activation: list of activations to be used for the two dilated convolutions inside the residual blcok
            residual_block_activation: final activation on the output of the residual block, i.e., o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            normalization: Whether to use batch, layer, weight or no normalization after the dilated convolutions in the residual block. A list of two elements, which can take values in ['batch', 'layer', 'weight', None]. 
            kwargs: Any initializers for Layer class.
        """

        assert (type(normalization) is tuple and
                len(normalization) == 2 and
                all(n in ['batch', 'layer', 'weight', None] for n in normalization)), "normalization must be a tuple with two elements, each one with a value in ['batch', 'layer', 'weight', None]"
        assert (type(conv_activation) is tuple and
                len(conv_activation) == 2), "conv_activation must be a tuple with two elements"

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv_activation = conv_activation
        self.residual_block_activation = residual_block_activation
        self.dropout_rate = dropout_rate
        self.normalization = normalization
        self.kernel_initializer = kernel_initializer
        self.last_block = last_block

        super(ResidualBlock, self).__init__(name=name, **kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.residual_layers.append(layer)
        self.residual_layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.residual_layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):
        #print("RB: input_shape", input_shape)

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.residual_layers = list()
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'rb_conv1D_{}'.format(k)
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    if self.normalization[k] == 'weight':
                        self._add_and_activate_layer(WeightNormalization(Conv1D(filters=self.nb_filters,
                                                            kernel_size=self.kernel_size,
                                                            dilation_rate=self.dilation_rate,
                                                            padding=self.padding,
                                                            name=name,
                                                            kernel_initializer=self.kernel_initializer)))
                    else:
                        self._add_and_activate_layer(Conv1D(filters=self.nb_filters,
                                                            kernel_size=self.kernel_size,
                                                            dilation_rate=self.dilation_rate,
                                                            padding=self.padding,
                                                            name=name,
                                                            kernel_initializer=self.kernel_initializer))

                if self.normalization[k] == 'batch':
                    self._add_and_activate_layer(BatchNormalization())
                elif self.normalization[k] == 'layer':
                    self._add_and_activate_layer(LayerNormalization())

                self._add_and_activate_layer(Activation(self.conv_activation[k]))
                self._add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))

            if not self.last_block:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'conv1D_{}'.format(k + 1)
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)

            else:
                self.shape_match_conv = Lambda(lambda x: x, name='identity')

            self.shape_match_conv.build(input_shape)
            self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = Activation(self.residual_block_activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force keras to add the layers in the list to self._layers
            for layer in self.residual_layers:
                self.__setattr__(layer.name, layer)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """

        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        for layer in self.residual_layers:
            if isinstance(layer, SpatialDropout1D):
                x = layer(x, training=training)
            else:
                x = layer(x)
            #print("  Residual Block - Shape after {}: {}".format(layer.name, x.shape))

        x2 = self.shape_match_conv(inputs)
        res_x = Add()([x2, x])
        return [self.final_activation(res_x), x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config['dilation_rate'] = self.dilation_rate
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['padding'] = self.padding
        config['conv_activation'] = self.conv_activation,
        config['residual_block_activation'] = self.residual_block_activation,
        config['dropout_rate'] = self.dropout_rate
        config['normalization'] = self.normalization
        config['kernel_initializer'] = self.kernel_initializer
        config['last_block'] = self.last_block

        return config


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class TCN(Layer):
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            input_timesteps: Length of history this TCN should be able to process. If specified, either kernel size or both dilations and nb_residualblocks should not be specified and will be automatically chosen to achieve a receptive field size greater than input_timesteps  
            kernel_size: The size of the kernel to use in each convolutional layer.
            nb_residualblocks: If specified, each stack will consist of nb_residualblocks residual blocks and their dilations will increase exponentially, i.e., d_i = 2^(i-1), for i=1, .., nb_residualblocks
            dilations: The list of dilations for the residual blocks inside one stack. Example is: [1, 2, 4, 8, 16, 32, 64]. Ignored if nb_residualblocks is specified.
            nb_stacks : The number of stacks of residual blocks to use.
            nb_filters: Number of filters to use in the convolutional layers. If a scalar, use same number of filters in all conv layers. If a list, ensure that len(nb_filters)=nb_residualblocks (+ 1, if use_input_conv=True) 
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Either None, for no skip connections, 'wavenet', for using skip connections ala Wavenet, or 'rb_output', for using skip connections on the residual blocks' outputs
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            conv_activation: list of activations to be used for the two dilated convolutions inside the residual blcok
            residual_block_activation: final activation on the output of the residual block, i.e., o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            normalization: Whether to use batch, layer, weight or no normalization after the dilated convolutions in the residual block. A list of two elements, which can take values in ['batch', 'layer', 'weight', None].
            use_input_conv: Whether to pass the input data through a conv1D layer with a kernel size of 1
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.

        Returns:
            A TCN layer.
        """
    def __init__(self,
                 input_timesteps=None,
                 kernel_size=2,
                 dilations=None,
                 nb_residualblocks=4,
                 nb_stacks=1,
                 nb_filters=64,
                 padding='causal',
                 use_skip_connections=None,
                 dropout_rate=0.0,
                 return_sequences=False,
                 conv_activation=('relu', 'relu'),
                 residual_block_activation=None,
                 kernel_initializer='he_normal',
                 normalization=('layer', 'layer'),
                 use_input_conv=False,
                 name='TCN',
                 **kwargs):
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.residual_block_activation = residual_block_activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.normalization = normalization
        self.use_input_conv = use_input_conv

        assert use_skip_connections in [None, 'wavenet', 'rb_output'], "Valid values for use_skip_connections are None, 'wavenet' or 'rb_output'"

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

        # initialize parent class
        super(TCN, self).__init__(name=name, **kwargs)

        if nb_residualblocks:
            print("WARNING: both nb_residualblocks and dilations have been specified. The latter will be ignored")
            self.dilations = [2**i for i in range(nb_residualblocks)]
        else:
            if not input_timesteps:
                assert dilations and type(dilations) in [list, tuple], 'A list of dilations must be specified when nb_residualblocks is None'
            self.dilations = dilations


        if input_timesteps:
            assert kernel_size or dilations or nb_residualblocks, 'When input_timesteps is specified, either kernel_size or dilations/nb_residualblocks should also be specified'
            assert (kernel_size and not dilations and not nb_residualblocks) or (not kernel_size and (
                dilations or nb_residualblocks)), 'When input_timesteps is specified, kernel_size and dilations/nb_residualblocks cannot be both specified'

            if kernel_size:
                # Need to choose dilations. Using exponentially increasing dilations, we need to choose nb_residualblocks,
                # such that the receptive field (RF) is greater than input_timesteps
                # For exponentially increasing dilations, we have RF = B * (1 + 2(k -1)*L) ,
                # with B = nb_stacks, k = kernel_size and L = nb_residualblocks
                # (See https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7)
                # We set 5 as the maximum value for L
                B = 0; L = 6; T = input_timesteps; k = kernel_size
                while L > 5:
                    B += 1
                    L = math.log(1 + (T/B - 1)/(2 * (k -1)), 2)
                    L = math.ceil(L)
                self.dilations = [2 ** i for i in range(L)]
                self.nb_stacks = B
                nb_residualblocks = L
            else:
                # Need to choose kernel_size. We set 10 as maximum kernel size
                D = sum(self.dilations)
                B = 0; k = 11; T = input_timesteps
                while k > 10:
                    B += 1
                    k = 1 + (T/B - 1)/(2 * D)
                    k = math.ceil(k)
                self.kernel_size = k
                self.nb_stacks = B
                nb_residualblocks = len(self.dilations)

            print("TCN parameters:")
            print(" - nb_stacks: {}".format(self.nb_stacks))
            print(" - nb_residual_blocks: {}".format(nb_residualblocks))
            print(" - kernel_size: {}".format(self.kernel_size))
            print(" - dilations: {}".format(list(self.dilations)))
            print("")
            print(" - Nr. of convolutional layers: {}".format(self.nb_stacks * 2 * nb_residualblocks))
            print(" - Receptive field: {}".format(self.get_receptive_field_size()))

        if isinstance(nb_filters, int):
            self.nb_filters = [nb_filters]*nb_residualblocks
        else:
            assert len(nb_filters) == nb_residualblocks + use_input_conv, 'The length ({}) of the nb_filters list does not match the value of nb_residualblocks ({}), plus 1 if use_input_conv is True'.format(len(nb_filters), nb_residualblocks)
            self.nb_filters = nb_filters

    def build(self, input_shape):
        if self.use_input_conv:
            self.main_conv1D = Conv1D(filters=self.nb_filters[0],
                                  kernel_size=1,
                                  padding=self.padding,
                                  kernel_initializer=self.kernel_initializer)
            self.main_conv1D.build(input_shape)

            # member to hold current output shape of the layer for building purposes
            self.build_output_shape = self.main_conv1D.compute_output_shape(input_shape)
        else:
            self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.residual_blocks = list()
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for d, nb_filters in zip(self.dilations, self.nb_filters[1:]):
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=nb_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          conv_activation = self.conv_activation,
                                                          residual_block_activation = self.residual_block_activation,
                                                          dropout_rate=self.dropout_rate,
                                                          normalization=self.normalization,
                                                          kernel_initializer=self.kernel_initializer,
                                                          last_block=len(self.residual_blocks) + 1 == total_num_blocks,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        # Author: @karolbadowski.
        output_slice_index = int(self.build_output_shape.as_list()[1] / 2) if self.padding == 'same' else -1
        self.lambda_layer = Lambda(lambda tt: tt[:, output_slice_index, :])
        self.lambda_ouput_shape = self.lambda_layer.compute_output_shape(self.build_output_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            return self.lambda_ouput_shape
        else:
            return self.build_output_shape

    def call(self, inputs, training=None):
        if self.use_skip_connections == 'wavenet':
            return self._call_skip_connections_wavenet_style(inputs, training=training)
        else:
            return self._call_no_skip_connections_or_rb_output(inputs, training=training)

    def _call_skip_connections_wavenet_style(self, inputs, training=None):
        '''
        Use skip connection ala Wavenet, i.e., inside the residual blocks
        '''
        x = inputs
        if self.use_input_conv:
            x = self.main_conv1D(x)
        skip_connections = list()
        for rb in self.residual_blocks:
            x, skip_out = rb(x, training=training)
            #print("TCN - Shape after Residual Block", x.shape)
            skip_connections.append(skip_out)

        x = Add()(skip_connections)
        if not self.return_sequences:
            x = self.lambda_layer(x)
        return x

    def _call_no_skip_connections_or_rb_output(self, inputs, training=None):
        '''
        No skip connections or skip connections on the output of each residual block and use their sum as output, ala Wavenet
        '''
        x = inputs
        if self.use_input_conv:
            x = self.main_conv1D(x)
        skip_connections = list()
        for rb in self.residual_blocks:
            x, _ = rb(x, training=training)
            #print("TCN - Shape after Residual Block", x.shape)
            skip_connections.append(x)

        if self.use_skip_connections is 'rb_output':
            x = Add()(skip_connections)
        if not self.return_sequences:
            x = self.lambda_layer(x)
        return x


    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config()
        config['return_sequences'] = self.return_sequences
        config['dropout_rate'] = self.dropout_rate
        config['use_skip_connections'] = self.use_skip_connections
        config['nb_stacks'] = self.nb_stacks
        config['kernel_size'] = self.kernel_size
        config['conv_activation'] = self.conv_activation,
        config['residual_block_activation'] = self.residual_block_activation,
        config['padding'] = self.padding
        config['kernel_initializer'] = self.kernel_initializer
        config['normalization'] = self.normalization
        config['use_input_conv'] = self.use_input_conv
        config['dilations'] = self.dilations
        config['nb_filters'] = self.nb_filters

        return config

    def get_receptive_field_size(self):
        """
        See https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7 
        https://distill.pub/2019/computing-receptive-fields/
        and Wavenet paper
        """
        return self.nb_stacks * (1 + 2*(self.kernel_size - 1) * sum(self.dilations))

def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 output_len=1,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=True,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='linear',  # type:str,
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False):
    # type: (...) -> Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.

    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        name: Name of the model. Useful when having multiple TCN.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        opt: Optimizer name.
        lr: Learning rate.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
        use_layer_norm: Whether to use layer normalization in the residual layers or not.
    Returns:
        A compiled keras TCN.
    """

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
                 conv_activation=('relu', 'relu'),
                 residual_block_activation=None,
                 kernel_initializer=kernel_initializer,
                 normalization=('batch', 'batch'), name=name)(input_layer)

    print('tcn.shape=', x.shape)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(lr=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(lr=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), loss='sparse_categorical_crossentropy', metrics=[accuracy])
    else:
        # regression
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')
    print('model.x = {}'.format(input_layer.shape))
    print('model.y = {}'.format(output_layer.shape))
    model.summary()

    return model


def compiled_elu_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 output_len=1,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=True,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='linear',  # type:str,
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False):
    # type: (...) -> Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.

    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        name: Name of the model. Useful when having multiple TCN.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        opt: Optimizer name.
        lr: Learning rate.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
        use_layer_norm: Whether to use layer normalization in the residual layers or not.
    Returns:
        A compiled keras TCN.
    """

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, use_layer_norm,
            name=name)(input_layer)

    print('x.shape=', x.shape)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(lr=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(lr=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), loss='sparse_categorical_crossentropy', metrics=[accuracy])
    else:
        # regression
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')
    print('model.x = {}'.format(input_layer.shape))
    print('model.y = {}'.format(output_layer.shape))
    return model
