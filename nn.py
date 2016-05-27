import theano
import numpy
import theano.tensor as T

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and numpy.isnan(output[0]).any()):
            print('*** NaN detected ***')
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            break

class Layer(object):
    def collect_params(self):
        params = []
        for dependency in self.dependencies:
            for param in dependency.collect_params():
                params.append(param)
        for param in self.trainable_params:
            params.append(param)
        return params

    def collect_inputs(self):
        inputs = []
        for dependency in self.dependencies:
            for input in dependency.collect_inputs():
                inputs.append(input)
        for input in self.raw_inputs:
            inputs.append(input)
        return inputs

class ConcatLayer(Layer):
    def __init__(self, input1, input2):
        self.input1 = input2
        self.input2 = input2

        # The output lengths should really be the same
        self.output_length = input1.output_length + input2.output_length

        self.trainable_params = []
        self.dependencies = [input1, input2]
        self.raw_inputs = []

        self.output = T.concat([input1, input2])

class InputLayer(Layer):
    def __init__(self, input, length):
        self.output_length = length

        self.trainable_params = []
        self.dependencies = []
        self.raw_inputs = [input]

        self.output = input

class MultiplyLayer(Layer):
    def __init__(self, input1, input2):
        self.input1 = input2
        self.input2 = input2

        # The output lengths should really be the same
        self.output_length = input1.output_length

        self.trainable_params = []
        self.dependencies = [input1, input2]
        self.raw_inputs = []

        self.output = input1 * input2

class AddLayer(Layer):
    def __init__(self, input1, input2):
        self.input1 = input2
        self.input2 = input2

        # The output lengths should really be the same
        self.output_length = input1.output_length

        self.trainable_params = []
        self.dependencies = [input1, input2]
        self.raw_inputs = []

        self.output = input1 + input2


class TransformLayer(Layer):
    def __init__(self,
                rng, # Initialization RNG
                input, # Input layer
                output_length, # Output dimensionality
                activation=T.tanh, # Activation function
                W = None, # Initial edges, if desired
                B = None # Initial bias, if desired
            ):
        # Remember inputs
        self.input = input
        self.output_length = output_length

        # Initialize W randomly.
        if W is None:
            # W is an (input length) x (output length) matrix
            if activation is T.nnet.softmax:
                W_values = numpy.zeros((input.output_length, output_length), dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (input.output_length + output_length)),
                        high=numpy.sqrt(6. / (input.output_length + output_length)),
                        size=(input.output_length, output_length)
                    ),
                    dtype=theano.config.floatX
                )

            # Apparently, when using a sigmoid activation function,
            # you want to use initlialization values four times greater in
            # magnitude.
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize B as zeros.
        if B is None:
            # b is a vectory of size (output length)
            B_values = numpy.zeros((output_length,), dtype=theano.config.floatX)
            B = theano.shared(value=B_values, name='b', borrow=True)

        self.B = B
        self.W = W

        self.trainable_params = [
            W,
            B
        ]
        self.dependencies = [
            input
        ]
        self.raw_inputs = []

        lin_output = T.dot(input.output, self.W) + self.B
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

class CrossEntropyLayer(Layer):
    def __init__(self, input, compare):
        self.trainable_params = []
        self.dependencies = [
            input
        ]
        self.raw_inputs = [
            compare
        ]

        self.output = T.nnet.categorical_crossentropy(input.output, compare).mean()

class Collector(Layer):
    def __init__(self, collection, cost, learning_rate = 0.01):
        # Remember inputs and outputs
        self.collection = collection
        self.inputs = sum([x.collect_inputs() for x in collection], [])
        self.params = sum([x.collect_params() for x in collection], [])

        self.output = cost([x.output for x in collection])

        self.gparams = [T.grad(self.output, param) for param in self.params]

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, self.gparams)
        ]

        self.train = theano.function(
            inputs=self.inputs,
            outputs=self.output,
            updates=updates
        )

class RecurrentCollector(Layer):
    def __init__(self, collection, recurrence, learning_rate = 0.01):
        # Remember inputs and outputs
        self.collection = collection
        self.inputs = collection.collect_inputs()
        self.params = collection.collect_params()

        self.output = collection.output

        self.gparams = [T.grad(self.output, param) for param in self.params]


        # Recurrence is a tuple linking an output layer to an input layer
        recurrent_grad = theano.scan(
            lambda i: T.grad(recurrence[1].output[i], recurrence[0].output),
            T.arange(recurrence[1].output.shape[0])
        )

        grad_addend = map(
            lambda param: theano.scan(
                lambda i: T.grad(recurrence[1].output[i], param),
                T.arange(recurrence[1].output.shape[0])
            ),
            self.params
        )

        # old_recurrent_grad and new_recurrent_grad are supposed to be arrays of
        # tensors, each with recurrent_grad[i] = gradient(recurrence_layer, params[i])
        #
        # Frank -- can you fix old_recurrent_grad so that it is actually like this
        old_recurrent_grad = [T.vector()]
        new_recurrent_grad = map(
            lambda el, i: T.dot(recurrent_grad, el) + grad_addend[i],
            old_recurrent_grad
        )

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, new_recurrent_grad)
        ]

        self.inputs.append(old_recurrent_grad)

        # Frank -- can you add old_recurrent_grad as another set of inputs to the train functino
        self.train = theano.function(
            inputs=self.inputs,
            outputs=(self.output, new_recurrent_grad),
            updates=updates
        )

MINIBATCH_LENGTH = 1000

# Build the neural network
if __name__ == '__main__':
    # Characters:
    x = T.matrix('x')

    # Hidden layer:
    h = T.matrix('h')

    # Classifications
    y = T.ivector('y')

    # Seeded random number generator
    rng = numpy.random.RandomState(1234)

    # Build the neural network
    char_in = InputLayer(x, 256)
    hidden_in = InputLayer(h, 1500)

    # Compute the new value we might want to place
    # in the GRU. First, we have a layer to determine
    # how much of the historical hidden input we want to use
    use_history_signal = TransformLayer(rng, all_info, 1500, activation = T.nnet.sigmoid)
    used_history = MultiplyLayer(use_history_signal, hidden_in)

    # Then we collect that much of the hidden input
    # with the actual input
    new_info = ConcatLayer(hidden_in, used_history)

    # Then we transform with tanh to get the new values
    new_hidden_value = TransformLayer(rng, new_info, 1500)

    # Now we have the update gate to determine
    # how much of the new hidden value we should actually put in
    all_info = ConcatLayer(char_in, hidden_in)

    update_signal = TransformLayer(rng, all_info, 1500, activation = T.nnet.sigmoid)
    inverse_update_signal = MapLayer(update_signal, lambda x: 1 - x) # Frank -- can you implement MapLayer here so that it does what we want

    filtered_values = MultiplyLayer(inverse_update_signal, hidden_in)
    values_to_insert = MultiplyLayer(update_signal, new_hidden_value)

    hidden_out = AddLayer(filtered_values, values_to_insert)

    # Finally, a softmax layer from the hidden output to a character
    output = TransformLayer(rng, hidden_out, 256, activation = T.nnet.softmax)

    # Our training thing is a recurrent collector
    collector = RecurrentCollector(output, (hidden_in, hidden_out))

    '''
    print(collector.inputs)
    print(collector.params)

    string = open("OANC.txt", "r").read()

    def clamped_ord(char):
      return max(0, min(ord(char), 255))

    def assemble_minibatch(length):
        value = numpy.zeros([length, 256 * 2], dtype=theano.config.floatX)
        output = numpy.zeros(length, dtype=numpy.uint8)
        for i in range(0, length):
            # Get a random sample
            index = rng.randint(len(string) - 2)
            value[i][clamped_ord(string[index])] = 1
            value[i][256 + clamped_ord(string[index + 1])] = 1
            output[i] = clamped_ord(string[index + 2])
        return value, output

    def generate(length, begin = 'th'):
        value = numpy.zeros([1, 256 * 2], dtype=theano.config.floatX)

        string = begin
        cursor = begin
        for i in range(2, length):
            value[0][ord(cursor[0])] = 1
            value[0][256 + ord(cursor[1])] = 1

            # Softmax
            result = forward(value)[0]
            char = chr(numpy.random.choice(len(result), p = result / sum(result)))

            value[0][ord(cursor[0])] = 0
            value[0][256 + ord(cursor[1])] = 0

            string += char
            cursor = cursor[1] + char

        return string

    gradient = theano.function([x, y], collector.gparams[0])

    error = float('inf')
    while True:
       print("'%r'" % generate(80))
       value, output = assemble_minibatch(MINIBATCH_LENGTH)
       error = collector.train(value, output)
       print(error)
    '''
