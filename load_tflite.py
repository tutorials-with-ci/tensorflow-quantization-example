import numpy as np
from tensorflow.contrib.lite.python.interpreter import Interpreter
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

model_path = './final.tflite'
with open(model_path, 'rb') as f:
    model_content = f.read()

interpreter = Interpreter(model_content=model_content)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

batch = mnist.train.next_batch(1)

test_input = np.array(batch[0] * 255, dtype=np.uint8)
test_input = test_input.reshape((1, 28, 28, 1))

interpreter.resize_tensor_input(input_details[0]['index'], np.array(test_input.shape, dtype=np.int32))
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("TF-Lite Output: {}".format(output_data))
print("Ground Truth: {}".format(batch[1]))
print("Right? {}".format(np.argmax(output_data) == np.argmax(batch[1])))
