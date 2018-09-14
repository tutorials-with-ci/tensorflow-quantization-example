import numpy as np
from tensorflow.contrib.lite.python.interpreter import Interpreter
from tensorflow.examples.tutorials.mnist import input_data


class TfLiteModel:
    def __init__(self, model_content):
        self.model_content = bytes(model_content)
        self.interpreter = Interpreter(model_content=self.model_content)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        print(input_details)
        self.input_index = input_details[0]['index']
        self.output_index = output_details[0]['index']

        self.input_scale, self.input_zero_point = input_details[0]['quantization']
        self.output_scale, self.output_zero_point = output_details[0]['quantization']

        self.interpreter.allocate_tensors()

    def forward(self, data_in):
        test_input = np.array(data_in / self.input_scale + self.input_zero_point, dtype=np.uint8).reshape(1, -1)
        self.interpreter.set_tensor(self.input_index, test_input)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_index)[0]
        return (np.array(output_data, dtype=np.float32) - self.output_zero_point) * self.output_scale


mnist = input_data.read_data_sets('MNIST-data', one_hot=True)
batch = mnist.train.next_batch(1)
image, label = batch[0], batch[1]

model_path = './final.tflite'
with open(model_path, 'rb') as f:
    model_content = f.read()

model = TfLiteModel(model_content)
predict = model.forward(image)

print("TF-Lite Output: {}".format(predict))
print("Ground Truth: {}".format(label))
print("Right? {}".format(np.argmax(predict) == np.argmax(label)))
