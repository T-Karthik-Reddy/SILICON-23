import tensorflow as tf
import numpy as np
import os

model_path = "/Users/theepireddykarthikreddy/Desktop/SILICON/silicon-22/int8_model.tflite"

if not os.path.exists(model_path):
    print(f"File not found: {model_path}")
    exit()

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

details = interpreter.get_tensor_details()
print(f"--- Tensor Analysis for {os.path.basename(model_path)} ---")

peak_mem = 0
for i, tensor in enumerate(details):
    shape = tensor['shape']
    dtype = tensor['dtype']
    name = tensor['name']
    

    size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
    




    if size_bytes > 5000:
        print(f"ID: {i:<3} | Shape: {str(shape):<15} | Dtype: {str(dtype):<15} | Size: {size_bytes/1024:>8.2f} KB | Name: {name}")




