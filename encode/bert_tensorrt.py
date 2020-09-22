import data_processing_new as dpn
import tokenization

import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

import ctypes
nvinfer =  ctypes.CDLL("libnvinfer_plugin.so", mode = ctypes.RTLD_GLOBAL)
cm = ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libcommon.so", mode = ctypes.RTLD_GLOBAL) 
pg = ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libbert_plugins.so", mode = ctypes.RTLD_GLOBAL) 

bert_engine = '/workspace/TensorRT/demo/BERT/python/bert_base.engine'
vocab_file = '/workspace/models/fine-tuned/chinese_bert_base/vocab.txt'
batch_size = 1

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

max_query_length = 64

max_seq_length = 128

eval_start_time = time.time()
input_features = dpn.convert_examples_to_features(text, None, tokenizer, max_seq_length)
time.time() - eval_start_time

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

max_batch_size = 1

text = '早安'
eval_start_time = time.time()
input_features = dpn.convert_examples_to_features(text, None, tokenizer, max_seq_length)

with open("./bert_base.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, \
    runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

    print("List engine binding:")
    for binding in engine:
        print(" - {}: {}, Shape {}, {}".format(
            "Input" if engine.binding_is_input(binding) else "Output",
            binding,
            engine.get_binding_shape(binding),
            engine.get_binding_dtype(binding)))

    def binding_nbytes(binding):
        return trt.volume(engine.get_binding_shape(binding)) * engine.get_binding_dtype(binding).itemsize
    
    d_inputs = [cuda.mem_alloc(binding_nbytes(binding)) for binding in engine if engine.binding_is_input(binding)]
    h_output = cuda.pagelocked_empty(tuple(engine.get_binding_shape(3)), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    print("\nRunning Inference...")
    
    cuda.memcpy_htod_async(d_inputs[0], input_features["input_ids"], stream)
    cuda.memcpy_htod_async(d_inputs[1], input_features["segment_ids"], stream)
    cuda.memcpy_htod_async(d_inputs[2], input_features["input_mask"], stream)

    context.execute_async(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    eval_time_elapsed = time.time() - eval_start_time


print(eval_time_elapsed * 1000)

a = h_output.reshape(128, 768)[0, :]