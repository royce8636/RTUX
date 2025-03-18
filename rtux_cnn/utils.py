import inspect
import builtins
import tensorflow as tf
import os
import numpy as np
import tensorrt as trt
from cuda import cudart
import sys
import subprocess
sys.path.insert(1, "onnx_to_tf/")
import common
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_adb_devices():
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        devices = []
        for line in output[1:]:
            if 'device' in line:
                device_info = line.split('\t')
                devices.append(device_info[0])
        if len(devices) == 0:
            print("No adb devices found")
            exit(1)
        elif len(devices) > 1:
            print(f"Multiple adb devices found, using the first device {devices[0]}\n"
                  f"Provide the device with -D to use an other device")
        return devices[0]
    except FileNotFoundError:
        print("No adb devices found")
        exit(1)
    

class SlidingWindow:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = [None for _ in range(maxsize)]
        self._is_notifying = False
        self.lock = threading.Lock()

    def on_change(self, func):
        def wrapper(*args, **kwargs):
            self._is_notifying = False
            func(*args, **kwargs)
        self._on_change_callback = wrapper
        return wrapper

    def _notify(self):
        if hasattr(self, "_on_change_callback"):
            self._is_notifying = True
            self._on_change_callback(self.queue)

    def add(self, key, value):
        new_item = {key: value}
        with self.lock:
            self.queue = [new_item] + self.queue[:-1] if len(self.queue) == self.maxsize else [new_item] + self.queue

    def update_val(self, index, key, value):
        if 0 <= index < len(self.queue):
            with self.lock:
                self.queue[index] = {key: value}
            self._notify()

    def update_to_true(self, model_name, index, notify=True):
        if index not in range(len(self.queue)):
            print("Index out of range")
            return

        with self.lock:
            try:
                current_model = list(self.queue[index])[0] if self.queue[index] else None
                
                if current_model != model_name:
                    matching_indices = [
                        i for i, item in enumerate(self.queue)
                        if item is not None and list(item)[0] == model_name
                    ]
                    if matching_indices:
                        index = matching_indices[0]
                    else:
                        print(f"Model {model_name} not found in queue")
                        return
                
                self.queue[index] = {model_name: True}
                
                if notify:
                    threading.Thread(target=self._notify).start()
                    
            except (IndexError, KeyError, AttributeError) as e:
                print(f"Error updating queue: {e}")
                return

        print(f"update_to_true DONE ({index}:{model_name})")


    def update_to_false(self, index, notify=True):
        if 0 <= index < len(self.queue):
            if self.queue[index] is None:
                threading.Thread(target=self._notify).start()
                return
            with self.lock:
                key = list(self.queue[index].keys())[0]
                self.queue[index] = {key: False}
                
                if notify:
                    threading.Thread(target=self._notify).start()
                print(f"update_to_false DONE ({index})")

    def get_state(self, index):
        with self.lock:
            try:
                state = list(self.queue[index].values())[0]
            except (AttributeError, IndexError) as e:
                state = False
        return state
    
    def reset_window(self):
        print(f"Resetting Window")
        for i in range(len(self.queue)):
            self.queue[i] = None

    def get_all_values(self):
        with self.lock:
            values = [list(item.values())[0] if item is not None else False for item in self.queue]
        return values
    
    def get_all_tasks(self):
        with self.lock:
            tasks = [list(item.keys())[0] if item is not None else None for item in self.queue]
        return tasks

    def set_all_false(self):
        with self.lock:
            for i in range(len(self.queue)):
                if self.queue[i] is not None:
                    key = list(self.queue[i].keys())[0]
                    self.queue[i] = {key: False}


    def change_all(self, lst):
        if len(lst) != self.maxsize:
            print(f"List length {len(lst)} does not match maxsize {self.maxsize}")
            sys.exit(1)
        with self.lock:
            self.queue = lst
            threading.Thread(target=self._notify).start()



class TensorRTModel:
    @classmethod
    def from_file(cls, engine_path):
        if os.path.isdir(engine_path):
            engine_path = f"{engine_path}.trt"
        logger = trt.Logger()
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            assert runtime
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        return TensorRTModel(engine, context)
    
    def __init__(self, engine, context):
        self.engine = engine
        self.context = context
        self.inputs, self.outputs, self.allocations, self.batch_size = self.binding_setup()
        self.output_format = np.zeros(self.outputs[0]["shape"], dtype=self.outputs[0]["dtype"])
        self.gpu_lock = threading.Lock()

    def predict(self, inp):
        self.gpu_lock.acquire()
        common.memcpy_host_to_device(self.inputs[0]["allocation"], inp)
        self.context.execute_v2(self.allocations)
        common.memcpy_device_to_host(self.output_format, self.outputs[0]["allocation"])
        self.gpu_lock.release()
        return self.output_format[0]

    def binding_setup(self):
        inputs, outputs, allocations = [], [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            allocations.append(allocation)
            if is_input:
                inputs.append(binding)
            else:
                outputs.append(binding)
        assert len(inputs) > 0 and len(outputs) > 0
        assert len(allocations) > 0 and batch_size > 0
        return inputs, outputs, allocations, batch_size

# Custom print function to print class name and function name
class CustomPrint:
    def __init__(self, classname, functionname):
        self.classname = classname
        self.functionname = functionname

    def __str__(self):
        color_start = '\033[0m'
        if "RtuxDetect" in self.classname:
            color_start = '\033[92m'  # Green color
        elif "task_manager" in self.functionname:
            color_start = '\033[93m' # Yellow color
        elif "queue_manager" in self.functionname:
            color_start = '\033[95m' # Red color
        else:
            color_start = '\033[94m'  # Blue color
        color_end = '\033[0m'
        return f"{color_start}[{self.classname}/{self.functionname}]{color_end} "

    def plain_str(self):
        return f"[{self.classname}/{self.functionname}]"


def print(*args, **kwargs):
    caller = inspect.stack()[1]
    if caller[0].f_locals.get("self") is not None:
        classname = caller[0].f_locals.get("self").__class__.__name__
        functionname = caller[3]
    else:
        classname = os.path.splitext(os.path.basename(caller[1]))[0]
        functionname = "main"
    prefix = CustomPrint(classname, functionname)

    output = ' '.join(map(str, args))
    if "ERROR" in output:
        output = f"\033[91;1m{output}\033[0m"

    builtins.print(prefix, output, **kwargs)
    
