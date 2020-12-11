import numpy as np
import math
import os

from .tflite import Model
from .tflite.BuiltinOperator import BuiltinOperator

# binary size for the TF-Lite Micro runtime, update this to match your setting
TFMicro_CodeSize = 96217


# Parse tflite to our model format
class TFMicroMemSimulator(object):
    def __init__(self, filepath):
        self.TFSize = 0
        self.Memory = {}

        # path to the tflite file
        self.filepath = filepath
        self.model = self.loadTFmodel(filepath)
        self.subgraph = self.model.Subgraphs(0)
        self.builtin_op_code = self.__build_str_map(BuiltinOperator())

    # public functions
    def loadTFmodel(self, filepath):
        buf = open(filepath, 'rb').read()
        self.TFSize = os.stat(filepath).st_size
        return Model.Model.GetRootAsModel(buf, 0)

    def getFlash(self):
        return self.TFSize + TFMicro_CodeSize

    # store the size of required memory of each data structure on microcontrollers, need to update/use another table if the platform (ISA) change
    SizeTable = {
        'CreateInPlaceSimpleMemoryAllocator': 24,
        'tensor': 64,
        'TfLiteAffineQuantization': 12,
        'IntBytes': 4,
        'FloatBytes': 4,
        'ArrayBytes': 4,
        'Node': 40,
        # OpData
        'CONV_2D': 24,
        'ADD': 4,
        'DEPTHWISE_CONV_2D': 28,
        'AVERAGE_POOL_2D': 40,
        'SOFTMAX': 4,
        'PAD': 0,
        'FULLY_CONNECTED': 12,
        'RESHAPE': 36,
        # This is just temporary memory, which does not count for peak memory
        'allocation_info': 20

    }

    def simulateMemoryAllocation(self):
        self.Memory['CreateInPlaceSimpleMemoryAllocator'] = self.SizeTable['CreateInPlaceSimpleMemoryAllocator']
        self.Memory['tensors'] = self.SizeTable['tensor'] * self.subgraph.TensorsLength()
        runtimeTensor = 0            
        for i in range(self.subgraph.TensorsLength()):
            tensor = self.subgraph.Tensors(i)
            q = tensor.Quantization()
            # for micro_allocator to copy the quantization information from the serialized data.

            if q.ScaleLength() > 0:
                runtimeTensor += self.SizeTable['TfLiteAffineQuantization']
                # zero_point
                runtimeTensor += self.SizeTable['IntBytes'] * q.ScaleLength() + self.SizeTable['ArrayBytes'] #Int array
                # scale
                runtimeTensor += self.SizeTable['FloatBytes'] * q.ScaleLength() + self.SizeTable['ArrayBytes'] #Float array

        self.Memory['runtimeTensor'] = runtimeTensor

        # AllocateNodeAndRegistrations
        # Node
        self.Memory['Node'] = self.SizeTable['Node'] * self.subgraph.OperatorsLength()
        # OpData
        OpData = 0
        for i in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(i)
            op_code_str = self.__getOpCodeStr(op)
            # Need to include all possiblely used layer Opdata in the SizeTable
            alignment = 8 # Memory alignment
            OpSize = self.SizeTable[op_code_str]
            if (sum(self.Memory.values()) + OpData) % alignment :
                OpData += math.ceil(OpSize / alignment) * alignment
            else: 
                OpData += OpSize

        self.Memory['OpData'] = OpData
        # FinishTensorAllocation
        # allocation_info
        self.Memory['allocation_info'] = self.SizeTable['allocation_info'] * self.subgraph.TensorsLength()

        # Activation
        layerAc = []
        layerOutIdx = []
        operators_len = self.subgraph.OperatorsLength()
        for i in range(operators_len):
            op = self.subgraph.Operators(i)
            op_code_str = self.__getOpCodeStr(op)
            input_tensors = self.__get_input_tensors(op)
            output_tensors = self.__get_output_tensors(op)
            act = 0
            input_size = input_tensors[0].tensor.ShapeAsNumpy().prod()
            output_size = output_tensors[0].tensor.ShapeAsNumpy().prod()
            act += input_size + output_size

            # residual, has two inputs and need to trace back
            output_idx = output_tensors[0].tensor_idx
            layerOutIdx.append(output_idx)
            if(op_code_str == "ADD"):
                input2_size = input_tensors[1].tensor.ShapeAsNumpy().prod()
                input2_idx = input_tensors[1].tensor_idx
                act += input2_size
                for j in range(i-1, 0, -1):
                    # find the layer of which output is the second input
                    if(input2_idx == layerOutIdx[j]):
                        break
                    layerAc[j] += input2_size

            layerAc.append(act)

        self.Memory['Activation'] = max(layerAc)

    def getActivationMemory(self):
        return sum(self.Memory.values())

    def __getOpCodeStr(self, op):
        op_code_list_idx = op.OpcodeIndex()
        op_code_id = self.model.OperatorCodes(
            op_code_list_idx).DeprecatedBuiltinCode()
        return self.builtin_op_code[op_code_id]

    def __build_str_map(self, obj):
        ret = {}
        for field_name in dir(obj):
            if not field_name.startswith("_"):
                field_value = getattr(obj, field_name)
                if isinstance(field_value, int):
                    ret[field_value] = field_name
        return ret

    def __getOpCodeStr(self, op):
        op_code_list_idx = op.OpcodeIndex()
        op_code_id = self.model.OperatorCodes(
            op_code_list_idx).DeprecatedBuiltinCode()
        return self.builtin_op_code[op_code_id]

    def __get_input_tensors(self, op):
        operator_inputs = op.InputsAsNumpy()
        return self.__get_tensors(operator_inputs)

    def __get_output_tensors(self, op):
        operator_outputs = op.OutputsAsNumpy()
        return self.__get_tensors(operator_outputs)

    def __get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                # TFLite supports both per-tensor and per-axis (aka channel) quantization.  For
                # per-tensor quantization, scale and zero points are scalar values.  For per-axis
                # quantization, scale and zero points for the weights are tensors (activations are
                # per-tensor quantized). However, the TFLite quantization spec puts restrictions on
                # zero points for per-axis quantization.  Specifically, the zero point is a tensor
                # but all values are 0. More information can be found here -
                # https://www.tensorflow.org/lite/performance/quantization_spec

                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True

                # Handle Per-axis and per-tensor cases
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)

                    # Tensor - Per-axis quantization
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        zero_point = tflite_zero_point
                        zero_point = int(zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        raise NotImplementedError(
                            "Quantized type {} (scale) and  {} (zero point) not supported".format(
                                type(tflite_scale), type(tflite_zero_point)
                            )
                        )
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    # Handle corner case for ops like quantized reshape whose second operand (shape)
                    # has zero scale and zero zero point. This is not used.
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError(
                        "Quantized type {} not supported".format(
                            type(tflite_scale))
                    )

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = scale
                    qnn_params["zero_point"] = zero_point
            return_list.append(TensorWrapper(
                tensor_idx, tensor, buffer, qnn_params))
        return return_list

# from TVM
class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params
