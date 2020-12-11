import numpy as np
import math

from .tflite import Model
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.BuiltinOptions import BuiltinOptions
from .tflite.ActivationFunctionType import ActivationFunctionType
from .tflite.TensorType import TensorType
from .tflite.Conv2DOptions import Conv2DOptions
from .tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from .tflite.Padding import Padding
from .tflite.Pool2DOptions import Pool2DOptions
from .tflite.SoftmaxOptions import SoftmaxOptions

# Parse tflite to our model format


class TfliteConvertor(object):
    def __init__(self, filepath):
        # path to the tflite file
        self.filepath = filepath
        self.model = self.loadTFmodel(filepath)
        self.subgraph = self.model.Subgraphs(0)
        self.builtin_op_code = self.__build_str_map(BuiltinOperator())
        self.layer = []  # list to store each layer which will be a dict
        self.tmpPADIndice = None

    # public functions
    def loadTFmodel(self, filepath):
        #buf = open('imagenet_ram320_flash1024.tflite', 'rb').read()
        buf = open(filepath, 'rb').read()
        return Model.Model.GetRootAsModel(buf, 0)

    def dumpModelInfo(self):
        version = self.model.Version()
        print("Model version:", version)
        description = self.model.Description().decode('utf-8')
        print("Description:", description)
        subgraph_len = self.model.SubgraphsLength()
        print("Subgraph length:", subgraph_len)

        self.dumpLayerInfo()

    def dumpLayerInfo(self):
        print('Layer length:', len(self.layer))

        # print brief info about each layer
        for i in range(len(self.layer)):
            if self.layer[i]['op'] == 'ADD':
                print("op:", self.layer[i]['op'], ",input_idx:", self.layer[i]['input_idx'],
                      ",input2_idx:", self.layer[i]['input2_idx'], "output_idx:", self.layer[i]['output_idx'])
            else:
                print("op:", self.layer[i]['op'], ",input_idx:", self.layer[i]
                      ['input_idx'], "output_idx:", self.layer[i]['output_idx'])

    def parseOperatorInfo(self):
        operators_len = self.subgraph.OperatorsLength()

        for i in range(operators_len):
            op = self.subgraph.Operators(i)

            # parse the op
            self.__handleOperator(op)

    # private functions
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

    def __getTensorTypeStr(self, type):
        if TensorType.INT8 == type:
            return "int8"
        if TensorType.UINT8 == type:
            return "uint8"
        if TensorType.FLOAT32 == type:
            return "float32"

    def __getMultiplierShift(self, effective_scale):
        significand = np.zeros(len(effective_scale), dtype='int32')
        shift = np.zeros(len(effective_scale), dtype='int32')

        for i, s in enumerate(effective_scale):
            if s == 0:
                significand[i] = 0
                shift[i] = 0
            else:
                sig, shi = math.frexp(s)
                sig = int(round(sig * 2 ** 31))

                if sig == 2 ** 31:
                    sig /= 2
                    shi += 1
                if shi < -31:
                    shi = 0
                    sig = 0

                significand[i] = sig
                shift[i] = shi

        return significand, shift

    def __getSigShift(self, s):
        sig, shi = math.frexp(s)
        sig = int(round(sig * 2 ** 31))
        if sig == 2 ** 31:
            sig /= 2
            shi += 1
        if shi < -31:
            shi = 0
            sig = 0

        return sig, shi

    def __getADDMultiplierShift(self, input_scale, input2_scale, output_scale):
        left_shift = 20

        twice_max_input_scale = 2 * np.double(max(input_scale, input2_scale))
        real_input1_multiplier = np.double(input_scale / twice_max_input_scale)
        real_input2_multiplier = np.double(input2_scale / twice_max_input_scale)
        real_output_multiplier = np.double(twice_max_input_scale / ((1 << left_shift) * output_scale))

        input_multiplier, input_shift = self.__getSigShift(real_input1_multiplier)
        input2_multiplier, input2_shift = self.__getSigShift(real_input2_multiplier)
        output_multiplier, output_shift = self.__getSigShift(real_output_multiplier)

        return left_shift, input_multiplier, input_shift, input2_multiplier, input2_shift, output_multiplier, output_shift

    def __preprocessSoftmaxScaling(self, beta, input_scale, input_integer_bits):

        input_beta_real_multiplier = min(
            beta * input_scale * (1 << (31 - input_integer_bits)), (1 << 31) - 1.0)

        multiplier, shift = self.__getSigShift(input_beta_real_multiplier)

        return multiplier, shift

    # follow TFlite implementation
    def __calculateInputRadius(self, input_integer_bits, input_left_shift, total_signed_bits=31):
        max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) * (
            1 << (total_signed_bits - input_integer_bits)) / (1 << input_left_shift)
        return math.floor(max_input_rescaled)

    # converting tflite fuctions
    def __convert_covolutions(self, op):
        # operator
        op_code_str = self.__getOpCodeStr(op)

        # get input, weight, and output tensors
        input_tensors = self.__get_input_tensors(op)
        input_tensor_count = len(input_tensors)
        assert input_tensor_count >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        output_tensors = self.__get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # conv_2d options
        if op_code_str == 'CONV_2D':
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        if op_code_str == 'DEPTHWISE_CONV_2D':
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)

        # conv parameters
        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        # shapes
        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
        if op_code_str == 'CONV_2D':
            output_c, kernel_h, kernel_w, _ = weight_tensor.tensor.ShapeAsNumpy()
        elif op_code_str == 'DEPTHWISE_CONV_2D':
            _, kernel_h, kernel_w, output_c = weight_tensor.tensor.ShapeAsNumpy()
        _, output_h, output_w, output_c_dual = output_tensor.tensor.ShapeAsNumpy()
        assert output_c_dual == output_c, "output channels not match"

        # tensor types
        input_type = self.__getTensorTypeStr(input_tensor.tensor.Type())
        output_type = self.__getTensorTypeStr(output_tensor.tensor.Type())
        weight_type = self.__getTensorTypeStr(weight_tensor.tensor.Type())
        assert input_type == output_type == weight_type, "tensor type not consistent"

        # tensor value: weight, scalers
        weight_value = self.__get_tensor_value(weight_tensor)
        if input_tensor_count == 3:
            bias_tensor = input_tensors[2]
            bias = self.__get_tensor_value(bias_tensor)
        else:
            bias = None

        # quantized setting
        input_zero_point = input_tensor.qnn_params["zero_point"]
        output_zero_point = output_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
        weight_scale = weight_tensor.qnn_params["scale"]
        output_scale = output_tensor.qnn_params["scale"]
        effective_scale = np.double(
            input_scale) * np.double(weight_scale) / np.double(output_scale)

        # quantized inference, used for requantize
        multiplier, shift = self.__getMultiplierShift(effective_scale)

        # find previous layer and redirct the index and fuse pad into conv
        if self.tmpPADIndice != None:
            if self.tmpPADIndice.output_idx == input_tensor.tensor_idx:
                input_idx = self.tmpPADIndice.input_idx
            else:
                input_idx = input_tensor.tensor_idx
        else:
            input_idx = input_tensor.tensor_idx
        # clean the buffer
        self.tmpPADIndice = None

        params = {
            # operator
            "op": op_code_str,
            # conv
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "padding": math.floor(kernel_h/2),
            "stride_h": stride_h,
            "stride_w": stride_w,
            # tensor
            "input_idx": input_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_dim": 3,
            "output_dim": 3,
            # padding is fused
            "input_h": input_h - math.floor(kernel_h/2) * 2,
            # padding is fused
            "input_w": input_w - math.floor(kernel_h/2) * 2,
            "input_c": input_c,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "dtypte": input_type,
            # trainable parameters
            "weight_value": weight_value,
            "bias": bias,
            "input_zero_point": input_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "weight_scale": weight_scale,
            "output_scale": output_scale,
            # quantized infernece
            "multiplier": multiplier,
            "shift": shift,
        }

        self.layer.append(params)

    def __convert_ADD(self, op):
        # operator
        op_code_str = self.__getOpCodeStr(op)

        # get input, weight, and output tensors
        input_tensors = self.__get_input_tensors(op)
        input_tensor_count = len(input_tensors)
        assert input_tensor_count == 2, "input should be 2 tensors"

        input_tensor = input_tensors[0]
        input2_tensor = input_tensors[1]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.__get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # shapes
        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
        _, input2_h, input2_w, input2_c = input2_tensor.tensor.ShapeAsNumpy()
        _, output_h, output_w, output_c = output_tensor.tensor.ShapeAsNumpy()
        assert input_h == input2_h == output_h, "tensor shpae not consistent"
        assert input_w == input2_w == output_w, "tensor shpae not consistent"
        assert input_c == input2_c == output_c, "tensor shpae not consistent"

        # tensor types
        input_type = self.__getTensorTypeStr(input_tensor.tensor.Type())
        input_type2 = self.__getTensorTypeStr(input2_tensor.tensor.Type())
        output_type = self.__getTensorTypeStr(output_tensor.tensor.Type())
        assert input_type == input_type2 == output_type, "tensor type not consistent"

        # quantized setting
        input_zero_point = input_tensor.qnn_params["zero_point"]
        input2_zero_point = input2_tensor.qnn_params["zero_point"]
        output_zero_point = output_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
        input2_scale = input2_tensor.qnn_params["scale"]
        output_scale = output_tensor.qnn_params["scale"]

        # get multipliers and shifts
        left_shift, input_multiplier, input_shift, input2_multiplier, input2_shift, output_multiplier, output_shift = self.__getADDMultiplierShift(
            input_scale, input2_scale, output_scale)

        # assign params
        params = {
            # operator
            "op": op_code_str,
            # tensor
            "input_idx": input_tensor.tensor_idx,
            "input2_idx": input2_tensor.tensor_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_h": input_h,
            "input_w": input_w,
            "input_c": input_c,
            "input2_h": input_h,
            "input2_w": input_w,
            "input2_c": input_c,
            "input_dim": 3,
            "input2_dim": 3,
            "output_dim": 3,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "dtypte": input_type,
            # trainable parameters
            "input_zero_point": input_zero_point,
            "input2_zero_point": input2_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "input2_scale": input2_scale,
            "output_scale": output_scale,
            # quantized infernece
            "left_shift": left_shift,
            "input_multiplier": input_multiplier,
            "input2_multiplier": input2_multiplier,
            "input_shift": input_shift,
            "input2_shift": input2_shift,
            "output_multiplier": output_multiplier,
            "output_shift": output_shift,
        }

        self.layer.append(params)

    def __convert_AVERAGE_POOL_2D(self, op):
        # operator
        op_code_str = self.__getOpCodeStr(op)

        # get input, weight, and output tensors
        input_tensors = self.__get_input_tensors(op)
        input_tensor_count = len(input_tensors)
        assert input_tensor_count == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.__get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # shapes
        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
        _, output_h, output_w, output_c = output_tensor.tensor.ShapeAsNumpy()

        # tensor types
        input_type = self.__getTensorTypeStr(input_tensor.tensor.Type())
        output_type = self.__getTensorTypeStr(output_tensor.tensor.Type())
        assert input_type == output_type, "tensor type not consistent"

        # pool parameters
        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        # padding
        if padding == Padding.VALID:
            pad_h = 0
            pad_w = 0
        elif padding == Padding.SAME:
            pass  # no support for now

        # quantized setting
        input_zero_point = input_tensor.qnn_params["zero_point"]
        output_zero_point = output_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
        output_scale = output_tensor.qnn_params["scale"]

        params = {
            # operator
            "op": op_code_str,
            # pool parameters
            "filter_h": filter_h,
            "filter_w": filter_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            # tensor
            "input_idx": input_tensor.tensor_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_h": input_h,
            "input_w": input_w,
            "input_c": input_c,
            "input_dim": input_tensor.tensor.ShapeAsNumpy().size,
            "output_dim": output_tensor.tensor.ShapeAsNumpy().size,
            "output_h": output_h,
            "output_w": output_w,
            "output_c": output_c,
            "dtypte": input_type,
            # trainable parameters
            "input_zero_point": input_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "output_scale": output_scale,
        }

        self.layer.append(params)

    def __convert_SOFTMAX(self, op):
        # operator
        op_code_str = self.__getOpCodeStr(op)

        # get input, weight, and output tensors
        input_tensors = self.__get_input_tensors(op)
        input_tensor_count = len(input_tensors)
        assert input_tensor_count == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]

        output_tensors = self.__get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # shapes
        input_h, input_c = input_tensor.tensor.ShapeAsNumpy()
        output_h, output_c = output_tensor.tensor.ShapeAsNumpy()
        assert input_c == output_c, "channels not match"
        assert input_h == output_h, "dims not match"

        # tensor types
        input_type = self.__getTensorTypeStr(input_tensor.tensor.Type())
        output_type = self.__getTensorTypeStr(output_tensor.tensor.Type())
        assert input_type == output_type, "tensor type not consistent"

        # quantized setting
        input_zero_point = input_tensor.qnn_params["zero_point"]
        output_zero_point = output_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
        output_scale = output_tensor.qnn_params["scale"]

        # softmax options
        assert op.BuiltinOptionsType() == BuiltinOptions.SoftmaxOptions
        op_options = op.BuiltinOptions()
        softmax_options = SoftmaxOptions()
        softmax_options.Init(op_options.Bytes, op_options.Pos)

        beta = softmax_options.Beta()
        kScaledDiffIntegerBits = 5

        # multiplier and shift
        output_multiplier, output_shift = self.__preprocessSoftmaxScaling(
            beta, input_scale, kScaledDiffIntegerBits)
        diff_min = self.__calculateInputRadius(
            kScaledDiffIntegerBits, output_shift)

        params = {
            # operator
            "op": op_code_str,
            # tensor
            "input_idx": input_tensor.tensor_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_h": input_h,
            "input_w": None,
            "input_c": input_c,
            "input_dim": 2,
            "output_dim": 2,
            "output_h": output_h,
            "output_w": None,
            "output_c": output_c,
            "dtypte": input_type,
            # trainable parameters
            "input_zero_point": input_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "output_scale": output_scale,
            # quantized infernece
            "output_multiplier": output_multiplier,
            "output_shift": output_shift,
            "diff_min": diff_min,
        }

        self.layer.append(params)

    def __convert_PAD(self, op):
        # operator
        op_code_str = self.__getOpCodeStr(op)

        # get input, weight, and output tensors
        input_tensors = self.__get_input_tensors(op)
        input_tensor_count = len(input_tensors)
        input_tensor = input_tensors[0]

        output_tensors = self.__get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # fuse pad into conv
        self.tmpPADIndice = PAD_tensorIndice(
            input_tensor.tensor_idx, output_tensor.tensor_idx)

    def __convert_FULLY_CONNECTED(self, op):
        # operator
        op_code_str = self.__getOpCodeStr(op)

        # get input, weight, and output tensors
        input_tensors = self.__get_input_tensors(op)
        input_tensor_count = len(input_tensors)
        assert input_tensor_count == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        weight_tensor = input_tensors[1]
        bias_tensor = input_tensors[2]
        input_tensor_idx = input_tensor.tensor_idx
        weight = self.__get_tensor_value(weight_tensor)
        bias = self.__get_tensor_value(bias_tensor)

        output_tensors = self.__get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # shapes
        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
        output_c, input_c_dual = weight_tensor.tensor.ShapeAsNumpy()
        output_h, output_c_dual = output_tensor.tensor.ShapeAsNumpy()
        assert input_c_dual == input_c, "channels not match"
        assert output_c_dual == output_c, "channels not match"

        # tensor types
        input_type = self.__getTensorTypeStr(input_tensor.tensor.Type())
        output_type = self.__getTensorTypeStr(output_tensor.tensor.Type())
        assert input_type == output_type, "tensor type not consistent"

        # quantized setting
        input_zero_point = input_tensor.qnn_params["zero_point"]
        weight_zero_point = weight_tensor.qnn_params["zero_point"]
        output_zero_point = output_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
        weight_scale = weight_tensor.qnn_params["scale"]
        bias_scale = bias_tensor.qnn_params["scale"]
        output_scale = output_tensor.qnn_params["scale"]

        # TODO: verify this is correct, current approach follows tensorflow lite micro
        #output_multiplier, output_shift = self.__getSigShift((np.double(input_scale)*np.double(weight_scale))/np.double(output_scale))
        output_multiplier, output_shift = self.__getSigShift(
            np.double(bias_scale)/np.double(output_scale))

        params = {
            # operator
            "op": op_code_str,
            # tensor
            "input_idx": input_tensor.tensor_idx,
            "output_idx": output_tensor.tensor_idx,
            "input_h": input_h,
            "input_w": input_w,
            "input_c": input_c,
            "input_dim": 3,
            "output_dim": 2,
            "output_h": output_h,
            "output_w": None,
            "output_c": output_c,
            "dtypte": input_type,
            # trainable parameters
            "weight_value": weight,
            "bias": bias,
            "input_zero_point": input_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "output_scale": output_scale,
            # quantized infernece
            "output_multiplier": output_multiplier,
            "output_shift": output_shift,
        }

        self.layer.append(params)

    # handle one op and parse it into layers[] for supported operators
    def __handleOperator(self, op):
        op_code_str = self.__getOpCodeStr(op)
        if(op_code_str == "CONV_2D"):
            self.__convert_covolutions(op)
        if(op_code_str == "ADD"):
            self.__convert_ADD(op)
        if(op_code_str == "AVERAGE_POOL_2D"):
            self.__convert_AVERAGE_POOL_2D(op)
        if(op_code_str == "DEPTHWISE_CONV_2D"):
            self.__convert_covolutions(op)
        if(op_code_str == "SOFTMAX"):
            self.__convert_SOFTMAX(op)
        if(op_code_str == "PAD"):
            self.__convert_PAD(op)
        if(op_code_str == "FULLY_CONNECTED"):
            self.__convert_FULLY_CONNECTED(op)

    # utility functions referred from TVM
    def __get_tensor_type_as_numpy(self, tensor_wrapper):
        """Returns np.dtype out of TensorType"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        try:
            from .tflite.TensorType import TensorType

            return {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }[tensor_wrapper.tensor.Type()]
        except ImportError:
            raise ImportError("The tflite package must be installed")
        except KeyError:
            raise NotImplementedError(
                "Tensor type '{}' currently not supported".format(
                    tensor_wrapper.tensor.Type())
            )

    def __get_tensor_value(self, tensor_wrapper):
        """Get tensor buffer value from given tensor wrapper"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        dtype = self.__get_tensor_type_as_numpy(tensor_wrapper)
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = tensor_wrapper.tensor.ShapeAsNumpy()
        else:
            shape = []

        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def __get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from .tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.INT8:
            return "int8"
        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        if tensor_type == TensorType.BOOL:
            return "bool"
        raise NotImplementedError(
            "Tensor type {} is currently not supported".format(
                str(tensor_type))
        )

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
                        # Ensure that all zero points are zeros
                        zero_point = tflite_zero_point
                        if not np.all(zero_point == 0):
                            pass
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


class PAD_tensorIndice(object):
    """pading tensor indice"""

    def __init__(self, input_idx, output_idx):
        self.input_idx = input_idx
        self.output_idx = output_idx


# from TVM
class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params
