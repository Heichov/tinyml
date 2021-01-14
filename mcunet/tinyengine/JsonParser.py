import numpy as np
import pickle
import json
import math


class JsonParser(object):
    def __init__(self, data):
        self.layer = []
        assert isinstance(data, (dict, str))
        if isinstance(data, dict):  # already loaded
            self.data = data
        else:
            assert data.endswith('.json') or data.endswith('.pkl')
            if data.endswith('.pkl'):  # pickle file
                with open(data, 'rb') as f:
                    self.data = pickle.load(f)
            else:  # json file
                with open(data) as f:
                    self.data = json.load(f)

    def loadPickleModel(self):
        self.__loadFirstConv()
        self.__loadBlocks()
        self.__loadFeatureMix()
        self.__loadClassifier()

    def allocateMemory(self):
        pass

    def __loadFirstConv(self):
        layer_info = self.data['first_conv']
        self.layer.append(self.__convert_covolutions(layer_info))

    previousblock_lastlayer = 0

    def __loadBlocks(self):
        for i in range(len(self.data['blocks'])):
            if self.data['blocks'][i]['pointwise1'] != None:
                layer_info = self.data['blocks'][i]['pointwise1']
                self.layer.append(self.__convert_covolutions(layer_info))
            if self.data['blocks'][i]['depthwise'] != None:
                layer_info = self.data['blocks'][i]['depthwise']
                self.layer.append(self.__convert_covolutions(layer_info))
            if self.data['blocks'][i]['pointwise2'] != None:
                layer_info = self.data['blocks'][i]['pointwise2']
                self.layer.append(self.__convert_covolutions(layer_info))
            if self.data['blocks'][i]['residual'] != None:
                layer_info = self.data['blocks'][i]['residual']
                self.layer.append(self.__convert_ADD(layer_info))
                pass

            self.previousblock_lastlayer = len(self.layer) - 1
        pass

    def __convert_covolutions(self, layer_info):
        input_idx = len(self.layer) - 1
        output_idx = len(self.layer)
        # In case no params
        input_type = None
        weight_value_reorder = None
        bias = None
        input_zero_point = None
        output_zero_point = None
        input_scale = None
        output_scale = None
        weight_scale = None
        multiplier = None
        shift = None
        # If params
        if "params" in layer_info:
            params = layer_info['params']

            bias = params['bias']
            bias = bias.astype(int)
            weight_value = params['weight']
            if layer_info['depthwise']:
                weight_value_reorder = np.einsum('iohw->ohwi', weight_value)
            else:
                weight_value_reorder = np.einsum('iohw->ihwo', weight_value)

            input_zero_point = int(params['x_zero'])
            output_zero_point = int(params['y_zero'])
            input_scale = params['x_scale']
            weight_scale = params['w_scales']
            output_scale = params['y_scale']

            effective_scale = np.double(input_scale) * np.double(weight_scale) / np.double(output_scale)

            # quantized inference, used for requantize
            multiplier, shift = self.__getMultiplierShift(effective_scale)

        if layer_info['depthwise']:
            op = 'DEPTHWISE_CONV_2D'
        else:
            op = 'CONV_2D'

        params = {
            # operator
            "op": op,
            # conv
            "kernel_h": layer_info['kernel_size'],
            "kernel_w": layer_info['kernel_size'],
            "padding": math.floor(layer_info['kernel_size'] / 2),
            "stride_h": layer_info['stride'],
            "stride_w": layer_info['stride'],
            # tensor
            "input_idx": input_idx,
            "output_idx": output_idx,
            "input_dim": 3,
            "output_dim": 3,
            "input_h": layer_info['in_shape'],
            "input_w": layer_info['in_shape'],
            "input_c": layer_info['in_channel'],
            "output_h": layer_info['out_shape'],
            "output_w": layer_info['out_shape'],
            "output_c": layer_info['out_channel'],
            "dtypte": input_type,
            # trainable parameters
            "weight_value": weight_value_reorder,
            "bias": bias,
            "input_zero_point": input_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "output_scale": output_scale,
            "weight_scale": weight_scale,
            # quantized infernece
            "multiplier": multiplier,
            "shift": shift,
        }

        return params

    def __convert_ADD(self, layer_info):
        input_idx = len(self.layer) - 1
        input2_idx = self.previousblock_lastlayer
        output_idx = len(self.layer)
        # Incase no params
        input_type = None
        weight_value = None
        input_zero_point = None
        input2_zero_point = None
        output_zero_point = None
        input_scale = None
        input2_scale = None
        output_scale = None

        left_shift = None
        input_multiplier = None
        input_shift = None
        input2_multiplier = None
        input2_shift = None
        output_multiplier = None
        output_shift = None

        op = 'ADD'

        if "params" in layer_info:
            params = layer_info['params']

            input_scale = self.layer[len(self.layer) - 1]['output_scale']
            input_zero_point = int(self.layer[len(self.layer) - 1]['output_zero_point'])
            output_zero_point = int(params['y_zero'])
            output_scale = params['y_scale']
            input2_zero_point = int(params['x_zero'])
            input2_scale = params['x_scale']

            # get multipliers and shifts
            left_shift, input_multiplier, input_shift, input2_multiplier, input2_shift, output_multiplier, output_shift = self.__getADDMultiplierShift(
                input_scale, input2_scale, output_scale)

        params = {


            # operator
            "op": op,
            # tensor
            "input_idx": input_idx,
            "input2_idx": input2_idx,
            "output_idx": output_idx,
            "input_h": layer_info['in_shape'],
            "input_w": layer_info['in_shape'],
            "input_c": layer_info['in_channel'],
            "input2_h": layer_info['in_shape'],
            "input2_w": layer_info['in_shape'],
            "input2_c": layer_info['in_channel'],
            "input_dim": 3,
            "input2_dim": 3,
            "output_dim": 3,
            "output_h": layer_info['in_shape'],
            "output_w": layer_info['in_shape'],
            "output_c": layer_info['in_channel'],
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
        return params

    def __convert_Classifier(self, layer_info):
        ''' ADD a pool layer first '''
        input_idx = len(self.layer) - 1
        output_idx = len(self.layer)
        # Incase no params
        input_type = None
        input_zero_point = None
        output_zero_point = None
        input_scale = None
        output_scale = None

        op = "AVERAGE_POOL_2D"

        previousLayer = self.layer[len(self.layer) - 1]

        pool_params = {
            # operator
            "op": op,
            # pool parameters
            "filter_h": previousLayer['output_h'],
            "filter_w": previousLayer['output_w'],
            "stride_h": previousLayer['output_h'],
            "stride_w": previousLayer['output_w'],
            "pad_h": 0,
            "pad_w": 0,
            # tensor
            "input_idx": input_idx,
            "output_idx": output_idx,
            "input_h": previousLayer['output_h'],
            "input_w": previousLayer['output_w'],
            "input_c": previousLayer['output_c'],
            "input_dim": 3,
            "output_dim": 3,
            "output_h": 1,
            "output_w": 1,
            "output_c": layer_info['in_channel'],
            "dtypte": input_type,
            # trainable parameters
            "input_zero_point": input_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "output_scale": output_scale,
            # quantized infernece
        }

        # Parse the PW conv
        input_idx = len(self.layer)
        output_idx = len(self.layer) + 1
        # Incase no params
        input_type = None
        weight_value = None
        bias = None
        input_zero_point = None
        output_zero_point = None
        input_scale = None
        output_scale = None
        weight_scale = None
        multiplier = None
        shift = None
        if "params" in layer_info:
            params = layer_info['params']

            bias = params['bias']
            bias = bias.astype(int)
            weight_value = params['weight']
            input_zero_point = int(params['x_zero'])
            output_zero_point = int(params['y_zero'])
            input_scale = params['x_scale']
            weight_scale = params['w_scales']
            output_scale = params['y_scale']

            effective_scale = np.double(input_scale) * np.double(weight_scale) / np.double(output_scale)
            # since we use point-wise conv for classifier
            eff_arr = [effective_scale for i in range(
                layer_info['out_channel'])]

            # quantized inference, used for requantize
            multiplier, shift = self.__getMultiplierShift(eff_arr)

        op = 'CONV_2D'

        conv_params = {
            # operator
            "op": op,
            # conv
            "kernel_h": 1,
            "kernel_w": 1,
            "padding": 0,
            "stride_h": 1,
            "stride_w": 1,
            # tensor
            "input_idx": input_idx,
            "output_idx": output_idx,
            "input_dim": 3,
            "output_dim": 3,
            "input_h": 1,
            "input_w": 1,
            "input_c": layer_info['in_channel'],
            "output_h": 1,
            "output_w": 1,
            "output_c": layer_info['out_channel'],
            "dtypte": input_type,
            # trainable parameters
            "weight_value": weight_value,
            "bias": bias,
            "input_zero_point": input_zero_point,
            "output_zero_point": output_zero_point,
            "input_scale": input_scale,
            "output_scale": output_scale,
            "weight_scale": weight_scale,
            # quantized infernece
            "multiplier": multiplier,
            "shift": shift,
        }

        return pool_params, conv_params

    def __loadFeatureMix(self):
        if self.data['feature_mix'] != None:
            layer_info = self.data['feature_mix']
            self.layer.append(self.__convert_covolutions(layer_info))

    def __loadClassifier(self):
        if self.data['classifier'] != None:
            layer_info = self.data['classifier']
            pool_params, conv_params = self.__convert_Classifier(layer_info)
            self.layer.append(pool_params)
            self.layer.append(conv_params)

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
