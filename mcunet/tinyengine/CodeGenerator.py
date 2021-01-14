import numpy as np


class CodeGenerator(object):
    parse_count = 0
    header_handle = None
    source_handle = None

    def __init__(self, memsche, inplace):
        self.MemSche = memsche,
        self.header_handle = open("genModel.h", "w"),
        self.source_handle = open("genModel.cpp", "w"),
        self.inplace = inplace,

    def codeGeneration(self):
        # parse trainable parameters & assign the corresponding buffers for layers
        self.__parseTrainable()

        # end of the header find
        self.__finishHeaderEnd()

        # include all headers
        self.__includeHeaders()

        # generate invoke function
        self.__genInvoke()

    def __genInvoke(self):
        fp = self.source_handle[0]
        string = "void invoke(){\n"
        fp.write(string)

        schedule = self.MemSche[0]
        for i in range(len(schedule.layer)):
            string = "/* layer " + str(i) + ":" + schedule.layer[i]['op'] + " */\n"
            fp.write(string)

            if schedule.layer[i]['op'] == 'CONV_2D':
                kernel_h = schedule.layer[i]['kernel_h']
                if kernel_h == 1:
                    string = "convolve_1x1_s8(&buffer" + str(self.__getBufferIndex(
                        schedule.layer[i]['input_buf_add'])) + "[" + str(schedule.layer[i]['input_buf_add_offset']) + "],"
                    string += str(schedule.layer[i]['input_h']) + "," + str(schedule.layer[i]
                                                                            ['input_w']) + "," + str(schedule.layer[i]['input_c']) + ","
                    if self.inplace[0] == True and schedule.layer[i]['kernel_h'] == 1 and i > 0 and schedule.layer[i-1]['op'] == 'DEPTHWISE_CONV_2D':
                        string += "(const q7_t*) inplaceWeight" + str(schedule.layer[i]['parsed_trainable']) + ","
                    else:
                        string += "(const q7_t*) weight" + str(schedule.layer[i]['parsed_trainable']) + ","
                    string += "(int32*) bias" + str(schedule.layer[i]['parsed_trainable']) + ","
                    string += "shift" + str(schedule.layer[i]['parsed_trainable']) + "," + \
                        "multiplier" + str(schedule.layer[i]['parsed_trainable']) + ","
                    string += str(schedule.layer[i]['output_zero_point']) + "," + str(schedule.layer[i]['input_zero_point'] * -1) + ",-128,127," + "&buffer" + str(
                        self.__getBufferIndex(schedule.layer[i]['output_buf_add'])) + "[" + str(schedule.layer[i]['output_buf_add_offset']) + "],"
                    string += str(schedule.layer[i]['output_h']) + "," + str(schedule.layer[i]
                                                                             ['output_w']) + "," + str(schedule.layer[i]['output_c']) + ",sbuf);\n"

                    fp.write(string)

                elif kernel_h == 3 and schedule.layer[i]['stride_h'] == 2 and schedule.layer[i]['padding'] == 1:
                    string = "convolve_s8_kernel3_inputch3_stride2_pad1(&buffer" + str(self.__getBufferIndex(
                        schedule.layer[i]['input_buf_add'])) + "[" + str(schedule.layer[i]['input_buf_add_offset']) + "],"
                    string += str(schedule.layer[i]['input_h']) + "," + str(schedule.layer[i]
                                                                            ['input_w']) + "," + str(schedule.layer[i]['input_c']) + ","
                    string += "1," + \
                        "(const q7_t*) weight" + \
                        str(schedule.layer[i]['parsed_trainable']) + "," + str(schedule.layer[i]['output_c']) + ","
                    string += str(schedule.layer[i]['kernel_h']) + "," + str(schedule.layer[i]['kernel_w']) + ","
                    string += str(schedule.layer[i]['padding']) + "," + str(schedule.layer[i]['padding']) + \
                        "," + str(schedule.layer[i]['stride_h']) + "," + str(schedule.layer[i]['stride_w']) + ","
                    string += "(int32*) bias" + str(schedule.layer[i]['parsed_trainable']) + "," + "&buffer" + str(self.__getBufferIndex(
                        schedule.layer[i]['output_buf_add'])) + "[" + str(schedule.layer[i]['output_buf_add_offset']) + "],"
                    string += "shift" + str(schedule.layer[i]['parsed_trainable']) + "," + \
                        "multiplier" + str(schedule.layer[i]['parsed_trainable']) + ","
                    string += str(schedule.layer[i]['output_zero_point']) + "," + \
                        str(schedule.layer[i]['input_zero_point'] * -1) + ",-128,127,"
                    string += str(schedule.layer[i]['output_h']) + "," + str(schedule.layer[i]
                                                                             ['output_w']) + ",sbuf," + str(schedule.layer[i]['input_zero_point']) + ");\n"

                    fp.write(string)
                else:
                    print("unexpected kernel_h for conv_2d")

                self.parse_count += 1
            elif schedule.layer[i]['op'] == 'DEPTHWISE_CONV_2D':
                kernel_h = schedule.layer[i]['kernel_h']
                if self.inplace[0] == True:
                    string = "fast_depthwise_conv_s8_kernel" + str(kernel_h) + "_stride" + str(
                        schedule.layer[i]['stride_h']) + "_pad" + str(schedule.layer[i]['padding']) + "_a8w8_8bit_HWC_inplace"
                else:
                    string = "fast_depthwise_conv_s8_kernel" + \
                        str(kernel_h) + "_stride" + str(schedule.layer[i]['stride_h']) + \
                        "_pad" + str(schedule.layer[i]['padding']) + "_a8w8_8bit_HWC"
                string += "(&buffer" + str(self.__getBufferIndex(schedule.layer[i]['input_buf_add'])) + "[" + str(
                    schedule.layer[i]['input_buf_add_offset']) + "],"
                string += str(schedule.layer[i]['input_h']) + "," + str(schedule.layer[i]
                                                                        ['input_w']) + "," + str(schedule.layer[i]['input_c']) + ","
                if self.inplace[0] == True:
                    output_offset = schedule.buffers['input_output'] - \
                        schedule.layer[i]['output_w'] * schedule.layer[i]['output_h'] * 2
                    string += "(const q7_t*) CHWweight" + str(schedule.layer[i]['parsed_trainable']) + ","
                    string += "(int32*) CHWbias" + str(schedule.layer[i]['parsed_trainable']) + ","
                else:
                    string += "(const q7_t*) weight" + str(schedule.layer[i]['parsed_trainable']) + ","
                    string += "(int32*) bias" + str(schedule.layer[i]['parsed_trainable']) + ","

                string += "shift" + str(schedule.layer[i]['parsed_trainable']) + "," + \
                    "multiplier" + str(schedule.layer[i]['parsed_trainable']) + ","
                string += str(schedule.layer[i]['output_zero_point']) + "," + \
                    str(schedule.layer[i]['input_zero_point'] * -1) + ",-128,127,"
                string += "&buffer" + \
                    str(self.__getBufferIndex(schedule.layer[i]['output_buf_add'])) + \
                    "[" + str(schedule.layer[i]['output_buf_add_offset']) + "],"
                string += str(schedule.layer[i]['output_h']) + "," + str(schedule.layer[i]
                                                                         ['output_w']) + "," + str(schedule.layer[i]['output_c']) + ","
                string += "sbuf," + str(schedule.layer[i]['input_zero_point']) + ");\n"

                fp.write(string)
                self.parse_count += 1

            elif schedule.layer[i]['op'] == 'ADD':
                string = """input_shape.Resize(4);
ptr = input_shape.DimsData();\n"""
                string += "ptr[0] = 1;ptr[1] = " + str(schedule.layer[i]['input_h']) + ";ptr[2] = " + str(
                    schedule.layer[i]['input_w']) + ";ptr[3] = " + str(schedule.layer[i]['input_c']) + ";\n"
                string += """output_shape.Resize(4);
ptr = output_shape.DimsData();\n"""
                string += "ptr[0] = 1;ptr[1] = " + str(schedule.layer[i]['input_h']) + ";ptr[2] = " + str(
                    schedule.layer[i]['input_w']) + ";ptr[3] = " + str(schedule.layer[i]['input_c']) + ";\n"
                string += "add_op_params.left_shift = " + str(schedule.layer[i]['left_shift']) + ";add_op_params.input1_offset = " + str(
                    schedule.layer[i]['input_zero_point'] * -1) + ";add_op_params.input1_multiplier = " + str(schedule.layer[i]['input_multiplier'])
                string += ";add_op_params.input1_shift = " + str(schedule.layer[i]['input_shift']) + ";add_op_params.input2_offset = " + str(
                    schedule.layer[i]['input2_zero_point'] * -1) + ";add_op_params.input2_multiplier = " + str(schedule.layer[i]['input2_multiplier'])
                string += ";add_op_params.input2_shift = " + str(schedule.layer[i]['input2_shift']) + ";add_op_params.output_offset = " + str(
                    schedule.layer[i]['output_zero_point']) + ";add_op_params.output_multiplier = " + str(schedule.layer[i]['output_multiplier'])
                string += ";add_op_params.output_shift = " + \
                    str(schedule.layer[i]['output_shift']) + \
                    ";add_op_params.quantized_activation_max = 127;add_op_params.quantized_activation_min = -128;\n"
                string += "Add(add_op_params, input_shape, &buffer" + str(self.__getBufferIndex(schedule.layer[i]['input_buf_add'])) + "[" + str(schedule.layer[i]['input_buf_add_offset']) + "], input_shape, &buffer" + str(self.__getBufferIndex(
                    schedule.layer[i]['input2_buf_add'])) + "[" + str(schedule.layer[i]['input2_buf_add_offset']) + "]," + " output_shape, &buffer" + str(self.__getBufferIndex(schedule.layer[i]['output_buf_add'])) + "[" + str(schedule.layer[i]['output_buf_add_offset']) + "]);\n"

                fp.write(string)
            elif schedule.layer[i]['op'] == 'FULLY_CONNECTED':
                string = '''input_shape.Resize(4);
ptr = input_shape.DimsData();\n'''
                string += "ptr[0] = 1;ptr[1] = " + str(schedule.layer[i]['input_h']) + ";ptr[2] = " + str(
                    schedule.layer[i]['input_w']) + ";ptr[3] = " + str(schedule.layer[i]['input_c']) + ";\n"
                string += '''output_shape.Resize(2);
ptr = output_shape.DimsData();\n'''
                string += "ptr[0] = " + str(schedule.layer[i]['output_h']) + ";ptr[1] = " + \
                    str(schedule.layer[i]['output_c']) + ";\n"
                string += '''filter_shape.Resize(2);
ptr = filter_shape.DimsData();\n'''
                string += "ptr[0] = " + str(schedule.layer[i]['output_c']) + ";ptr[1] = " + \
                    str(schedule.layer[i]['input_c']) + ";\n"
                string += '''bias_shape.Resize(1);
ptr = bias_shape.DimsData();\n'''
                string += "ptr[0] = " + str(schedule.layer[i]['output_c']) + ";\n"
                string += "FC_op_params.input_offset = " + str(schedule.layer[i]['input_zero_point'] * -1) + ";FC_op_params.weights_offset = 0;FC_op_params.output_offset = " + str(schedule.layer[i]['output_zero_point']) + ";FC_op_params.output_multiplier = " + str(
                    schedule.layer[i]['output_multiplier']) + ";FC_op_params.output_shift = " + str(schedule.layer[i]['output_shift']) + ";FC_op_params.quantized_activation_min = -128;FC_op_params.quantized_activation_max = 127;\n"
                string += "FullyConnected(FC_op_params, input_shape, &buffer" + str(self.__getBufferIndex(schedule.layer[i]['input_buf_add'])) + "[" + str(schedule.layer[i]['input_buf_add_offset']) + "], filter_shape, (const q7_t*) weight" + str(
                    schedule.layer[i]['parsed_trainable']) + ", bias_shape, (int32*) bias" + str(schedule.layer[i]['parsed_trainable']) + ", output_shape, &buffer" + str(self.__getBufferIndex(schedule.layer[i]['output_buf_add'])) + "[" + str(schedule.layer[i]['output_buf_add_offset']) + "]);"
                fp.write(string)

            elif schedule.layer[i]['op'] == 'SOFTMAX':
                string = '''shape.Resize(4);
ptr = shape.DimsData();\n'''
                string += "ptr[0] = 1;ptr[1] = 1;ptr[2] = " + \
                    str(schedule.layer[i]['input_h']) + ";ptr[3] = " + str(schedule.layer[i]['input_c']) + ";\n"
                string += '''output_shape.Resize(2);
ptr = output_shape.DimsData();\n'''
                string += "soft_op_params.input_multiplier = " + str(schedule.layer[i]['output_multiplier']) + ";soft_op_params.input_left_shift = " + str(
                    schedule.layer[i]['output_shift']) + ";soft_op_params.diff_min = " + str(schedule.layer[i]['diff_min'] * -1) + ";\n"
                string += "Softmax(soft_op_params, shape, &buffer" + str(self.__getBufferIndex(schedule.layer[i]['input_buf_add'])) + "[" + str(schedule.layer[i]['input_buf_add_offset']) + "], shape, &buffer" + str(
                    self.__getBufferIndex(schedule.layer[i]['output_buf_add'])) + "[" + str(schedule.layer[i]['output_buf_add_offset']) + "]);\n"
                fp.write(string)

            elif schedule.layer[i]['op'] == 'AVERAGE_POOL_2D':
                string = """input_shape.Resize(4);
ptr = input_shape.DimsData();\n"""
                string += "ptr[0] = 1;ptr[1] = " + str(schedule.layer[i]['input_h']) + ";ptr[2] = " + str(
                    schedule.layer[i]['input_w']) + ";ptr[3] = " + str(schedule.layer[i]['input_c']) + ";\n"
                string += """output_shape.Resize(4);
ptr = output_shape.DimsData();\n"""
                string += "ptr[0] = 1;ptr[1] = 1;ptr[2] = 1;ptr[3] = " + str(schedule.layer[i]['input_c']) + ";\n"
                string += "pool_op_params.stride_height = " + str(schedule.layer[i]['stride_h']) + ";pool_op_params.stride_width = " + str(
                    schedule.layer[i]['stride_w']) + ";pool_op_params.filter_height  = " + str(schedule.layer[i]['filter_h'])
                string += ";pool_op_params.filter_width = " + str(schedule.layer[i]['filter_w']) + ";pool_op_params.padding_values.height = " + str(
                    schedule.layer[i]['pad_h']) + ";pool_op_params.padding_values.width = " + str(schedule.layer[i]['pad_h'])
                string += ";pool_op_params.quantized_activation_max = 127;pool_op_params.quantized_activation_min = -128;\n"
                string += "AveragePool(pool_op_params, input_shape, &buffer" + str(self.__getBufferIndex(schedule.layer[i]['input_buf_add'])) + "[" + str(
                    schedule.layer[i]['input_buf_add_offset']) + "]" + " , output_shape, &buffer" + str(self.__getBufferIndex(schedule.layer[i]['output_buf_add'])) + "[" + str(schedule.layer[i]['output_buf_add_offset']) + "]);\n"
                fp.write(string)

        string = "}\n"
        fp.write(string)

    def __getBufferIndex(self, location):
        if location == 'front':
            return 0
        elif location == 'end':
            return 0
        elif location == 'residual':
            return 1
        return None

    def __finishHeaderEnd(self):
        schedule = self.MemSche[0]
        # define output tensor
        last_index = len(schedule.layer) - 1
        string = "#define NNoutput &buffer0[" + str(schedule.layer[last_index]['output_buf_add_offset']) + "];"
        fp = self.header_handle[0]
        fp.write("\n" + string + "\n")

        # activation buffers
        string = "\n/* sram:" + str(schedule.peakmem) + ", flash:" + str(schedule.flash) + " */\n"
        fp.write(string + "\n")

        string = "static int16_t sbuf[" + str(int(schedule.buffers["im2col"]/2)) + "];\n"
        fp.write(string)
        string = "static int32_t kbuf[" + str(int(schedule.buffers["kernel"]/4)) + "];\n"
        fp.write(string)
        string = "static signed char buffer0[" + str(int(schedule.buffers["input_output"])) + "];\n"
        fp.write(string)
        string = "static signed char buffer1[" + str(int(schedule.buffers["residual"])) + "];\n"
        fp.write(string)
        string = "const int SBuffer_size = " + str(int(schedule.buffers["im2col"])) + ";\n"
        fp.write(string)
        string = "const int KBuffer_size = " + str(int(schedule.buffers["kernel"])) + ";\n"
        fp.write(string)

    def __includeHeaders(self):
        include_string = """/* Automatically generated source file */
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include "arm_nnfunctions.h"
#include "scratch_buffer.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/pad.h"

#include "genNN.h"
#include "genModel.h"

extern "C" { 
    #include "kernel_buffer.h"
    #include "tinyengine_function.h"
}
using namespace tflite;
using namespace tflite::reference_ops;
using namespace tflite::reference_integer_ops;
/* Variables for each op's parameters */
PadParams pad_op_params __attribute__((section(".dtcmvars.opparameters")));
signed char pad_value __attribute__((section(".dtcmvars.opparameters")));
tflite::ArithmeticParams add_op_params __attribute__((section(".dtcmvars.opparameters")));
PoolParams pool_op_params __attribute__((section(".dtcmvars.opparameters")));
tflite::FullyConnectedParams FC_op_params __attribute__((section(".dtcmvars.opparameters")));
tflite::SoftmaxParams soft_op_params __attribute__((section(".dtcmvars.opparameters")));

/* Variables used by all ops */
RuntimeShape input_shape __attribute__((section(".dtcmvars.opparameters")));
RuntimeShape output_shape __attribute__((section(".dtcmvars.opparameters")));
RuntimeShape filter_shape __attribute__((section(".dtcmvars.opparameters")));
RuntimeShape bias_shape __attribute__((section(".dtcmvars.opparameters")));
RuntimeShape shape __attribute__((section(".dtcmvars.opparameters")));
int32* ptr __attribute__((section(".dtcmvars.opparameters")));

signed char* getInput() {
	return buffer0;
}
signed char* getOutput() {
	return NNoutput;
}
void setupBuffer() {
	set_kernel_buffer(kbuf, KBuffer_size);
	set_scratch_buffer((int16_t*)sbuf, SBuffer_size);
}"""
        fp = self.source_handle[0]
        fp.write(include_string)

    def __parseTrainable(self):
        schedule = self.MemSche[0]
        for i in range(len(schedule.layer)):

            if schedule.layer[i]['op'] == 'CONV_2D':
                if self.inplace[0] == True and schedule.layer[i]['kernel_h'] == 1 and i > 0 and schedule.layer[i-1]['op'] == 'DEPTHWISE_CONV_2D':
                    # reorder PW's weight if needed
                    self.__parseReorderPWWeight(self.parse_count, schedule.layer[i]['weight_value'].flatten(
                    ), schedule.layer[i]['input_c'], schedule.layer[i]['output_c'])
                else:
                    self.__parseWeight(self.parse_count, schedule.layer[i]['weight_value'].flatten())
                self.__parseBias(self.parse_count, schedule.layer[i]['bias'].flatten())
                self.__parseRequantize(self.parse_count, schedule.layer[i]['shift'].flatten(
                ), schedule.layer[i]['multiplier'].flatten())

                schedule.layer[i]['parsed_trainable'] = self.parse_count
                self.parse_count += 1
            elif schedule.layer[i]['op'] == 'DEPTHWISE_CONV_2D':
                if self.inplace[0] == True:
                    self.__parseCHWWeight(
                        self.parse_count, schedule.layer[i]['weight_value'].flatten(), schedule.layer[i]['input_c'])
                    self.__parseCHWBias(self.parse_count, schedule.layer[i]['bias'].flatten(
                    ), schedule.layer[i]['input_zero_point'] * -1, schedule.layer[i]['weight_value'].flatten(), schedule.layer[i]['input_c'])
                else:
                    self.__parseWeight(self.parse_count, schedule.layer[i]['weight_value'].flatten())
                    self.__parseBias(self.parse_count, schedule.layer[i]['bias'].flatten())
                self.__parseRequantize(self.parse_count, schedule.layer[i]['shift'].flatten(
                ), schedule.layer[i]['multiplier'].flatten())

                schedule.layer[i]['parsed_trainable'] = self.parse_count
                self.parse_count += 1

            elif schedule.layer[i]['op'] == 'ADD':
                pass

            elif schedule.layer[i]['op'] == 'FULLY_CONNECTED':
                self.__parseWeight(self.parse_count, schedule.layer[i]['weight_value'].flatten())
                self.__parseBias(self.parse_count, schedule.layer[i]['bias'].flatten())

                schedule.layer[i]['parsed_trainable'] = self.parse_count
                self.parse_count += 1

            elif schedule.layer[i]['op'] == 'SOFTMAX':
                pass

    def __parseCHWWeight(self, Lindex, weight, channel):
        fp = self.header_handle[0]
        string = 'const unsigned char CHWweight' + str(Lindex) + '[' + str(len(weight)) + '] = {'
        fp.write(string)
        kernelsize = int(len(weight) / channel)
        for j in range(channel):
            for i in range(kernelsize):
                value = weight[i * channel + j]
                if value < 0:
                    value += 256
                fp.write(str(format(value, "#04x")) + ", ")
        fp.write("};\n")

    def __parseReorderPWWeight(self, Lindex, weight, input_c, output_c):
        fp = self.header_handle[0]
        string = 'const unsigned char inplaceWeight' + str(Lindex) + '[' + str(len(weight)) + '] = {'
        fp.write(string)

        for i in range(output_c):
            for j in range(2, input_c):
                value = weight[i * input_c + j]
                if value < 0:
                    value += 256
                fp.write(str(format(value, "#04x")) + ", ")

            value = weight[i * input_c]
            if value < 0:
                value += 256
            fp.write(str(format(value, "#04x")) + ", ")
            value = weight[i * input_c + 1]
            if value < 0:
                value += 256
            fp.write(str(format(value, "#04x")) + ", ")

        fp.write("};\n")

    def __parseWeight(self, Lindex, weight):
        fp = self.header_handle[0]
        string = 'const unsigned char weight' + str(Lindex) + '[' + str(len(weight)) + '] = {'
        fp.write(string)
        for i in range(len(weight)):
            value = weight[i]
            if value < 0:
                value += 256
            fp.write(str(format(value, "#04x")) + ", ")
        fp.write("};\n")

    def __parseCHWBias(self, Lindex, bias, input_offset, weight, channel):
        fp = self.header_handle[0]
        string = 'const int32 CHWbias' + str(Lindex) + '[' + str(len(bias)) + '] = {'
        fp.write(string)
        kernelsize = int(len(weight) / channel)
        for i in range(channel):
            tmpW = 0
            for j in range(kernelsize):
                tmpW += weight[j * channel + i]
            fp.write(str(bias[i] + tmpW * input_offset) + ", ")
        fp.write("};\n")

    def __parseBias(self, Lindex, bias):
        fp = self.header_handle[0]
        string = 'const int32 bias' + str(Lindex) + '[' + str(len(bias)) + '] = {'
        fp.write(string)
        for i in range(len(bias)):
            fp.write(str(bias[i]) + ", ")
        fp.write("};\n")

    def __parseRequantize(self, Lindex, shift, multiplier):
        fp = self.header_handle[0]
        string = 'const int32 shift' + str(Lindex) + '[' + str(len(shift)) + '] = {'
        fp.write(string)
        for i in range(len(shift)):
            fp.write(str(shift[i]) + ", ")
        fp.write("};\n")

        string = 'const int32 multiplier' + str(Lindex) + '[' + str(len(multiplier)) + '] = {'
        fp.write(string)
        for i in range(len(multiplier)):
            fp.write(str(multiplier[i]) + ", ")
        fp.write("};\n")

    def __closefp(self):
        self.header_handle[0].close()
        self.source_handle[0].close()
