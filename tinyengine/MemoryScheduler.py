import numpy as np


class MemoryScheduler(object):
    USE_INPLACE = True
    def __init__(self, layer):
        self.layer = layer
        self.buffers = {
            "input_output" : 0,
            "residual": 0,
            "im2col": 0, 
            "kernel": 0}
        self.peakmem = 0 #currently we only support MoblieNet-like models which only have 1 by-pass for a block
        self.flash = 0
        self.bias = 0
        self.scale = 0

        self.layermem = []
        
    # public functions 
    def allocateMemory(self):
        # varaiables to maintain tensors of previous layer 
        previous_output_add = 'front' # input is place at &buffer0[0]
        last_is_residual = False 

        
        ## For detailed memory
        for i in range(len(self.layer)):
            layermem = { }
            self.layermem.append(layermem)

        # go through all layers and figure out the placement of each tensors
        for i in range(len(self.layer)):
            ''' find the life cycle of the output '''
            output_idx = self.layer[i]['output_idx']
            # scan if the output is used for residual
            output_residual = False
            ## For detailed memory
            residual_index = 0
            for j in range(i+2, len(self.layer)):
                if self.layer[j]['input_idx'] == output_idx:
                    output_residual = True
                    residual_index = j
                    break
                if self.layer[j]['op'] == 'ADD' and self.layer[j]['input2_idx'] == output_idx:
                    output_residual = True
                    residual_index = j
                    break
                
            ## For detailed memory
            output_size = self.__flatsize(self.layer[i], "output")
            for j in range(i+1, residual_index):
                self.layermem[j]['residual'] = output_size

            ''' assign the output address '''
            if output_residual:
                self.layer[i]['output_buf_add'] = 'residual'  # place it in the residual buf
            else:
                if previous_output_add == 'end':
                    if self.layer[i]['op'] == 'DEPTHWISE_CONV_2D' and self.USE_INPLACE:
                        self.layer[i]['output_buf_add'] = 'end' # place it inplace
                    else:
                        self.layer[i]['output_buf_add'] = 'front' # place it inplace
                else:
                    if self.layer[i]['op'] == 'DEPTHWISE_CONV_2D' and self.USE_INPLACE:
                        self.layer[i]['output_buf_add'] = 'front' # place it inplace
                    else:
                        self.layer[i]['output_buf_add'] = 'end' # place it inplace

            ''' assign the input address and enlarge buffer '''
            input_size = self.__flatsize(self.layer[i], "input")
            if self.layer[i]['op'] == 'DEPTHWISE_CONV_2D' and self.USE_INPLACE:
                # we just need two channels for the inplace implementation
                output_size = self.layer[i]['input_h'] * self.layer[i]['input_w'] * 2
            else:
                output_size = self.__flatsize(self.layer[i], "output")

            # 1. enlarge the input_output buffer
            if output_residual:
                self.__enlargeBuffer("input_output", input_size)
            else:
                self.__enlargeBuffer("input_output", input_size + output_size)
            # 2. assign address and enlarge residual buffer if needed (e.g., ADD)
            if self.layer[i]['op'] == 'ADD':# two inputs
                self.layer[i]['input_buf_add'] = previous_output_add
                self.layer[i]['input2_buf_add'] = 'residual'

                input2_size = self.__flatsize(self.layer[i], "input") # same as input
                self.__enlargeBuffer("residual", input_size)

                ## For detailed memory
                self.layermem[i]['activation'] = input_size + output_size

            else:# one input
                self.layer[i]['input_buf_add'] = previous_output_add

                ## For detailed memory
                self.layermem[i]['activation'] = input_size + output_size
                
            ''' update previous output address '''
            previous_output_add = self.layer[i]['output_buf_add']

        # now we have the buffer size for input/output tensors
        # go through all layers again to (1) assign specific address for each tensor and (2) enlarge intermediate buffers
        for i in range(len(self.layer)):
            # (1) assign specific address for each tensor
            self.layer[i]['input_buf_add_offset'] = self.__getBufferAddress(self.layer[i]['input_buf_add'],  self.__flatsize(self.layer[i], "input"))
            if self.layer[i]['op'] == 'DEPTHWISE_CONV_2D' and self.USE_INPLACE:
                if self.layer[i]['output_buf_add'] == 'front':
                    self.layer[i]['output_buf_add_offset'] = self.__getBufferAddress('end',  self.__flatsize(self.layer[i], "output"))
                else:
                    self.layer[i]['output_buf_add_offset'] = self.__getBufferAddress('front',  self.__flatsize(self.layer[i], "output"))
            else:
                self.layer[i]['output_buf_add_offset'] = self.__getBufferAddress(self.layer[i]['output_buf_add'],  self.__flatsize(self.layer[i], "output"))
            if self.layer[i]['op'] == 'ADD':# two inputs
                self.layer[i]['input2_buf_add_offset'] = self.__getBufferAddress(self.layer[i]['input2_buf_add'],  self.__flatsize(self.layer[i], "input2"))
            else:
                # (2) enlarge intermediate buffers
                if self.layer[i]['op'] == 'DEPTHWISE_CONV_2D':
                    if self.USE_INPLACE:
                        im2col_size = 2 * (self.layer[i]['input_h'] + 2 * self.layer[i]['padding']) * (self.layer[i]['input_w'] + 2 * self.layer[i]['padding'])
                        kernel_size = 0
                    else:
                        im2col_size = 2 * self.layer[i]['kernel_h'] * self.layer[i]['kernel_w'] * self.layer[i]['input_c'] # 16 bit
                        kernel_size = 2 * (self.layer[i]['kernel_h'] * self.layer[i]['kernel_w'] + 1) * self.layer[i]['input_c'] # 16 bit
                    weight_size = self.layer[i]['kernel_h'] * self.layer[i]['kernel_w'] * self.layer[i]['input_c'] 

                    self.__enlargeBuffer('im2col', im2col_size)
                    self.__enlargeBuffer('kernel', kernel_size)

                    ## For detailed memory
                    self.layermem[i]['runtime'] = kernel_size + im2col_size
                    self.layermem[i]['weight'] = weight_size 
                    self.layermem[i]['bias'] = 4 * self.layer[i]['output_c'] # bias
                    self.layermem[i]['scale'] = 8 * self.layer[i]['output_c'] # shift and multiplier

                    self.__increaseFlash(weight_size)
                    self.__increaseFlash(3 * 4 * self.layer[i]['output_c'])# 32-bit bias, shift, multiplier
                elif self.layer[i]['op'] == 'CONV_2D':          
                    im2col_size = 2 * 2 * self.layer[i]['kernel_h'] * self.layer[i]['kernel_w'] * self.layer[i]['input_c'] # 16 bit
                    if self.layer[i]['kernel_h'] == 1:
                        kernel_size = 0
                    else:
                        kernel_size = 2 * self.layer[i]['kernel_h'] * self.layer[i]['kernel_w'] * self.layer[i]['input_c'] * self.layer[i]['output_c'] # 16 bit
                    weight_size = self.layer[i]['kernel_h'] * self.layer[i]['kernel_w'] * self.layer[i]['input_c'] * self.layer[i]['output_c']
                    
                    self.__enlargeBuffer('im2col', im2col_size)
                    self.__enlargeBuffer('kernel', kernel_size)
                    ## For detailed memory
                    self.layermem[i]['runtime'] = kernel_size + im2col_size
                    self.layermem[i]['weight'] = weight_size 
                    self.layermem[i]['bias'] = 4 * self.layer[i]['output_c'] # bias
                    self.layermem[i]['scale'] = 8 * self.layer[i]['output_c'] # shift and multiplier

                    self.__increaseFlash(weight_size)
                    self.__increaseFlash(3* 4 * self.layer[i]['output_c'])# 32-bit bias, shift, multiplier
                elif self.layer[i]['op'] == 'FULLY_CONNECTED':   
                    weight_size = self.layer[i]['input_c'] * self.layer[i]['output_c'] 
                    
                    ## For detailed memory
                    self.layermem[i]['weight'] = weight_size
                    self.layermem[i]['bias'] = 4 * self.layer[i]['output_c'] # bias

                    self.__increaseFlash(weight_size)
                    self.__increaseFlash(4 * self.layer[i]['output_c'])# 32-bit bias
            
        self.peakmem = self.buffers['im2col'] + self.buffers['kernel'] + self.buffers['input_output'] + self.buffers['residual'] 

    def dumpLayerMem(self):
        # header
        print("--------------------------------------------  Schedule Details --------------------------------------------")
        print("----------------------|                      SRAM                      ||              Flash               |")
        print("----------------------|  activation  |  runtime  |  residual  |  sum   ||   weight   |   bias   |  scale   |")

        string = "-------Schedule-------|"
        maxActive = self.buffers['input_output'] 
        maxRuntime = self.buffers['im2col'] + self.buffers['kernel']
        maxResidual = self.buffers['residual'] 
        maxWeight = self.__sumKey(self.layermem,'weight')
        maxBias = self.__sumKey(self.layermem,'bias')
        maxScale = self.__sumKey(self.layermem,'scale')
        string += str(maxActive).ljust(14) + "|"
        string += str(maxRuntime).ljust(11) + "|"
        string += str(maxResidual).ljust(12) + "|"
        string += str(maxActive+maxRuntime+maxResidual).ljust(8) + "||"
        string += str(maxWeight).ljust(12) + "|"
        string += str(maxBias).ljust(10) + "|"
        string += str(maxScale).ljust(10) + "|"
        print(string)
        for i in range(len(self.layermem)):
            string = ""
            string += str(i) + ":" + self.layer[i]['op']
            string = string.ljust(22) + "|"
            SRAM = 0
            if "activation" in self.layermem[i]:
                substr = str(self.layermem[i]['activation']) + " (" + "{:.0%}".format(self.layermem[i]['activation']/maxActive) + ")"
                string += substr.ljust(14) + "|"
                SRAM += self.layermem[i]['activation']
            if "runtime" in self.layermem[i]:
                substr = str(self.layermem[i]['runtime']) + " (" + "{:.0%}".format(self.layermem[i]['runtime']/maxRuntime) + ")"
                string += substr.ljust(11) + "|"
                SRAM += self.layermem[i]['runtime']
            else:
                #SRAM end
                string = string.ljust(49) + "|"
            if "residual" in self.layermem[i]:
                substr = str(self.layermem[i]['residual']) + "(" + "{:.0%}".format(self.layermem[i]['residual']/maxResidual) + ")"
                string += substr.ljust(12) + "|"
                SRAM += self.layermem[i]['residual']
            else:
                #SRAM end
                string = string.ljust(62) + "|"

            string += str(SRAM)
            string = string.ljust(71) + "||"

            if "weight" in self.layermem[i]:
                substr = str(self.layermem[i]['weight']) + " (" + "{:.0%}".format(self.layermem[i]['weight']/maxWeight) + ")"
                string += str(substr).ljust(12) + "|"
            if "bias" in self.layermem[i]: 
                substr = str(self.layermem[i]['bias']) + " (" + "{:.0%}".format(self.layermem[i]['bias']/maxBias) + ")"
                string += str(substr).ljust(10) + "|"
            if "scale" in self.layermem[i]:
                substr = str(self.layermem[i]['scale']) + " (" + "{:.0%}".format(self.layermem[i]['scale']/maxScale) + ")"
                string += str(substr).ljust(10) + "|"
            print(string)
        pass

    def __sumKey(self, layers, key):
        result = 0
        for l in range(len(layers)): 
            if key in layers[l]: 
                result += layers[l][key]

        return result

    def getBuffers(self):
        return self.buffers
    
    # Maximum binary size: This should be updated if any change in the inference side
    # TODO: Combine with code generation to get more accurate result
    BINARAY_PRESERVE = 110 * 1024
    def profileResult(self):
        return self.peakmem , self.flash + self.BINARAY_PRESERVE

    # private functions
    def __increaseFlash(self, size):
        self.flash += size

    def __getBufferAddress(self, location, tensorSize):
        if location == 'front':
            return 0
        elif location == 'end':
            return self.buffers['input_output'] - tensorSize
        elif location == 'residual':
            return 0
        else:
            assert 1 == 0, "unexpected tensor location"

    def __flatsize(self, params, target_str):
        ret_size = 0

        if target_str == "input":
            if params['input_dim'] == 3:
                ret_size = params['input_h'] * params['input_w'] * params['input_c']
            elif params['input_dim'] == 2:
                ret_size = params['input_h'] * params['input_c']
        elif target_str == "input2":
            if params['input2_dim'] == 3:
                ret_size = params['input2_h'] * params['input2_w'] * params['input_c']
            elif params['input2_dim'] == 2:
                ret_size = params['input2_h'] * params['input_c']
        elif target_str == "output":
            if params['output_dim'] == 3:
                ret_size = params['output_h'] * params['output_w'] * params['output_c']
            elif params['output_dim'] == 2:
                ret_size = params['output_h'] * params['output_c']

        return ret_size

    def __enlargeBuffer(self, buf_str, size):        
         self.buffers[buf_str] = max(self.buffers[buf_str], size)
         