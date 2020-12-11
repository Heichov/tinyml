import sys
import getopt
import os

from .TfliteConvertor import TfliteConvertor
from .MemoryScheduler import MemoryScheduler
from .CodeGenerator import CodeGenerator
from .JsonParser import JsonParser
from .TFMicroMemSimulator import TFMicroMemSimulator

MemoryScheduler.USE_INPLACE = True  # set it to use inplace depthwise update


def main(argv):
    try:
        opts, args = getopt.getopt(
            argv, '-h-t:-p:-m-d-g-T', ["help", "tflite", "pickle", "memory profile", "code generation", "TF-lite Micro memory profile"])
    except getopt.GetoptError:
        print('toolchain.py -t <inputfile> -m (optional: memory profiling) -g (optional: code generation) -T (optional: profile TF-lite Micro)')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('toolchain.py -t <inputfile> -m (optional: memory profiling) -g (optional: code generation) -T (optional: profile TF-lite Micro)')
            sys.exit()
        elif opt in ("-t"):
            tf_convertor = TfliteConvertor(arg)
            tf_convertor.parseOperatorInfo()
            memory_scheduler = MemoryScheduler(tf_convertor.layer)
            memory_scheduler.allocateMemory()
            sram, flash = memory_scheduler.profileResult()
            # For TFMicro
            TFMicroMemory = TFMicroMemSimulator(arg)
            TFMicroMemory.simulateMemoryAllocation()
            TFMicroSRAM = TFMicroMemory.getActivationMemory()
            TFMicroFlash = TFMicroMemory.getFlash()
        elif opt in ("-p"):
            pickle_parser = JsonParser(arg)
            pickle_parser.loadPickleModel()
            memory_scheduler = MemoryScheduler(pickle_parser.layer)
            memory_scheduler.allocateMemory()
            sram, flash = memory_scheduler.profileResult()
        elif opt in ("-m"):
            print("SRAM usage:" + str(sram) + ", Flash usage:" + str(flash))
        elif opt in ("-d"):
            memory_scheduler.dumpLayerMem()
        elif opt in ("-g"):
            code_generator = CodeGenerator(memory_scheduler, memory_scheduler.USE_INPLACE)
            code_generator.codeGeneration()
        elif opt in ("-T"):
            print("TFMicro SRAM usage:" + str(TFMicroSRAM) + ", Model size:" + str(TFMicroFlash))


if __name__ == '__main__':
    main(sys.argv[1:])
