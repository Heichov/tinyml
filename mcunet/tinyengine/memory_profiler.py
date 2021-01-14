# classes


def profile_tinyengine(model, data_shape=(1, 3, 224, 224)):
    from utils import get_network_config_with_activation_shape
    cfg = get_network_config_with_activation_shape(model, device='cpu', data_shape=data_shape)
    from .JsonParser import JsonParser
    from .MemoryScheduler import MemoryScheduler
    pickle_parser = JsonParser(cfg)
    pickle_parser.loadPickleModel()
    memory_scheduler = MemoryScheduler(pickle_parser.layer)
    memory_scheduler.allocateMemory()
    sram, flash = memory_scheduler.profileResult()
    return sram, flash


def profile_tflite(model, data_shape=(1, 3, 224, 224)):
    from tinynas.tf_codebase.generate_tflite import generate_tflite_with_weight
    # 1. convert the pt model to tflite with dummy weights
    import torch
    dummy_loader = [(torch.randn(*data_shape), None)]
    import random
    tmp_tflite_path = './tmp-{}.tflite'.format(random.randint(0, 99999999))
    # NOTE: this one is important, using dummy weight (e.g., all 0 bias) could lead to smaller size
    for p in model.parameters():
        p.data.random_()
    generate_tflite_with_weight(model, data_shape[-1], tmp_tflite_path, dummy_loader, n_calibrate_sample=1)

    # 2. evaluate
    from .TfliteConvertor import TfliteConvertor
    from .TFMicroMemSimulator import TFMicroMemSimulator
    from .MemoryScheduler import MemoryScheduler

    original_flag = MemoryScheduler.USE_INPLACE
    MemoryScheduler.USE_INPLACE = False
    tf_convertor = TfliteConvertor(tmp_tflite_path)
    tf_convertor.parseOperatorInfo()
    memory_scheduler = MemoryScheduler(tf_convertor.layer)
    memory_scheduler.allocateMemory()
    # sram, flash = memory_scheduler.profileResult()
    # For TFMicro
    TFMicroMemory = TFMicroMemSimulator(tmp_tflite_path)
    TFMicroMemory.simulateMemoryAllocation()
    TFMicroSRAM = TFMicroMemory.getActivationMemory()
    TFMicroFlash = TFMicroMemory.getFlash()
    # clean up
    import os
    os.system('rm {}'.format(tmp_tflite_path))
    MemoryScheduler.USE_INPLACE = original_flag
    return TFMicroSRAM, TFMicroFlash

