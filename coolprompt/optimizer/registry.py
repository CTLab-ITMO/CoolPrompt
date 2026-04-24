from coolprompt.optimizer.hype.hype_method import HyPEMethod
from coolprompt.optimizer.reflective_prompt.reflective_method import ReflectiveMethod
from coolprompt.optimizer.distill_prompt.distill_method import DistillMethod
from coolprompt.optimizer.regps.regps_method import ReGPSMethod
from coolprompt.optimizer.prompt_compressor.compressor_method import CompressorMethod

METHOD_REGISTRY = {
    "hype": HyPEMethod(),
    "reflective": ReflectiveMethod(),
    "distill": DistillMethod(),
    "regps": ReGPSMethod(),
    "compress": CompressorMethod(),
}