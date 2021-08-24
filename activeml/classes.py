from .types import *
from .utils import get_pathlike
from abc import ABC, abstractmethod
from itertools import chain


class ModelFramework(str, Enum):
    pt = 'pt'
    torch = 'pt'
    pytorch = 'pt'
    tf = 'tf'
    tensorflow = 'tf'
    dsparse = 'dparse'
    deepsparse = 'dsparse'
    dspeed = 'dspeed'
    deepspeed = 'dspeed'
    onnx = 'onnx'
    ort = 'onnx'

class ModelLoadMethod(str, Enum):
    pt = 'pt'
    torch = 'pt'
    pytorch = 'pt'
    tf = 'tf'
    tensorflow = 'tf'
    dsparse = 'dparse'
    deepsparse = 'dsparse'
    dspeed = 'dspeed'
    deepspeed = 'dspeed'
    onnx = 'onnx'
    ort = 'onnx'
    jax = 'jax'
    flax = 'flax'
    msgpk = 'flax'
    msgpack = 'flax'



ModelExtensions = {
    '.pt': ModelLoadMethod.pytorch,
    '.bin': ModelLoadMethod.pytorch,
    '.h5': ModelLoadMethod.tensorflow,
    '.onnx': ModelLoadMethod.onnx,
}

DefaultModelNames = {
    'tf_model.h5':  ModelLoadMethod.tensorflow,
    'pytorch_model.bin': ModelLoadMethod.pytorch,
    'flax_model.msgpack': ModelLoadMethod.flax
}


@lazyclass
@dataclass
class ModelFile:
    name: str
    src: PathIOLike
    framework: ModelFramework
    directory: PathIOLike
    
    @classmethod
    def load(cls, directory: Union[str, PathIOLike], filename: str = None, recursive=False):
        directory = get_pathlike(directory)
        if not directory.is_dir():
            return ModelFile(name = directory.parents[-1], src = directory, framework = ModelExtensions[directory.suffix], directory=directory.absolute_parent)
        filenames = directory.rglob('*') if recursive else directory.glob('*')
        for fname in filenames:
            if fname.is_file() and fname.suffix in ModelExtensions:
                if filename and filename not in fname.name:
                    continue
                return ModelFile(name = fname.parents[-1], src = fname, framework = ModelExtensions[fname.suffix], directory=directory)
        return None

