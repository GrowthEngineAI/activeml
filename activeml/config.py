import os
from fileio import PathIO, File


ONNX_CACHE_DIR = PathIO(os.environ.get('ONNX_CACHE_DIR', File.join(File.get_root(__file__), '.onnx')))
ONNX_CACHE_DIR.mkdir(exist_ok=True, parents=True)

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]