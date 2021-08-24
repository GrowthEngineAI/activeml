


import os
import sys
import csv
import numpy as np

from enum import Enum
from itertools import chain
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TypeVar

from lazyops import lazy_init, get_logger, timed_cache
from lazyops.mp_utils import _CPU_CORES as NUM_CPU_CORES
from lazyops.lazyio import gfile, File, LazyFile, LazyPickler, LazyJson

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryFile, NamedTemporaryFile, TemporaryDirectory

from transformers.file_utils import is_tf_available, is_torch_available

if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch
    

if TYPE_CHECKING:
    from transformers.modeling_tf_utils import TFPreTrainedModel
    from transformers.modeling_utils import PreTrainedModel


from transformers.file_utils import ModelOutput
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding
from transformers.configuration_utils import PretrainedConfig
from transformers.convert_graph_to_onnx import convert_pytorch, convert_tensorflow, infer_shapes
from transformers.data import SquadExample, squad_convert_examples_to_features
from transformers.file_utils import add_end_docstrings
from transformers.modelcard import ModelCard

from transformers.models.bert import BasicTokenizer
from transformers.models.auto import AutoConfig, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy

from transformers.convert_graph_to_onnx import (
    ORT_QUANTIZE_MINIMUM_VERSION,
    SUPPORTED_PIPELINES,
    ensure_valid_input,
    load_graph_from_args
)
