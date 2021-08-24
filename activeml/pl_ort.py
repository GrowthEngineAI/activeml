

import onnx
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from deepsparse import Engine, compile_model, cpu

from transformers.pipelines.base import (
    ArgumentHandler,
    PipelineException,
    infer_framework_load_model,
    PreTrainedFeatureExtractor
)
from .types import *
from ._base import NamedTemporaryFile, Pipeline, PreTrainedTokenizer, ModelCard, get_logger, File, torch, PreTrainedModel, TFPreTrainedModel
from .utils_ort import convert_tensorflow_to_onnx, convert_pytorch_to_onnx, convert_to_onnx, \
    load_onnx_model, save_onnx_model, onnx_inference_session, get_pathlike, PathIOLike, infer_shapes

from .config import SUPPORTED_ENGINES, DEEPSPARSE_ENGINE, ORT_ENGINE, ONNX_CACHE_DIR
from .utils_ops import save_active_trained

logger = get_logger('ActiveML', 'PipelineORT')

def overwrite_transformer_onnx_model_inputs(
    path: str,
    batch_size: int = 1,
    max_length: int = 128,
    output_path: Optional[str] = None,
    ) -> Tuple[Optional[str], List[str], Optional[NamedTemporaryFile]]:
    """
    Overrides an ONNX model's inputs to have the given batch size and sequence lengths.
    Assumes that these are the first and second shape indices of the given model inputs
    respectively
    :param path: path to the ONNX model to override
    :param batch_size: batch size to set
    :param max_length: max sequence length to set
    :param output_path: if provided, the model will be saved to the given path,
        otherwise, the model will be saved to a named temporary file that will
        be deleted after the program exits
    :return: if no output path, a tuple of the saved path to the model, list of
        model input names, and reference to the tempfile object will be returned
        otherwise, only the model input names will be returned
    """
    # overwrite input shapes
    model = load_onnx_model(path)
    #onnx.load(path)
    initializer_input_names = set([node.name for node in model.graph.initializer])
    external_inputs = [inp for inp in model.graph.input if inp.name not in initializer_input_names]
    input_names = []
    for external_input in external_inputs:
        external_input.type.tensor_type.shape.dim[0].dim_value = batch_size
        external_input.type.tensor_type.shape.dim[1].dim_value = max_length
        input_names.append(external_input.name)

    # Save modified model
    if output_path is None:
        tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
        onnx.save(model, tmp_file.name)
        return tmp_file.name, input_names, tmp_file
    else:
        save_onnx_model(model, output_path)
        #onnx.save(model, output_path)
        return input_names


def _create_model(
    model_path: str,
    engine_type: str,
    num_cores: Optional[int],
    num_sockets: Optional[int],
    provider: str = None,
    max_length: int = 128) -> Tuple[Union[Engine, "InferenceSession"], List[str]]:

    onnx_path, input_names, _ = overwrite_transformer_onnx_model_inputs(model_path, max_length=max_length)
    if engine_type == DEEPSPARSE_ENGINE:
        model = compile_model(onnx_path, batch_size=1, num_cores=num_cores, num_sockets=num_sockets)
    elif engine_type == ORT_ENGINE:
        assert provider in get_all_providers(), f"Provider {provider} not found, {get_all_providers()}"
        sess_options = SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (GraphOptimizationLevel.ORT_ENABLE_ALL)
        model = onnx_inference_session(onnx_path, sess_options, providers=[provider])

    return model, input_names

def create_model_for_provider(model_path: Union[str, PathIOLike], provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"Provider {provider} not found, {get_all_providers()}"
    model_path = get_pathlike(model_path)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    
    model_path_or_data = model_path.read_bytes() if model_path.is_gcs else model_path.as_posix()
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path_or_data, options, providers=[provider])
    session.disable_fallback()

    return session


class ActivePipeline(Pipeline):
    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        engine_type: str = ORT_ENGINE,
        graph_path: Optional[PathIOLike] = None,
        ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            feature_extractor=feature_extractor,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
        )
        self.onnx = engine_type is not None
        #bool(engine_type == 'onnx')
        self.engine_type = engine_type
        self.graph_path = get_pathlike(graph_path)
    
        if self.onnx:
            input_names_path = self.graph_path.parent.joinpath(self.graph_path.name + '.input_names.json')
                #f"{os.path.basename(graph_path)}.input_names.json")
            if not graph_path.exists() or not input_names_path.exists(): self._export_onnx_graph(input_names_path)
            logger.info(f"Loading onnx graph from {self.graph_path.as_posix()}")
            self.onnx_model = create_model_for_provider(self.graph_path, "CPUExecutionProvider")
            self.input_names = File.jsonload(str(input_names_path))
            self.framework = "np"
            self._warmup_onnx_graph()
        
    
    def save_pretrained(self, save_directory: str):
        """
        Save the pipeline's model and tokenizer.
        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
        save_active_trained(save_directory, self.model, self.tokenizer, self.feature_extractor, self.modelcard, overwrite=True)

    def _export_onnx_graph(self, input_names_path: Union[str, PathIOLike]):
        # if graph exists, but we are here then it means something went wrong in previous load
        # so delete old graph
        if self.graph_path.exists(): self.graph_path.unlink()

        input_names_path = get_pathlike(input_names_path)
        if input_names_path.exists(): input_names_path.unlink()
        
        # create parent dir
        if not self.graph_path.parent.exists(): self.graph_path.parent.mkdir(parents=True, exist_ok=True)
       
        logger.info(f"Saving onnx graph at { self.graph_path.as_posix()}")

        if self.framework == "pt":
            convert_pytorch_to_onnx(self, opset=11, output=self.graph_path, use_external_format=False)
        else:
            convert_tensorflow_to_onnx(self, opset=11, output=self.graph_path)

        # save input names
        self.input_names = infer_shapes(self, "pt")[0]
        File.jsondump(self.input_names, input_names_path)

    def _forward_engine(self, inputs, return_tensors=False):
        if self.engine_type == ORT_ENGINE:
            inputs = {k: v for k, v in inputs.items() if k in self.input_names}
            return self.onnx_model.run(None, inputs)
        elif self.engine_type == DEEPSPARSE_ENGINE:
            return self.onnx_model.run([inputs[name] for name in self.input_names])

    def _warup_onnx_graph(self, n=10):
        model_inputs = self.tokenizer("My name is Bert", return_tensors="np")
        for _ in range(n):
            self._forward_engine(model_inputs)
    
    def __call__(self, inputs, *args, **kwargs):
        try:
            model_inputs = self._parse_and_tokenize(inputs, *args, **kwargs)
            return self._forward_engine(model_inputs) if self.onnx else self._forward(model_inputs)
        except ValueError:
            # XXX: Some tokenizer do NOT have a pad token, hence we cannot run the inference
            # in a batch, instead we run everything sequentially
            if isinstance(inputs, list):
                values = []
                for input_ in inputs:
                    model_input = self._parse_and_tokenize(input_, padding=False, *args, **kwargs)
                    value = self._forward_engine(model_input) if self.onnx else self._forward(model_input)
                    values.append(value.squeeze(0))
            else:
                model_input = self._parse_and_tokenize(inputs, padding=False, *args, **kwargs)
                values = self._forward_engine(model_input) if self.onnx else self._forward(model_input)
            return values

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching
        Args:
            inputs: dict holding all the keyword arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array
        Returns:
            Numpy array
        """
        with self.device_placement():
            if self.framework == "tf":
                predictions = self.model(inputs.data, training=False)[0]
            else:
                with torch.no_grad():
                    inputs = self.ensure_tensor_on_device(**inputs)
                    predictions = self.model(**inputs)[0].cpu()

        if return_tensors:
            return predictions
        else:
            return predictions.numpy()

