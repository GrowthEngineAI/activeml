
from io import BytesIO
from onnx import save_model as _save_onnx_model
from onnx import load_model as _load_onnx_model
from onnx import load as load_onnx
from onnxruntime import InferenceSession, SessionOptions

from .types import *
from ._base import is_tf_available, is_torch_available, NamedTemporaryFile, infer_shapes, Pipeline, ensure_valid_input, load_graph_from_args
from .utils import logger, Union, PathIO, PathIOLike, get_pathlike, lazy_init, generate_identified_filename



def load_onnx_model(filepath: Union[str, PathIOLike], model_cls=None, load_external_data=True):
    filepath = get_pathlike(filepath)
    logger.info(f'Loading ONNX Model from {filepath.as_posix()}')
    if filepath.is_gcs:
        return _load_onnx_model(filepath.read_bytes(), format=model_cls, load_external_data=load_external_data)
    return _load_onnx_model(filepath.as_posix(), format=model_cls, load_external_data=load_external_data)


def onnx_inference_session(filepath: Union[str, PathIOLike], sess_opts=None, **kwargs):
    filepath = get_pathlike(filepath)
    sess_opts = SessionOptions if sess_opts is None else sess_opts
    if filepath.is_gcs:
        return InferenceSession(filepath.read_bytes(), sess_opts, **kwargs)
    return InferenceSession(filepath.as_posix(), sess_opts, **kwargs)


def save_onnx_model(model, filepath: Union[str, PathIOLike]):
    filepath = get_pathlike(filepath)
    if filepath.is_gcs:
        model_data = BytesIO()
        _save_onnx_model(model, model_data)
        filepath.write_bytes(model_data.getvalue())
    else:
        _save_onnx_model(model, filepath)
    logger.info(f'Saved Model to {filepath.as_posix()}')


def convert_pytorch_to_onnx(nlp: Pipeline, opset: int, output: Union[str, PathIOLike], use_external_format: bool):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR
    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB
    Returns:
    """
    if not is_torch_available():
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")

    import torch
    from torch.onnx import export

    logger.info(f"Using framework PyTorch: {torch.__version__}")

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
        ordered_input_names, model_args = ensure_valid_input(nlp.model, tokens, input_names)
        output = get_pathlike(output)
        #f_writer = gfile(output.as_posix(), 'wb+')
        export(nlp.model, model_args, f=output.write_bytes, input_names=ordered_input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, do_constant_folding=True, use_external_data_format=use_external_format, enable_onnx_checker=True, opset_version=opset,
        )


def convert_tensorflow_to_onnx(nlp: Pipeline, opset: int, output: Union[str, PathIOLike]):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR
    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow
    """
    if not is_tf_available():
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")

    logger.warning("/!\\ Please note TensorFlow doesn't support exporting model > 2Gb /!\\")

    #gfile(output.as_posix(), 'wb+')
    lazy_init('keras2onnx')
    try:
        import tensorflow as tf
        from keras2onnx import __version__ as k2ov
        from keras2onnx import convert_keras

        logger.info(f"Using framework TensorFlow: {tf.version.VERSION}, keras2onnx: {k2ov}")

        # Build
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "tf")

        # Forward
        nlp.model.predict(tokens.data)
        onnx_model = convert_keras(nlp.model, nlp.model.name, target_opset=opset)
        save_onnx_model(onnx_model, output)


    except ImportError as e:
        raise Exception(f"Cannot import {e.name} required to convert TF model to ONNX. Please install {e.name} first.")


def convert_to_onnx(framework: str, model: str, output: Union[str, PathIOLike],
    opset: int, tokenizer: Optional[str] = None,
    use_external_format: bool = False, pipeline_name: str = "feature-extraction",
    overwrite: bool = True, **model_kwargs
):
    """
    Convert the pipeline object to the ONNX Intermediate Representation (IR) format
    Args:
        framework: The framework the pipeline is backed by ("pt" or "tf")
        model: The name of the model to load for the pipeline
        output: The path where the ONNX graph will be stored
        opset: The actual version of the ONNX operator set to use
        tokenizer: The name of the model to load for the pipeline, default to the model's name if not provided
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB (PyTorch only)
        pipeline_name: The kind of pipeline to instantiate (ner, question-answering, etc.)
        model_kwargs: Keyword arguments to be forwarded to the model constructor
    Returns:
    """
    logger.info(f"ONNX opset version set to: {opset}")

    # Load the pipeline
    nlp = load_graph_from_args(pipeline_name, framework, model, tokenizer, **model_kwargs)
    output = get_pathlike(output)
    if not output.parent.exists():
        logger.info(f"Creating folder {output.parent}")
        output.mkdir(parents=True, exist_ok=True)

    elif len([f for f in output.glob('*')]) > 0 and not overwrite:
        raise Exception(f"Folder {output.parent.as_posix()} is not empty, and Overwrite = False. Aborting conversion")

    # Export the graph
    if framework == "pt":
        convert_pytorch_to_onnx(nlp, opset, output, use_external_format)
    else:
        convert_tensorflow_to_onnx(nlp, opset, output)


def optimize_onnx_model(onnx_model_path: Union[str, PathIOLike]) -> PathIOLike:
    """
    Load the model at the specified path and let onnxruntime look at transformations on the graph to enable all the
    optimizations possibl
    Args:
        onnx_model_path: filepath where the model binary description is stored
    Returns: Path where the optimized model binary description has been saved
    """
    # Generate model name with suffix "optimized"
    opt_model_path = generate_identified_filename(onnx_model_path, "-optimized")
    sess_option = SessionOptions()
    if opt_model_path.is_gcs:
        with NamedTemporaryFile() as tmp:
            logger.info(f'Writing Temporary File: {tmp.name}')
            sess_option.optimized_model_filepath = tmp.name
            _ = InferenceSession(onnx_model_path.read_bytes(), sess_option)
            opt_model_path.write_bytes(PathIO(tmp.name).read_bytes())

    else:
        sess_option.optimized_model_filepath = opt_model_path.as_posix()
        _ = InferenceSession(onnx_model_path.as_posix(), sess_option)

    logger.info(f"Optimized model has been written at {opt_model_path}: \N{heavy check mark}")
    logger.info("/!\\ Optimized model contains hardware specific operators which might not be portable. /!\\")

    return opt_model_path


def quantize_onnx_model(onnx_model_path: Union[str, PathIOLike]) -> PathIOLike:
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    from onnxruntime.quantization import QuantizationMode, quantize

    onnx_model = load_onnx_model(onnx_model_path)

    # Discussed with @yufenglee from ONNX runtime, this will be address in the next release of onnxruntime
    logger.info(
        "As of onnxruntime 1.4.0, models larger than 2GB will fail to quantize due to protobuf constraint.\n"
        "This limitation will be removed in the next release of onnxruntime."
    )

    quantized_model = quantize(
        model=onnx_model,
        quantization_mode=QuantizationMode.IntegerOps,
        force_fusions=True,
        symmetric_weight=True,
    )

    # Append "-quantized" at the end of the model's name
    quantized_model_path = generate_identified_filename(onnx_model_path, "-quantized")

    # Save model
    logger.info(f"Quantized model has been written at {quantized_model_path}: \N{heavy check mark}")
    save_onnx_model(quantized_model, quantized_model_path)

    return quantized_model_path


def verify_onnx_model(path: Union[str, PathIOLike]):
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
    logger.info(f"Checking ONNX model loading from: {path} ...")
    try:
        _ = onnx_inference_session(path, providers=["CPUExecutionProvider"])
        logger.info(f"Model {path} correctly loaded: \N{heavy check mark}")
    except RuntimeException as re:
        logger.info(f"Error while loading the model {re}: \N{heavy ballot x}")

