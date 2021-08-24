from .types import *
from .pl_ort import ActivePipeline, _create_model
from ._basecls import *
from ._base import TruncationStrategy, tf
from ._base import PretrainedConfig, AutoTokenizer, AutoConfig, PreTrainedTokenizer
from .config import SUPPORTED_ENGINES, DEEPSPARSE_ENGINE

class Text2TextGenerationPipeline(ActivePipeline):
    return_name = "generated"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING if self.framework == "tf" else MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)
    
    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        return True

    def _parse_and_tokenize(self, *args, truncation):
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            assert (self.tokenizer.pad_token_id is not None), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`")
        inputs = super()._parse_and_tokenize(*args, padding=padding, truncation=truncation)
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs: del inputs["token_type_ids"]
        return inputs

    def __call__(
        self,
        *args,
        return_tensors=False,
        return_text=True,
        clean_up_tokenization_spaces=False,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        **generate_kwargs
    ):
        r"""
        Generate the output text(s) using text(s) given as inputs.
        Args:
            args (:obj:`str` or :obj:`List[str]`):
                Input text for the encoder.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).
        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:
            - **generated_text** (:obj:`str`, present when ``return_text=True``) -- The generated text.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated text.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        with self.device_placement():
            inputs = self._parse_and_tokenize(*args, truncation=truncation)
            return self._generate(inputs, return_tensors, return_text, clean_up_tokenization_spaces, generate_kwargs)

    def _generate(
        self, inputs, return_tensors: bool, return_text: bool, clean_up_tokenization_spaces: bool, generate_kwargs
    ):
        if self.framework == "pt":
            inputs = self.ensure_tensor_on_device(**inputs)
            input_length = inputs["input_ids"].shape[-1]
        elif self.framework == "tf":
            input_length = tf.shape(inputs["input_ids"])[-1].numpy()

        min_length = generate_kwargs.get("min_length", self.model.config.min_length)
        max_length = generate_kwargs.get("max_length", self.model.config.max_length)
        self.check_inputs(input_length, min_length, max_length)

        generate_kwargs.update(inputs)
        generations = self.model.generate(**generate_kwargs)
        results = []
        for generation in generations:
            record = {}
            if return_tensors: record[f"{self.return_name}_token_ids"] = generation
            if return_text: record[f"{self.return_name}_text"] = self.tokenizer.decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            results.append(record)
        return results


@dataclass
class TaskInfo:
    """
    Information about an NLP task
    :param pipeline_constructor: reference to constructor for the given pipeline task
    :param default model name: the transformers canonical name for the default model
    :param base_stub: sparsezoo stub path for the base model for this task
    :param default_pruned_stub: sparsezoo stub path for the default pruned model
        for this task
    :param default_quant_stub: sparsezoo stub path for the default quantized model
        for this task
    """

    pipeline_constructor: Callable[[Any], ActivePipeline]
    default_model_name: str
    base_stub: Optional[str] = None
    default_pruned_stub: Optional[str] = None
    default_quant_stub: Optional[str] = None



def activepipeline(
    task: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    engine_type: str = DEEPSPARSE_ENGINE,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    max_length: int = 128,
    num_cores: Optional[int] = None,
    num_sockets: Optional[int] = None,
    **kwargs,
) -> ActivePipeline:
    """
    Utility factory method to build a Pipeline
    :param task: name of the task to define which pipeline to create. Currently
        supported task - "question-answering"
    :param model_name: canonical name of the hugging face model this model is based on
    :param model_path: path to (ONNX) model file to run
    :param engine_type: inference engine name to use. Supported options are 'deepsparse'
        and 'onnxruntime'
    :param config: huggingface model config, if none provided, default will be used
    :param tokenizer: huggingface tokenizer, if none provided, default will be used
    :param max_length: maximum sequence length of model inputs. default is 128
    :param num_cores: number of CPU cores to run engine with. Default is the maximum
        available
    :param num_sockets: number of CPU sockets to run engine with. Default is the maximum
        available
    :param kwargs: additional key word arguments for task specific pipeline constructor
    :return: Pipeline object for the given taks and model
    """
    if engine_type not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unsupported engine {engine_type}, supported engines "
            f"are {SUPPORTED_ENGINES}"
        )

    #task_info = SUPPORTED_TASKS[task]
    #model_path = model_path #or _get_default_model_path(task_info)
    #model_name = model_name #or task_info.default_model_name

    # default the tokenizer and config to given model name
    tokenizer = tokenizer or model_name
    config = config or model_name

    # create model
    #if model_path.startswith("zoo:"):
    #    model_path = _download_zoo_model(model_path)
    model, input_names = _create_model(
        model_path, engine_type, num_cores, num_sockets, max_length
    )

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate config if needed
    if config is not None and isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    task_info = TaskInfo(
        pipeline_constructor=Text2TextGenerationPipeline,
        default_model_name=model_name,
    )
    return task_info.pipeline_constructor(
        model=model,
        tokenizer=tokenizer,
        config=config,
        engine_type=engine_type,
        max_length=max_length,
        input_names=input_names,
        **kwargs,
    )
