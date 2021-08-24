
from tempfile import TemporaryDirectory
from classes import ModelLoadMethod
from utils import *

def load_active_trained(load_directory: Union[str, PathIOLike], model=None, tokenizer=None, method: ModelLoadMethod = ModelLoadMethod.onnx, **kwargs):
    assert method in ModelLoadMethod, 'Invalid Load Method.'



def save_active_trained(save_directory: Union[str, PathIOLike], model=None, tokenizer=None, feature_extractor=None, modelcard=None, overwrite=True, **kwargs):
    save_directory = get_pathlike(save_directory)
    if save_directory.is_file: save_directory = save_directory.absolute_parent()
    save_directory.mkdir(parents=True, exist_ok=True)
    tmp_dir = None
    if save_directory.is_gcs:
        copied_files = []
        tmp_dir = TemporaryDirectory()
        logger.info(f'Saving Model to {tmp_dir.name} as TempDir')
    save_path = tmp_dir.name if tmp_dir else save_directory.as_posix()
    if model is not None: model.save_pretrained(save_path)
    if tokenizer is not None: tokenizer.save_pretrained(save_path)
    if feature_extractor is not None: feature_extractor.save_pretrained(save_path)
    if modelcard is not None: modelcard.save_pretrained(save_path)
    if tmp_dir:
        for fname in PathIO(tmp_dir.name).rglob('*'):
            dest_path = save_directory.joinpath(fname.relative_to(tmp_dir.name))
            if fname.is_file():
                fname.copy(dest_path.as_posix(), overwrite=overwrite)
                copied_files.append(dest_path.as_posix())
            
            elif fname.is_dir():
                dest_path.mkdir(parents=True, exist_ok=True)
        logger.info('Copied Pretrained Active Model')
        for fname in copied_files:
            logger.info(f'- {fname}')
        tmp_dir.cleanup()
    logger.info(f'Saved Pretrained Active Model to {save_directory.as_posix()}')

