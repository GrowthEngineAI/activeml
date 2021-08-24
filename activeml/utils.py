from typing import Union
from fileio import PathIO, PathIOLike
from lazyops import get_logger, lazy_init

logger = get_logger('ActiveML', 'Utils')

def get_pathlike(path: Union[str, PathIOLike]):
    if isinstance(path, str): path = PathIO(path)
    return path

def generate_identified_filename(filename: Union[str, PathIOLike], identifier: str) -> PathIOLike:
    """
    Append a string-identifier at the end (before the extension, if any) to the provided filepath
    Args:
        filename: pathlib.Path The actual path object we would like to add an identifier suffix
        identifier: The suffix to add
    Returns: String with concatenated identifier at the end of the filename
    """
    filename = get_pathlike(filename)
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)
