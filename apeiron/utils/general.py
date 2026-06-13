from pathlib import Path
import numpy as np
import os
from PIL import Image
import json
import re
import pandas as pd
from typing import Dict, Any
from typing import List, Literal, Union
import matplotlib.pyplot as plt
import tifffile
import torch

def save_and_show_plot(fig, save_path=None, show=True):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def convert_to_list(x):
    """Wrap a scalar value in a list; pass through lists/tuples and None unchanged.

    Args:
        x: Any value. None is returned as-is.

    Returns:
        list or None: ``[x]`` if scalar, ``x`` if already a list/tuple, None if None.
    """
    if x is None:
        return x 
    elif not isinstance(x, (tuple, list, np.ndarray)):
        return [x]
    return x

def to_raw_list(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, (np.ndarray, torch.tensor)):
        return list(x)

def mkdir(dir_path: Path) -> None:
    """Create a directory if it does not exist."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)

def search_dir(folder, name, extensions):
    """Search for the first file matching a stem name with a valid extension.

    Recursively walks ``folder`` and returns the first file whose stem
    matches ``name`` and whose extension is in ``extensions``.

    Args:
        folder (str or Path): Directory to search.
        name (str): Target filename stem (without extension).
        extensions (set[str]): Valid extensions including the dot (e.g. ``{'.json', '.tiff'}``).

    Returns:
        Path or None: Full path to the matching file, or None if not found.
    """
    stem_name = str(name)
    
    for root, _, files in os.walk(folder):
        for fname in files:
            # Use os.path.splitext for speed (it's faster than Path.stem for simple loops)
            name, ext = os.path.splitext(fname)
            
            if name == stem_name and ext in extensions:
                return Path(root) / fname
    return None

def match_file(folder, name, extensions):
    for ext in extensions:
        match = Path(folder) / f'{str(name)}{ext}'
        if match.is_file():
            return match
    return None

def save_img(img_arr, path):
    """Save numpy array as image file.
    
    Args:
        img_arr (np.ndarray): Image array to save
        path (str or Path): Destination file path
    """
    Image.fromarray(img_arr).save(path)
    
def load_img(path):
    """Load image file as numpy array.
    
    Args:
        path (str or Path): Image file path
    
    Returns:
        np.ndarray: Loaded image array
    """
    if Path(path).is_file():
        return np.array(Image.open(path))
    else:
        return None

def load_tiff(path):
    """Load a TIFF file as a numpy array.

    Args:
        path (str or Path): Path to the TIFF file.

    Returns:
        np.ndarray or None: Loaded image array, or None if file does not exist.
    """
    if Path(path).is_file():
        return tifffile.imread(path)
    else:
        return None
    
def read_json(json_file):
    """Read JSON file and return parsed data.
    
    Args:
        json_file (str or Path): Path to JSON file
    
    Returns:
        dict or list: Parsed JSON data
    """
    if json_file and Path(json_file).is_file():
        with open(json_file, "r") as file:
            return json.load(file)
    else:
        return {}
        
def save_json(json_file, data, indent=2, io_mode='w'):
    """Save data to JSON file with formatting.
    
    Args:
        json_file (str or Path): Destination file path
        data (dict or list): Data to serialize
        indent (int): JSON indentation level. Default 2
        io_mode ('w' or 'r'): read or write operation (generally is write lol)
    """
    if io_mode == 'w':
        with open(json_file, "w") as file:
            json.dump(data, file, indent=indent)

def inc_str_suffix(s: str, num: int = None) -> str:
    """Increment numeric suffix of a string or add '_1' if none exists.
    
    Args:
        s (str): Input string (e.g., 'file_1', 'data')
    
    Returns:
        str: String with incremented suffix (e.g., 'file_2', 'data_1')
    
    Examples:
        >>> inc_str_suffix('ext_1')
        'ext_2'
        >>> inc_str_suffix('data')
        'data_1'
    """
    # Match trailing digits
    match = re.search(r'(\d+)$', s)
    if match:
        # Replace the old number with the new one
        number = num if num else int(match.group(1)) + 1
        return s[:match.start(1)] + str(number)
    else:
        # If no digits at the end, just append "_1"
        number = num if num else 1
        return s + f"_{number}"
    
def deep_get(dictionary, keys, default=None):
    """Safely navigate nested dictionaries using a list of keys.
    
    Args:
        dictionary (dict): Dictionary to navigate
        keys (list): List of keys to traverse (e.g., ['a', 'b', 'c'])
        default: Value to return if path doesn't exist. Default None
    
    Returns:
        Value at the nested path or default if not found
    
    Example:
        >>> d = {'a': {'b': {'c': 42}}}
        >>> deep_get(d, ['a', 'b', 'c'])
        42
    """
    for key in keys:
        if isinstance(dictionary, dict):
            dictionary = dictionary.get(key, default)
        else:
            return default
    return dictionary

def deep_assign(dictionary, keys, value):
    """Assign value to nested dictionary path, creating intermediate dicts as needed.
    
    Args:
        dictionary (dict): Dictionary to modify (modified in-place)
        keys (list): List of keys defining the path (e.g., ['a', 'b', 'c'])
        value: Value to assign at the path
    
    Example:
        >>> d = {}
        >>> deep_assign(d, ['a', 'b', 'c'], 42)
        >>> d
        {'a': {'b': {'c': 42}}}
    """
    for key in keys[:-1]:  # Go up to the second-to-last key
        # dictionary.setdefault returns the value if key exists, 
        # otherwise sets it to {} and returns {}
        dictionary = dictionary.setdefault(key, {})
    
    # Set the value at the very last key
    dictionary[keys[-1]] = value
    
def read_csv(path):
    """Read a CSV file into a DataFrame, returning None if the file does not exist.

    Args:
        path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame, or None if file is missing.
    """
    if Path(path).is_file():
        return pd.read_csv(path)
    else:
        return None

def update_dict(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Return a shallow copy of ``base`` with ``overrides`` applied.

    Args:
        base (dict): Original dictionary.
        **overrides: Key-value pairs to override or add.

    Returns:
        dict: New dictionary with overrides merged in.
    """
    return {**base, **overrides}

def extend_dict(data, **kwargs):
    # Iterate through the passed arguments
    # Only set if the key doesn't exist
    for key, value in kwargs.items():
        data.setdefault(key, value)
    return data

def get_device(device=None, desc=None):
    if device:
        print(f"{desc} -- Device: {device}")
        return torch.device(device)

    if torch.cuda.is_available():
        print(f"{desc} -- Device: GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    else:
        print(f"{desc} -- Device: CPU")
        return torch.device("cpu")

def np_unsqueeze(x):
    if x is not None:
        return np.array([x])
    return x

def to_cpu(x: torch, numpy=False):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        if numpy:
            return x.numpy()
        else:
            return x
    else:
        return None

def is_truly_empty(x):
    # 1. Check if it's None
    if x is None:
        return True
    
    # 2. Check if it's a collection (list, tuple, etc.)
    if isinstance(x, (list, tuple, set, dict)):
        # If it has length, check if all its elements are also "empty"
        # If it has no length (len == 0), it's empty
        return len(x) == 0 or all(is_truly_empty(i) for i in x)
    
    # 3. If it's not a collection and not None, it has a value
    return False