import h5py

def save_h5_datas(file_path, group_parts, data_dicts, attributes=None):
    """Save numpy arrays to HDF5 file in hierarchical structure.
    
    Organizes data as: extractions/{ext_id}/{sub_ext}/{dataset_name}
    Overwrites existing datasets if they exist.
    
    Args:
        file_path (str or Path): Path to HDF5 file
        group_parts (list[list]): list of [list of parts to form the group_path]
        data_dicts (list[dict)]: list of Dictionary of numpy arrays to save
        attributes (list[dict], optional): list of Metadata to attach as HDF5 attributes
    """
    # If no attributes provided, create empty dicts for each group 
    if attributes is None: 
        attributes = [{} for _ in range(len(group_parts))]
    
    with h5py.File(file_path, 'a') as f:
        for grp_parts, data_dict, attr in zip(group_parts, data_dicts, attributes):

            h5_group_path = "/".join([str(gp) for gp in grp_parts])

            group = f.require_group(h5_group_path)

            # Save each key/value as its own dataset
            for k, v in data_dict.items():
                if k in group:
                    # Delete existing dataset before recreating
                    del group[k]
                group.create_dataset(k, data=v)

            # Attach metadata as attributes
            if attr:
                for k, v in attr.items():
                    group.attrs[k] = v
                    
def save_h5_data(file_path, group_parts, data_dict, attributes=None):
    """Save a single data dict to an HDF5 group. Convenience wrapper around ``save_h5_datas``."""
    save_h5_datas(file_path, [group_parts], [data_dict], attributes=[attributes])


def load_h5_datas(file_path, group_parts):
    """Load numpy arrays and attributes from HDF5 file.
    
    Reads data from: extractions/{ext_id}/{sub_ext}/
    
    Args:
        file_path (str or Path): Path to HDF5 file
        group_parts (list[list]): list of [list of parts to form the group_path]
    """
    data_dicts, attributes = [], []
    with h5py.File(file_path, 'r') as f:
        for grp_parts in group_parts:
            h5_group_path = "/".join([str(gp) for gp in grp_parts])
                
            if h5_group_path not in f:
                print(f"Group {h5_group_path} not found in {file_path}")
                return {}, {}
            
            group = f[h5_group_path]

            # Load datasets into a dict
            data_dicts.append({k: v[()] for k, v in group.items() if isinstance(v, h5py.Dataset)})

            # Load attributes into a dict
            attributes.append(dict(group.attrs))

    return data_dicts, attributes

def load_h5_data(file_path, group_parts):
    """Load a single HDF5 group. Convenience wrapper around ``load_h5_datas``.

    Returns:
        tuple: (data_dict, attributes) for the requested group.
    """
    data_dicts, attributes = load_h5_datas(file_path, [group_parts])
    return data_dicts[0], attributes[0]


def list_h5_paths(file_path):
    """Print the hierarchical structure of an HDF5 file for debugging.

    Args:
        file_path (str or Path): Path to the HDF5 file.
    """
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[Dataset]  {name} (shape: {obj.shape}, type: {obj.dtype})")
        elif isinstance(obj, h5py.Group):
            print(f"[Group]    {name}")

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Structure of {file_path}:")
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error reading file: {e}")