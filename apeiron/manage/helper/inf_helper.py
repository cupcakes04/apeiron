from apeiron.utils import read_json, save_json, inc_str_suffix, mkdir, deep_get, mkdir, convert_to_list
from typing import Literal, List
from pathlib import Path


def is_manifest_match(target_manifest, current_manifest, can_train=True):
    
    def is_dicts_diff(tgt: dict, cur: dict, keys):
        return any(tgt.get(key) != cur.get(key) for key in keys)

    # 1. Must match these exactly
    keys = ['ext_enc', 'ext_mpp', 'ext_model', 'in_features', 'inf_models']
    if is_dicts_diff(target_manifest, current_manifest, keys=keys):
        return False

    # 2. Match annotation and label configs
    if can_train:
        keys = ['lbl_class_id_map', 'lbl_loss_type', 'lbl_cls_weights', 'ann_class_id_map', 'ann_loss_type', 'ann_cls_weights']  
    else:
        keys = ['lbl_class_id_map', 'ann_class_id_map']
    if is_dicts_diff(target_manifest, current_manifest, keys=keys):
        return False

    # 3. Match feats_configs
    tgt_feats_configs = target_manifest['feats_configs']
    cur_feats_configs = current_manifest['feats_configs']
    
    if tgt_feats_configs['window_level'] == 'patch':
        if is_dicts_diff(tgt_feats_configs, cur_feats_configs, keys=['window_level']):
            return False
    else:
        if is_dicts_diff(tgt_feats_configs, cur_feats_configs, keys=['patch_to_tile']):
            return False
    
    # 4. Match Optimizer
    if can_train:
        keys = ['lr', 'optimizer', 'weight_decay', 'scheduler']
        if is_dicts_diff(target_manifest, current_manifest, keys=keys):
            return False
    
    # All matches (no dicts different)
    return True


def find_epoch(valid_history, load_epoch: Literal['best', 'last'] = 'best'):
    if isinstance(load_epoch, (int, tuple)):
        return int(load_epoch)
        
    final_loss = None
    chosen_epoch = 0
    for epoch, history in valid_history.items():
        epoch = int(epoch)
        new_final_loss = deep_get(history, keys=['loss', 'composite', 'final_loss'])
        if load_epoch == 'last':
            continue
        elif final_loss is None or new_final_loss < final_loss:
            final_loss = new_final_loss
            chosen_epoch = epoch
    return chosen_epoch


def new_chkp_path(chkp_path, epoch):
    """also mkdir if doesnt exists"""
    chkp_path = Path(chkp_path)
    new_stem = inc_str_suffix(chkp_path.stem, epoch)
    mkdir(chkp_path.parent)
    return chkp_path.parent / (new_stem + chkp_path.suffix)


def get_inf_metadata(inferencer_folder, cur_configs=None, 
                     load_epoch: int | float | Literal["best", "last"] = 'best',
                     inf_id: int | str = None):

    inferencer_folder = Path(inferencer_folder)
    mkdir(inferencer_folder)
    manifest = read_json(inferencer_folder / 'manifest.json')

    matched = False
    mnf_id = 'inf_0'
    
    # --- 1. Targeted Load (inf_id priority) ---
    if inf_id is not None:
        target_key = f"inf_{inf_id}" if isinstance(inf_id, int) else inf_id
        
        if target_key in manifest:
            mnf_id = target_key
            # We override cur_configs with whatever was stored in this ID
            cur_configs = manifest[mnf_id].get('configs', cur_configs)
            valid_history = manifest[mnf_id].get('valid_history', {})
            
            cur_epoch = find_epoch(valid_history, load_epoch)
            chkp_path = inferencer_folder / mnf_id / f'checkpoint_{cur_epoch}.pth'
            matched = True
            print(f"Force-loading {mnf_id}. Configs updated to match manifest.")
        else:
            # Requested ID doesn't exist, we will create it as a new entry
            mnf_id = target_key

    # --- 2. Automatic Match (Fallback if no inf_id or ID not found) ---
    if not matched:
        # Search for an existing config match
        for m_id, tgt_manifest in manifest.items():
            tgt_configs = tgt_manifest.get('configs', {})
            if is_manifest_match(tgt_configs, cur_configs, can_train=True):
                valid_history = tgt_manifest.get('valid_history', {})
                cur_epoch = find_epoch(valid_history, load_epoch)
                chkp_path = inferencer_folder / m_id / f'checkpoint_{cur_epoch}.pth'
                mnf_id = m_id
                matched = True
                break
        
        # --- 3. Finalizing New Entry (If still no match) ---
        if not matched:
            # If inf_id wasn't provided, increment from the last known ID
            if inf_id is None:
                # Find the highest existing ID to increment correctly
                existing_ids = [k for k in manifest.keys() if k.startswith('inf_')]
                last_id = sorted(existing_ids, key=lambda x: int(x.split('_')[1]))[-1] if existing_ids else 'inf_0'
                mnf_id = inc_str_suffix(last_id)
            
            manifest.setdefault(mnf_id, {})
            manifest[mnf_id]['configs'] = cur_configs
            cur_epoch = 0 
            chkp_path = inferencer_folder / mnf_id / f'checkpoint_{cur_epoch}.pth'

    print(f"Result -> mnf_id: {mnf_id}, epoch: {cur_epoch}")
    # Return cur_configs so the calling script updates its state
    return chkp_path, manifest, mnf_id, cur_epoch, cur_configs