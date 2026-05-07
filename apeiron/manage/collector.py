
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from .manager import Manager
from .searcher import Searcher
from typing import Literal, List
from apeiron.utils import convert_to_list, deep_assign, deep_get
import random
from sklearn.model_selection import train_test_split
from apeiron.utils import np_unsqueeze, extend_dict
from itertools import zip_longest
from .helper.inf_helper import *
from apeiron.utils import to_cpu
from dataclasses import dataclass

@dataclass
class SimDF:
    feat_res: pd.DataFrame | None = None
    roi_res: pd.DataFrame | None = None
    wrd_res: pd.DataFrame | None = None


class Collector(Manager, Searcher):
    """Collects slide and tile features as memory-efficient generators.

    Extends Manager to provide generator-based access to processed features,
    annotations, and labels for downstream training tasks.  Each call to
    ``slide_features_collector`` or ``tile_features_collector`` yields
    dictionaries containing feature arrays, coordinates, annotations, and
    labels — avoiding the need to load everything into memory at once.

    Args:
        **kwargs: Forwarded to Manager (requires ``bacbone``, ``root_dir``,
            ``project_path``).

    Attributes:
        slides_collected_ann (dict): Cache of slide annotations keyed by slide_id.
        tiles_collected_data (dict): Cache of tile features for repeated access.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_collector()
        self.manifest = {}
        self.mnf_id: int = 'inf_1'
        self.inf_mode: Literal['slide', 'tile'] = None
        self.train_history = {}
        self.valid_history = {}
        self.index_built = False

    def reset_collector(self):
        """Reset all collected data caches."""
        self.slides_collected_ann = {}
        self.tiles_collected_data = {}
        self.index_built = False

    # |-----------------------------------------------|
    # |------------ Collect Processed Data -----------|
    # |-----------------------------------------------|

    def slide_features_collector(self, collect_ids: List = None, shuffle=False, batch_size=1, cache=False):
        """Generate per-slide feature dictionaries for downstream training.

        For each selected slide, loads embeddings, computes features at the
        configured window level, and loads annotations.  Yields one or more
        dictionaries per slide depending on annotation bags and grid grouping:

        1. **Bagged regions** — if ``supervision`` produced annotation bags, each
           bag is yielded as a separate pseudo-slide.
        2. **Grid groups** — if ``window_level='grid'``, each grid group is
           yielded separately.
        3. **Full slide** — always yielded as the final entry.

        Args:
            collect_ids (list, optional): Subset of slide UUIDs to collect.
                If None, collects from all selected slides.
            cache (bool): Cache annotations to avoid recomputation on
                repeated calls. Default False.

        Yields:
            dict: Feature dictionary with keys:
                - ``id`` (str): Slide UUID.
                - ``coords`` (np.ndarray): (N, 2) tile coordinates.
                - ``features`` (np.ndarray): (N, F) feature vectors.
                - ``annotation`` (np.ndarray): (N, C) class fractions.
                - ``label`` (np.ndarray): (C,) slide-level label vector.
        """
        if not self.available_modes['slide']: return
        collect_ids = convert_to_list(collect_ids)
        label_cols = [f"class{lbl}" for lbl in list(self.slide_gt_configs['label_configs']['class_id_map'].keys())]
        
        # Filter dataset to get total count
        df_filtered = self.selected_slide_dataset
        if collect_ids:
            df_filtered = df_filtered[df_filtered['slide_id'].astype(str).isin(collect_ids)]
        total_slides = len(df_filtered)

        # Iterate through unique slides in your dataframe
        slide_count = 0
        for _, row in self.shuffle_df(df_filtered, shuffle).iterrows():
            slide_id = str(row['slide_id'])
            slide_label = row[label_cols].values.astype(np.float16)
            slide_label = np_unsqueeze(slide_label)   # (C,) -> (1,C)
            text = row['text'] if 'text' in row else None

            # 1. Initialize Analyzer for the specific slide
            self.serve_slide_analyzer(slide_id, data_modes='embeddings')

            # 2. Process Annotations
            if cache and slide_id in self.slides_collected_ann:
                self.analyzer.load_annotations(**self.slides_collected_ann[slide_id])
            else:
                self.serve_slide_analyzer(slide_id, data_modes='annotation')
                if cache:
                    self.slides_collected_ann[slide_id] = {
                        "annotation": self.analyzer.annotation,
                        "objects": self.analyzer.objects,
                        "supervision": self.analyzer.supervision,
                    }

            # 3. Generators
            slide_count += 1
            propagate_loss = (slide_count % batch_size == 0) or (slide_count == total_slides)
            predata = self.analyzer.slide_preprocessor()
            for data in predata.generator(batch_size=batch_size, shuffle=shuffle, propagate_loss=propagate_loss):
                yield extend_dict(data, id=slide_id, label=slide_label, text=text)


    def tile_features_collector(self, collect_ids: List = None, shuffle=False, batch_size=300, cache=True):
        """Generate per-tile feature dictionaries for downstream training.

        Loads embeddings from all tile classes, computes features, and matches
        labels from the tile dataset.  Supports both standalone and windowed
        tile modes.

        For **standalone** tiles each yield is a single tile::

            {'id': str, 'coords': (2,), 'features': (F,), 'label': (C,)}

        For **windowed** tiles (grouped by window_id) each yield is a
        pseudo-slide::

            {'id': str, 'coords': (N, 2), 'features': (N, F), 'label': (C,)}

        Args:
            collect_ids (list, optional): Subset of tile UUIDs to collect.
                If None, collects from all selected tiles.
            cache (bool): Cache concatenated features to avoid
                recomputation on repeated calls. Default False.

        Yields:
            dict: Feature dictionary (format depends on tile mode).
        """
        if not self.available_modes['tile']: return
        collect_ids = convert_to_list(collect_ids)
        label_cols = [f"class{lbl}" for lbl in list(self.tile_gt_configs['label_configs']['class_id_map'].keys())]

        # 1. Index the dataframe for fast ID-based matching
        df_indexed = self.selected_tile_dataset.set_index('tile_id')
        for tile_class, tile_data in self.tile_data_paths.items():

            # 2. Match Labels using the ID order from the Analyzer
            new_tile_ids = np.array(tile_data['tile_ids'])
            labels = df_indexed.loc[new_tile_ids, label_cols].values.astype(np.float16)
            text = df_indexed.loc[new_tile_ids, 'text'].to_numpy() if 'text' in df_indexed else np.array([None] * len(new_tile_ids))
            
            # 3. Extract Features from Analyzer
            self.serve_tile_analyzer(tile_class)

            # 4. Filter with requested collect_ids
            if collect_ids:
                indices = np.where(np.isin(new_tile_ids, collect_ids))[0]
                if len(indices) == 0:
                    continue
                new_tile_ids = new_tile_ids[indices]
                labels = labels[indices]
            else:
                indices = None
                
            # 5. Create Generator
            if not cache or tile_class not in self.tiles_collected_data:
                self.tiles_collected_data[tile_class] = []
                
                predata = self.analyzer.tile_preprocessor(indices=indices)
                for data in predata.generator(batch_size=batch_size, shuffle=shuffle):
                    split_id = data['split_id']
                    self.tiles_collected_data[tile_class].append(
                        extend_dict(data, id=new_tile_ids[split_id], label=labels[split_id], text=text[split_id])
                    )

        for data in self.random_batch_generator(self.tiles_collected_data, shuffle=shuffle):
            yield data


    @staticmethod
    def shuffle_df(df, shuffle=True):
        if shuffle:
            return df.sample(frac=1).reset_index(drop=True)
        return df

    @staticmethod
    def random_batch_generator(data_dict, shuffle=True):
        """
        Flattens everything and shuffles. 
        Class B will be scattered randomly among Class A.
        """
        # 1. Combine all batches into one big list
        all_batches = []
        for batches in data_dict.values():
            all_batches.extend(batches)
        
        # 2. Shuffle everything
        if shuffle:
            random.shuffle(all_batches)
        
        # 3. Yield them
        for batch in all_batches:
            yield batch


    # |-----------------------------------------------|
    # |--------------- Prepare & Train ---------------|
    # |-----------------------------------------------|


    def get_slide_splits(self, df, mode: Literal['slide', 'tile'], id_col: Literal['slide_id', 'tile_id'], train_ratio=0.8, auto_split=True):
        """
        Splits IDs for training and validation while maintaining class distribution.
        
        Args:
            df (pd.DataFrame): The dataset.
            id_col (str): Column to split on (e.g., 'slide_id').
            train_ratio (float): Fraction for training.
        """
        # Dynamically find the class columns based on your config
        if mode == 'slide':
            label_cols = [f"class{lbl}" for lbl in list(self.slide_gt_configs['label_configs']['class_id_map'].keys())]
        if mode == 'tile':
            label_cols = [f"class{lbl}" for lbl in list(self.tile_gt_configs['label_configs']['class_id_map'].keys())]
        
        if auto_split:
            # 1. Create a temporary 'stratify_label' by taking the argmax of the class columns
            # This ensures we have one discrete class per slide for the split logic
            df_temp = df.copy()
            df_temp['temp_label'] = df[label_cols].idxmax(axis=1)

            # 2. Get unique slides and their corresponding single label
            # We drop duplicates so we have 1 row per Slide ID
            slide_labels = df_temp[[id_col, 'temp_label']].drop_duplicates(subset=[id_col])
            
            unique_ids = slide_labels[id_col].values
            stratify_values = slide_labels['temp_label'].values

            # 3. Perform Stratified Split
            train_slides, valid_slides = train_test_split(
                unique_ids, 
                train_size=train_ratio, 
                stratify=stratify_values, 
                random_state=42
            )

            # 4. Update the DataFrame columns
            df['train'] = df[id_col].isin(train_slides).astype(int)
            df['valid'] = df[id_col].isin(valid_slides).astype(int)

            # --- 5. Print Distribution Stats ---
            print(f"\n{'='*30}")
            print(f"SPLIT SUMMARY (By Slide ID)")
            print(f"{'='*30}")
            print(f"Total Slides: {len(unique_ids)}")
            print(f"Train: {len(train_slides)} | Valid: {len(valid_slides)}")
            
            # Calculate percentages for the printout
            train_counts = slide_labels[slide_labels[id_col].isin(train_slides)]['temp_label'].value_counts(normalize=True)
            valid_counts = slide_labels[slide_labels[id_col].isin(valid_slides)]['temp_label'].value_counts(normalize=True)
            
            stats_df = pd.DataFrame({
                'Train Portion (%)': (train_counts * 100).round(2),
                'Valid Portion (%)': (valid_counts * 100).round(2)
            }).fillna(0) # In case a class is missing in one set
            
            print(stats_df)
            print(f"{'='*30}\n")

        # 6. Final Extraction
        self.train_ids = df[df['train'] == 1][id_col].unique().tolist()
        self.valid_ids = df[df['valid'] == 1][id_col].unique().tolist()

    def intitalise_inferencer(self, mode: Literal['slide', 'tile'], load_epoch='best', inf_id: int = None):
        if not self.available_modes[mode]: return
        self.inf_mode = mode

        if mode == 'slide':
            feats_configs = self.slide_feats_configs
            label_configs = self.slide_gt_configs.get('label_configs', {})
            ann_configs   = self.slide_gt_configs.get('ann_configs', {})
            model_configs = self.slide_downstream_configs['model_configs']
            train_configs = self.slide_downstream_configs['train_configs']
            split_configs = self.slide_downstream_configs['split_configs']
            self.data_collector = self.slide_features_collector
            self.get_slide_splits(self.selected_slide_dataset, mode='slide', id_col='slide_id', **split_configs)
            
        elif mode == 'tile':
            feats_configs = self.tile_feats_configs
            label_configs = self.tile_gt_configs.get('label_configs', {})
            ann_configs   = {}
            model_configs = self.tile_downstream_configs['model_configs']
            train_configs = self.tile_downstream_configs['train_configs']
            split_configs = self.tile_downstream_configs['split_configs']
            self.data_collector = self.tile_features_collector
            self.get_slide_splits(self.selected_tile_dataset, mode='tile', id_col='tile_id', **split_configs)

        # Pass configs for setup
        cur_configs = self.analyzer.prepare_inferencer(
            mode, feats_configs,
            lbl_class_id_map = label_configs.get('class_id_map'), ann_class_id_map = ann_configs.get('class_id_map'),
            lbl_cls_weights = label_configs.get('class_id_weights'), ann_cls_weights = ann_configs.get('class_id_weights'),
            **model_configs, **train_configs, return_cfgs=True,
        )
        
        # Get inf manifest and setup
        self.chkp_path, self.manifest, self.mnf_id, self.cur_epoch, cur_configs = get_inf_metadata(
            self.inferencer_folder, cur_configs, load_epoch, inf_id)
        self.analyzer.setup_inferencer(**cur_configs, chkp_path=self.chkp_path)


    def train(self, n_epochs=100, batch_size=1, verbose=True, **kwargs):
        """Simple training loop over the configured feature collector.

        If its full slide, batch size will be set to 1 internally
        Processes 1 sample per forward pass (since N varies across slides)

        Args:
            n_epochs (int): Number of training epochs. Default ``100``.
            batch_size (int): Gradient accumulation steps. Default ``1``.
            verbose (bool): Print epoch summaries. Default ``True``.

        Returns:
            dict: Training history with per-epoch loss dicts.
        """
        self.analyzer.setup_optimizer(n_epochs=n_epochs, **kwargs)
        self.train_history = {}
        self.valid_history = {}

        def get_loss(d, epoch):
            return deep_get(d, [epoch, 'loss', 'composite', 'final_loss'])

        for epoch in range(self.cur_epoch, n_epochs):
            epoch += 1

            # --- Train ---
            train_collector = self.data_collector(collect_ids=self.train_ids, shuffle=True, batch_size=batch_size, cache=True)
            self.train_history[epoch] = self.analyzer.train_epoch(train_collector)

            # --- Validate ---
            has_val = self.valid_ids is not None
            if has_val:
                valid_collector = self.data_collector(collect_ids=self.valid_ids, shuffle=False, batch_size=batch_size, cache=True)
                self.valid_history[epoch] = self.analyzer.eval_epoch(valid_collector)
                self.analyzer.optimizer.step_scheduler(metric=get_loss(self.valid_history, epoch=epoch))
            else:
                self.analyzer.optimizer.step_scheduler()

            if verbose:
                msg = f"Epoch {epoch}/{n_epochs} | train loss: {get_loss(self.train_history, epoch=epoch):.4f}"
                if has_val:
                    msg += f" | valid loss: {get_loss(self.valid_history, epoch=epoch):.4f}"
                msg += f" | lr: {self.analyzer.optimizer.lr:.2e}"
                print(msg)
                
            self.chkp_path = new_chkp_path(self.chkp_path, epoch=epoch)
            self.analyzer.save_inferencer(self.chkp_path, epoch=epoch)
            deep_assign(self.manifest, [self.mnf_id, 'train_history', epoch], value=self.train_history[epoch])
            deep_assign(self.manifest, [self.mnf_id, 'valid_history', epoch], value=self.valid_history[epoch])
            save_json(self.inferencer_folder / 'manifest.json', self.manifest)


    def evaluate(self, batch_size=1, eval_ids: Literal['valid', 'train', 'all'] = 'valid'):
        """
        1. Run evaluation and return losses + predictions.
        2. Works similarly with `collector`, as front end API
        3. To use analyzer, input only 1 eval_ids (list or str), only for mode='slide'

        Args:
            valid_ids (list, optional): subset to evaluate.

        Returns:
            tuple: (losses_dict, predictions_list)
        """
        if eval_ids == 'valid':
            eval_ids = self.valid_ids
        elif eval_ids == 'train':
            eval_ids = self.train_ids
        elif eval_ids == 'all':
            eval_ids = None

        # Collect data and evaluate
        eval_collector = self.data_collector(collect_ids=eval_ids, shuffle=False, batch_size=batch_size, cache=True)
        return self.analyzer.eval_epoch(eval_collector)


    def plot_history(self, figsize=(15, 5)):
        """Plot the training and validation history across epochs.
        
        Plots main modality losses, composite loss, and overall accuracy metrics.
        """
        # Pull from manifest if available (preserves history across sessions)
        mnf = self.manifest.get(self.mnf_id, {})
        train_hist = mnf.get('train_history', self.train_history)
        valid_hist = mnf.get('valid_history', self.valid_history)

        if not train_hist:
            print("No training history available to plot.")
            return

        # Ensure epochs are sorted integers
        epochs = sorted([int(k) for k in train_hist.keys()])
        if not epochs: return
        
        # 1. Extract data points
        # Structure: plots_data[base_name] = {'train': (epochs, values), 'valid': (epochs, values)}
        plots_data = {}
        
        for epoch in epochs:
            # Manifest keys might be strings if loaded from JSON
            str_epoch = str(epoch)
            
            # Helper to get the right epoch key (int or str)
            def get_epoch_data(hist):
                if epoch in hist: return hist[epoch]
                if str_epoch in hist: return hist[str_epoch]
                return None

            for split, hist in [('train', train_hist), ('valid', valid_hist)]:
                epoch_data = get_epoch_data(hist)
                if not epoch_data: continue
                
                # Extract Losses
                for modality, res in epoch_data.get('loss', {}).items():
                    for metric_name, val in res.items():
                        if modality == 'composite' and metric_name != 'final_loss':
                            continue
                        base_name = f"Loss | {modality.capitalize()}: {metric_name}"
                        if base_name not in plots_data: plots_data[base_name] = {'train': ([], []), 'valid': ([], [])}
                        plots_data[base_name][split][0].append(epoch)
                        plots_data[base_name][split][1].append(val)
                        
                # Extract Metrics (Include all aggregate scalars, exclude per-class for clarity)
                import re
                for modality, res in epoch_data.get('metric', {}).items():
                    for metric_name, val in res.items():
                        if isinstance(val, (int, float)):
                            # Filter out per-class metrics to avoid clutter like f1_c0, acc_c1
                            if re.search(r'(_c\d+|class_|_cls)', metric_name.lower()):
                                continue
                            base_name = f"Metric | {modality.capitalize()}: {metric_name}"
                            if base_name not in plots_data: plots_data[base_name] = {'train': ([], []), 'valid': ([], [])}
                            plots_data[base_name][split][0].append(epoch)
                            plots_data[base_name][split][1].append(val)

        # 2. Plotting
        total_plots = len(plots_data)
        if total_plots == 0:
            return
            
        cols = min(3, total_plots)
        rows = (total_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if total_plots == 1:
            axes = [axes]
        elif hasattr(axes, 'flatten'):
            axes = axes.flatten()
            
        for i, (base_name, data) in enumerate(plots_data.items()):
            ax = axes[i]
            for split, (eps, vals) in data.items():
                if eps:
                    linestyle = '-' if split == 'train' else '--'
                    ax.plot(eps, vals, label=split.capitalize(), linestyle=linestyle)
            ax.set_title(base_name, fontsize=11)
            ax.set_xlabel('Epoch', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=9)
            
        # Hide any unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()


    # |-----------------------------------------------|
    # |-------------- Similarity Search --------------|
    # |-----------------------------------------------|
    
    def get_descriptor_generator(self, mode: Literal['slide', 'tile']):
        if mode != self.inf_mode:
            self.intitalise_inferencer(mode=mode)
            self.index_built = False

        self.inf_mode = mode
        
        if mode == 'slide':
            def descriptor_generator():
                for _, row in self.shuffle_df(self.selected_slide_dataset, shuffle=True).iterrows():
                    slide_id = str(row['slide_id'])
                    self.serve_slide_analyzer(slide_id, data_modes='embeddings')
                    img_emb = self.analyzer.get_contrastive_embeddings(features=self.analyzer.proc_ext.features).get('img_emb')
                    yield {
                        'id': slide_id, 'features': self.analyzer.proc_ext.features, 
                        'coords': self.analyzer.proc_ext.coords, "img_emb": to_cpu(img_emb).numpy()
                    }

        elif mode == 'tile':
            def descriptor_generator():
                for tile_class, tile_data in self.tile_data_paths.items():
                    new_tile_ids = np.array(tile_data['tile_ids'])
                    self.serve_tile_analyzer(tile_class)
                    img_emb = self.analyzer.get_contrastive_embeddings(features=self.analyzer.proc_ext.features).get('img_emb')
                    yield {
                        'id': new_tile_ids, 'features': self.analyzer.proc_ext.features, 
                        'coords': self.analyzer.proc_ext.coords, "img_emb": to_cpu(img_emb).numpy()
                    }

        return descriptor_generator


    def similarity_search(self,
        mode: Literal['slide', 'tile'],
        query_mode: List[Literal['feat', 'roi', 'wrd', 'img']],
        query_feat_id: List[int] = None,
        query_roi_id: List[int] = None, 
        query_text: List[str] = None,
        rebuild_index=False,
        # Additional
        top_k: int = 5,
        similarity_threshold: float = 0.8,
        ):
        if not self.available_modes[mode]: return
        query_mode = convert_to_list(query_mode)

        # Defaults to build index if no query passed
        descriptor_generator = self.get_descriptor_generator(mode)

        if not self.index_built or rebuild_index:
            self.fit_from_generator(descriptor_generator())
            self.build_index(descriptor_generator(), compute=bool(mode=='slide'))
            self.index_built = True

        feat_res = None
        roi_res = None
        wrd_res = None

        # 1. Search for similar features
        if 'feat' in query_mode:
            feat_res = self.find_similar_feat(query_feat_id, top_k=top_k)

        # 2. Search for similar ROI in features
        if 'roi' in query_mode and feat_res is not None:
            self.serve_slide_analyzer(query_feat_id, data_modes='embeddings')
            query_features = self.analyzer.proc_ext.features[query_roi_id]

            # Loop starting from the second row (index 1 onwards)
            dfs_dict = {}
            for _, row in feat_res.iloc[1:].iterrows():
                tgt_emb_id = row['id']
                self.serve_slide_analyzer(tgt_emb_id, data_modes='embeddings')
                dfs_dict[tgt_emb_id] = self.find_similar_regions(
                    target_features=self.analyzer.proc_ext.features,
                    target_coords=self.analyzer.proc_ext.coords,
                    query_features=query_features,
                    similarity_threshold=similarity_threshold
                )
            roi_res = pd.concat(dfs_dict)

        # 3. Search for similar text among VLM embeddings
        if 'wrd' in query_mode or 'img' in query_mode:
            txt_mode = 'img' if 'img' in query_mode else 'wrd'
            img_emb = convert_to_list(query_feat_id)
            wrd_emb = self.analyzer.get_contrastive_embeddings(text=query_text).get('wrd_emb')
            wrd_res = self.find_similar_text(mode=txt_mode, img_emb=img_emb, wrd_emb=to_cpu(wrd_emb).numpy())
            
        return SimDF(feat_res=feat_res, roi_res=roi_res, wrd_res=wrd_res)