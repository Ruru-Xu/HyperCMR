import glob
import logging
import math
import os
import pickle
import h5py
import numpy as np
import torch
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Union, Tuple
from torch import Tensor
import os.path
from pathlib import Path
from typing import Callable, Optional, Union
import pytorch_lightning as pl
import contextlib
import time

class CMRxReconRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class CmrxReconSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root, fileName, transform=None, use_dataset_cache=False, dataset_cache_file="dataset_cache.pkl", num_cols=None, raw_sample_filter=None, num_adj_slices=3):
        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset"
        self.num_adj_slices = num_adj_slices
        self.recons_key = "reconstruction_rss"
        self.raw_samples = []
        self.raw_sample_filter = raw_sample_filter or (lambda raw_sample: True)

        # Ensure the directory for dataset_cache_file exists
        if not self.dataset_cache_file.parent.exists():
            self.dataset_cache_file.parent.mkdir(parents=True, exist_ok=True)

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            try:
                with open(self.dataset_cache_file, "rb") as f:
                    dataset_cache = pickle.load(f)
            except EOFError:
                # print(f"Error: The file {self.dataset_cache_file} is empty or corrupted.")
                dataset_cache = {}
            except Exception as e:
                # print(f"An error occurred while loading the dataset cache: {e}")
                dataset_cache = {}
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        files = sorted(glob.glob(os.path.join(root, "**/*" + fileName + "*.h5"), recursive=True))
        for fname in files:
            if dataset_cache.get(fname) is None or not use_dataset_cache:
                with h5py.File(fname, 'r') as hf:
                    num_slices = hf["kspace"].shape[0]
                    metadata = {**hf.attrs}
                new_raw_samples = [
                    CMRxReconRawDataSample(fname, slice_ind, metadata)
                    for slice_ind in range(num_slices)
                    if self.raw_sample_filter(CMRxReconRawDataSample(fname, slice_ind, metadata))
                ]
                self.raw_samples += new_raw_samples

            if dataset_cache.get(fname) is None and use_dataset_cache:
                dataset_cache[fname] = self.raw_samples
                # logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
            else:
                # logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
                self.raw_samples = dataset_cache[fname]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex.metadata["encoding_size"][1] in num_cols  # type: ignore
            ]

    def _get_frames_indices(self, dataslice, num_slices_in_volume, num_t_in_volume=None):
        '''
        when we reshape t, z to one axis in preprocessing, we need to get the indices of the slices in the original t, z axis;
        then find the adjacent slices in the original z axis
        '''
        ti = dataslice//num_slices_in_volume
        zi = dataslice - ti*num_slices_in_volume

        zi_idx_list = [zi]

        ti_idx_list = [ (i+ti)%num_t_in_volume for i in range(-2,3)]
        output_list = []

        for zz in zi_idx_list:
            for tt in ti_idx_list:
                output_list.append(tt*num_slices_in_volume + zz)

        return output_list
    def _get_frames_indices_mapping(self, dataslice, num_slices_in_volume, num_t_in_volume=None, isT2=False):
        '''
        when we reshape t, z to one axis in preprocessing, we need to get the indices of the slices in the original t, z axis;
        then find the adjacent slices in the original z axis
        '''
        ti = dataslice//num_slices_in_volume
        zi = dataslice - ti*num_slices_in_volume

        zi_idx_list = [zi]

        if isT2: # only 3 nw in T2, so we repeat adjacent for 3 times
            ti_idx_list = [ (i+ti)%num_t_in_volume for i in range(-1,2)]
            ti_idx_list = 1*ti_idx_list[0:1] + ti_idx_list + ti_idx_list[2:3]*1
        else:
            ti_idx_list = [ (i+ti)%num_t_in_volume for i in range(-2,3)]
        output_list = []

        for zz in zi_idx_list:
            for tt in ti_idx_list:
                output_list.append(tt*num_slices_in_volume + zz)

        return output_list

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]
        isT2=True if 'T2' in fname else False
        kspace = []
        with h5py.File(str(fname), 'r') as hf:
            kspace_volume = hf["kspace"]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)

            num_slices = attrs['shape'][1]
            num_t = attrs['shape'][0]
            slice_idx_list = self._get_frames_indices_mapping(dataslice, num_slices,num_t,isT2=isT2)
            for idx in slice_idx_list:
                kspace.append(kspace_volume[idx])
            kspace = np.concatenate(kspace, axis=0)

        sample = self.transform(kspace, mask, target, attrs, str(fname), dataslice)

        return sample


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)
class MaskFunc:

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):

        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:

        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:

    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


class PromptMrDataTransform:
    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> tuple[Any, Any, Any, Any]:
        target_torch = to_tensor(target)
        # max_value = attrs["max"]
        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname)) # so in validation, the same fname (volume) will have the same acc
        acq_start = 0
        acq_end = attrs["padding_right"]
        masked_kspace, mask_torch, num_low_frequencies = apply_mask(
            kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
        )
        return kspace_torch, masked_kspace, mask_torch.to(torch.bool), target_torch




def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[CmrxReconSliceDataset] = worker_info.dataset  # pylint: disable=no-member
    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member
    data.transform.mask_func.rng.seed(base_seed % (2**32 - 1))


class CmrxReconDataModule(pl.LightningDataModule):
    def __init__(self, data_path, fileName, train_transform, val_transform, batch_size, num_workers, use_dataset_cache_file=True):
        super().__init__()

        self.data_path = data_path
        self.filename = fileName
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:
        is_train = data_partition == "train"
        data_path = os.path.join(self.data_path, data_partition)

        dataset = CmrxReconSliceDataset(
            root=data_path,
            fileName=self.filename,
            transform=data_transform,
            use_dataset_cache=self.use_dataset_cache_file,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=is_train,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            data_paths = [
                os.path.join(self.data_path, "train"),
                os.path.join(self.data_path, "val"),
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                    zip(data_paths, data_transforms)
            ):
                _ = CmrxReconSliceDataset(
                    root=data_path,
                    fileName=self.fileName,
                    transform=data_transform,
                    use_dataset_cache=self.use_dataset_cache_file,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")



class FixedLowEquiSpacedMaskFunc(MaskFunc):
    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        return center_mask, acceleration_mask, num_low_frequencies

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


def to_tensor(data: np.ndarray) -> torch.Tensor:
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def matlab_round(n):
    if n > 0:
        return int(n + 0.5)
    else:
        return int(n - 0.5)


def _crop(a, crop_shape):
    indices = [
        (math.floor(dim/2) + math.ceil(-crop_dim/2),
         math.floor(dim/2) + math.ceil(crop_dim/2))
        for dim, crop_dim in zip(a.shape, crop_shape)
    ]
    return a[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1]]

def crop_submission(a):
    b, sx,sy = a.shape
    b = _crop(a,(b, matlab_round(sx/3), matlab_round(sy/2)))
    return b
