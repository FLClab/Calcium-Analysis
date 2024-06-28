
import numpy
import h5py
import os, glob
import multiprocessing
import json
import tifffile
import scipy

from skimage import measure, filters
from tqdm.auto import tqdm
from tensorflow.keras.utils import Sequence

def baseline(y, lam=1e3, ratio=1e-6):
    """
    Provient de https://github.com/charlesll/rampy/blob/master/rampy/baseline.py
    """
    N = len(y)
    D = scipy.sparse.csc_matrix(numpy.diff(numpy.eye(N), 2))
    w = numpy.ones(N)
    MAX_ITER = 100

    for _ in range(MAX_ITER):
        W = scipy.sparse.spdiags(w, 0, N, N)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w * y)
        d = y - z
        # make d- and get w^t with m and s
        dn = d[d < 0]
        m = numpy.mean(dn)
        s = numpy.std(dn)
        wt = 1.0 / (1 + numpy.exp(2 * (d - (2 * s - m)) / s))
        # check exit condition and backup
        if numpy.linalg.norm(w - wt) / numpy.linalg.norm(w) < ratio:
            break
        w = wt

    return z

def preprocess_stream(stream, method='moving_statistic', kernel_size=None):
    """
    Encapsulate all preprocessing steps
    """
    stream = stream.astype(numpy.float32)
    if method == 'moving_statistic':
        # Find threshold
        threshold = filters.threshold_triangle(stream)
        foreground_intensity = numpy.mean(stream[stream > threshold])
        mean_by_frame = numpy.mean(stream, axis=(1, 2))
        mean_average = baseline(mean_by_frame)
        mean_average = mean_average[..., numpy.newaxis, numpy.newaxis]
        stream = (stream - mean_average) / foreground_intensity
    return stream

def get_slice(event_center, max_shape):
    """
    Creates a `slice` of the event

    :param event_center: A `tuple` of the center of the event
    """
    return tuple(
        slice(max(0, c - s // 2), min(_max, c + s // 2)) for c, s, _max in zip(event_center, (64, 64, 64), max_shape)
    )

def get_crop(file, info, label):
    path = "/".join((info["fold"], info["group"]))
    event = info["event"]
    id, event = event[:1], event[1:]
    shape = info["shape"]

    event_center = event.reshape(-1, 2)
    event_center = (numpy.sum(event_center, axis=-1) / event_center.shape[-1]).astype(int)
    slc = get_slice(event_center, shape)

    crop = file[path][label][slc]
    return crop

class MSCTSequence(Sequence):
    """
    Creates a `Sequence` to load data from a `h5` dataset

    Note: Using `use_cache` speeds up the iteration process through the dataset
    but it assumes that the crops in cache are the ones that will be used.
    """
    def __init__(self, h5file, folds, label, crop_size=64, max_cache_size=30e+9, samples_pu=None, return_full=False, cache_mode="normal"):
        """
        Instantiates `MSCTSequence`

        :param h5file: A `str` of the h5file
        :param folds: A `list` of the accessible folds
        :param label: A `str` of the input type {"input", "label"}
        :param crop_size: An `int` of the size of the crop
        :param use_cache: A `bool` whether to load crops from cache
        """
        self.h5file = h5file
        if isinstance(folds, type(None)):
            folds = []
        self.folds = folds
        self.label = label
        assert self.label in {"input", "label"}, "Invalid input type"
        self.return_positive_only = False

        if isinstance(crop_size, int):
            crop_size = tuple(crop_size for _ in range(3))
        self.crop_size = crop_size

        self.return_full = return_full
        self.cache_mode = cache_mode
        avail_cache_mode = ["normal", "full", "crop"]
        assert self.cache_mode in avail_cache_mode, f"This is not a valid cache mode: {avail_cache_mode}"
        self.max_cache_size = max_cache_size
        self.cache = {}              

        # Updates samples if positive_samples and unlabeled_ratio are given
        self.samples_pu = {}
        if isinstance(samples_pu, dict):
            self.samples_pu = json.load(open(samples_pu["path"], "r"))[f"{samples_pu['positive']}-1:{samples_pu['unlabeled']}"]

        if self.samples_pu:
            if self.cache_mode == "full":
                self.info = self.get_file_info_pu_cache_mode_full()
            else:
                self.info = self.get_file_info_pu()
        else:
            # This is assumed that HDF5 file only contains crops in
            # data/label keys
            if self.cache_mode == "crop":
                self.info = self.get_file_info_cache_mode_crop()
            else:
                self.info = self.get_file_info()            

    def _getsizeof(self, obj):
        """
        Returns the size of an array
        :param ary: A `numpy.ndarray`
        :returns : The size of the array in `bytes`
        """
        if isinstance(obj, (list, tuple)):
            return sum([self._getsizeof(o) for o in obj])
        if isinstance(obj, str):
            return len(str)
        return obj.size * obj.dtype.itemsize        
        
    def _calc_current_cache_size(self, info):
        """
        Calculates the current cache size
        """
        return sum([di["datasize"] 
                    for di in info 
                    if isinstance(di["cache-idx"], str)])     
    
    def _calc_cached_items(self, info):
        return sum([1 
                    for di in info 
                    if isinstance(di['cache-idx'], str)])

    def cache_data(self, file, info, data):
        """
        Caches the data
        """
        # Creates local path variables
        path = "/".join((info["fold"], info["group"]))
        group = file[path]
        cache_path = f"cache-{self.label}"

        # Verify if already a cache group exists
        if cache_path not in group:
            cache_group = file.create_group("/".join((path, cache_path)))
        else:
            cache_group = group[cache_path]

        # Verify if data already in cache if not then adds it
        event_idx = str(info["event-idx"])
        info["cache"] = "/".join((path, cache_path, event_idx))
        if event_idx not in cache_group:
            cache_group.create_dataset(
                event_idx, data=data, compression="gzip"
            )

    def add_data_cache(self):
        """
        Loads data into cache
        """
        with h5py.File(self.h5file, "r+") as file:
            for info in tqdm(self.info, f"Loading in cache {self.label}"):
                if not info["cache"]:
                    crop = get_crop(file, info, self.label)
                    self.cache_data(file, info, crop)

    # def get_file_info(self):
    #     """
    #     Extracts the file information
    #     """
    #     info = []
    #     with h5py.File(self.h5file, "r") as file:
    #         for fold in self.folds:
    #             for group in sorted(file[fold].keys(), key=lambda key : int(key)):
    #                 for idx, event in enumerate(file[fold][group]["events"]):
    #                     cache, idx = None, str(idx)
    #                     if f"cache-{self.label}" in file[fold][group]:
    #                         if idx in file[fold][group][f"cache-{self.label}"]:
    #                             cache = "/".join((
    #                                 fold, group, f"cache-{self.label}", idx
    #                             ))
    #                     info.append({
    #                         "fold" : fold,
    #                         "group" : group,
    #                         "event" : event,
    #                         "shape" : file[fold][group][self.label].shape,
    #                         "cache" : cache,
    #                         "event-idx" : idx
    #                     })
    #     return info

    # def get_file_info_pu(self):
    #     """
    #     Extracts the file information by assuming a positive unlabeled ratio
    #     """
    #     info = []
    #     with h5py.File(self.h5file, "r") as file:
    #         for fold in self.folds:
    #             if isinstance(fold, str):
    #                 # Adds positive samples
    #                 for sample in tqdm(self.samples_pu["positive"][fold], desc=f"{fold} -- positive"):
    #                     group = sample["neuron"]
    #                     idx = sample["event-id"]
    #                     event = file[fold][group]["events"][idx]
    #                     cache, idx = None, str(idx)
    #                     if f"cache-{self.label}" in file[fold][group]:
    #                         if idx in file[fold][group][f"cache-{self.label}"]:
    #                             cache = "/".join((
    #                                 fold, group, f"cache-{self.label}", idx
    #                             ))
    #                     info.append({
    #                         "fold" : fold,
    #                         "group" : group,
    #                         "event" : event,
    #                         "shape" : file[fold][group][self.label].shape,
    #                         "cache" : cache,
    #                         "event-idx" : idx,
    #                         "type" : "positive"
    #                     })

    #                 if not self.return_positive_only:
    #                     # Adds unlabeled samples
    #                     for idx, sample in enumerate(tqdm(self.samples_pu["negative"][fold], desc=f"{fold} -- negative")):
    #                         group = sample["neuron"]
    #                         coord = sample["coord"]
    #                         cache = None
    #                         idx = str(coord)
    #                         if f"cache-unlabeled-{self.label}" in file[fold][group]:
    #                             if idx in file[fold][group][f"cache-unlabeled-{self.label}"]:
    #                                 cache = "/".join((
    #                                     fold, group, f"cache-unlabeled-{self.label}", idx
    #                                 ))
    #                         info.append({
    #                             "fold" : fold,
    #                             "group" : group,
    #                             "event" : coord,
    #                             "shape" : file[fold][group][self.label].shape,
    #                             "cache" : cache,
    #                             "event-idx" : idx, # use coord to avoid multiple versions of same event for different PU ratios
    #                             "type" : "unlabeled"
    #                         })

    #     return info

    def get_file_info(self):
        """
        Extracts the file information
        """

        info = []
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            print(f"Getting dataset: {self.label}")
            for fold in self.folds:
                print(f"Getting fold: {fold}")
                if isinstance(fold, str):
                    for gg, group in enumerate(tqdm(sorted(file[fold].keys(), key=lambda key : int(key)))):
                        if self.return_full:
                            event = file[fold][group][self.label]
                            info.append({
                                "fold" : fold,
                                "group" : group,
                                "shape" : file[fold][group][self.label].shape,
                                "event-idx" : None,
                                "key" : "/".join((fold, group, self.label)),
                                "datasize" : self._getsizeof(event),
                                "cache-idx" : None,
                                "is-empty" : False
                            })                                
                        else:
                            if f"cache-{self.label}" in file[fold][group]:
                                for ee, (idx, event) in enumerate(tqdm(file[fold][group][f"cache-{self.label}"].items(), leave=False)):
                                    cache_idx = None
                                    is_empty = not numpy.any(event)
                                    current_cache_size = self._calc_current_cache_size(info)
                                    if (not is_empty) and (current_cache_size < self.max_cache_size):
                                        cache_idx = "/".join((fold, group, f"cache-{self.label}", idx))                                        
                                        self.cache[cache_idx] = event[()]
                                    info.append({
                                        "fold" : fold,
                                        "group" : group,
                                        "shape" : file[fold][group][f"cache-{self.label}"][idx].shape,
                                        "event-idx" : idx,
                                        "key" : "/".join((fold, group, f"cache-{self.label}", idx)),
                                        "datasize" : self._getsizeof(event),
                                        "cache-idx" : cache_idx,
                                        "is-empty" : is_empty
                                    })
                            else:
                                for ee, (idx, event) in enumerate(tqdm(file[fold][group][self.label].items(), leave=False)):
                                    cache_idx = None
                                    is_empty = not numpy.any(event)
                                    current_cache_size = self._calc_current_cache_size(info)
                                    if (not is_empty) and (current_cache_size < self.max_cache_size):
                                        cache_idx = "/".join((fold, group, self.label, idx))                                        
                                        self.cache[cache_idx] = event[()]
                                    info.append({
                                        "fold" : fold,
                                        "group" : group,
                                        "shape" : file[fold][group][self.label][idx].shape,
                                        "event-idx" : idx,
                                        "key" : "/".join((fold, group, self.label, idx)),
                                        "datasize" : self._getsizeof(event),
                                        "cache-idx" : cache_idx,
                                        "is-empty" : is_empty
                                    })
    #                             print(f"[----] Neuron: {group} ({100 * (gg + 1) / len(file[fold]):0.1f});")
                            print(f"Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")
        return info
    
    def get_file_info_cache_mode_crop(self):
        """
        Extracts the file information
        """

        info = []
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            print(f"Getting dataset: {self.label}")
            for fold in self.folds:
                print(f"Getting fold: {fold}")
                if isinstance(fold, str):
                    for gg, group in enumerate(tqdm(sorted(file[fold].keys(), key=lambda key : int(key)))):
                        
                        for event in file[fold][group]["events"]:
                            event_idx, event = str(event[:1]), event[1:]

                            event_center = event.reshape(-1, 2)
                            event_center = numpy.mean(event_center, axis=-1).astype(int)
                            slc = tuple(
                                slice(max(0, c - s // 2), min(_max, c + s // 2)) for c, s, _max in zip(event_center, self.crop_size, file[fold][group][self.label].shape)
                            )

                            event = file[fold][group][self.label][slc]

                            cache_idx = None
                            is_empty = not numpy.any(event)
                            current_cache_size = self._calc_current_cache_size(info)
                            if (not is_empty) and (current_cache_size < self.max_cache_size):
                                cache_idx = "/".join((fold, group, self.label, event_idx))
                                self.cache[cache_idx] = event

                            info.append({
                                "fold" : fold,
                                "group" : group,
                                "shape" : self.crop_size,
                                "event-idx" : event_idx,
                                "key" : "/".join((fold, group, self.label)),
                                "datasize" : self._getsizeof(event),
                                "cache-idx" : cache_idx,
                                "is-empty" : is_empty,
                                "slice" : slc
                            })

                        print(f"Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")
        return info    
    
    def get_file_info_pu(self):
        info = []
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            print(f"Getting dataset: {self.label}")
            for fold in self.folds:
                print(f"Getting fold: {fold}")
                if isinstance(fold, str):
                    # Adds positive samples
                    for sample in self.samples_pu["positive"][fold]:
                        cache_idx = None
                        group = sample["neuron"]
                        idx = sample["event-id"]
                        event = file[fold][group]["events"][idx]

                        cache_idx = "/".join((fold, group, f"cache-{self.label}", str(idx)))
                        event = file[cache_idx]
                        is_empty = not numpy.any(event)

                        current_cache_size = self._calc_current_cache_size(info)
                        if (not is_empty) and (current_cache_size < self.max_cache_size):
                            cache_idx = "/".join((fold, group, f"cache-{self.label}", str(idx)))
                            self.cache[cache_idx] = file[cache_idx][()]
                        else:
                            cache_idx = None
                                                
                        idx = str(idx)
                        info.append({
                            "fold" : fold,
                            "group" : group,
                            "shape" : file[fold][group][f"cache-{self.label}"][idx].shape,
                            "event-idx" : idx,
                            "key" : "/".join((fold, group, f"cache-{self.label}", idx)),
                            "datasize" : self._getsizeof(event),
                            "cache-idx" : cache_idx,
                            "is-empty" : is_empty                                
                        })

                    print(f"[----] Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")

                    # Adds unlabeled samples
                    for idx, sample in enumerate(tqdm(self.samples_pu["negative"][fold])):
                        group = sample["neuron"]
                        coord = sample["coord"]
                        idx = str(coord)

                        cache_idx = "/".join((fold, group, f"cache-unlabeled-{self.label}", idx))
                        current_cache_size = self._calc_current_cache_size(info)
                        if not (cache_idx in file):
                            cache_idx = None
                            is_empty = False
                            slc = tuple(slice(coord[i], coord[i] + self.crop_size[i]) for i in range(3))
                            datasize = 0
                            if current_cache_size < self.max_cache_size:
                                cache_idx = "/".join((fold, group, f"cache-unlabeled-{self.label}", idx))
                                event = file["/".join((fold, group, self.label))][slc]
                                is_empty = not numpy.any(event)
                                if not is_empty:
                                    self.cache[cache_idx] = event
                                else:
                                    cache_idx = None

                                datasize = self._getsizeof(event)
                                
                            info.append({
                                "fold" : fold,
                                "group" : group,
                                "shape" : self.crop_size,
                                "event-idx" : idx,
                                "key" : "/".join((fold, group, self.label)),
                                "datasize" : datasize,
                                "cache-idx" : cache_idx,
                                "is-empty" : is_empty,
                                "slice" : slc
                            })
                        else:
                            event = file[cache_idx]
                            is_empty = not numpy.any(event)
                            if (not is_empty) and (current_cache_size < self.max_cache_size):
                                self.cache[cache_idx] = event[()]
                            else:
                                cache_idx = None
                        
                            info.append({
                                "fold" : fold,
                                "group" : group,
                                "shape" : file[fold][group][f"cache-unlabeled-{self.label}"][idx].shape,
                                "event-idx" : idx,
                                "key" : "/".join((fold, group, f"cache-unlabeled-{self.label}", idx)),
                                "datasize" : self._getsizeof(event),
                                "cache-idx" : cache_idx,
                                "is-empty" : is_empty                                    
                            })

                    print(f"[----] Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")
        return info     

    def get_file_info_pu_cache_mode_full(self):
        info = []
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            print(f"Getting dataset: {self.label}")
            for fold in self.folds:
                print(f"Getting fold: {fold}")
                if isinstance(fold, str):
                    # Adds positive samples
                    for sample in self.samples_pu["positive"][fold]:
                        cache_idx = None
                        group = sample["neuron"]
                        idx = sample["event-id"]
                        event = file[fold][group]["events"][idx]

                        cache_idx = "/".join((fold, group, f"cache-{self.label}", str(idx)))
                        event = file[cache_idx]
                        is_empty = not numpy.any(event)

                        current_cache_size = self._calc_current_cache_size(info)
                        if (not is_empty) and (current_cache_size < self.max_cache_size):
                            cache_idx = "/".join((fold, group, f"cache-{self.label}", str(idx)))
                            self.cache[cache_idx] = file[cache_idx][()]
                        else:
                            cache_idx = None
                                                
                        idx = str(idx)
                        info.append({
                            "fold" : fold,
                            "group" : group,
                            "shape" : file[fold][group][f"cache-{self.label}"][idx].shape,
                            "event-idx" : idx,
                            "key" : "/".join((fold, group, f"cache-{self.label}", idx)),
                            "datasize" : self._getsizeof(event),
                            "cache-idx" : cache_idx,
                            "is-empty" : is_empty                                
                        })

                    print(f"[----] Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")

                    # Adds unlabeled samples
                    for idx, sample in enumerate(tqdm(self.samples_pu["negative"][fold])):
                        group = sample["neuron"]
                        coord = sample["coord"]
                        idx = str(coord)

                        slc = tuple(slice(coord[i], coord[i] + self.crop_size[i]) for i in range(3))

                        # current_cache_size = self._calc_current_cache_size(info)
                        cache_idx = "/".join((fold, group, self.label))
                        if not (cache_idx in self.cache) and True:#(current_cache_size < self.max_cache_size):
                            self.cache[cache_idx] = file[fold][group][self.label][()]
                            datasize = self._getsizeof(self.cache[cache_idx])
                        elif (cache_idx in self.cache):
                            datasize = 0
                        else:
                            datasize = 0
                            cache_idx = None

                        info.append({
                            "fold" : fold,
                            "group" : group,
                            "shape" : self.crop_size,
                            "event-idx" : idx,
                            "key" : "/".join((fold, group, self.label)),
                            "datasize" : datasize,
                            "cache-idx" : cache_idx,
                            "is-empty" : False,
                            "slice" : slc
                        })


                    print(f"[----] Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")
        return info         

    def get_slice(self, event_center, max_shape):
        """
        Creates a `slice` of the event

        :param event_center: A `tuple` of the center of the event
        """
        return tuple(
            slice(max(0, c - s // 2), min(_max, c + s // 2)) for c, s, _max in zip(event_center, self.crop_size, max_shape)
        )

    def __getitem__(self, index):
        """
        Implements the getitem method of the `MSCTSequence`

        :param index: An `int` of the index
        """
        # Handles slicing
        if isinstance(index, slice):
            # start = index.start if isinstance(index.start, int) else 0
            # stop = index.stop if isinstance(index.stop, int) else len(self.info)
            # step = index.step if isinstance(index.step, int) else 1
            self.info = self.info[index]
            return self
        elif isinstance(index, (list, tuple, numpy.ndarray)):
            self.info = [self.info[idx] for idx in index]
            return self

        info = self.info[index]
        if info["is-empty"]:
            crop = numpy.zeros(self.crop_size, dtype=numpy.uint8)
        elif isinstance(info["cache-idx"], str):
            crop = self.cache[info["cache-idx"]]  
            if "slice" in info:
                crop = crop[info["slice"]]                   
        else:
            with h5py.File(self.h5file, "r") as file:
                if "slice" in info:
                    crop = file[info["key"]][info["slice"]]
                else:
                    crop = file[info["key"]][()]

        # Pads crop if not good size
        if not self.return_full:
            if crop.size != numpy.prod(self.crop_size):
                crop = numpy.pad(
                    crop,
                    [(0, cs - current) for cs, current in zip(self.crop_size, crop.shape)],
                    mode="symmetric"
                )

        if self.label == "label":
            crop = crop.astype(numpy.uint8)

        return crop

    def __len__(self):
        return len(self.info)

class TestMSCTSequence:
    def __init__(self, *args, **kwargs):

        test_data = kwargs.get("test-data", "./data/testset")

        self.inputs = glob.glob(os.path.join(test_data, "raw-input", "*.tif"))
        self.labels = [name.replace("/raw-input/", "/minifinder/").replace(".tif", ".npy") for name in self.inputs]

    def __getitem__(self, index):
        if self.inputs[index].endswith(".npy"):
            # Assumes file is already processed
            image = numpy.load(self.inputs[index])
        else:
            # Normalizes the file
            image = tifffile.imread(self.inputs[index])
            image = preprocess_stream(image)
        label = numpy.load(self.labels[index])

        label = measure.label(label)
        regionprops = measure.regionprops(label)
        return os.path.basename(self.inputs[index]), image, regionprops

    def __len__(self):
        return len(self.inputs)
    
class FolderMSCTSequence:
    def __init__(self, *args, **kwargs):

        test_data = kwargs.get("test-data", "./data/testset")

        self.inputs = glob.glob(os.path.join(test_data, "**", "*.tif"), recursive=True)

    def __getitem__(self, index):
        if self.inputs[index].endswith(".npy"):
            # Assumes file is already processed
            image = numpy.load(self.inputs[index])
        else:
            # Normalizes the file
            image = tifffile.imread(self.inputs[index])
            image = preprocess_stream(image)
        
        return self.inputs[index], image, None

    def __len__(self):
        return len(self.inputs)    

if __name__ == "__main__":

    from tqdm.auto import tqdm, trange

    h5path = "./data/calcium-dataset.h5"

    loader = MSCTSequence(h5path, folds=["foldA", "foldB", "foldC"], label="label")
    for i in trange(len(loader)):
        loader[i]
