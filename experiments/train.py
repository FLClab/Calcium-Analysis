#@markdown ##<font color=orange>Install Network and dependencies

#Libraries contains information of certain topics.

#Put the imported code and libraries here

Notebook_version = ['1.12'] #Contact the ZeroCostDL4Mic team to find out about the version number
Network = "Calcium U-Net 3D"

from builtins import any as b_any

def get_requirements_path():
    # Store requirements file in 'contents' directory
    current_dir = os.getcwd()
    dir_count = current_dir.count('/') - 1
    path = '../' * (dir_count) + 'requirements.txt'
    return path

def filter_files(file_list, filter_list):
    filtered_list = []
    for fname in file_list:
        if b_any(fname.split('==')[0] in s for s in filter_list):
            filtered_list.append(fname)
    return filtered_list

def build_requirements_file(before, after):
    path = get_requirements_path()

    # Exporting requirements.txt for local run
    # !pip freeze > $path

    # Get minimum requirements file
    df = pandas.read_csv(path, delimiter = "\n")
    mod_list = [m.split('.')[0] for m in after if not m in before]
    req_list_temp = df.values.tolist()
    req_list = [x[0] for x in req_list_temp]

    # Replace with package name and handle cases where import name is different to module name
    mod_name_list = [['sklearn', 'scikit-learn'], ['skimage', 'scikit-image']]
    mod_replace_list = [[x[1] for x in mod_name_list] if s in [x[0] for x in mod_name_list] else s for s in mod_list]
    filtered_list = filter_files(req_list, mod_replace_list)

    file=open(path,'w')
    for item in filtered_list:
        file.writelines(item + '\n')

    file.close()

import sys
before = [str(m) for m in sys.modules]

import os
import glob
import itertools
import numpy
import pandas
import shutil
import h5py
import random
import tifffile
import yaml
import scipy
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose
from collections import defaultdict
from matplotlib import pyplot
from skimage import filters, io, measure
from sklearn.metrics import precision_recall_curve
from scipy.spatial import distance
from metrics import CentroidDetectionError

import time
import subprocess
from datetime import datetime
# from fpdf import FPDF, HTMLMixin
from pip._internal.operations.freeze import freeze

from ipywidgets import interact
from ipywidgets import interactive
from ipywidgets import fixed
from ipywidgets import interact_manual 
import ipywidgets as widgets

from tqdm import tqdm, trange

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

################################################################################################
# MODEL DEFINITION
################################################################################################

class DownConv(nn.Module):
    """
    Module pour faire 2 convolutions et un max pooling.
    ReLU ou LReLU et BatchNorm apres chaque convolution.
    """

    def __init__(self, inC, outC, kernel_size=3, pooling=True,
                 use_leaky_relu=False, use_batch_norm=False):
        super().__init__()
        self.inC = inC
        self.outC = outC
        self.pooling = pooling
        self.use_leaky_relu = use_leaky_relu
        self.use_batch_norm = use_batch_norm

        if use_leaky_relu:
            relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            relu = nn.ReLU(inplace=True)

        self.convs = []
        self.convs.append(nn.Conv3d(inC, outC, kernel_size=kernel_size, padding=kernel_size // 2))
        if use_batch_norm:
            self.convs.append(nn.BatchNorm3d(outC))
        self.convs.append(relu)
        self.convs.append(nn.Conv3d(outC, outC, kernel_size=kernel_size, padding=kernel_size // 2))
        if use_batch_norm:
            self.convs.append(nn.BatchNorm3d(outC))
        self.convs.append(relu)
        self.convs = nn.ModuleList(self.convs)

        if pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        for i, module in enumerate(self.convs):
            x = module(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UpConv(nn.Module):
    """
    Module pour faire une upconv et 2 convolutions.
    ReLU et BatchNorm apres chaque convolution.
    """

    def __init__(self, inC, outC, use_batch_norm=False):
        super().__init__()
        self.inC = inC
        self.outC = outC
        self.use_batch_norm = use_batch_norm

        relu = nn.ReLU(inplace=True)

        self.upconv = nn.ConvTranspose3d(inC, outC, kernel_size=3,
                                         stride=2, output_padding=1,
                                         padding=1)

        self.convs = []
        self.convs.append(nn.Conv3d(inC, outC, kernel_size=3, padding=1))
        if self.use_batch_norm:
            self.convs.append(nn.BatchNorm3d(outC))
        self.convs.append(relu)
        self.convs.append(nn.Conv3d(outC, outC, kernel_size=3, padding=1))
        if self.use_batch_norm:
            self.convs.append(nn.BatchNorm3d(outC))
        self.convs.append(relu)

        self.convs = nn.ModuleList(self.convs)

    def forward(self, from_enco, from_deco):
        from_deco = self.upconv(from_deco)
        x = torch.cat([from_enco, from_deco], 1)
        for module in self.convs:
            x = module(x)
        return x

class BottleNeck(nn.Module):
    """
    Module pour faire les opérations dans le bas
    du réseau (le bottleneck). Il y a une conv,
    des convolutions atrous (concatened together)
    suivi d'une conv. Avec r=1, il s'agit d'une
    conv normale.
    """

    def __init__(self, inC, outC, r=1):
        super().__init__()
        self.inC = inC
        self.outC = outC
        self.r = r

        self.dilated_conv = []
        for i in range(r):
            dilation_conv = nn.Conv3d(inC, outC, kernel_size=3,
                                      padding=i + 1, dilation=i + 1)
            self.dilated_conv.append(dilation_conv)
        self.dilated_conv = nn.ModuleList(self.dilated_conv)
        self.conv2 = nn.Conv3d(outC * r, outC, kernel_size=3, padding=1)

    def forward(self, x):
        x = [F.relu(d_conv(x)) for d_conv in self.dilated_conv]
        x = torch.cat(x, 1)
        x = F.relu(self.conv2(x))
        return x

class UNet3D(nn.Module):
    """
    Implémentation de http://www.nature.com/articles/s41598-018-34817-6
    """

    def __init__(self, config):
        """Defines the UNet network.
        """
        super().__init__()

        self.config = config
        model_config = self.config["model_config"]

        self.first_kernel_size = model_config["first_kernel_size"]
        self.n_layer = model_config["n_layer"]
        self.nbf = model_config["nbf"]
        self.inC = model_config["inC"]
        self.n_layer = model_config["n_layer"]
        self.nbf = model_config["nbf"]
        self.outC = model_config["outC"]
        self.r = model_config["r"]
        self.use_batch_norm = model_config["use_batch_norm"]
        self.use_leaky_relu = model_config["use_leaky_relu"]

        self.down_convs = []
        self.up_convs = []
        
        for i in range(self.n_layer):
            ins = self.inC if i == 0 else outs
            outs = self.nbf * (2**i)
            if i == 0:  # first conv
                down_conv = DownConv(ins, outs, kernel_size=self.first_kernel_size,
                                     use_leaky_relu=self.use_leaky_relu, use_batch_norm=self.use_batch_norm)
            else:
                down_conv = DownConv(ins, outs,
                                     use_leaky_relu=self.use_leaky_relu, use_batch_norm=self.use_batch_norm)
            self.down_convs.append(down_conv)

        ins = outs
        outs = self.nbf * (2**(i + 1))
        self.bottleneck = BottleNeck(ins, outs, r=self.r)

        for i in range(self.n_layer):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, use_batch_norm=self.use_batch_norm)
            self.up_convs.append(up_conv)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.final_conv = nn.Conv3d(outs, self.outC, kernel_size=1)

    def forward(self, x):
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        x = self.bottleneck(x)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 1)]
            x = module(before_pool, x)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out

    def predict_stream(self, streamNPZ, step=None, batch_size=1, num_workers=1, device='cpu'):
        """Fonction pour effectuer une prédiction sur un stream.

        Inputs:
            streamNPZ (NpzFile): Fichier Npz contenant le nécessaire pour faire la prédiction.
            step (list of int): Défini les step d'évaluation dans les 3 axes du stream
            batch_size (int): La batch size a utiliser pour inférer.
        """
        stream = streamNPZ['input']
        shape2crop = streamNPZ['shape2crop']
        stream = numpy.pad(
            stream, tuple((s, s) for s in shape2crop), mode="symmetric"
        )
        stream_ds = PredictStream3D(stream, shape2crop, step=step)
        stream_dl = DataLoader(stream_ds, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)

        # Matrices pour sauvegarder les prédictions
        nbPixel = numpy.zeros_like(stream, dtype=numpy.uint8)
        probAccumulator = numpy.zeros_like(stream, dtype=numpy.float32)

        for batch_of_crops, batch_of_corners in tqdm(stream_dl, desc="Prediction: "):
            batch_of_crops = batch_of_crops.to(device)
            batch_of_corners = batch_of_corners.numpy()
            preds = self.predict(batch_of_crops)
            preds = preds.cpu().numpy()
            for b in range(batch_of_crops.shape[0]):
                corner = batch_of_corners[b]
                slicesTHW = (slice(corner[0], corner[0] + shape2crop[0]),
                             slice(corner[1], corner[1] + shape2crop[1]),
                             slice(corner[2], corner[2] + shape2crop[2]))
                probAccumulator[slicesTHW] += preds[b, 0]
                nbPixel[slicesTHW] += 1
        
        slc = tuple(slice(s, -s) for s in shape2crop)
        probAccumulator = probAccumulator[slc]
        nbPixel = nbPixel[slc]
        return probAccumulator / nbPixel
    
    def optimize_threshold(self,
                          valid_dataset):
        
        training_config = self.config["training_config"]
        valid_loader = DataLoader(valid_dataset, batch_size=training_config["batch_size"],
                                num_workers=training_config["num_workers"], shuffle=True,
                                drop_last=training_config["drop_last"], pin_memory=True)           
        
        self.eval()

        thresholds = []
        for batch in tqdm(valid_loader, desc="Validation Loader: ", leave=False):
            
            event, mask, index = [t.to(DEVICE) for t in batch]

            # defines x and y
            x = event
            y = mask

            pred = self.forward(x)

            thresholds.extend(compute_optimal_thresholds(mask, pred))
            
        threshold = numpy.mean(thresholds)
                
        return threshold
    
    def optimize_threshold_detection(self,
                          valid_dataset,
                          **kwargs):
        
        self.eval()

        patch_size = kwargs.pop("patch_size", 32)
        
        thresholds = []
        for event, mask, index in tqdm(valid_dataset, desc="Validation Dataset: ", leave=False):
            event, mask = event.squeeze(), mask.squeeze()
            pred = self.predict_stream({
                "input" : event,
                "shape2crop" : numpy.array([patch_size * 2, patch_size*2, patch_size*2])
                },
                **kwargs
            )
            
            thresholds.append(compute_optimal_thresholds_detection(mask, pred))
                                
        threshold = numpy.mean(thresholds)            

        torch.cuda.empty_cache()
        return threshold    

    def train_model(self, 
              train_dataset, 
              valid_dataset, 
              model_path, 
              model_name, 
              ckpt_period=1,
              save_best_ckpt_only=False, 
              ckpt_path=None):
        
        # Create quality control folder
        quality_control_path = os.path.join(model_path, model_name, "Quality Control")
        if not os.path.isdir(quality_control_path):
            os.makedirs(quality_control_path, exist_ok=True)
            
        # Export configuration file
        yaml.dump(self.config, open(os.path.join(model_path, model_name, "config.yaml"), "w"))
        
        training_config = self.config["training_config"]

        optimizer = optim.Adam(self.parameters(), lr=training_config["optimizer"]["lr"])
        criterion = nn.MSELoss()

        if isinstance(ckpt_path, str):
            checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"]
            stats = checkpoint["stats"]
        else:
            start_step = 0
            stats = defaultdict(list)

        train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"],
                                num_workers=training_config["num_workers"], shuffle=True,
                                drop_last=training_config["drop_last"], pin_memory=True)
        
        valid_loader = DataLoader(valid_dataset, batch_size=training_config["batch_size"],
                                num_workers=training_config["num_workers"], shuffle=True,
                                drop_last=training_config["drop_last"], pin_memory=True)   

        train_loader_iter = iter(train_loader)
        for step in trange(start_step, training_config["num_steps"], desc="Steps: "):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)
            
            self.train()

            event, mask, index = [t.to(DEVICE) for t in batch]

            # defines x and y
            x = event
            y = mask

            optimizer.zero_grad()

            pred = self.forward(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            stats["train-loss"].append({
               "step" : step,
               "loss" : loss.item()
            })

            if (step + 1) % training_config["valid_interval"] == 0:
                self.eval()

                valid_loss = []
                for batch in valid_loader:
                   
                    event, mask, index = [t.to(DEVICE) for t in batch]

                    # defines x and y
                    x = event
                    y = mask

                    pred = self.forward(x)
                    loss = criterion(pred, y)  

                    valid_loss.append(loss.item())

                stats["valid-loss"].append({
                    "step" : step,
                    "loss" : numpy.mean(valid_loss),
                    "min-loss" : numpy.min(valid_loss),
                    "max-loss" : numpy.max(valid_loss),
                    "std-loss" : numpy.std(valid_loss)
                })

                # Save if best model so far
                if (len(stats["valid-loss"]) == 1) or (stats["valid-loss"][-1]["loss"] < numpy.min([l["loss"] for l in stats["valid-loss"][:-1]])):
                    checkpoint = {}
                    checkpoint['model'] = self.state_dict()
                    checkpoint['optimizer'] = optimizer.state_dict()
                    checkpoint['step'] = step + 1
                    checkpoint['stats'] = stats
                    torch.save(checkpoint, os.path.join(model_path, model_name, "results.pt"))

            if not save_best_ckpt_only:
                if (step + 1) % ckpt_period == 0:
                    checkpoint = {}
                    checkpoint['model'] = self.state_dict()
                    checkpoint['optimizer'] = optimizer.state_dict()
                    checkpoint['step'] = step + 1
                    checkpoint['stats'] = stats
                    torch.save(checkpoint, os.path.join(model_path, model_name, "checkpoint.pt"))               

            df = pandas.DataFrame.from_dict(stats["train-loss"])
            df.to_csv(os.path.join(quality_control_path, "train-stats.csv"))

            df = pandas.DataFrame.from_dict(stats["valid-loss"])
            df.to_csv(os.path.join(quality_control_path, "valid-stats.csv"))
    
    @classmethod
    def from_path(cls, model_path, model_name, checkpoint="results.pt"):
        
        checkpoint = torch.load(os.path.join(model_path, model_name, checkpoint), map_location=torch.device('cpu'))
        config = yaml.load(open(os.path.join(model_path, model_name, "config.yaml"), "r"), Loader=yaml.Loader)
        
        model = cls(config)
        model.load_state_dict(checkpoint["model"])
        
        return model

################################################################################################
# LOADER DEFINITION
################################################################################################

def prepare_for_torch(image, full3D=False):
    """Prepare the image for torch training

    Args:
        image (numpy.array): the image to prepare

    Returns:
        image_t (torch.FloatTensor): the image prepared for torch
    """
    if full3D:
        image = image[numpy.newaxis]
        image_t = torch.from_numpy(numpy.ascontiguousarray(image)).float()
        return image_t

    if image.ndim == 2:  # pour le cas du mask
        image = image[numpy.newaxis]
    image_t = torch.from_numpy(numpy.ascontiguousarray(image).astype(numpy.float32))
    return image_t

class PredictStream3D(Dataset):
    """Defines the PredictStream dataset

    This dataset is useful to help infer full stream.
    """

    def __init__(self, stream, shape2crop, step=None):
        self.shape2crop = shape2crop
        if step is None:
            step = numpy.array(shape2crop) // 2
        self.step = step

        deltaT, deltaH, deltaW = step

        T, H, W = stream.shape

        limites = ((0, T - shape2crop[0]),
                   (0, H - shape2crop[1]),
                   (0, W - shape2crop[2]))

        T_step = numpy.arange(limites[0][0], limites[0][1], deltaT)
        if T_step[-1] is not limites[0][1]:
            T_step = numpy.append(T_step, limites[0][1])

        H_step = numpy.arange(limites[1][0], limites[1][1], deltaH)
        if H_step[-1] is not limites[1][1]:
            H_step = numpy.append(H_step, limites[1][1])

        W_step = numpy.arange(limites[2][0], limites[2][1], deltaW)
        if W_step[-1] is not limites[2][1]:
            W_step = numpy.append(W_step, limites[2][1])

        # On calcul tous les coins de crop à inférer
        corners = list(itertools.product(*[T_step, H_step, W_step]))

        self.corners = corners
        self.stream = stream

    def __getitem__(self, index):
        corner = self.corners[index]
        slicesTHW = (slice(corner[0], corner[0] + self.shape2crop[0]),
                     slice(corner[1], corner[1] + self.shape2crop[1]),
                     slice(corner[2], corner[2] + self.shape2crop[2]))
        crop = self.stream[slicesTHW]
        crop = crop[numpy.newaxis]
        return prepare_for_torch(crop), torch.from_numpy(numpy.array(corner))

    def __len__(self):
        return len(self.corners)

class MSCTDataset(Dataset):
    """
    Creates a `Dataset` to load data from a `h5` dataset
    """
    def __init__(self, h5file, folds, crop_size=64, max_cache_size=64e+9, samples_pu=None, return_full=False, cache_mode="normal"):
        """
        Instantiates `MSCTDataset`

        :param h5file: A `h5py.File` of the h5file
        :param folds: A `list` of the accessible folds
        :param crop_size: An `int` of the size of the crop
        """
        super().__init__()
        self.h5file = h5file
        self.folds = folds

        if isinstance(crop_size, int):
            crop_size = tuple(crop_size for _ in range(3))
        self.crop_size = crop_size
        self.return_full = return_full

        self.cache_mode = cache_mode
        avail_cache_mode = ["normal", "full", "crop"]
        assert self.cache_mode in avail_cache_mode, f"This is not a valid cache mode: {avail_cache_mode}"
        self.max_cache_size = max_cache_size
        self.cache = {}        
        
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
                    for data_infos in info.values()
                    for di in data_infos 
                    if isinstance(di["cache-idx"], str)])     
    
    def _calc_cached_items(self, info):
        return sum([1 
                    for data_infos in info.values()
                    for di in data_infos 
                    if isinstance(di['cache-idx'], str)])

    def get_file_info(self):
        """
        Extracts the file information
        """

        info = {
            "input" : [],
            "label" : []
        }
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            for key in info.keys():
                print(f"Getting dataset: {key}")
                for fold in self.folds:
                    print(f"Getting fold: {fold}")
                    if isinstance(fold, str):
                        for gg, group in enumerate(tqdm(sorted(file[fold].keys(), key=lambda key : int(key)))):
                            if self.return_full:
                                event = file[fold][group][key]
                                info[key].append({
                                    "fold" : fold,
                                    "group" : group,
                                    "shape" : file[fold][group][key].shape,
                                    "event-idx" : None,
                                    "key" : "/".join((fold, group, key)),
                                    "datasize" : self._getsizeof(event),
                                    "cache-idx" : None,
                                    "is-empty" : False
                                })                                
                            else:
                                if f"cache-{key}" in file[fold][group]:
                                    for ee, (idx, event) in enumerate(tqdm(file[fold][group][f"cache-{key}"].items(), leave=False)):
                                        cache_idx = None
                                        is_empty = not numpy.any(event)
                                        current_cache_size = self._calc_current_cache_size(info)
                                        if (not is_empty) and (current_cache_size < self.max_cache_size):
                                            cache_idx = "/".join((fold, group, f"cache-{key}", idx))                                        
                                            self.cache[cache_idx] = event[()]
                                        info[key].append({
                                            "fold" : fold,
                                            "group" : group,
                                            "shape" : file[fold][group][f"cache-{key}"][idx].shape,
                                            "event-idx" : idx,
                                            "key" : "/".join((fold, group, f"cache-{key}", idx)),
                                            "datasize" : self._getsizeof(event),
                                            "cache-idx" : cache_idx,
                                            "is-empty" : is_empty
                                        })
                                else:
                                    for ee, (idx, event) in enumerate(tqdm(file[fold][group][key].items(), leave=False)):
                                        cache_idx = None
                                        is_empty = not numpy.any(event)
                                        current_cache_size = self._calc_current_cache_size(info)
                                        if (not is_empty) and (current_cache_size < self.max_cache_size):
                                            cache_idx = "/".join((fold, group, key, idx))                                        
                                            self.cache[cache_idx] = event[()]
                                        info[key].append({
                                            "fold" : fold,
                                            "group" : group,
                                            "shape" : file[fold][group][key][idx].shape,
                                            "event-idx" : idx,
                                            "key" : "/".join((fold, group, key, idx)),
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

        info = {
            "input" : [],
            "label" : []
        }
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            for key in info.keys():
                print(f"Getting dataset: {key}")
                for fold in self.folds:
                    print(f"Getting fold: {fold}")
                    if isinstance(fold, str):
                        for gg, group in enumerate(tqdm(sorted(file[fold].keys(), key=lambda key : int(key)))):
                            
                            for event in file[fold][group]["events"]:
                                event_idx, event = str(event[:1]), event[1:]

                                event_center = event.reshape(-1, 2)
                                event_center = numpy.mean(event_center, axis=-1).astype(int)
                                slc = tuple(
                                    slice(max(0, c - s // 2), min(_max, c + s // 2)) for c, s, _max in zip(event_center, self.crop_size, file[fold][group][key].shape)
                                )

                                event = file[fold][group][key][slc]

                                cache_idx = None
                                is_empty = not numpy.any(event)
                                current_cache_size = self._calc_current_cache_size(info)
                                if (not is_empty) and (current_cache_size < self.max_cache_size):
                                    cache_idx = "/".join((fold, group, key, event_idx))
                                    self.cache[cache_idx] = event

                                info[key].append({
                                    "fold" : fold,
                                    "group" : group,
                                    "shape" : self.crop_size,
                                    "event-idx" : event_idx,
                                    "key" : "/".join((fold, group, key)),
                                    "datasize" : self._getsizeof(event),
                                    "cache-idx" : cache_idx,
                                    "is-empty" : is_empty,
                                })

                            print(f"Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")
        return info    
    
    def get_file_info_pu(self):
        info = {
            "input" : [],
            "label" : []
        }
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            for key in info.keys():
                print(f"Getting dataset: {key}")
                for fold in self.folds:
                    print(f"Getting fold: {fold}")
                    if isinstance(fold, str):
                        # Adds positive samples
                        for sample in self.samples_pu["positive"][fold]:
                            cache_idx = None
                            group = sample["neuron"]
                            idx = sample["event-id"]
                            event = file[fold][group]["events"][idx]

                            cache_idx = "/".join((fold, group, f"cache-{key}", str(idx)))
                            event = file[cache_idx]
                            is_empty = not numpy.any(event)

                            current_cache_size = self._calc_current_cache_size(info)
                            if (not is_empty) and (current_cache_size < self.max_cache_size):
                                cache_idx = "/".join((fold, group, f"cache-{key}", str(idx)))
                                self.cache[cache_idx] = file[cache_idx][()]
                            else:
                                cache_idx = None
                                                   
                            idx = str(idx)
                            info[key].append({
                                "fold" : fold,
                                "group" : group,
                                "shape" : file[fold][group][f"cache-{key}"][idx].shape,
                                "event-idx" : idx,
                                "key" : "/".join((fold, group, f"cache-{key}", idx)),
                                "datasize" : self._getsizeof(event),
                                "cache-idx" : cache_idx,
                                "is-empty" : is_empty                                
                            })

                        print(f"[----] Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")

                        # Adds unlabeled samples
                        for idx, sample in enumerate(self.samples_pu["negative"][fold]):
                            group = sample["neuron"]
                            coord = sample["coord"]
                            idx = str(coord)

                            cache_idx = "/".join((fold, group, f"cache-unlabeled-{key}", idx))
                            current_cache_size = self._calc_current_cache_size(info)
                            if not (cache_idx in file):
                                cache_idx = None
                                is_empty = False
                                slc = tuple(slice(coord[i], coord[i] + self.crop_size[i]) for i in range(3))
                                datasize = 0
                                if current_cache_size < self.max_cache_size:
                                    cache_idx = "/".join((fold, group, f"cache-unlabeled-{key}", idx))
                                    event = file["/".join((fold, group, key))][slc]
                                    is_empty = not numpy.any(event)
                                    if not is_empty:
                                        self.cache[cache_idx] = event
                                    else:
                                        cache_idx = None

                                    datasize = self._getsizeof(event)
                                    
                                info[key].append({
                                    "fold" : fold,
                                    "group" : group,
                                    "shape" : self.crop_size,
                                    "event-idx" : idx,
                                    "key" : "/".join((fold, group, key)),
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
                            
                                info[key].append({
                                    "fold" : fold,
                                    "group" : group,
                                    "shape" : file[fold][group][f"cache-unlabeled-{key}"][idx].shape,
                                    "event-idx" : idx,
                                    "key" : "/".join((fold, group, f"cache-unlabeled-{key}", idx)),
                                    "datasize" : self._getsizeof(event),
                                    "cache-idx" : cache_idx,
                                    "is-empty" : is_empty                                    
                                })

                        print(f"[----] Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")
        return info     

    def get_file_info_pu_cache_mode_full(self):
        info = {
            "input" : [],
            "label" : []
        }
        print("Getting file information... This may take a while...")
        with h5py.File(self.h5file, "r") as file:
            for key in info.keys():
                print(f"Getting dataset: {key}")
                for fold in self.folds:
                    print(f"Getting fold: {fold}")
                    if isinstance(fold, str):
                        # Adds positive samples
                        for sample in self.samples_pu["positive"][fold]:
                            cache_idx = None
                            group = sample["neuron"]
                            idx = sample["event-id"]
                            event = file[fold][group]["events"][idx]

                            cache_idx = "/".join((fold, group, f"cache-{key}", str(idx)))
                            event = file[cache_idx]
                            is_empty = not numpy.any(event)

                            current_cache_size = self._calc_current_cache_size(info)
                            if (not is_empty) and (current_cache_size < self.max_cache_size):
                                cache_idx = "/".join((fold, group, f"cache-{key}", str(idx)))
                                self.cache[cache_idx] = file[cache_idx][()]
                            else:
                                cache_idx = None
                                                   
                            idx = str(idx)
                            info[key].append({
                                "fold" : fold,
                                "group" : group,
                                "shape" : file[fold][group][f"cache-{key}"][idx].shape,
                                "event-idx" : idx,
                                "key" : "/".join((fold, group, f"cache-{key}", idx)),
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
                            cache_idx = "/".join((fold, group, key))
                            if not (cache_idx in self.cache) and True:#(current_cache_size < self.max_cache_size):
                                self.cache[cache_idx] = file[fold][group][key][()]
                                datasize = self._getsizeof(self.cache[cache_idx])
                            elif (cache_idx in self.cache):
                                datasize = 0
                            else:
                                datasize = 0
                                cache_idx = None

                            info[key].append({
                                "fold" : fold,
                                "group" : group,
                                "shape" : self.crop_size,
                                "event-idx" : idx,
                                "key" : "/".join((fold, group, key)),
                                "datasize" : datasize,
                                "cache-idx" : cache_idx,
                                "is-empty" : False,
                                "slice" : slc
                            })


                        print(f"[----] Current cache size: {self._calc_current_cache_size(info) * 1e-9:0.2f}G; Cached items: {self._calc_cached_items(info)});")
        return info        

    def __getitem__(self, index):
        """
        Implements the getitem method of the `MSCTSequence`

        :param index: An `int` of the index
        """

        info_input = self.info["input"][index]
        info_label = self.info["label"][index]

        # Crop input
        if isinstance(info_input["cache-idx"], str):
            crop_input = self.cache[info_input["cache-idx"]]    
            if "slice" in info_input:
                crop_input = crop_input[info_input["slice"]]
        else:
            with h5py.File(self.h5file, "r") as file:
                if "slice" in info_input:
                    crop_input = file[info_input["key"]][info_input["slice"]]     
                else:
                    crop_input = file[info_input["key"]][()]
        
        # Crop label
        if info_label["is-empty"]:
            crop_label = numpy.zeros(self.crop_size, dtype=numpy.uint8)
        elif isinstance(info_label["cache-idx"], str):
            crop_label = self.cache[info_label["cache-idx"]]  
            if "slice" in info_label:
                crop_label = crop_label[info_label["slice"]]                   
        else:
            with h5py.File(self.h5file, "r") as file:
                if "slice" in info_label:
                    crop_label = file[info_label["key"]][info_label["slice"]]
                else:
                    crop_label = file[info_label["key"]][()]

        if not self.return_full:
            # Pads crop if not good size
            if crop_input.size != numpy.prod(self.crop_size):
                crop_input = numpy.pad(
                    crop_input,
                    [(0, cs - current) for cs, current in zip(self.crop_size, crop_input.shape)],
                    mode="symmetric"
                )
            if crop_label.size != numpy.prod(self.crop_size):
                crop_label = numpy.pad(
                    crop_label,
                    [(0, cs - current) for cs, current in zip(self.crop_size, crop_label.shape)],
                    mode="symmetric"
                )
        volume = crop_input.astype(numpy.float32)
        label = crop_label.astype(numpy.uint8)

        return {
            "input" : volume,
            "label" : label,
            "detection" : None,
            "index" : index,
            "shape2crop" : numpy.array([s // 2 for s in self.crop_size])
        }
    
    def __len__(self):
        return len(self.info["input"])

class LoadedEvent3D(Dataset):
    """
    Defines the LoadedEvent dataset. Si on veut ajouter un channel de detection
    en entrainement, on va tout simplement retourner un mask de deux channels

    :param data: `list` List of loaded events
    :param transform: (torchvision.transforms): Transforms to apply on the dataset
    :param center_crop: (bool): Whether a center crop is forced
    """

    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        event_dict = self.data[index]
        
        event = event_dict['input']
        mask = event_dict['label']
        eventIndex = event_dict['index']

        if self.transforms:
            event, mask = self.transforms((event, mask))

        event = prepare_for_torch(event, full3D=True)
        mask = prepare_for_torch(mask, full3D=True)

        return event, mask, int(eventIndex)

    def __len__(self):
        return len(self.data)

class MSCTDatasetFromFolder(Dataset):
    """
    Creates a `Dataset` to load data from a folder of tifffiles
    """
    def __init__(self, source_folder, target_folder, crop_size=64):
        """
        Instantiates `MSCTDatasetFromFolder`

        :param folds: A `list` of the accessible folds
        :param crop_size: An `int` of the size of the crop
        """
        super().__init__()
        self.source_folder = source_folder
        self.target_folder = target_folder
        
        self.source_files = glob.glob(os.path.join(self.source_folder, "*.tif"))
        self.target_files = [file.replace(self.source_folder, self.target_folder) for file in self.source_folder]
        
        remove_idx = []
        for i, (source_file, target_file) in enumerate(zip(self.source_files, self.target_files)):
            if not os.path.isfile(target_file):
                remove_idx.append(i)
                
        for idx in reversed(remove_idx):
            print(f"Source file: {self.source_files[idx]} does not have an annotated file... Removing")
            del self.source_files[idx]
            del self.target_files[idx]
        
    def __getitem__(self, index):
        """
        Implements the getitem method of the `MSCTSequence`

        :param index: An `int` of the index
        """
        
        crop_input = tifffile.imread(self.source_files[index])
        crop_label = tifffile.imread(self.target_files[index]) > 0

        # Pads crop if not good size
        if crop_input.size != numpy.prod(self.crop_size):
            crop_input = numpy.pad(
                crop_input,
                [(0, cs - current) for cs, current in zip(self.crop_size, crop_input.shape)],
                mode="symmetric"
            )
        if crop_label.size != numpy.prod(self.crop_size):
            crop_label = numpy.pad(
                crop_label,
                [(0, cs - current) for cs, current in zip(self.crop_size, crop_label.shape)],
                mode="symmetric"
            )
        volume = crop_input.astype(numpy.float32)
        label = crop_label.astype(numpy.uint8)

        return {
            "input" : volume,
            "label" : label,
            "detection" : None,
            "index" : index,
            "shape2crop" : numpy.array([s // 2 for s in self.crop_size])
        }
    
    def __len__(self):
        return len(self.source_files)
    
class RandomFlip(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.possible_flips = (-2, -1, None) # 1 flipud, 2 fliplr, None nothing

    def forward(self, x):
        x, y = x
        
        flipping_mode = random.choice(self.possible_flips)
        if flipping_mode:
            x = numpy.flip(x, flipping_mode)
            y = numpy.flip(y, flipping_mode) 

        return (x, y)   

class RandomCrop(nn.Module):
    def __init__(self, crop_size) -> None:
        super().__init__()

        self.crop_size = crop_size
        if isinstance(self.crop_size, int):
            self.crop_size = numpy.array([self.crop_size for _ in range(3)])
        
    def forward(self, x):
        
        x, y = x

        eventShape = numpy.array(x.shape)
        choices = eventShape - self.crop_size + 1
        corner = numpy.array([numpy.random.randint(low=0, high=d) for d in choices])

        slicesTHW = tuple([slice(corner[i], corner[i] + self.crop_size[i]) for i in range(3)])

        x = x[slicesTHW]
        y = y[slicesTHW]
        return (x, y)
    
class CenterCrop(nn.Module):
    def __init__(self, crop_size) -> None:
        super().__init__()

        self.crop_size = crop_size
        if isinstance(self.crop_size, int):
            self.crop_size = numpy.array([self.crop_size for _ in range(3)])
        
    def forward(self, x):
        
        x, y = x

        eventShape = numpy.array(x.shape)
        center = eventShape // 2
        corner = center - self.crop_size // 2

        slicesTHW = tuple([slice(corner[i], corner[i] + self.crop_size[i]) for i in range(3)])

        x = x[slicesTHW]
        y = y[slicesTHW]
        return (x, y)    

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

def filter_regionprops(regionprops, constraints):
    updated_regionprops, remove_coords = [], []
    for rprop in regionprops:
        t1, h1, w1, t2, h2, w2 = rprop.bbox
        lenT = t2 - t1
        lenH = h2 - h1
        lenW = w2 - w1
        should_remove = False
        if "minimal_time" in constraints:
            if lenT < constraints["minimal_time"]:
                should_remove = True
            if lenH < constraints["minimal_height"]:
                should_remove = True
            if lenW < constraints["minimal_width"]:
                should_remove = True
        if should_remove:
            remove_coords.extend(rprop.coords)
        else:
            updated_regionprops.append(rprop)
    return updated_regionprops

def compute_optimal_thresholds_detection(truth, pred):
    """
    Computes the optimal threshold to apply the prediction to match the ground
    truth mask

    :param truth: A `numpy.ndarray` of the ground truth mask
    :param pred: A `numpy.ndarray` of the predicted segmentation

    :returns : A `float` of the threshold
    """    
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().data.numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().data.numpy()
        
    thresholds = numpy.linspace(0.01, 1, 100)
    optimal_threshold, best_f1 = thresholds[0], -1
    for t in tqdm(thresholds, desc="Thresholds", leave=False):
        f1 = _compute_optimal_thresholds_detection(truth, pred, t)
        if f1 >= best_f1:
            optimal_threshold = t
            best_f1 = f1
    return optimal_threshold

def _compute_optimal_thresholds_detection(truth, pred, threshold):
    """
    Computes the optimal threshold to apply the prediction to match the ground
    truth mask

    :param truth: A `numpy.ndarray` of the ground truth mask
    :param pred: A `numpy.ndarray` of the predicted segmentation

    :returns : A `float` of the threshold
    """    
    truth, pred = truth.squeeze(), pred.squeeze()
    
    pred = pred > threshold
    
    # There are no predictions
    if not numpy.any(pred):
        return 0
    
    truth_label = measure.label(truth)
    truth_rprops = measure.regionprops(truth_label)
    truth_centroids = [rprop.centroid for rprop in truth_rprops]
    
    pred_label = measure.label(pred)
    pred_rprops = measure.regionprops(pred_label)
    
    # Filter small regions
    POSTPROCESS_PARAMS = {
        "minimal_time" : 2,
        "minimal_height" : 3,
        "minimal_width" : 3
    }
    pred_rprops = filter_regionprops(pred_rprops, POSTPROCESS_PARAMS)
    
    pred_centroids = []
    for rprop in pred_rprops:
        try:
            centroid = rprop.centroid
        except:
            centroid = rprop.bbox.reshape(-1, 2)
            centroid = numpy.mean(centroid, axis=-1)
        pred_centroids.append(centroid)
    # pred_centroids = [rprop.centroid for rprop in pred_rprops]
    
    scorer = CentroidDetectionError(truth_centroids, pred_centroids, threshold=6)
    
    return scorer.f1_score

def compute_optimal_thresholds(y_true, y_pred):
    """
    Computes the optimal thresholds for every item using a precision recall curve

    :param y_true: A `torch.Tensor` of ground truth
    :param y_pred: A `torch.Tensor` of prediction

    :returns : A `list` of thresholds
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().data.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().data.numpy()

    thresholds = []
    for truth, pred in zip(y_true, y_pred):
        if numpy.any(truth):
            thresholds.append(_compute_optimal_threshold(truth, pred))
    return thresholds

def _compute_optimal_threshold(truth, pred):
    """
    Computes the optimal threshold to apply the prediction to match the ground
    truth mask

    :param truth: A `numpy.ndarray` of the ground truth mask
    :param pred: A `numpy.ndarray` of the predicted segmentation

    :returns : A `float` of the threshold
    """
    precision, recall, thresholds = precision_recall_curve(truth.ravel(), pred.ravel())
    distances = distance.cdist(numpy.stack((precision, recall), axis=-1), numpy.array([[1., 1.]])).ravel()
    return thresholds[distances.argmin()]

# Below are templates for the function definitions for the export
# of pdf summaries for training and qc. You will need to adjust these functions
# with the variables and other parameters as necessary to make them
# work for your project
from datetime import datetime

def pdf_export(trained = False, augmentation = False, pretrained_model = False):
    pass
    # save FPDF() class into a
    # variable pdf
    #from datetime import datetime

#     class MyFPDF(FPDF, HTMLMixin):
#         pass

#     pdf = MyFPDF()
#     pdf.add_page()
#     pdf.set_right_margin(-1)
#     pdf.set_font("Arial", size = 11, style='B')

#     Network = "Your network's name"
#     day = datetime.now()
#     datetime_str = str(day)[0:10]

#     Header = 'Training report for '+Network+' model ('+model_name+')\nDate: '+datetime_str
#     pdf.multi_cell(180, 5, txt = Header, align = 'L')

#     # add another cell
#     if trained:
#       training_time = "Training time: "+str(hour)+ "hour(s) "+str(mins)+"min(s) "+str(round(sec))+"sec(s)"
#       pdf.cell(190, 5, txt = training_time, ln = 1, align='L')
#     pdf.ln(1)

#     Header_2 = 'Information for your materials and methods:'
#     pdf.cell(190, 5, txt=Header_2, ln=1, align='L')

#     all_packages = ''
#     for requirement in freeze(local_only=True):
#       all_packages = all_packages+requirement+', '
#     #print(all_packages)

#     #Main Packages
#     main_packages = ''
#     version_numbers = []
#     for name in ['tensorflow','numpy','Keras','csbdeep']:
#       find_name=all_packages.find(name)
#       main_packages = main_packages+all_packages[find_name:all_packages.find(',',find_name)]+', '
#       #Version numbers only here:
#       version_numbers.append(all_packages[find_name+len(name)+2:all_packages.find(',',find_name)])

#     cuda_version = subprocess.run('nvcc --version',stdout=subprocess.PIPE, shell=True)
#     cuda_version = cuda_version.stdout.decode('utf-8')
#     cuda_version = cuda_version[cuda_version.find(', V')+3:-1]
#     gpu_name = subprocess.run('nvidia-smi',stdout=subprocess.PIPE, shell=True)
#     gpu_name = gpu_name.stdout.decode('utf-8')
#     gpu_name = gpu_name[gpu_name.find('Tesla'):gpu_name.find('Tesla')+10]
#     #print(cuda_version[cuda_version.find(', V')+3:-1])
#     #print(gpu_name)

#     # pyplot.imread(Training_source+'/'+os.listdir(Training_source)[1]).shape
#     # dataset_size = len(os.listdir(Training_source))
#     shape = (32, 32, 32)
#     dataset_size = len(train_generator) + len(valid_generator)

    # text = 'The '+Network+' model was trained from scratch for '+str(number_of_epochs)+' epochs on '+str(dataset_size*number_of_patches)+' paired image patches (image dimensions: '+str(shape)+', patch size: ('+str(patch_size)+','+str(patch_size)+')) with a batch size of '+str(batch_size)+' and a '+config.train_loss+' loss function, using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). Key python packages used include tensorflow (v '+version_numbers[0]+'), Keras (v '+version_numbers[2]+'), csbdeep (v '+version_numbers[3]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

    # if pretrained_model:
    #   text = 'The '+Network+' model was trained for '+str(number_of_epochs)+' epochs on '+str(dataset_size*number_of_patches)+' paired image patches (image dimensions: '+str(shape)+', patch size: ('+str(patch_size)+','+str(patch_size)+')) with a batch size of '+str(batch_size)+' and a '+config.train_loss+' loss function, using the '+Network+' ZeroCostDL4Mic notebook (v '+Notebook_version[0]+') (von Chamier & Laine et al., 2020). The model was re-trained from a pretrained model. Key python packages used include tensorflow (v '+version_numbers[0]+'), Keras (v '+version_numbers[2]+'), csbdeep (v '+version_numbers[3]+'), numpy (v '+version_numbers[1]+'), cuda (v '+cuda_version+'). The training was accelerated using a '+gpu_name+'GPU.'

    # pdf.set_font('')
    # pdf.set_font_size(10.)
    # pdf.multi_cell(190, 5, txt = text, align='L')
    # pdf.set_font('')
    # pdf.set_font('Arial', size = 10, style = 'B')
    # pdf.ln(1)
    # pdf.cell(28, 5, txt='Augmentation: ', ln=0)
    # pdf.set_font('')
    # if augmentation:
    #   aug_text = 'The dataset was augmented by a factor of '+str(Multiply_dataset_by)+' by'
    #   if rotate_270_degrees != 0 or rotate_90_degrees != 0:
    #     aug_text = aug_text+'\n- rotation'
    #   if flip_left_right != 0 or flip_top_bottom != 0:
    #     aug_text = aug_text+'\n- flipping'
    #   if random_zoom_magnification != 0:
    #     aug_text = aug_text+'\n- random zoom magnification'
    #   if random_distortion != 0:
    #     aug_text = aug_text+'\n- random distortion'
    #   if image_shear != 0:
    #     aug_text = aug_text+'\n- image shearing'
    #   if skew_image != 0:
    #     aug_text = aug_text+'\n- image skewing'
    # else:
    #   aug_text = 'No augmentation was used for training.'
    # pdf.multi_cell(190, 5, txt=aug_text, align='L')
    # pdf.set_font('Arial', size = 11, style = 'B')
    # pdf.ln(1)
    # pdf.cell(180, 5, txt = 'Parameters', align='L', ln=1)
    # pdf.set_font('')
    # pdf.set_font_size(10.)
    # if Use_Default_Advanced_Parameters:
    #   pdf.cell(200, 5, txt='Default Advanced Parameters were enabled')
    # pdf.cell(200, 5, txt='The following parameters were used for training:')
    # pdf.ln(1)
    # html = """
    # <table width=40% style="margin-left:0px;">
    #   <tr>
    #     <th width = 50% align="left">Parameter</th>
    #     <th width = 50% align="left">Value</th>
    #   </tr>
    #   <tr>
    #     <td width = 50%>number_of_epochs</td>
    #     <td width = 50%>{0}</td>
    #   </tr>
    #   <tr>
    #     <td width = 50%>patch_size</td>
    #     <td width = 50%>{1}</td>
    #   </tr>
    #   <tr>
    #     <td width = 50%>number_of_patches</td>
    #     <td width = 50%>{2}</td>
    #   </tr>
    #   <tr>
    #     <td width = 50%>batch_size</td>
    #     <td width = 50%>{3}</td>
    #   </tr>
    #   <tr>
    #     <td width = 50%>number_of_steps</td>
    #     <td width = 50%>{4}</td>
    #   </tr>
    #   <tr>
    #     <td width = 50%>percentage_validation</td>
    #     <td width = 50%>{5}</td>
    #   </tr>
    #   <tr>
    #     <td width = 50%>initial_learning_rate</td>
    #     <td width = 50%>{6}</td>
    #   </tr>
    # </table>
    # """.format(number_of_epochs,str(patch_size)+'x'+str(patch_size),number_of_patches,batch_size,number_of_steps,percentage_validation,initial_learning_rate)
    # pdf.write_html(html)

    # #pdf.multi_cell(190, 5, txt = text_2, align='L')
    # pdf.set_font("Arial", size = 11, style='B')
    # pdf.ln(1)
    # pdf.cell(190, 5, txt = 'Training Dataset', align='L', ln=1)
    # pdf.set_font('')
    # pdf.set_font('Arial', size = 10, style = 'B')
    # pdf.cell(29, 5, txt= 'Training_source:', align = 'L', ln=0)
    # pdf.set_font('')
    # pdf.multi_cell(170, 5, txt = Training_source, align = 'L')
    # pdf.set_font('')
    # pdf.set_font('Arial', size = 10, style = 'B')
    # pdf.cell(27, 5, txt= 'Training_target:', align = 'L', ln=0)
    # pdf.set_font('')
    # pdf.multi_cell(170, 5, txt = Training_target, align = 'L')
    # #pdf.cell(190, 5, txt=aug_text, align='L', ln=1)
    # pdf.ln(1)
    # pdf.set_font('')
    # pdf.set_font('Arial', size = 10, style = 'B')
    # pdf.cell(22, 5, txt= 'Model Path:', align = 'L', ln=0)
    # pdf.set_font('')
    # pdf.multi_cell(170, 5, txt = model_path+'/'+model_name, align = 'L')
    # pdf.ln(1)
    # pdf.cell(60, 5, txt = 'Example Training pair', ln=1)
    # pdf.ln(1)
    # exp_size = io.imread("/content/NetworkNameExampleData.png").shape
    # pdf.image("/content/NetworkNameExampleData.png", x = 11, y = None, w = round(exp_size[1]/8), h = round(exp_size[0]/8))
    # pdf.ln(1)
    # ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
    # pdf.multi_cell(190, 5, txt = ref_1, align='L')
    # ref_2 = '- Your networks name: first author et al. "Title of publication" Journal, year'
    # pdf.multi_cell(190, 5, txt = ref_2, align='L')
    # if augmentation:
    #   ref_3 = '- Augmentor: Bloice, Marcus D., Christof Stocker, and Andreas Holzinger. "Augmentor: an image augmentation library for machine learning." arXiv preprint arXiv:1708.04680 (2017).'
    #   pdf.multi_cell(190, 5, txt = ref_3, align='L')
    # pdf.ln(3)
    # reminder = 'Important:\nRemember to perform the quality control step on all newly trained models\nPlease consider depositing your training dataset on Zenodo'
    # pdf.set_font('Arial', size = 11, style='B')
    # pdf.multi_cell(190, 5, txt=reminder, align='C')

    # pdf.output(model_path+'/'+model_name+'/'+model_name+"_training_report.pdf")


#Make a pdf summary of the QC results

def qc_pdf_export():
    pass
#   class MyFPDF(FPDF, HTMLMixin):
#     pass

#   pdf = MyFPDF()
#   pdf.add_page()
#   pdf.set_right_margin(-1)
#   pdf.set_font("Arial", size = 11, style='B')

#   Network = "Your network's name"
#   #model_name = os.path.basename(full_QC_model_path)
#   day = datetime.now()
#   datetime_str = str(day)[0:10]

#   Header = 'Quality Control report for '+Network+' model ('+QC_model_name+')\nDate: '+datetime_str
#   pdf.multi_cell(180, 5, txt = Header, align = 'L')

#   all_packages = ''
#   for requirement in freeze(local_only=True):
#     all_packages = all_packages+requirement+', '

#   pdf.set_font('')
#   pdf.set_font('Arial', size = 11, style = 'B')
#   pdf.ln(2)
#   pdf.cell(190, 5, txt = 'Development of Training Losses', ln=1, align='L')
#   pdf.ln(1)
#   exp_size = io.imread(full_QC_model_path+'Quality Control/QC_example_data.png').shape
#   if os.path.exists(full_QC_model_path+'Quality Control/lossCurvePlots.png'):
#     pdf.image(full_QC_model_path+'Quality Control/lossCurvePlots.png', x = 11, y = None, w = round(exp_size[1]/10), h = round(exp_size[0]/13))
#   else:
#     pdf.set_font('')
#     pdf.set_font('Arial', size=10)
#     pdf.multi_cell(190, 5, txt='If you would like to see the evolution of the loss function during training please play the first cell of the QC section in the notebook.', align='L')
#   pdf.ln(2)
#   pdf.set_font('')
#   pdf.set_font('Arial', size = 10, style = 'B')
#   pdf.ln(3)
#   pdf.cell(80, 5, txt = 'Example Quality Control Visualisation', ln=1)
#   pdf.ln(1)
#   exp_size = io.imread(full_QC_model_path+'Quality Control/QC_example_data.png').shape
#   pdf.image(full_QC_model_path+'Quality Control/QC_example_data.png', x = 16, y = None, w = round(exp_size[1]/10), h = round(exp_size[0]/10))
#   pdf.ln(1)
#   pdf.set_font('')
#   pdf.set_font('Arial', size = 11, style = 'B')
#   pdf.ln(1)
#   pdf.cell(180, 5, txt = 'Quality Control Metrics', align='L', ln=1)
#   pdf.set_font('')
#   pdf.set_font_size(10.)

#   pdf.ln(1)
#   html = """
#   <body>
#   <font size="7" face="Courier New" >
#   <table width=94% style="margin-left:0px;">"""
#   with open(full_QC_model_path+'Quality Control/QC_metrics_'+QC_model_name+'.csv', 'r') as csvfile:
#     metrics = csv.reader(csvfile)
#     header = next(metrics)
#     image = header[0]
#     precision = header[1]
#     recall = header[2]
#     iou = header[3]
#     header = """
#     <tr>
#       <th width = 10% align="left">{0}</th>
#       <th width = 15% align="left">{1}</th>
#       <th width = 15% align="center">{2}</th>
#       <th width = 15% align="left">{3}</th>
#     </tr>""".format(image, precision, recall, iou)
#     html = html+header
#     for row in metrics:
#       image = row[0]
#       precision = row[1]
#       recall = row[2]
#       iou = row[3]
#       cells = """
#         <tr>
#           <td width = 10% align="left">{0}</td>
#           <td width = 15% align="center">{1}</td>
#           <td width = 15% align="center">{2}</td>
#           <td width = 15% align="center">{3}</td>
#         </tr>""".format(image,str(round(float(precision),3)),str(round(float(recall),3)),str(round(float(iou),3)))
#       html = html+cells
#     html = html+"""</body></table>"""

#   pdf.write_html(html)

#   pdf.ln(1)
#   pdf.set_font('')
#   pdf.set_font_size(10.)
#   ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
#   pdf.multi_cell(190, 5, txt = ref_1, align='L')
#   ref_2 = '- Your networks name: first author et al. "Title of publication" Journal, year'
#   pdf.multi_cell(190, 5, txt = ref_2, align='L')

#   pdf.ln(3)
#   reminder = 'To find the parameters and other information about how this model was trained, go to the training_report.pdf of this model which should be in the folder of the same name.'

#   pdf.set_font('Arial', size = 11, style='B')
#   pdf.multi_cell(190, 5, txt=reminder, align='C')
#   print("CALLED")
#   pdf.output(full_QC_model_path+'Quality Control/'+QC_model_name+'_QC_report.pdf')

print("Depencies installed and imported.")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of configuration file")
    parser.add_argument("--seed", type=int, default=None, help="Overwrites the default random seed from the configuration file")    
    args = parser.parse_args()

    print(f"Loading from configuration file: `{args.config}`")
    CONFIG = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    if isinstance(args.seed, int):
        CONFIG["seed"] = args.seed
    

    # Build requirements file for local run
    # -- the developers should leave this below all the other installations
    after = [str(m) for m in sys.modules]
    # build_requirements_file(before, after)

    class bcolors:
        WARNING = '\033[31m'
        NORMAL = '\033[0m'  # white (normal)

    #@markdown ###Path to training images:

    Training_source = "" #@param {type:"string"}
    Training_source = "../data/80-20_calcium_dataset.h5"

    # Ground truth images
    Training_target = "" #@param {type:"string"}
    Training_target = "../data/80-20_calcium_dataset.h5"

    # model name and path
    #@markdown ###Name of the model and path to model folder:
    model_name = "" #@param {type:"string"}
    model_path = "" #@param {type:"string"}
    
    model_name = f"unet3D-ZeroCostDL4Mic_{CONFIG['training_config']['samples_pu']['positive']}_1-{CONFIG['training_config']['samples_pu']['unlabeled']}_{CONFIG['seed']}"
    model_path = "../data/baselines/ZeroCostDL4Mic"

    # other parameters for training.
    #@markdown ### Training Parameters
    #@markdown Number of steps:
    number_of_steps = 100000 #@param {type:"number"}

    #@markdown Other parameters, add as necessary
    patch_size =  32#@param {type:"number"} # in pixels


    #@markdown ###Advanced Parameters

    Use_Default_Advanced_Parameters = True #@param {type:"boolean"}
    #@markdown ###If not, please input:

    random_seed = CONFIG["seed"] #@param {type:"number"}
    batch_size =  128 #@param {type:"number"}
    initial_learning_rate =  0.0002 #@param {type:"number"}
    valid_interval = 100 #@param {type:"number"}
    ckpt_period = 100 #@param {type:"number"}

    if (Use_Default_Advanced_Parameters):
        print("Default advanced parameters enabled")
        batch_size = 128
        initial_learning_rate =  0.0002
        valid_interval = 100
        ckpt_period = 100
        percentage_validation = 0.2

    #Here we define the percentage to use for validation
    percentage = percentage_validation/100

    #here we check that no model with the same name already exist, if so delete
    if os.path.exists(model_path+'/'+model_name):
        shutil.rmtree(model_path+'/'+model_name)

    print("Parameters initiated.")

    # This will display a randomly chosen dataset input and output
    if os.path.isfile(Training_source):
        with h5py.File(Training_source, "r") as file:
            neuron_id = random.choice(list(file["train"].keys()))
            try:
                event_id = random.choice(list(file["train"][neuron_id]["input"].keys()))
                x = file["train"][neuron_id]["input"][event_id][()]
                y = file["train"][neuron_id]["label"][event_id][()]
            except AttributeError:
                event = random.choice(list(file["train"][neuron_id]["events"]))
                
                event_center = event[1:].reshape(-1, 2)
                event_center = (numpy.sum(event_center, axis=-1) / event_center.shape[-1]).astype(int)

                slc = tuple(
                    slice(max(0, c - s // 2), min(_max, c + s // 2)) for c, s, _max in zip(event_center, (64, 64, 64), file["train"][neuron_id]["input"].shape)
                )
                x = file["train"][neuron_id]["input"][slc]
                y = file["train"][neuron_id]["label"][slc]

    else:
        random_choice = random.choice(os.listdir(Training_source))
        x = tifffile.imread(os.path.join(Training_source, random_choice))
        y = tifffile.imread(os.path.join(Training_target, random_choice))

    # Here we check that the input images contains the expected dimensions
    if len(x.shape) == 3:
        print("Image dimensions (y,x)",x.shape)

    if not len(x.shape) == 3:
        print(bcolors.WARNING +"Your images appear to have the wrong dimensions. Image dimension",x.shape)

    #Find image XY dimension
    Image_Z = x.shape[0]
    Image_Y = x.shape[1]
    Image_X = x.shape[2]

    #Hyperparameters failsafes

    # Here we check that patch_size is smaller than the smallest xy dimension of the image

    if patch_size > min(Image_Z, Image_Y, Image_X):
        patch_size = min(Image_Y, Image_X)
        print (bcolors.WARNING + " Your chosen patch_size is bigger than the xy dimension of your image; therefore the patch_size chosen is now:",patch_size)

    # Here we check that patch_size is divisible by 8
    if not patch_size % 8 == 0:
        patch_size = ((int(patch_size / 8)-1) * 8)
        print (bcolors.WARNING + " Your chosen patch_size is not divisible by 8; therefore the patch_size chosen is now:",patch_size)

    #We save the example data here to use it in the pdf export of the training
    # pyplot.savefig('/content/NetworkNameExampleData.png', bbox_inches='tight', pad_inches=0)

    # Sets training seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)
    random.seed(random_seed)

    config = {
        "seed" : random_seed,
        "model_config": {
            "first_kernel_size": 3,
            "inC": 1,
            "n_layer": 5,
            "nbf": 8,
            "outC": 1,
            "r": 1,
            "use_batch_norm": True,
            "use_leaky_relu": True
        },
        "training_config": {
            "batch_size": batch_size,
            "drop_last": False,
            "inference_batch_size": 4,
            "num_steps": number_of_steps,
            "num_workers": 4,#os.cpu_count() - 1,
            "optimizer": {
            "lr": initial_learning_rate
            },
            "valid_interval": valid_interval,
            "valid_proportion": 0.2,
        }
    }
    model = UNet3D(config)
    model = model.to(DEVICE)

    def scroll_in_z(z):
        f=pyplot.figure(figsize=(16,8))
        pyplot.subplot(1,2,1)
        pyplot.imshow(x[z-1, :, :], cmap='gray', vmin=0, vmax=x.max())
        pyplot.title('Sample augmented source (z = ' + str(z) + ')', fontsize=15)
        pyplot.axis('off')

        pyplot.subplot(1,2,2)
        pyplot.imshow(y[z-1, :, :], cmap='gray')
        pyplot.title('Sample training target (z = ' + str(z) + ')', fontsize=15)
        pyplot.axis('off')

    print('This is what the augmented training images will look like with the chosen settings')
    # interact(scroll_in_z, z=widgets.IntSlider(min=1, max=x.shape[0], step=1, value=0));

    augmentations = []

    #@markdown ###<font color = orange>Add any further useful augmentations
    Use_Data_augmentation = False #@param{type:"boolean"}
    Use_Data_augmentation = True 

    #@markdown Select this option if you want to use augmentation to increase the size of your dataset

    #@markdown **Rotate each image 3 times by 90 degrees.**
    # Rotation = True #@param{type:"boolean"}

    #@markdown **Flip each image once around the x axis of the stack.**
    Flip = True #@param{type:"boolean"}
    if Flip:
        augmentations.append(RandomFlip())

    Crop = True #@param{type:"boolean"}
    if Crop:
        augmentations.append(RandomCrop(patch_size))
    else:
        augmentations.append(CenterCrop(patch_size))

    #@markdown **Would you like to save your augmented images?**

    # Save_augmented_images = False #@param {type:"boolean"}

    # Saving_path = "" #@param {type:"string"}

    # if not Save_augmented_images:
    #   Saving_path= "/content"

    # Samples Positive-Unlabeled
    Samples_PU = None #@param{type:"boolean"}
    Samples_PU = CONFIG["training_config"]["samples_pu"]

    if os.path.isfile(Training_source):
        train_data = MSCTDataset(Training_source, ["train"], samples_pu=Samples_PU, cache_mode="full")
        valid_data = MSCTDataset(Training_source, ["valid"], samples_pu=Samples_PU)
    else:
        train_valid_data = MSCTDatasetFromFolder(Training_source, Training_target)
        
        # Split train/valid
        numpy.random.seed(random_seed)
        indices = numpy.arange(len(train_valid_data))
        indices.shuffle()
        length = percentage * len(indices)
        
        train_data = Subset(train_valid_data, indices[length:])
        valid_data = Subset(train_valid_data, indices[:length])

    train_generator = LoadedEvent3D(train_data, transforms=Compose(augmentations))
    valid_generator = LoadedEvent3D(valid_data, transforms=CenterCrop(patch_size))

    if Use_Data_augmentation:
        print('Data augmentation enabled.')
        sample_src_aug, sample_tgt_aug, _ = train_generator[random.randint(0, len(train_generator))]

        def scroll_in_z(z):
            f=pyplot.figure(figsize=(16,8))
            pyplot.subplot(1,2,1)
            pyplot.imshow(sample_src_aug[0, z-1, :, :], cmap='gray', vmin=0, vmax=sample_src_aug.max())
            pyplot.title('Sample augmented source (z = ' + str(z) + ')', fontsize=15)
            pyplot.axis('off')

            pyplot.subplot(1,2,2)
            pyplot.imshow(sample_tgt_aug[0, z-1, :, :], cmap='gray')
            pyplot.title('Sample training target (z = ' + str(z) + ')', fontsize=15)
            pyplot.axis('off')

        print('This is what the augmented training images will look like with the chosen settings')
        #   interact(scroll_in_z, z=widgets.IntSlider(min=1, max=sample_src_aug.shape[3], step=1, value=0));

    else:
        print(bcolors.WARNING+"Data augmentation disabled" + bcolors.NORMAL)

    # @markdown ##Loading weights from a pre-trained network

    Use_pretrained_model = False #@param {type:"boolean"}
    Use_pretrained_model = False 

    pretrained_model_choice = "Model_from_file" #@param ["Model_from_file"]

    Weights_choice = "best" #@param ["checkpoint", "best"]
    if Weights_choice == "best":
        Weights_choice = "results"

    #@markdown ###If you chose "Model_from_file", please provide the path to the model folder:
    pretrained_model_path = "" #@param {type:"string"}
    pretrained_model_path = "../networks/unet3D/"

    # --------------------- Check if we load a previously trained model ------------------------
    if Use_pretrained_model:

    # --------------------- Load the model from the choosen path ------------------------
        if pretrained_model_choice == "Model_from_file":

            checkpoint_path = os.path.join(pretrained_model_path, Weights_choice+".pt")


    # --------------------- Download the a model provided in the XXX ------------------------

        if pretrained_model_choice == "Model_name":
            pretrained_model_name = "Model_name"
            pretrained_model_path = "/content/"+pretrained_model_name
            print("Downloading the 2D_Demo_Model_from_Stardist_2D_paper")
            if os.path.exists(pretrained_model_path):
                shutil.rmtree(pretrained_model_path)
            os.makedirs(pretrained_model_path)
            wget.download("", pretrained_model_path)
            wget.download("", pretrained_model_path)
            wget.download("", pretrained_model_path)
            wget.download("", pretrained_model_path)
            checkpoint_path = os.path.join(pretrained_model_path, "weights_"+Weights_choice+".h5")

    # --------------------- Add additional pre-trained models here ------------------------

    # --------------------- Check the model exist ------------------------
    # If the model path chosen does not contain a pretrain model then use_pretrained_model is disabled,
        if not os.path.exists(checkpoint_path):
            print(bcolors.WARNING+f'WARNING: {checkpoint_path} pretrained model does not exist' + bcolors.NORMAL)
            Use_pretrained_model = False


    # If the model path contains a pretrain model, we load the training rate,
        if os.path.exists(checkpoint_path):
        #Here we check if the learning rate can be loaded from the quality control folder
            if os.path.exists(os.path.join(pretrained_model_path, 'Quality Control', 'training_evaluation.csv')):

                with open(os.path.join(pretrained_model_path, 'Quality Control', 'training_evaluation.csv'),'r') as csvfile:
                    csvRead = pandas.read_csv(csvfile, sep=',')
                    #print(csvRead)

                if "learning rate" in csvRead.columns: #Here we check that the learning rate column exist (compatibility with model trained un ZeroCostDL4Mic bellow 1.4)
                    print("pretrained network learning rate found")
                    #find the last learning rate
                    lastLearningRate = csvRead["learning rate"].iloc[-1]
                    #Find the learning rate corresponding to the lowest validation loss
                    min_val_loss = csvRead[csvRead['val_loss'] == min(csvRead['val_loss'])]
                    #print(min_val_loss)
                    bestLearningRate = min_val_loss['learning rate'].iloc[-1]

                if Weights_choice == "last":
                    print('Last learning rate: '+str(lastLearningRate))

                if Weights_choice == "best":
                    print('Learning rate of best validation loss: '+str(bestLearningRate))

                if not "learning rate" in csvRead.columns: #if the column does not exist, then initial learning rate is used instead
                    bestLearningRate = initial_learning_rate
                    lastLearningRate = initial_learning_rate
                    print(bcolors.WARNING+'WARNING: The learning rate cannot be identified from the pretrained network. Default learning rate of '+str(bestLearningRate)+' will be used instead' + bcolors.NORMAL)

        #Compatibility with models trained outside ZeroCostDL4Mic but default learning rate will be used
            if not os.path.exists(os.path.join(pretrained_model_path, 'Quality Control', 'training_evaluation.csv')):
                print(bcolors.WARNING+'WARNING: The learning rate cannot be identified from the pretrained network. Default learning rate of '+str(initial_learning_rate)+' will be used instead'+ bcolors.NORMAL)
                bestLearningRate = initial_learning_rate
                lastLearningRate = initial_learning_rate


    # Display info about the pretrained model to be loaded (or not)
    if Use_pretrained_model:
        print('Weights found in:')
        print(checkpoint_path)
        print('will be loaded prior to training.')

    else:
        checkpoint_path = None
        print(bcolors.WARNING+'No pretrained nerwork will be used.' + bcolors.NORMAL)


    #@markdown ###<font color=orange> You will need to add or replace the code that loads any previously trained weights to the notebook here.

    import time
    import csv

    # Export the training parameters as pdf (before training, in case training fails)
    pdf_export(augmentation = Use_Data_augmentation, pretrained_model = Use_pretrained_model)

    start = time.time()

    #@markdown ## <font color=orange>Start training

    resume_training = False
    full_model_path = os.path.join(model_path, model_name)
    #here we check that no model with the same name already exist, if so delete
    if not resume_training and os.path.exists(full_model_path): 
        shutil.rmtree(full_model_path)
        print(bcolors.WARNING+'!! WARNING: Folder already exists and has been overwritten !!'+bcolors.NORMAL) 

    if not os.path.exists(full_model_path):
        os.makedirs(full_model_path)

    # Start Training

    # Sets training seed again to ensure reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)
    random.seed(random_seed)

    #Insert the code necessary to initiate training of your model
    model.train_model(
        train_generator,
        valid_generator,
        model_path = model_path,
        model_name = model_name,
        ckpt_path = checkpoint_path,
        save_best_ckpt_only = False,
        ckpt_period = ckpt_period,
    )

    #Note that the notebook should load weights either from the model that is
    #trained from scratch or if the pretrained weights are used (3.3.)

    # Displaying the time elapsed for training
    dt = time.time() - start
    mins, sec = divmod(dt, 60)
    hour, mins = divmod(mins, 60)
    print("Time elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)")

    # Export the training parameters as pdf (after training)
    pdf_export(trained = True, augmentation = Use_Data_augmentation, pretrained_model = Use_pretrained_model)
