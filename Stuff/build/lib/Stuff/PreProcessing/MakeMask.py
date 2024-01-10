import SimpleITK as sitk, numpy as np, platipy.dicom.io.rtstruct_to_nifti as plat, glob, os, scipy.ndimage as ndimg, pydicom as pdcm, nibabel as nib
from pathlib import Path
from typing import Optional, Union, Any

from monai.transforms import ScaleIntensity
intensity_scaler = ScaleIntensity(minv=0, maxv=1)

from threading import Thread

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

################################################################
######## String handling #######################################
################################################################

class SearchDict(dict):
    def __init__(self, base:dict=None, **kwargs):
        # Initializing the dictionary
        if isinstance(base, dict): 
            for key, item in base.items(): self.__setitem__(key, item)
        
        # Initial search for conflicts
        self.find_conflict()

    def find_conflict(self):
        keys = list(self.keys())
        self.conf = {k: [] for k in keys}

        for i in range(len(keys)):
            # isolate a key and get a list of remainder keys
            k = keys[i]
            _list = keys.copy()
            _list.pop(i)

            # check to see if there is a potential search conflict
            for k_ in _list:
                if k in k_: self.conf[k].append(k_)

    # as a function
    def __call__(self, key:str=None, value:Any=None) -> Union[Any, str]:
        if not key and not value: ValueError("Must have either a key or a value")
        if key: return super().__getitem__(key)
        if value: return list(self.keys())[list(self.values()).index(value)]

    # When element of the dictionary is called
    def __getitem__(self, k:str) -> Any:
        if not isinstance(k, str): TypeError(f'Input {k} must be a string')        
        # uniformity
        k = k.upper()

        # HARD CODE FIX
        if k[-4:] == '_RV1': 
            k = ''.join([*k[:-4]])


        # iterate over all known keys (kk)
        known_keys = list(self.keys())
        for kk in known_keys:

            # if one of the known keys (kk) is in the search key (k): 
            if kk in k:

                # Looking for a competing interst (ci) conflict
                if self.conf[kk]:
                    ci = False
                    for c in self.conf[kk]:
                        if c in k:
                            ci = True

                    # only if no competing interest (ci) return matched key (kk)
                    if not ci: return super().__getitem__(kk)

                # if no conflict, return matched key (kk)
                else: return super().__getitem__(kk)
            
        # if no matching key found
        return False


    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str): TypeError('String keys only supported')
        super().__setitem__(key.upper(), value)
        # find any conflicts within the keys
        self.find_conflict()   
        return None


def conflict(search_list: list[str]):

    conf = {k: [] for k in search_list}

    for i in range(len(search_list)):
        # isolate a key and get a list of remainder keys
        k = search_list[i]
        _list = search_list.copy()
        _list.pop(i)

        # check to see if there is a potential search conflict
        for k_ in _list:
            if k in k_: conf[k].append(k_)

    return conf


def search(search_key: str, search_list: list[str]):

    if not isinstance(search_key, str): TypeError(f'Input {search_key} must be a string')

    conf = conflict(search_list)

    # uniformity
    k = search_key.upper()
    # iterate over search list (sl)

    for sl in search_list:

        # if one of the known keys (kk) is in the search key (k): 
        if sl in k:

            # Looking for a competing interst (ci) conflict
            if conf[sl]:
                ci = False
                for c in conf[sl]:
                    if c in k:
                        ci = True

                # only if no competing interest (ci) return True
                if not ci: return True

            # if no conflict, return True
            else: return True
        
    # if no matching key found
    return False


################################################################
######## File handling #########################################
################################################################


def file_from_dir(
    path: Path, 
    file_identity: str = '*.dcm'
    ) -> Union[Path, list[Path]]:
    """
    Helper function to retrieve a file from a directory. If the directory has more than one file that matches the file_identity, then it will return a list of all matches

    ---
    Inputs:
    * path: path to the directory
    * file_identity: an identity to search for. Must include the variable *

    ---
    Returns:
    * path to a file or list of paths
    """
    assert os.path.isdir(path)
    assert '*' in file_identity
    file = glob.glob(os.path.join(path, file_identity))

    if len(file) == 1: return file[0]
    else:
        print(f'Path {path} returns more than one file with file_identity {file_identity}\nReturning all files that match')
        return file


file_search = lambda parent, search='*': sorted(glob.glob(os.path.join(parent, search)))
"""
Concatinating a function to return an interatable list of paths within a directory.
Inputs:
* parent: a path to the directory in question
* search: a string that defines the searchable parameter to narrow down the search 
"""


def label_crawl(
    src: Path,
    identifier_rtstruct: Optional[Union[str, list[str]]]=None,
    nested: Optional[int]=0,
    ) -> list:
    """
    Function to go through a directory and sub directories to exmaine the different RTStruct label names.
    This function presumes the target files are in the same "depth" of directory, i.e. they aren't necessary in the same directory, but they are in the same number of subdirectories.

    ---
    Inputs:
    *scr: Path to the starting, parent directory
    *identifier_rtstruct: str identifiers that narrow down the search. Must include or be the variable "*" and must have an element for every nest
    *nested: int that defines how many nested directories to look through

    ---
    Outputs:
    A list of all unique names found in the RTStruct files.
    """

    if not identifier_rtstruct: identifier_rtstruct=['*'] * nested 
    if isinstance(identifier_rtstruct, str): identifier_rtstruct = [identifier_rtstruct]
    rts_list = {-1: [src]}

    for i in range(nested):
        _rts_list = rts_list[i-1]
        rts_list[i] = []
        for dir in _rts_list:
            rts_list[i].extend([sub_dir for sub_dir in file_search(dir, identifier_rtstruct[i])])

    names = []
    for rts_dir in rts_list[nested - 1]:
        if os.path.isdir(rts_dir): rts_dir = file_from_dir(rts_dir)

        dicom_rtstruct = pdcm.read_file(rts_dir, force=True)
        struct_names = [ROI.ROIName for ROI in dicom_rtstruct.StructureSetROISequence]
        names.extend([name for name in struct_names if name not in names])

    return names


def check_defined_labels(
    labeled_names: Union[dict, SearchDict],
    ignored_names: Union[dict, SearchDict],
    src: Path,
    identifier_rtstruct: Optional[Union[str, list[str]]]=None,
    nested: Optional[int]=0,
    ) -> Union[bool, list]: 
    """
    A function to check if there are any missing defined labeles.
    """
    if not isinstance(labeled_names, dict) or not isinstance(labeled_names, SearchDict): TypeError('labeled_names must be a dict or SeachDict')
    if isinstance(labeled_names, dict): labeled_names=SearchDict(labeled_names)

    if not isinstance(ignored_names, dict) or not isinstance(ignored_names, SearchDict): TypeError('ignored_names must be a dict or SearchDict')
    if isinstance(ignored_names, dict): ignored_names=SearchDict(ignored_names)

    if not identifier_rtstruct: identifier_rtstruct=['*'] * nested 
    if isinstance(identifier_rtstruct, str): identifier_rtstruct = [identifier_rtstruct]
    rts_list = {-1: [src]}

    for i in range(nested):
        _rts_list = rts_list[i-1]
        rts_list[i] = []
        for dir in _rts_list:
            rts_list[i].extend([sub_dir for sub_dir in file_search(dir, identifier_rtstruct[i])])
    
    unknown_names = []
    for rts_dir in rts_list[nested - 1]:
        if os.path.isdir(rts_dir): rts_dir = file_from_dir(rts_dir)

        dicom_rtstruct = pdcm.read_file(rts_dir, force=True)
        struct_names = [ROI.ROIName for ROI in dicom_rtstruct.StructureSetROISequence]

        for name in struct_names:
            if not labeled_names[name] and not ignored_names[name] and not name in unknown_names: unknown_names.append(name)

    if unknown_names: return unknown_names
    else: return True


################################################################
######## Img / Mask Creation ###################################
################################################################


def get_images(dicom_img_paths = list[Path]) -> list[Any]:
    """
    Returns a list of nifti images from a list of paths
    """
    if not isinstance(dicom_img_paths, list): dicom_img_paths = [dicom_img_paths]
    for path in dicom_img_paths: assert os.path.exists(path)
    imgs = [sitk.GetArrayFromImage(plat.read_dicom_image(path)) for path in dicom_img_paths]
    return imgs


def extract_mask_ordered(
    dicom_image:Path,
    dicom_rtstruct:Path,
    label_dictionary:Optional[Union[SearchDict[str, int], dict[str, int]]]=None,
    output_img:Optional[bool]=False,
    channeled_label: Optional[bool]=False,
    single_channel: bool = False,
    expected_number_of_labels: int = 13,
    ) -> Union[np.ndarray, tuple[np.ndarray]]:
     # The image must be a dir
    assert os.path.isdir(dicom_image)
    # The RT struct must be a file. If it's a directory, try and return a .dcm file
    if not os.path.isfile(dicom_rtstruct): dicom_rtstruct = file_from_dir(dicom_rtstruct, file_identity='*.dcm')
    assert os.path.isfile(dicom_rtstruct)

    # Getting images, structures and their names from dicom files using platipy
    dicom_image = plat.read_dicom_image(dicom_image)
    dicom_rtstruct = plat.read_dicom_struct_file(dicom_rtstruct)
    struct_list, struct_name_sequence = plat.transform_point_set_from_dicom_struct(dicom_image, dicom_rtstruct)

    # Checking the label dictionary
    if label_dictionary:
        if isinstance(label_dictionary, dict): label_dictionary = SearchDict(label_dictionary)
        elif not isinstance(label_dictionary, SearchDict): TypeError(f'Label dictionary must be a dict or SearchDict, found {type(label_dictionary)}')
    
    else: label_dictionary = {_name: i+1 for i, _name in enumerate(struct_name_sequence)}

    # compiling a single channeled label
    if not channeled_label:
        # Initializing a list of all labels
        _structs = [None for i in range(expected_number_of_labels)]
        # Iterating over the name label pairs
        for _struct, _name in zip(struct_list, struct_name_sequence):
            map_value = label_dictionary[_name]
            if map_value:
                # Get the binary array of the structure
                _struct = sitk.GetArrayViewFromImage(_struct)
                _structs[map_value-1] = [map_value, _struct]
            
        if None in _structs:
            for i, _struct in enumerate(_structs):
                if _structs is None:
                    print(f'No structure found for number {i}')
        assert None not in _structs # Making sure all structs have been accounted for

        # Initializing the master label
        label = None

        # If only a sinle channel is desired as output -> concatinate them into a single list
        if single_channel:
            # Iterating through structures
            for map_value, _struct in _structs:
                if label is None: label = _struct
                label = np.where(_struct == 1, map_value, label) if label is not None else _struct * map_value
                
                print(np.unique(label))

        else:
            label = []
            for map_value, _struct in _structs:
                label.append(_struct)
            label = np.stack(label, axis=0)

    # compiling a multi-channel label
    else:
        # Initializing the master label
        shape = tuple([len(struct_list)]) + sitk.GetArrayViewFromImage(struct_list[0]).shape
        label = np.zeros((shape))
        # Iterating over the name label pairs
        for _struct, _name in zip(struct_list, struct_name_sequence):
            _struct = sitk.GetArrayViewFromImage(_struct)
            label[label_dictionary[_name] - 1, ...] = _struct

    # returning either the label or the image, label pair
    if output_img: return label, sitk.GetArrayFromImage(dicom_image)
    else: return label

    
def extract_mask(
    dicom_image:Path,
    dicom_rtstruct:Path,
    label_dictionary:Optional[Union[SearchDict[str, int], dict[str, int]]]=None,
    output_img:Optional[bool]=False,
    channeled_label: Optional[bool]=False,
    ) -> Union[np.ndarray, tuple[np.ndarray]]:
    """
    Creates a raw numpy segmentation mask from a dicom RTStruct and paired dicom image file. 
    Works from Platipy's backbone.

    ---
    Inputs:
    * dicom_image: path to the dicom image
    * dicom_rtstruct: path to the dicom RTStruct file that contains the segmentation structures
    * label_dictionary: optional dictionary that dictates the numerical value that corresponds to what structure
    * output_img: optional flag to output the original image volume along side the segmentation structure
    * channeled_label: optional flag to output the label as a multi-channel array instead of all stacked together

    ---
    Outputs:
    * Default: The segmentation mask as a np.ndarray
    * If output_img flag is True: the segmentation mask and the original image volume as np.ndarrays
    """
    # The image must be a dir
    assert os.path.isdir(dicom_image)
    # The RT struct must be a file. If it's a directory, try and return a .dcm file
    if not os.path.isfile(dicom_rtstruct): dicom_rtstruct = file_from_dir(dicom_rtstruct, file_identity='*.dcm')
    assert os.path.isfile(dicom_rtstruct)

    # Getting images, structures and their names from dicom files using platipy
    dicom_image = plat.read_dicom_image(dicom_image)
    dicom_rtstruct = plat.read_dicom_struct_file(dicom_rtstruct)
    struct_list, struct_name_sequence = plat.transform_point_set_from_dicom_struct(dicom_image, dicom_rtstruct)

    # Checking the label dictionary

    if label_dictionary:
        if isinstance(label_dictionary, dict): label_dictionary = SearchDict(label_dictionary)
        elif not isinstance(label_dictionary, SearchDict): TypeError(f'Label dictionary must be a dict or SearchDict, found {type(label_dictionary)}')
    
    else: label_dictionary = {_name: i+1 for i, _name in enumerate(struct_name_sequence)}

    # compiling a single channeled label
    if not channeled_label:
        # Initializing the master label
        label = None
        # Iterating over the name label pairs
        for _struct, _name in zip(struct_list, struct_name_sequence):
            map_value = label_dictionary[_name]
            if map_value:
                # Get the binary array of the structure
                _struct = sitk.GetArrayViewFromImage(_struct)
                if label is None: label = _struct
                label = np.where(_struct == 1, map_value, label) if label is not None else _struct * map_value
            else: print(_name)
    # compiling a multi-channel label
    else:
        # Initializing the master label
        shape = tuple([len(struct_list)]) + sitk.GetArrayViewFromImage(struct_list[0]).shape
        label = np.zeros((shape))
        # Iterating over the name label pairs
        for _struct, _name in zip(struct_list, struct_name_sequence):
            _struct = sitk.GetArrayViewFromImage(_struct)
            label[label_dictionary[_name] - 1, ...] = _struct

    # returning either the label or the image, label pair
    if output_img: return label, sitk.GetArrayFromImage(dicom_image)
    else: return label


def crop_np_by_array(
    array:np.ndarray, 
    coords:list[list[int]], 
    pad_to_shape: bool=True
    ) -> np.ndarray:
    """
    Function to crop a numpy array by a given list of indicies. If the indicies are outside the shape of the array, then the array will either trunkated or padded with zeros if toggled

    ---
    Inputs:
    * array: np.ndarray that contains N dimensions that will be cropped
    * coords: list of N sub-lists, each which contain the lower bound and the higher bound indicies to be cropped to
    * pad_to_shape: toggle to allow for a zero padding around the volume to achieve the desired indexing

    ---
    Output:
    * np.ndarray that is cropped and padded to match the sizes as defined by the coords
    """
    num_coords = len(coords)

    pad = []
    for i in range(num_coords):
        # Looking at one axis at a time
        d, u = coords[num_coords-1-i] # down, up
        array = np.moveaxis(array, -1, 0) # change the order of the array such that the first channel has the dim of interst

        # Ensuring the index doesn't go out of range, tracking remainders
        _s = array.shape[0] # The shape
        _ur, _dr = 0, 0 # up remainder, down remainder
        if u > _s: # If up is out of the range of the shape
            _ur = u - _s# - int((u + d)/2) # Find the difference between the two
            u = _s # Cap the up at the outer region
        if d < 0: # if down is below the range of the shape
            _dr = abs(d) # Find the abs value -> remainder
            d = 0 # Cap the down at the lowest possible value

        # Saving changes
        array = array[d:u] # Change the array to that difference
        print(array.shape)
        pad.append((_dr, _ur))

    # reverse the list because the loop went backwards
    pad.reverse()
    
    # pad the shape if specified
    print(pad)
    print(array.shape)
    if pad_to_shape: array = np.pad(array, tuple(pad), mode='constant', constant_values=0)
    return array

def crop_to_center_of_mass(
    img:np.ndarray, 
    dest_shape:list[int],
    pad_to_shape:bool=True,
    paired_imgs:Optional[list[np.ndarray]]=None
    ) -> list[np.ndarray]:

    # only supports length 2 or 3 images
    assert (len(img.shape) == 2) or (len(img.shape) == 3)
    # dimensions must be the same
    assert len(img.shape) == len(dest_shape)
    # Getting the center of the image through scipy
    CoM = ndimg.center_of_mass(img)

    # getting the list of cropped index arrays
    _c = []
    for i in range(len(CoM)):
        h = round(CoM[i]) + round(dest_shape[i]/2)
        l = round(CoM[i]) - round(dest_shape[i]/2)
        _c.append([l, h])


    print(_c)
    # Applying the crop to the img and any additional images
    out = crop_np_by_array(img, _c, pad_to_shape=pad_to_shape)
    if paired_imgs is None: return [out]
    else:
        if isinstance(paired_imgs, np.ndarray): paired_imgs = [paired_imgs]
        out_list = [out]
        for _img in paired_imgs:
            out_list.append(crop_np_by_array(_img, _c, pad_to_shape=pad_to_shape))
        
        return out_list


def resample(
    img:np.ndarray,
    src_voxel_size:Union[float, list[float]],
    dest_voxel_size:Union[float, list[float]]=1,
    order:int=3,
    mode:str='constant',
) -> np.ndarray:
    """
    Resamples the image from it's current voxel size to a destination voxel size. Based off of scipy.ndimage.zoom

    ---
    Inputs:
    * img: np.ndarray image / array to be resized
    * src_voxel_size: either a float or a list of floats that represent the current image's voxel size
    * dest_voxel_size: either a float or a list of floats that represent the target voxel size

    ---
    Outputs:
    * np.ndarray with the correct voxel spacing
    """

    # number of dimentions for the image
    ndim = len(img.shape)

    # Making the src_voxel_size a list of floats with length ndim
    if isinstance(src_voxel_size, int): src_voxel_size = float(src_voxel_size)
    if isinstance(src_voxel_size, float): src_voxel_size = [src_voxel_size for _ in range(ndim)]
    assert len(src_voxel_size) == ndim

    # Making the dest_voxel_size a list of floats with length ndim
    if isinstance(dest_voxel_size, int): dest_voxel_size = float(dest_voxel_size)
    if isinstance(dest_voxel_size, float): dest_voxel_size = [dest_voxel_size for _ in range(ndim)]
    assert len(dest_voxel_size) == ndim

    # Applying a zoom across the src and the destination
    zoom = [x/y for x, y in zip(src_voxel_size, dest_voxel_size)]
    resized_img = ndimg.zoom(img, zoom, order=order, mode=mode)

    return resized_img


def zero_mean_normalization(
    img:np.ndarray
    ) -> np.ndarray:
    """
    Function to apply a zero mean normalization to an image / array

    (array - mean) / subtraction
    """
    assert isinstance(img, np.ndarray)

    return (img - img.mean()) / img.std()

def ImgMaskPair(
    dicom_image:Path,
    dicom_rtstruct:Path,
    dest_shape: Union[int, list[int]],
    dest_voxel_size: Union[int, list[int]]=1,
    label_dictionary:Optional[dict[str, int]]=None,
    verbose:bool=False,
) -> list[np.ndarray]:
    """
    Function to generate a cropped, resampled, and zero-mean normalized image, mask pairs. 

    ---
    Inputs:
    * dicom_image: path to a directory that contains the .dcm files for an image volume
    * dicom_rtstruct: path to a directory or file that contains / is the RTStruct file (must only contain the RTStruct file)
    * dest_shape: the desired shape of the output image
    * dest_voxel_size: the desired spatial size of the voxels
    * label_dictionary: a dictionary that defines what value corresponds to each label name (names defined in the rt-struct file)
    * verbose: toggle for print statements to help debugging

    ---
    Output:
    * [image, mask] pair that has been cropped, resampled, and normalized
    """

    # Generate the image, mask pairs
    if verbose: print(f'Getting the raw image and mask pair')
    raw_mask, raw_img = extract_mask(
        dicom_image=dicom_image,
        dicom_rtstruct=dicom_rtstruct,
        label_dictionary=label_dictionary,
        output_img=True, # Hard Code
        channeled_label=False, # Hard Code
        )
    
    # Normalize the image
    if verbose: print('Normalzing the image')
    norm_img = zero_mean_normalization(raw_img)

    # Getting the voxel sizes:
    if verbose: print("Getting the source voxel size")
    src_voxel_size = list(plat.read_dicom_image(dicom_image).GetSpacing())
    src_voxel_size.reverse()

    # Resample both the normalized image and the raw mask the the desired size
    if verbose: print('Resampling the mask')
    resample_mask = resample(
        img=raw_mask,
        src_voxel_size=src_voxel_size,
        dest_voxel_size=dest_voxel_size,
        order=0,
    )

    if verbose: print('Resampling the image')
    resample_img = resample(
        img=norm_img,
        src_voxel_size=src_voxel_size,
        dest_voxel_size=dest_voxel_size,
        order=3
    )

    # Crop the the resampled image and resampled mask to the center of the mask
    if verbose: print('Cropping the image and mask pair')
    crop_mask, crop_img = crop_to_center_of_mass(
        img=resample_mask,
        dest_shape=dest_shape,
        pad_to_shape=True, # Hard Code
        paired_imgs=[resample_img]
    )

    return crop_img, crop_mask

def ImgMaskPlus(
    dicom_image:Path,
    dicom_rtstruct:Path,
    dest_shape: Union[int, list[int]],
    dest_voxel_size: Union[int, list[int]]=1,
    label_dictionary:Optional[dict[str, int]]=None,
    add_images:Optional[list[Path]]=None,
    verbose:bool=False,
    expected_number_of_labels = 13,
) -> list[np.ndarray]:
    """
    Function to generate a cropped, resampled, and zero-mean normalized image, mask pairs. 

    ---
    Inputs:
    * dicom_image: path to a directory that contains the .dcm files for an image volume
    * dicom_rtstruct: path to a directory or file that contains / is the RTStruct file (must only contain the RTStruct file)
    * dest_shape: the desired shape of the output image
    * dest_voxel_size: the desired spatial size of the voxels
    * label_dictionary: a dictionary that defines what value corresponds to each label name (names defined in the rt-struct file)
    * verbose: toggle for print statements to help debugging

    ---
    Output:
    * [image, mask] pair that has been cropped, resampled, and normalized
    """

    # Generate the image, mask pairs
    if verbose: print(f'Getting the raw image and mask pair')
    raw_mask, raw_img = extract_mask_ordered(
        dicom_image=dicom_image,
        dicom_rtstruct=dicom_rtstruct,
        label_dictionary=label_dictionary,
        output_img=True, # Hard Code
        channeled_label=False, # Hard Code
        expected_number_of_labels=expected_number_of_labels, # Hard Code
        )
    # raw_mask, raw_img = extract_mask(
    #     dicom_image=dicom_image,
    #     dicom_rtstruct=dicom_rtstruct,
    #     label_dictionary=label_dictionary,
    #     output_img=True, # Hard Code
    #     channeled_label=False, # Hard Code
    #     )
    
    # Generate any additional images if they are present
    if add_images: add_images = get_images(add_images)

    # Getting the voxel sizes:
    if verbose: print("Getting the source voxel size")
    src_voxel_size = list(plat.read_dicom_image(dicom_image).GetSpacing())
    src_voxel_size.reverse()

    # Resample both the normalized image and the raw mask the the desired size
    if verbose: print('Resampling the mask')
    resample_mask = resample(
        img=raw_mask,
        src_voxel_size=src_voxel_size,
        dest_voxel_size=dest_voxel_size,
        order=3,
    )
    print(np.unique(resample_mask))
    resample_mask = np.round(resample_mask)
    print(np.unique(resample_mask))

    if verbose: print('Resampling the image')
    resample_img = resample(
        img=raw_img,
        src_voxel_size=src_voxel_size,
        dest_voxel_size=dest_voxel_size,
        order=3
    )

    # Resampling any additional images
    if add_images: add_images = [resample(
            img=img,
            src_voxel_size=src_voxel_size,
            dest_voxel_size=dest_voxel_size,
            order=3
        ) for img in add_images]

    
    # Scale the image
    scaled_img = intensity_scaler(resample_img[None, ...])[0]
    if add_images: add_images = [intensity_scaler(img[None, ...])[0] for img in add_images]

    # Putting all images into a single list
    if add_images: paired_imgs = [scaled_img] + add_images
    else: paired_imgs = [scaled_img]

    # Crop the the resampled image and resampled mask to the center of the mask
    if verbose: print('Cropping the image and mask pair')
    cropped = crop_to_center_of_mass(
        img=resample_mask,
        dest_shape=dest_shape,
        pad_to_shape=True, # Hard Code
        paired_imgs=paired_imgs)

    return cropped

def ImgMaskPlusPlus(
    dicom_image:Path,
    dicom_rtstruct:Path,
    dest_shape: Union[int, list[int]],
    dest_voxel_size: Union[int, list[int]]=1,
    label_dictionary:Optional[dict[str, int]]=None,
    add_images:Optional[list[Path]]=None,
    verbose:bool=False,
    expected_number_of_labels = 13,
) -> list[np.ndarray]:
    """
    Function to generate a cropped, resampled, and zero-mean normalized image, mask pairs. 

    ---
    Inputs:
    * dicom_image: path to a directory that contains the .dcm files for an image volume
    * dicom_rtstruct: path to a directory or file that contains / is the RTStruct file (must only contain the RTStruct file)
    * dest_shape: the desired shape of the output image
    * dest_voxel_size: the desired spatial size of the voxels
    * label_dictionary: a dictionary that defines what value corresponds to each label name (names defined in the rt-struct file)
    * verbose: toggle for print statements to help debugging

    ---
    Output:
    * [image, mask] pair that has been cropped, resampled, and normalized
    """

    # Generate the image, mask pairs
    if verbose: print(f'Getting the raw image and mask pair')
    raw_mask, raw_img = extract_mask_ordered(
        dicom_image=dicom_image,
        dicom_rtstruct=dicom_rtstruct,
        label_dictionary=label_dictionary,
        output_img=True, # Hard Code
        channeled_label=False, # Hard Code
        expected_number_of_labels=expected_number_of_labels, # Hard Code
        single_channel=False,
        )
    print(raw_mask.shape)
    # raw_mask, raw_img = extract_mask(
    #     dicom_image=dicom_image,
    #     dicom_rtstruct=dicom_rtstruct,
    #     label_dictionary=label_dictionary,
    #     output_img=True, # Hard Code
    #     channeled_label=False, # Hard Code
    #     )
    
    # Generate any additional images if they are present
    if add_images: add_images = get_images(add_images)

    # Getting the voxel sizes:
    if verbose: print("Getting the source voxel size")
    src_voxel_size = list(plat.read_dicom_image(dicom_image).GetSpacing())
    src_voxel_size.reverse()

    # Resample both the normalized image and the raw mask the the desired size
    if verbose: print('Resampling the mask')
    # Multithread the process for faster processing
    mask_threads = []
    for sub_mask in raw_mask:
        mask_threads.append(
            ThreadWithReturnValue(target=resample, args = (sub_mask, src_voxel_size, dest_voxel_size, 0, 'constant'))
        )
        mask_threads[-1].start()
    mask_out = []
    for thread in mask_threads:
        mask_out.append(thread.join())

    # Combine the masks into a single, multivalue mask
    resampled_mask = None
    for i, sub_mask in enumerate(mask_out):
        sub_mask = np.round(sub_mask)
        if resampled_mask is None:
            resampled_mask = sub_mask * (i + 1)
        else:
            resampled_mask = np.where(sub_mask == 1, (i+1), resampled_mask)

    print(np.unique(resampled_mask))
    print(resampled_mask.shape)

    # print(np.unique(resample_mask))
    # resample_mask = np.round(resample_mask)
    # print(np.unique(resample_mask))

    if verbose: print('Resampling the image')
    resample_img = resample(
        img=raw_img,
        src_voxel_size=src_voxel_size,
        dest_voxel_size=dest_voxel_size,
        order=3
    )

    # Resampling any additional images
    if add_images: add_images = [resample(
            img=img,
            src_voxel_size=src_voxel_size,
            dest_voxel_size=dest_voxel_size,
            order=3
        ) for img in add_images]

    
    # Scale the image
    scaled_img = intensity_scaler(resample_img[None, ...])[0]
    if add_images: add_images = [intensity_scaler(img[None, ...])[0] for img in add_images]

    # Putting all images into a single list
    if add_images: paired_imgs = [scaled_img] + add_images
    else: paired_imgs = [scaled_img]

    # Crop the the resampled image and resampled mask to the center of the mask
    if verbose: print('Cropping the image and mask pair')
    cropped = crop_to_center_of_mass(
        img=resampled_mask,
        dest_shape=dest_shape,
        pad_to_shape=True, # Hard Code
        paired_imgs=paired_imgs)

    return cropped



################################################################
######## Volume Modification ###################################
################################################################



################################################################
######## Volume saving #########################################
################################################################

def save(
    img:Union[np.ndarray, list[np.ndarray]], 
    name:Union[str, list[str]], 
    dest:Union[Path, list[Path]],
    file_type:str='.nii.gz') -> None:

    # Make everything a list to handle multi-inputs
    if isinstance(img, np.ndarray): img = [img]
    if isinstance(name, str): name = [name]
    if isinstance(dest, Path): dest = [dest]

    # Everthing must be a list
    if not (isinstance(img, list) and isinstance(name, list) and isinstance(dest, list)): TypeError('Check input types')

    # if one destination is given, make everything save there
    if len(dest) == 1: dest = [dest for _ in range(len(img))]

    # Everything must have a set of img, name, dest
    if not (len(img) == len(dest)) or not (len(img) == len(name)): TypeError(f'Must have an equal amount of each input. {len(img)} images, {len(name)} names, {len(dest)} destinations')

    # loop through and save as nib imaegs
    for i in range(len(img)):
        # turn numpy array into nib image
        _nib = nib.Nifti1Image(img[i], np.eye(4))
        nib.save(_nib, os.path.join(dest[i], name[i]+file_type))

    return None

# hard coded cardiac substructure dictionary
Substructures = SearchDict({
    'RA': 1,
    'LA': 2,
    'RV': 3,
    'LV': 4,
    'AA': 5,
    'SVC': 6,
    'IVC': 7,
    'PA': 8,
    'PV': 9,
    'LMCA': 10,
    'LADA': 11,
    'RCA': 12,
})
    

if __name__ == '__main__':
    pass