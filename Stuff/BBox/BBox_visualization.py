import numpy as np
from matplotlib import patches, pyplot as plt
from typing import Optional
import scipy.ndimage as ndimg

from BBox_functions import bounding_box

import sys

if sys.version_info >= (3, 9):
    import matplotlib
    matplotlib.use("TkAgg")
   
def bbox_view(src_img: np.ndarray, pred: np.ndarray, w: list[float] = [1, 1, 1], ratios: list[float] = [1, 1, 1], save: Optional[str] = None):
    fig, axs = plt.subplots(ncols=3, dpi=200, figsize=(6,3))

    # plot 1: axial
    img = src_img.copy()
    plab = pred.copy()

    bbox = bounding_box(plab, option='cube', scale_to_1=False)
    z = int(bbox[2] + bbox[5]/2)

    img = img[:, :, z]
    _max = img.max() *w[0]
    img[img > _max] = _max

    axial_rect = {"xy": [*bbox[:2]], 'width': bbox[3], 'height': bbox[4]}
    axial_rect = patches.Rectangle(**axial_rect, linewidth=1, edgecolor='r', facecolor='none')

    ax: plt.Axes = axs[0]
    ax.imshow(img, cmap='gray', aspect=ratios[0])
    ax.add_artist(axial_rect)
    ax.set_title(f'Axial slice')
    ax.axis('off')

    # plot 2: Sagital
    img = src_img.copy()
    plab = pred.copy()
    img = np.rot90(img, k=1, axes=(0, 2))
    plab = np.rot90(plab, k=1, axes=(0, 2))

    z = int(bbox[1] + bbox[4]/2)

    img = img[:, z, :]
    _max = img.max() * w[1]
    img[img > _max] = _max

    bbox = bounding_box(plab, option='cube', scale_to_1=False)

    axial_rect = {"xy": [*bbox[:2]], 'width': bbox[3], 'height': bbox[4]}
    axial_rect = patches.Rectangle(**axial_rect, linewidth=1, edgecolor='r', facecolor='none')

    ax: plt.Axes = axs[1]
    ax.imshow(img, cmap='gray', aspect=ratios[1])
    ax.add_artist(axial_rect)
    ax.set_title(f'Sagital slice')
    ax.axis('off')


    # plot 2: Coronal
    img = src_img.copy()
    plab = pred.copy()
    img = np.rot90(img, k=1, axes=(1, 2))
    plab = np.rot90(plab, k=1, axes=(1, 2))

    z = int(bbox[0] + bbox[3]/2)

    img = img[z, :, :]
    _max = img.max() * w[2]
    img[img > _max] = _max

    bbox = bounding_box(plab, option='cube', scale_to_1=False)

    axial_rect = {"xy": [*bbox[:2]], 'width': bbox[3], 'height': bbox[4]}
    axial_rect = patches.Rectangle(**axial_rect, linewidth=1, edgecolor='r', facecolor='none')

    ax: plt.Axes = axs[2]
    ax.imshow(img, cmap='gray', aspect=ratios[2])
    ax.add_artist(axial_rect)
    ax.set_title(f'Coronal slice')
    ax.axis('off')


    plt.tight_layout()

    if save:
        plt.savefig(save, format='pdf')
    plt.show()


def windowed(arr:np.ndarray, w:float=1.0) -> np.ndarray:
    _max = arr.max() * w
    return np.where(arr > _max, _max, arr)


class QA_bbox:
    def __init__(self, image:np.ndarray, bbox: list[int]) -> list[int]:
        self.img = image
        self.bbox = bbox
        self.initial_bbox = bbox.copy()

        # plottable figure with interaction
        self.fig, self.ax = plt.subplots(figsize=(5,6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        # Adding in direction text
        self.top_line = self.fig.text(0.01, 0.99, "", ha='left', va='top')

        text1 = "INCREASE bbox value: [1,2,3,4,5,6], uniform expansion: 8\n"
        text2 = "DECREASE bbox value: [!,@,#,$,%,^], uniform reduction: *\n"
        text3 = "increase / decrease z: right / left\nQuit: q\nincrease / decrease w: w / e"
        text = text1 + text2 + text3
        self.top_line.set_text(text)

        # variables that will effect the plotting
        self.z = 0
        self._z = image.shape[-1]
        
        self.w = 1
        self.wimg = windowed(self.img, self.w)

        # running loop
        self.running = True
        while self.running:
            self.update()
            
    def get_difference(self) -> list[int]:
        return [nb - ib for ib, nb in zip(self.initial_bbox, self.bbox)]

    # updating the graph
    def update(self):
        # resetting it
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_title(f'bbox: {self.bbox}, z:{self.z}, w:{self.w}')
        
        self.ax.imshow(self.wimg[..., self.z], cmap='gray', vmin=self.wimg.min(), vmax=self.wimg.max())
        
        # getting the bounding box
        if (self.z > self.bbox[2]) & (self.z < (self.bbox[2] + self.bbox[5])):
            rect_args = {"xy": [*self.bbox[:2]], 'width': self.bbox[3], 'height': self.bbox[4]}
            rect = patches.Rectangle(**rect_args, linewidth=2, edgecolor='r', facecolor='none')
            self.ax.add_artist(rect)
        
        # pushing it
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()    
        plt.pause(0.00001)

    def on_press(self, event):

        current_w = self.w
        if event.key == '1': 
            self.bbox[0] += 1
            self.bbox[3] -= 1
        elif event.key == '2': 
            self.bbox[1] += 1
            self.bbox[4] -= 1
        elif event.key == '3': 
            self.bbox[2] += 1
            self.bbox[5] -= 1
        elif event.key == '4': 
            self.bbox[3] += 1
        elif event.key == '5': 
            self.bbox[4] += 1
        elif event.key == '6': 
            self.bbox[5] += 1

        elif event.key == '!': 
            self.bbox[0] -= 1
            self.bbox[3] += 1
        elif event.key == '@': 
            self.bbox[1] -= 1
            self.bbox[4] += 1
        elif event.key == '#': 
            self.bbox[2] -= 1
            self.bbox[5] += 1
        elif event.key == '$': 
            self.bbox[3] -= 1
        elif event.key == '%': 
            self.bbox[4] -= 1
        elif event.key == '^': 
            self.bbox[5] -= 1

        elif event.key == '8':
            self.bbox = list(np.asarray(self.bbox) + np.asarray([-1, -1, -1, 2, 2, 2]))
        elif event.key == '*':
            self.bbox = list(np.asarray(self.bbox) + np.asarray([1, 1, 1, -2, -2, -2]))

        elif event.key == 'right': self.z += 1
        elif event.key == 'left': self.z -= 1

        elif event.key == 'w': self.w -= 0.05
        elif event.key == 'e': self.w += 0.05

        elif event.key == 'q': self.running = False

        self.z = self.z % self._z
        if self.w < 0.05: self.w = 0.05
        if self.w > 1: self.w = 1
        self.w = round(self.w, 3)

        if self.w != current_w:
            self.wimg = windowed(self.img, w=self.w)


class view_pred:
    def __init__(self, image:np.ndarray, pred: np.ndarray):
        self.img = image
        self.pred = np.ma.masked_where(pred==0, pred)

        # plottable figure with interaction
        self.fig, self.ax = plt.subplots(figsize=(5,6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        
        # variables that will effect the plotting
        self.z = int(ndimg.center_of_mass(np.where(pred!=0, 1, 0))[-1])
        self._z = image.shape[-1]
        
        self.w = 1
        self.wimg = windowed(self.img, self.w)

        # running loop
        self.running = True
        while self.running:
            self.update()
            
    # updating the graph
    def update(self):
        # resetting it
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_title(f'z:{self.z}, w:{self.w}')

        # getting the image
        self.ax.imshow(self.wimg[..., self.z], cmap='gray', vmin=self.wimg.min(), vmax=self.wimg.max())
        self.ax.imshow(self.pred[..., self.z], cmap='cool', vmin=0, vmax=self.pred.max())
        
        # pushing it
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()    
        plt.pause(0.00001)

    def on_press(self, event):

        current_w = self.w
        if event.key == 'right': self.z += 1
        elif event.key == 'left': self.z -= 1

        elif event.key == 'w': self.w -= 0.05
        elif event.key == 'e': self.w += 0.05

        elif event.key == 'q': self.running = False

        self.z = self.z % self._z
        if self.w < 0.05: self.w = 0.05
        if self.w > 1: self.w = 1
        self.w = round(self.w, 3)

        if self.w != current_w:
            self.wimg = windowed(self.img, self.w)