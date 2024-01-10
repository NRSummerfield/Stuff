import os, glob, numpy as np, matplotlib.pyplot as plt, nibabel as nib, sys, pandas as pd
from typing import Optional
import argparse

if sys.version_info >= (3, 9):
    import matplotlib
    matplotlib.use("TkAgg")

def windowed(arr:np.ndarray, w:float=1.0) -> np.ndarray:
    _max = arr.max() * w
    return np.where(arr > _max, _max, arr)

class QA:
    def __init__(self, image_path:str, label_path:Optional[str] = None):

        self.labels_exist = False if not label_path else True
        self.type = None

        # finding the file type
        working_types = ['.nii.gz', '.npy']
        if os.path.isdir(image_path):
            _path, _file = os.path.split(next(iter(glob.glob(os.path.join(image_path, "*")))))
        else:
            _path, _file = os.path.split(image_path)

        for extension in working_types:
            if _file[-len(extension):] == extension:
                self.type = extension
                print(f'File extension {extension} found!')
        if not self.type: raise ValueError(f"src type not known, must be either '.nii.gz' OR '.npy'")


        if self.type == ".nii.gz":
            if self.labels_exist:
                images:list[str] = sorted(glob.glob(os.path.join(image_path, '*.nii.gz')))
                labels:list[str] = sorted(glob.glob(os.path.join(label_path, '*.nii.gz')))
                self.store:list[list[str]] = [[img, lab] for img, lab in zip(images, labels)]

            else: 
                self.store:list[str] = sorted(glob.glob(os.path.join(image_path, '*.nii.gz')))

        elif self.type == '.npy':
            if self.labels_exist:
                images = np.load(image_path, allow_pickle=True)
                labels = np.load(label_path, allow_pickle=True)
                self.store: list[np.ndarray] = [[img, lab] for img, lab in zip(images, labels)]
            
            else:
                self.store: list[np.ndarray] = np.load(image_path, allow_pickle=True)



        # Define the indicies 
        self.i:int = 0
        self.i_ = len(self.store)

        self.unpack() # Get the first set of images / names
        self.z = 0
        self.z_ = self.img.shape[-1]
        
        self.c = 0
        self.c_ = self.img.shape[0]

        self.w = 1
        
        # Setting up the figure
        self.fig, self.ax = plt.subplots(figsize=(9,6), ncols=2) if self.labels_exist else plt.subplots(figsize=(5,6), ncols=1)
        # Connecting the figure to the keyboard for user entry
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        # Adding text to the figure
        self.z_text = self.fig.text(0.01, 0.94, '', ha='left', va='top')
        self.i_text = self.fig.text(0.01, 0.89, '', ha='left', va='top')
        self.c_text = self.fig.text(0.01, 0.84, '', ha='left', va='top')
        self.w_text = self.fig.text(0.01, 0.06, '', ha='left', va='top')

        # QA_labs = self.fig.text(0.9, 0.94, '', ha='center', va='top')
        # QA_arts = self.fig.text(0.775, 0.94, '', ha='center', va='top')
        # QA_grad = self.fig.text(0.65, 0.94, '', ha='center', va='top')
        # self.QA_list = [QA_labs, QA_arts, QA_grad]

        # Loop that controls the running sequence
        self.running: bool = True
        while self.running: self.draw_image()
        # try:
            
        # except: 
        #     pass

    # Function to unpack the pair of images into data arrays / masks / names
    def unpack(self: list[list[str]]) -> list[np.ndarray, np.ndarray, str]:

        if self.type == '.nii.gz':
            if self.labels_exist:
                img, lab = self.store[self.i]
                self.name = os.path.split(img)[-1].split('.')[1]
                self.img = nib.load(img).get_fdata()
                self.lab = nib.load(lab).get_fdata()
                self.lab = np.ma.masked_where(self.lab==0, self.lab)
            else:
                img = self.store[self.i]
                self.name = os.path.split(img)[-1].split('.')[1]
                self.img = nib.load(img).get_fdata()
        
        else:
            self.name = f'Volume {self.i:03d}'
            if self.labels_exist:
                img, lab = self.store[self.i]
                self.img = img
                self.lab = np.ma.masked_where(lab==0, lab)
            else:
                self.img = self.store[self.i]

    
    # Function that updates the figure with the images
    def draw_image(self):

        # Update the text
        self.z_text.set_text(f'Current slice: {self.z + 1}/{self.z_}    Press right or left ["z" to reset to 0]')
        self.c_text.set_text(f'Current channel: {self.c + 1}/{self.c_}    Press up or down')
        self.i_text.set_text(f'Current image: {self.name} ({self.i + 1}/{self.i_})    Press "a" or "d" ["i" to reset to 0]')
        self.w_text.set_text(f'Current window level: {self.w}, press "w" or "e" to lower / raise')

        _img = windowed(self.img[self.c], self.w)
        
        # Clear the images
        if self.labels_exist:
            for ax in self.ax:
                ax.clear()
                ax.axis('off')

            # Show the updated images
            self.ax[0].imshow(_img[..., self.z], cmap='gray', vmin=_img.min(), vmax=_img.max())
            self.ax[1].imshow(_img[..., self.z], cmap='gray', vmin=_img.min(), vmax=_img.max())
            self.ax[1].imshow(self.lab[0, ..., self.z], cmap='Reds', vmin=self.lab.min(), vmax=self.lab.max(), interpolation='none')

            # Adjust and draw the images
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top = 0.8, wspace=0.025, hspace=0)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()    
            plt.pause(0.001)
        
        else:
            self.ax.clear()
            self.ax.axis('off')
            self.ax.imshow(_img[..., self.z], cmap='gray', vmin=_img.min(), vmax=_img.max())
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top = 0.8, wspace=0.025, hspace=0)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()    
            plt.pause(0.001)

    # command that prints out the QA dict & saves to CSV file
    def print(self):

        out_dict = {'Names': [],'Correct Mask?': [], 'Artifact?': [], 'Gradient?': []}

        for k in self.qa_dict.keys():
            out_dict['Names'].append(k)
            print(f'{k}:')
            for kk in self.qa_dict[k].keys():
                out_dict[kk].append(self.qa_dict[k][kk])
                print(f'\t{kk} = {self.qa_dict[k][kk]}')

        df = pd.DataFrame.from_dict(out_dict, orient='index')
        df.to_csv(self.csv_save)


    # Commands that are followed when a key is pressed
    def on_press(self, event):
        
        # This group changes the image / slice
        rec_i = self.i
        if event.key == 'q':
            self.running = False

        if event.key == 'left':
            self.z -= 1
        
        if event.key == 'right':
            self.z += 1

        if event.key == 'z':
            self.z = 0

        if event.key == 'down':
            self.c -= 1

        if event.key == 'up':
            self.c += 1


        if event.key == 'a':
            self.i += 1
        
        if event.key == 'd':
            self.i -= 1

        if event.key == 'i':
            self.i = 0

        if event.key == "w":
            self.w -= 0.05
        
        if event.key == 'e':
            self.w += 0.05

        self.z = self.z % self.z_
        self.i = self.i % self.i_
        self.c = self.c % self.c_
        
        self.w = round(self.w, 3)
        if self.w < 0.1: self.w = 0.1
        if self.w > 1: self.w = 1

        if self.i - rec_i: self.unpack()

            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('Images')
    parser.add_argument('-l', '--labels', default=None)

    args = parser.parse_args()

    QA(image_path=args.Images, label_path=args.labels)