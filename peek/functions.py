import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from pathlib import Path
from scipy.special import entr

def compute_PEEK(feature_maps):
    # make feature map positive
    positivized_maps = feature_maps + torch.abs(torch.min(feature_maps)) + 1e-10
    # compute entropy maps
    ## dim = 1 for pytorch friends, dim= -1 is for the widths
    peek_map = -torch.sum(positivized_maps * torch.log(positivized_maps), dim=1)

    peek_map = peek_map.unsqueeze(1)  # Add the missing dimension back
    return peek_map
    
    return peek_map

def plot_PEEK(modules, frame_paths, feature_folder, save_path=False, run_path=False, verbose=False):
        
    # loop over the frames for which we plot PEEK maps
    for frame_path in frame_paths:
        # find filename and path to feature maps
        frame_filename = os.path.split(frame_path)[-1]
        # frame_filename = os.path.split(image_path)[-1].split('.')[0]

        feature_map_path = f"{feature_folder}/{frame_filename.split('.')[0]}.pkl"
        
        # make subplots -- 3 columns if we include inferred images
        if run_path:
            cols=3
            fig, axes = plt.subplots(len(modules), cols)

        else:
            cols=2
            fig, axes = plt.subplots(len(modules), cols)
        
        # read image and get dimensions
        image = plt.imread(frame_path)
        h, w, _ = image.shape
         
        # load all feature maps for the image
        with open(feature_map_path, 'rb') as f:
            loaded_feature_maps = pickle.load(f)
            
        # plot for each module
        for i, layer in enumerate(modules):
            # plot the original image
            axes[i,0].imshow(image)
            
            # compute PEEK map
            feature_maps = loaded_feature_maps[layer][0].cpu().numpy()
            feature_maps = np.moveaxis(feature_maps, 0, -1)
            peek_map = compute_PEEK(feature_maps, h, w)
            
            # plot the original image with semi-transparent PEEK map on top
            axes[i,1].imshow(image)
            axes[i,1].imshow(peek_map, alpha=0.7, cmap='jet')
            
            # add column titles
            if i==0:
                axes[i,0].set_title('Input')
                axes[i,1].set_title('PEEK')
                if run_path: axes[i,2].set_title('Predictions')
               
            # add row title
            axes[i,0].set_ylabel(f'Module {layer}')
            
            # plot the image with its bounding boxes
            if run_path:
                inferred_image = plt.imread(f'{run_path}/{frame_filename}')
                axes[i,2].imshow(inferred_image)
            
        # print status updates
        if verbose:
            print(f'Finished with frame {frame_path}.')
            if save_path: print(f'Saving figure to {save_path}/{frame_filename}')
            
        # Add row labels and hide all axes
        for i in range(len(modules)):                
            for j in range(cols):  # Iterate through columns
                axes[i, j].set_xticks([])  # Hide the x-axis ticks and labels
                axes[i, j].set_yticks([])  # Hide the y-axis ticks and labels
            
        # tighten the layout
        fig.tight_layout()
        
        # save figures
        if save_path:
            # create folder if it does not exist
            save_fig_path = f'{save_path}/{frame_filename}'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # save figure
            fig.savefig(save_fig_path)
            fig.clear()

class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self, weights='DEFAULT'):
        super(VGG16FeatureExtractor, self).__init__()
        self.vgg16 = models.vgg16(weights=weights).features
        # Automatically collect indices of all convolutional layers
        self.conv_layers = [i for i, layer in enumerate(self.vgg16) if isinstance(layer, torch.nn.Conv2d)]

    def forward(self, x):
        features = []
        for layer_index, layer in enumerate(self.vgg16):
            x = layer(x)
            if layer_index in self.conv_layers:
                features.append(x)
        return features

    def load_image(self, image_path):
        # Load an image and transform it to the format required by VGG16
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    def save_features(self, frame_folder):
        _, save_folder = frame_folder.split('/')
        feature_folder = f'feature_maps/{save_folder}'

        Path(feature_folder).mkdir(parents=True, exist_ok=True)

        image_filepaths = sorted(glob.glob(f'{frame_folder}/*'))

        for image_path in image_filepaths:
            
            input_tensor = self.load_image(image_path)
            
            with torch.no_grad():
                features = self.forward(input_tensor)

            # Create a base filename for saving features without the original extension
            base_filename = os.path.split(image_path)[-1].split('.')[0]
            
            filename = f'feature_maps/{save_folder}/{base_filename}.pkl'
            with open(filename, "wb") as f:
                pickle.dump([feature for feature in features], f)
            print(f"Saved all features to {filename}")

class check_PEEKs(Exception):
    def __init__(self, message = "PEEK lists must be the same length"):
        self.message = message
        super().__init__(self.message)

    @staticmethod
    def check_length(a, b):
        if len(a) != len(b):
            raise unequal_PEEKs(self.message)

class PEEKLossWrapper(nn.Module):
    def __init__(self, base_loss_fn, custom_weight=1.0, pos=2, alpha=1.0, beta=1.0, lam=1.0):
        """
        Args:
            base_loss_fn (callable): The base loss function (e.g., nn.CrossEntropyLoss()).
            custom_weight (float): Weight for the custom PEEK loss term.
            pos (int): Interval for grouping PEEK outputs.
            beta (float): Weight for the difference term in PEEK loss.
            lam (float): Weight for the conservation term in PEEK loss.
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.custom_weight = custom_weight
        self.pos = pos
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def forward(self, input, target, current_PEEKs, past_PEEKs):
        """
        Args:
            input (Tensor): Model predictions for the base loss.
            target (Tensor): Ground truth targets for the base loss.
            PEEKs (PEEK maps from models): Extra model outputs used for computing the custom PEEK loss.
        """
        # Compute the base loss using the provided loss function
        base_loss, loss_items = self.base_loss_fn(input, target)

        try:
            check_PEEKs.check_length(current_PEEKs, past_PEEKs)
        except check_PEEKs as e:
            print("Exception:", e)

        PEEKs = []

        for i in range(0, len(current_PEEKs)):
            PEEKs.append(current_PEEKs[i])
            PEEKs.append(past_PEEKs[i])

        # Compute PEEK loss
        Sigmoid = nn.Sigmoid()
        PEEK_criterion = nn.L1Loss()
        custom_loss = 0.0

        # Process PEEKs in groups defined by self.pos
        for i in range(0, len(PEEKs), self.pos):
            PEEK_dif = 0
            PEEK_conservation = 0
            for j in range(1, self.pos):
                current_PEEK, prev_PEEK = Sigmoid(PEEKs[i]), Sigmoid(PEEKs[i+j])
                PEEK_dif += PEEK_criterion(current_PEEK, prev_PEEK)
                PEEK_conservation += torch.sum(current_PEEK) - torch.sum(prev_PEEK)

            PEEK_dif = (PEEK_dif)/(self.pos-1) # Bring it between [0,1]

            custom_loss += -1*((self.beta)*PEEK_dif)+(self.lam)*PEEK_conservation #MAE encourages the images to be similar,
            # so taking the negative does the opposite i.e: encourages them to be different.

        # Combine the base loss with the weighted custom loss term
        total_loss = base_loss + self.custom_weight * custom_loss

        return total_loss, loss_items
