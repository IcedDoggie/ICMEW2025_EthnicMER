import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayer

from torch.autograd import Function
import torch
import torch.nn.functional as F
import torchvision.models as models

import torch
import torch.nn.functional as F

from torchvision import transforms


from collections import OrderedDict
def onehot_reshape_transform():
    # TODO
    return 1


def resnet_reshape_transform(x):
    """
    Reshape and transform activations for ResNet-18 to match a specific target size.
    Args:
        x (dict): Dictionary of layer activations.

    Returns:
        torch.Tensor: Concatenated tensor of resized activations.
    """
    # target_size = list(x.values())[0].size()[-2:]  # Target size from the first activation
    target_size = list(x)[0].size()[-2:]
    activations = []
    # for key, value in x.items():
    activations.append(F.interpolate(torch.abs(x), target_size, mode='bilinear'))
    activations = torch.cat(activations, dim=1)  # Concatenate along the channel axis
    return activations


class AblationLayerResNet(AblationLayer):
    def __init__(self):
        super(AblationLayerResNet, self).__init__()

    def set_next_batch(self, input_batch_index, activations, num_channels_to_ablate):
        """
        Extract the next batch member from activations and repeat it num_channels_to_ablate times.
        Args:
            input_batch_index (int): Index of the batch to ablate.
            activations (dict): Dictionary of layer activations.
            num_channels_to_ablate (int): Number of channels to ablate.
        """
        self.activations = OrderedDict()
        for key, value in activations.items():
            activation = value[input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations[key] = activation.repeat(num_channels_to_ablate, 1, 1, 1)

    def __call__(self, x):
        """
        Apply ablation by modifying activation indices.
        Args:
            x: Input to the layer (not used here).
        Returns:
            dict: Modified activations after ablation.
        """
        result = self.activations
        num_channels_to_ablate = next(iter(result.values())).size(0)
        for i in range(num_channels_to_ablate):
            layer_name = list(result.keys())[0]  # Single layer key for ResNet
            index_in_layer = self.indices[i]
            result[layer_name][i, index_in_layer, :, :] = -1000  # Ablation value
        return result



def reshape_transform(tensor, height=7, width=7):
    # result = tensor[:, 1 :  , :].reshape(tensor.size(0),
    #     height, width, tensor.size(2)) # this is to ignore class token
    result = tensor[:, :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))    

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def get_input_feats(of, mag):
    outpath = '/home/hq/Documents/Weights/Visualization/'

    of = of[0].cpu().numpy().transpose((1, 2, 0))
    mag = mag[0].cpu().detach().numpy().transpose((1, 2, 0))

    plt.imshow(of)
    plt.axis('off')
    plt.savefig(outpath + 'of' + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    norm_of = (of - of.min()) / (of.max() - of.min())
    plt.imshow(norm_of)
    plt.axis('off')
    plt.savefig(outpath + 'norm_of' + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()    

    for i in range(mag.shape[2]):
        curr = mag[:, :, i]
        
        plt.imshow(curr)
        plt.axis('off')
        plt.savefig(outpath + str(i) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()
        a = 1

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.hook_layer()

    def hook_layer(self):
        # Register the forward hook to store the activations
        def forward_hook(module, input, output):
            self.activations = output

        # Register the backward hook to store the gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Access the target layer
        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def forward(self, inputs):
        # Unpack the inputs
        return self.model(*inputs)

    def __call__(self, inputs, class_idx=None):
        # Forward pass
        output = self.forward(inputs)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()

        # Gradients and activations
        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Compute the Grad-CAM
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Global average pooling of gradients
        cam = torch.sum(weights * activations, dim=1)  # Weighted combination of activations

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize the CAM
        cam -= cam.min()
        cam /= cam.max()

        return cam
    
from collections import OrderedDict

def get_resnet_activations(model, input_tensor):
    activations = OrderedDict()
    gradients = OrderedDict()

    # Attach hooks to desired layers
    layer_names = ['spatial_model.layer1', 'spatial_model.layer2', 'spatial_model.layer3', 'spatial_model.layer4']  # Example layers

    def hook(name):
        def hook_fn(module, input, output):
            activations[name] = output
        return hook_fn
    def backward_hook(name):
        def hook_fn(module, input, output):
            gradients[name] = output
        return hook_fn    

    # model = models.resnet18(pretrained=True).cuda()

    hooks = []
    hooks_backward = []
    for name, layer in model.named_modules():
        if name in layer_names:
            hooks.append(layer.register_forward_hook(hook(name)))
            hooks_backward.append(layer.register_backward_hook(backward_hook(name)))

    # Forward pass to capture activations
    model(input_tensor)



    # Remove hooks
    for h in hooks:
        h.remove()
    a = 1

    return activations    


def denormalization(single_image):
    #### Normalization parameters ####
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()  # Shape (C, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()   # Shape (C, 1, 1)

    # Reverse normalization
    single_image = single_image * std + mean        
    trans = transforms.ToPILImage()
    single_image = trans(single_image)
    single_image = np.array(single_image)
    single_image = single_image.astype(np.float32)
    single_image = single_image/255      
    ###################################     

    return single_image


def au_to_emotion(predicted_classes, groundtruth_classes, au_classes):
    # classes = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17']
    # emotion classes = ['Happiness', 'Sadness', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']
    au_classes = np.asarray(au_classes)
    for i in range(len(predicted_classes)):
        pred = predicted_classes[i]
        gt = groundtruth_classes[i]
        pred_emo = au_classes[np.argwhere(pred == 1).flatten()]
        gt_emo = au_classes[np.argwhere(gt == 1).flatten()]
        a = 1

    return 1