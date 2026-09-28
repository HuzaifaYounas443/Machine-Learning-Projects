import numpy as np
import cv2
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM implementation for model interpretability. Registers hooks on a
    target convolutional layer, backprops the predicted class, and produces
    a class activation heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        modules = dict(self.model.named_modules())
        layer = modules.get(self.target_layer)

        if layer is None:
            # Fall back to the last Conv2d layer in the network
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    self.target_layer = name
                    layer = module
                    break

        if layer is None:
            raise ValueError("No suitable convolutional layer found for Grad-CAM")

        self._hook_handles.append(layer.register_forward_hook(forward_hook))
        self._hook_handles.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam, target_class, output

    def overlay_heatmap(self, image, cam, alpha=0.45, colormap=cv2.COLORMAP_JET):
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, colormap)

        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
        return overlay, heatmap
