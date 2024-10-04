'''
We optimized the Eagle Loss function: https://github.com/sypsyp97/Eagle_Loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Eagle_Loss(nn.Module):
    def __init__(self, patch_size, cpu=False, cutoff=0.35, epsilon=1e-8):
        super(Eagle_Loss, self).__init__()
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        self.cutoff = cutoff
        self.epsilon = epsilon

        # Scharr kernel for the gradient map calculation
        kernel_values = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        self.kernel_x = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)
        self.kernel_y = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)

        # Operation for unfolding image into non-overlapping patches
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size).to(self.device)

    def forward(self, output, target):
        output, target = output.to(self.device), target.to(self.device)
        if output.size(1) != 1 or target.size(1) != 1:
            raise ValueError("Input 'output' and 'target' should be grayscale")

        # Gradient maps calculation
        gx_output, gy_output = self.calculate_gradient(output)
        gx_target, gy_target = self.calculate_gradient(target)

        # Unfolding and variance calculation
        eagle_loss = self.calculate_patch_loss(gx_output, gx_target) + \
                     self.calculate_patch_loss(gy_output, gy_target)

        return eagle_loss

    def calculate_gradient(self, img):
        img = img.to(self.device)
        gx = F.conv2d(img, self.kernel_x, padding=1, groups=img.shape[1])
        gy = F.conv2d(img, self.kernel_y, padding=1, groups=img.shape[1])
        return gx, gy

    def calculate_patch_loss(self, output_gradient, target_gradient):
        output_gradient, target_gradient = output_gradient.to(self.device), target_gradient.to(self.device)
        batch_size = output_gradient.size(0)
        num_channels = output_gradient.size(1)
        patch_size_squared = self.patch_size * self.patch_size

        # Unfold the gradient tensors into patches
        output_patches = self.unfold(output_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        target_patches = self.unfold(target_gradient).view(batch_size, num_channels, patch_size_squared, -1)

        # Compute the variance for each patch
        var_output = torch.var(output_patches, dim=2, keepdim=True) + self.epsilon
        var_target = torch.var(target_patches, dim=2, keepdim=True) + self.epsilon

        shape0, shape1 = output_gradient.shape[-2] // self.patch_size, output_gradient.shape[-1] // self.patch_size

        # Compute the FFT-based loss on the variance maps
        return self.fft_loss(var_output.view(batch_size, num_channels, shape0, shape1),
                            var_target.view(batch_size, num_channels, shape0, shape1))

    def gaussian_highpass_weights2d(self, size):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = torch.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        return 1 - weights  # Inverted for high pass

    def fft_loss(self, pred, gt):
        pred, gt = pred.to(self.device), gt.to(self.device)

        pred_padded, unpad_pred = self.pad_to_pow2(pred)
        gt_padded, unpad_gt = self.pad_to_pow2(gt)

        pred_fft = torch.fft.fft2(pred_padded)
        gt_fft = torch.fft.fft2(gt_padded)

        # Compute FFT magnitudes
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2 + self.epsilon)
        gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2 + self.epsilon)

        # Normalize FFT magnitudes
        pred_mag = (pred_mag - pred_mag.mean()) / (pred_mag.std() + self.epsilon)
        gt_mag = (gt_mag - gt_mag.mean()) / (gt_mag.std() + self.epsilon)

        # Apply high-pass filter
        # weights = self.gaussian_highpass_weights2d(pred_padded.size()[2:]).to(pred.device)
        weights = self.butterworth_highpass_weights2d(pred_padded.size()[2:], cutoff=self.cutoff, order=2).to(
            pred.device)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        pred_mag_unpadded = unpad_pred(weighted_pred_mag)
        gt_mag_unpadded = unpad_gt(weighted_gt_mag)

        l1_loss_val = F.l1_loss(pred_mag_unpadded, gt_mag_unpadded)
        return l1_loss_val

    def pad_to_pow2(self, x):
        h, w = x.shape[-2:]
        new_h = 1 << (h - 1).bit_length()
        new_w = 1 << (w - 1).bit_length()
        padding = (0, new_w - w, 0, new_h - h)
        padded_x = F.pad(x, padding)

        def unpad(y):
            return y[..., :h, :w]

        return padded_x, unpad

    def butterworth_highpass_weights2d(self, size, cutoff=0.35, order=2):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = 1 / (1 + (cutoff / (freq_mag + self.epsilon)) ** (2 * order))

        return 1 - weights  # Inverted for high-pass filtering







def enhance_contrast(image, lower_percentile=1, upper_percentile=99):
    """Enhances the contrast of the image by clipping the intensities."""
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image

def plot_weights(weights, pred_mag_unpadded, gt_mag_unpadded):
    weights_np = weights.cpu().detach().numpy()
    pred_mag_unpadded = pred_mag_unpadded.cpu().detach().numpy()
    gt_mag_unpadded = gt_mag_unpadded.cpu().detach().numpy()

    # Enhance contrast for better visualization
    pred_mag_unpadded = enhance_contrast(pred_mag_unpadded[0, 0])
    gt_mag_unpadded = enhance_contrast(gt_mag_unpadded[0, 0])

    plt.figure(figsize=(12, 5), dpi=400)

    # Plot the high-pass filter weights
    plt.subplot(1, 3, 1)
    plt.imshow(weights_np, interpolation='nearest')
    plt.title('Weights (High-Pass Filter)', fontsize=12)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # Plot the enhanced predicted magnitude in k-space
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mag_unpadded, cmap='hot', interpolation='nearest')
    plt.title('Predicted Magnitude', fontsize=12)
    plt.axis('off')

    # Plot the enhanced ground truth magnitude in k-space
    plt.subplot(1, 3, 3)
    plt.imshow(gt_mag_unpadded, cmap='hot', interpolation='nearest')
    plt.title('Ground Truth Magnitude', fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.show()




# plt.figure(dpi=500)
# img = gx_output[0,0].cpu().detach().numpy()
# plt.imshow(img, cmap='gray')
# plt.clim(vmin=np.percentile(img, 5), vmax=np.percentile(img, 95))  # Adjust contrast using percentiles
# plt.colorbar()
# plt.show()


# plt.figure(dpi=500)
# plt.imshow(np.log(torch.abs(pred_fft[0,0]).cpu().detach().numpy()), cmap='viridis')
# plt.show()






# # Clip the values with a slightly larger minimum value to avoid log issues
# pred_mag_np = np.clip(pred_mag_np, a_min=1e-5, a_max=None)  # Clip to avoid log(0) issues
# gt_mag_np = np.clip(gt_mag_np, a_min=1e-5, a_max=None)      # Clip to avoid log(0) issues
#
# # Plot the predicted magnitude
# plt.figure(dpi=300)
# plt.subplot(1, 2, 1)
# plt.title("Predicted FFT Magnitude")
# plt.imshow(np.log(pred_mag_np[0, 0]), cmap='inferno')  # Log with adjusted clipping values
# plt.colorbar()
# plt.axis('off')
#
# # Plot the ground truth magnitude
# plt.subplot(1, 2, 2)
# plt.title("Ground Truth FFT Magnitude")
# plt.imshow(np.log(gt_mag_np[0, 0]), cmap='inferno')  # Log with adjusted clipping values
# plt.colorbar()
# plt.axis('off')
# plt.show()