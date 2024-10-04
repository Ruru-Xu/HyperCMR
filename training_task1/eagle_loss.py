'''
We optimized the Eagle Loss function: https://github.com/sypsyp97/Eagle_Loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return self.fft_loss(var_target.view(batch_size, num_channels, shape0, shape1),
                             var_output.view(batch_size, num_channels, shape0, shape1))

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

