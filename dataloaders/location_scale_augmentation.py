import numpy as np
import random
from scipy.special import comb
import numpy as np
import random
from scipy.special import comb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
# import bezier
import numpy as np
from kornia.geometry.transform import get_tps_transform, warp_image_tps
# from thin_plate_spline import thin_plate_spline_transform  # Ensure this function is correctly implemented and available

class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01):
        self.vrange = vrange
        self.background_threshold = background_threshold
    def thin_plate_spline_transform(self, x, masks, control_points=9):
        print("Before conversion - Shape of x:", x.shape, type(x))  # Debugging
        
        if isinstance(x, np.ndarray):  # If x is a NumPy array
            x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
        
        print("After conversion - Shape of x:", x.shape, type(x))  # Debugging
        print("Before conversion - Shape of mask:", masks.shape, type(masks))  # Debugging
        
        if isinstance(masks, np.ndarray):  # If x is a NumPy array
            masks = torch.tensor(masks).permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
        
        print("After conversion - Shape of mask:", masks.shape, type(masks))  # Debugging
        B, C, H, W = x.shape
        device = x.device
        
        # Generate control points in grid pattern
        n_per_side = int(np.sqrt(control_points))
        points_src = torch.zeros(B, n_per_side**2, 2, device=device)
        points_idx = 0
        for i in range(n_per_side):
            for j in range(n_per_side):
                points_src[:, points_idx, 0] = 2 * i / (n_per_side - 1) - 1  # Range [-1, 1]
                points_src[:, points_idx, 1] = 2 * j / (n_per_side - 1) - 1
                points_idx += 1
        
        # Generate anatomically plausible displacements
        # Different organs have different elasticity - use masks to adjust displacement
        # Create organ-specific displacement maps
        displacements = torch.zeros(B, n_per_side**2, 2, device=device)
        
        # Anatomical constraints (based on common abdominal organ elasticity)
        # Values based on medical literature for abdominal organs
        elasticity_map = {
            0: 0.05,  # Background - minimal deformation
            1: 0.12,  # Liver - moderate deformation
            2: 0.15,  # R-Kidney - more deformation
            3: 0.15,  # L-Kidney - more deformation
            4: 0.08,  # Spleen - less deformation
        }
        
        # Get majority organ per control point region to determine deformation magnitude
        for b in range(B):
            for i in range(n_per_side):
                for j in range(n_per_side):
                    idx = i * n_per_side + j
                    
                    # Determine which part of the image this control point affects most
                    # by creating a simple gaussian mask around the control point
                    y_coord = int((i / (n_per_side - 1)) * (H - 1))
                    x_coord = int((j / (n_per_side - 1)) * (W - 1))
                    
                    # Sample 5x5 region around point to determine dominant organ
                    y_min, y_max = max(0, y_coord-2), min(H, y_coord+3)
                    x_min, x_max = max(0, x_coord-2), min(W, x_coord+3)
                    
                    # Get region from mask
                    region = masks[b, :, y_min:y_max, x_min:x_max].sum(dim=(1,2))
                    dominant_organ = torch.argmax(region).item()
                    
                    # Get elasticity factor for this organ
                    elasticity = elasticity_map.get(dominant_organ, 0.10)
                    
                    # Generate random displacement scaled by elasticity
                    displacements[b, idx, 0] = torch.randn(1, device=device) * elasticity
                    displacements[b, idx, 1] = torch.randn(1, device=device) * elasticity
        
        # Apply smoothness constraint to prevent unrealistic deformations
        # Ensure neighboring control points have similar displacements
        smoothed_displacements = displacements.clone()
        for i in range(1, n_per_side-1):
            for j in range(1, n_per_side-1):
                idx = i * n_per_side + j
                # Average with neighbors for smoothness
                neighbors_idx = [
                    (i-1)*n_per_side + j, # top
                    (i+1)*n_per_side + j, # bottom
                    i*n_per_side + (j-1), # left
                    i*n_per_side + (j+1)  # right
                ]
                for b in range(B):
                    # Apply smoothing (80% original + 20% neighbors average)
                    neighbor_avg = torch.stack([displacements[b, n_idx] for n_idx in neighbors_idx]).mean(dim=0)
                    smoothed_displacements[b, idx] = 0.8 * displacements[b, idx] + 0.2 * neighbor_avg
        
        points_dst = points_src + smoothed_displacements
        
        # Use the correct function from kornia
        # from kornia.geometry.transform import thin_plate_spline
        # grid = thin_plate_spline.thin_plate_spline(points_src, points_dst, (H, W))
        kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)
        warped_image = warp_image_tps(x, points_src, kernel_weights, affine_weights, align_corners = True)
        # Apply the transformation with gradient tracking
        # transformed = F.grid_sample(
        #     x, grid, mode='bilinear', padding_mode='border', align_corners=True
        # )
        # Convert tensor back to NumPy
        warped_image = warped_image.squeeze(0).permute(1, 2, 0).numpy()  # (1, C, H, W) → (H, W, C)

        print("Converted NumPy shape:", warped_image.shape)  # Expected: (192, 192, 1)

        return warped_image

    def non_linear_transformation(self, image, mask):
        temp = self.thin_plate_spline_transform(image, mask)  # Replacing Bézier-based transformation with TPS

        return temp

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs * scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image, mask):
        image = self.non_linear_transformation(image, mask)
        image = self.location_scale_transformation(image).astype(np.float32)
        return image

    def Local_Location_Scale_Augmentation(self, image, mask):
        """
        Performs augmentation while preserving certain regions based on the mask.

        - Background (mask == 0): Always undergoes non-linear + scale transformations.
        - Foreground (mask > 0): Has a 50% probability of undergoing non-linear transformation with inversion.
        - Low-intensity pixels (<= background_threshold): Are preserved.

        Args:
            image (numpy.ndarray): Input image.
            mask (numpy.ndarray): Mask with same dimensions as image.
        
        Returns:
            numpy.ndarray: Augmented image.
        """
        output_image = np.copy(image)

        # Apply non-linear and scale transformation to the background (mask == 0)
        output_image[mask == 0] = self.location_scale_transformation(
            self.non_linear_transformation(image, mask)
        )

        # Apply transformations to foreground regions (mask > 0) with 50% probability of inversion
        for c in range(1, np.max(mask) + 1):
            if (mask == c).sum() == 0:
                continue
            
            transformed_region = self.non_linear_transformation(image, mask)

            # Apply 50% probability of inversion
            if random.random() < 0.5:
                transformed_region = self.vrange[1] - transformed_region  # Invert the transformation
            
            output_image[mask == c] = self.location_scale_transformation(transformed_region)

        # Preserve low-intensity pixels (≤ background_threshold)
        if self.background_threshold >= self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image
