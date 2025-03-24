import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsamplingCNN(nn.Module):
    def __init__(self):
        super(UpsamplingCNN, self).__init__()
        
        # Initial feature extraction from single channel
        self.conv_init = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(64)
        
        # First convolutional block at the same resolution
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # First upsampling block: 192 -> 384
        # For exact doubling: stride=2, kernel=4, padding=1
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_up1 = nn.BatchNorm2d(32)
        
        # Second convolutional block at 384x384
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Second upsampling block: 384 -> 512
        # We need to go from 384 to 512, which is not a multiple
        # We'll use a 4x4 kernel with stride=1 and padding=0 which adds (kernel_size-1)=3 pixels per side
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=0)
        self.bn_up2 = nn.BatchNorm2d(16)
        
        # Final output convolution to get 3 channels
        self.conv_final = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial feature extraction
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        
        # First upsampling: 192 -> 384
        x = F.relu(self.bn_up1(self.upconv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Calculate exact padding needed
        current_size = x.size(-1)
        target_size = 512
        # Create a target size tensor for adaptive pooling
        if current_size != 384:
            x = F.adaptive_avg_pool2d(x, (384, 384))
        
        # Second upsampling with precise padding
        # We're using the fact that upconv with k=4, s=1, p=0 expands by 3 pixels per side
        # So 384+3+3 = 390, and we need to get to 512
        x = F.relu(self.bn_up2(self.upconv2(x)))
        
        # Apply center crop to get exactly 512x512
        # If size is smaller than 512, we pad
        current_size = x.size(-1)
        if current_size < target_size:
            # Padding needed
            pad_size = (target_size - current_size) // 2
            x = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        elif current_size > target_size:
            # Center crop needed
            start = (current_size - target_size) // 2
            x = x[:, :, start:start+target_size, start:start+target_size]
        
        # Final output convolution
        x = self.conv_final(x)
        
        return x

# Function to use the model
def upsample_image(image):
    """
    Upsample a single image from 192x192x1 to 512x512x3
    
    Args:
        image: PyTorch tensor of shape [1, 1, 192, 192]
        
    Returns:
        PyTorch tensor of shape [1, 3, 512, 512]
    """
    model = UpsamplingCNN()
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        upsampled_image = model(image)
        
        # Ensure the output is exactly 512x512
        if upsampled_image.shape[-1] != 512 or upsampled_image.shape[-2] != 512:
            # Force resize to exactly 512x512 as a last resort
            upsampled_image = F.adaptive_avg_pool2d(upsampled_image, (512, 512))
    
    # Verify the output size
    assert upsampled_image.shape[1:] == (3, 512, 512), f"Output shape {upsampled_image.shape} does not match expected (batch, 3, 512, 512)"
    
    return upsampled_image

# Example usage
if __name__ == "__main__":
    # Create a dummy input tensor of shape [1, 1, 192, 192]
    input_tensor = torch.randn(1, 1, 192, 192)
    
    # Upsample the image
    output_tensor = upsample_image(input_tensor)
    
    # Print the output shape
    print("Output shape:", output_tensor.shape)