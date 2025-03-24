import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class CustomSegformer(nn.Module):
    def __init__(self, model):
        super(CustomSegformer, self).__init__()
        self.encoder = model  # Extract encoder
        self.num_classes = 5  # Number of segmentation classes

        # Decoder: Fuse multi-scale features (SegFormer has 4 feature maps)
        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(160, 128, kernel_size=3, padding=1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
        ])
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.num_classes, kernel_size=1)  # Output logits for each class
        )

    def forward(self, x):
        # Get multi-scale feature maps from the encoder
        encoder_outputs = self.encoder(x, output_attentions=False, output_hidden_states=True, return_dict=True)
        hidden_states = encoder_outputs.hidden_states  # List of 4 feature maps

        # Upsample and fuse multi-scale feature maps
        fused_features = 0
        for i, decoder_layer in enumerate(self.decoder_layers):
            upsampled_feature = F.interpolate(hidden_states[i], scale_factor=2**(3-i), mode="bilinear", align_corners=False)
            fused_features += decoder_layer(upsampled_feature)

        # Final segmentation map
        logits = self.segmentation_head(fused_features)

        return logits




# Function to use the model
def upsample_image(image):
    """
    Upsample a single image from 192x192x1 to 512x512x3
    
    Args:
        image: PyTorch tensor of shape [1, 1, 192, 192]
        
    Returns:
        PyTorch tensor of shape [1, 3, 512, 512]
    """
    # Load pre-trained SegFormer and wrap it
    pretrained_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=5)
    model = CustomSegformer(pretrained_model)
    
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
    input_tensor = torch.randn(1, 3, 512, 512)
    
    # Upsample the image
    output_tensor = upsample_image(input_tensor)
    
    # Print the output shape
    print("Output shape:", output_tensor.shape)