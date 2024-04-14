from torchsr.datasets import Div2K
from torchsr.models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor

# Div2K dataset
dataset = Div2K(root="./data", scale=2, download=True)

# Get the first image in the dataset (High-Res and Low-Res)
hr, lr = dataset[0]

# Download a pretrained NinaSR model
model = ninasr_b0(scale=2, pretrained=True)

# Run the Super-Resolution model
lr_t = to_tensor(lr).unsqueeze(0)
sr_t = model(lr_t)
sr = to_pil_image(sr_t.squeeze(0).clamp(0, 1))
sr.show()