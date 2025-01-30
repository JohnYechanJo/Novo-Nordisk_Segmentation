import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import trange
import matplotlib.pyplot as plt

# Define Dataset Class
class RetinaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load Image & Mask files
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Change Mask into Binary
        mask = (mask > 0).float()  # 0 or 1
        return image, mask

# 데이터 경로 설정
extract_path = "./dataset_L"
TRAIN_IMAGE_PATH = os.path.join(extract_path, "retina/train")
TRAIN_MASK_PATH = os.path.join(extract_path, "mask/train")
TEST_IMAGE_PATH = os.path.join(extract_path, "retina/test")
TEST_MASK_PATH = os.path.join(extract_path, "mask/test")
VALID_IMAGE_PATH = os.path.join(extract_path, "retina/validation")
VALID_MASK_PATH = os.path.join(extract_path, "mask/validation")


# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# 데이터 로더 준비
train_dataset = RetinaDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transform=transform)
test_dataset = RetinaDataset(TEST_IMAGE_PATH, TEST_MASK_PATH, transform=transform)
valid_dataset = RetinaDataset(VALID_IMAGE_PATH, VALID_MASK_PATH, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim

class PatchMaskedAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, patch_size=16, image_size=512, mask_ratio=0.75):
        """
        patch_size: size of each patch (e.g., 16)
        image_size: assume input images are (image_size x image_size)
        mask_ratio: fraction of patches to mask out
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        x: (N, 3, H, W)
        """
        with torch.no_grad():
            latent = self.encoder(x)  # (N, C, H, W)
        reconstructed = self.decoder(latent)   # should come back to (N, 3, H, W)

        return reconstructed


import torch.nn as nn
import torch.optim as optim

##############################################################################
# 1. Residual Block
##############################################################################
class ResidualBlock(nn.Module):
    """
    A standard residual block with two 3x3 convolutions and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If shapes differ (due to stride), we use the 'downsample' layer
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out

##############################################################################
# 2. Modern Encoder
##############################################################################
class ModernEncoder(nn.Module):
    """
    A ResNet-like encoder that progressively downsamples.
    You can adjust `layers` to control depth, e.g. [2,2,2,2].
    """
    def __init__(self, layers=[2, 2, 2, 2], base_channels=64, in_channels=3):
        super(ModernEncoder, self).__init__()
        self.in_channels = base_channels
        
        # Initial convolution and pooling (similar to ResNet stem)
        self.conv1 = nn.Conv2d(in_channels, base_channels, 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(base_channels,   layers[0], stride=1)  # 1/4 scale
        self.layer2 = self._make_layer(base_channels*2, layers[1], stride=2)  # 1/8 scale
        self.layer3 = self._make_layer(base_channels*4, layers[2], stride=2)  # 1/16 scale
        self.layer4 = self._make_layer(base_channels*8, layers[3], stride=2)  # 1/32 scale

    def _make_layer(self, out_channels, blocks, stride=1):
        """
        Create a stack of residual blocks, including a 'downsample' 
        layer if channel dimension or stride is changed.
        """
        
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # First block in this layer
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Downsampling stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

##############################################################################
# 3. Modern Decoder
##############################################################################
class ModernDecoder(nn.Module):
    """
    A decoder that uses transpose convolutions (or could use upsampling + conv).
    Mirrors the encoder in reverse, but you can also add skip connections 
    from the encoder if desired (like U-Net).
    """
    def __init__(self, base_channels=64, out_channels=3):
        super(ModernDecoder, self).__init__()

        # The decoder channels should match the encoder's last layer, i.e. base_channels*8
        # if you used the default layers=[2,2,2,2]. Adjust accordingly.
        
        self.up1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.res1 = ResidualBlock(base_channels*4, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.res2 = ResidualBlock(base_channels*2, base_channels*2)

        self.up3 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.res3 = ResidualBlock(base_channels, base_channels)

        # One more up if you want to get back to the original scale 
        # (depending on the input image resolution).
        self.up4 = nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=2, stride=2)
        self.res4 = ResidualBlock(base_channels // 2, base_channels // 2)

        self.up5 = nn.ConvTranspose2d(base_channels//2, base_channels//4, 2, 2)
        self.res5 = ResidualBlock(base_channels//4, base_channels//4)

        # Final 1x1 convolution to map to the desired number of output channels
        self.final_conv = nn.Conv2d(base_channels // 2, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.up1(x)
        x = self.res1(x)

        x = self.up2(x)
        x = self.res2(x)

        x = self.up3(x)
        x = self.res3(x)

        x = self.up4(x)
        x = self.res4(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x

# Instantiate
encoder = ModernEncoder()
decoder = ModernDecoder()
model = PatchMaskedAutoEncoder(
    encoder, decoder,
    patch_size=16,
    image_size=512,
    mask_ratio=0.5
)
print(model)

model.load_state_dict(torch.load('model.pt'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion = nn.MSELoss()
# Loss: Focal Tversky Loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.3):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True positives, false positives, false negatives
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        tversky = (TP + 1e-10) / (TP + self.alpha*FP + self.beta*FN + 1e-10)
        focal = (1 - tversky)**self.gamma

        return focal
    
criterion = FocalTverskyLoss()
optimizer = optim.Adam(model.parameters(), lr=10e-4)

validation_losses = []
training_losses = []
num_epochs = 250
for epoch in trange(num_epochs):
    model.train()
    epoch_loss = 0.0
    for images, seg in train_loader:  # from your RetinaDataset
        images = images.to(device)
        seg = seg.to(device)

        predicted_seg = model(images)

        loss = criterion(predicted_seg, seg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    training_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    # Validation step
    model.eval()
    val_epoch_loss = 0.0
    with torch.no_grad():
        for val_images, val_seg in valid_loader:
            val_images = val_images.to(device)
            val_seg = val_seg.to(device)
            val_out = model(val_images)
            val_loss = criterion(val_out, val_seg)
            val_epoch_loss += val_loss.item()
    validation_losses.append(val_epoch_loss/len(valid_loader))
    print(f"Validation Loss: {val_epoch_loss/len(valid_loader):.4f}")

# Save validation losses
torch.save(validation_losses, 'validation_losses.pt')
torch.save(training_losses, 'training_losses.pt')

# 绘制 validation_losses 折线图
plt.figure(figsize=(10, 5))
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.savefig('validation_losses.png')
plt.show()

# 绘制 training_losses 折线图
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('training_losses.png')
plt.show()

import matplotlib.pyplot as plt

# Put model in evaluation mode
model.eval()
images, seg = next(iter(test_loader))
images = images.to(device)
with torch.no_grad():
    reconstructed = model(images)
# Move images back to CPU for visualization
images = images.cpu()
reconstructed = reconstructed.cpu()

# Plot original, masked, and reconstructed images
plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(3, 4, i+5)
    plt.imshow(seg[i].squeeze(), cmap='gray')
    plt.title("GT")
    plt.axis('off')

    plt.subplot(3, 4, i+9)
    plt.imshow(reconstructed[i].permute(1, 2, 0))
    plt.title("Predicted")
    plt.axis('off')

plt.tight_layout()
# plt.show()
plt.savefig('wowow.png')

