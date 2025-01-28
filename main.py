import os
import glob
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# 1. Dataset for Training & Testing
# -----------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Collect all jpg/png/jpeg paths
        self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) \
                            + glob.glob(os.path.join(image_dir, '*.png')) \
                            + glob.glob(os.path.join(image_dir, '*.jpeg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# -----------------------------
# 2. Patch Embedding
# -----------------------------
class PatchEmbed(nn.Module):
    """
    Splits an image into patches, then projects to an embedding dimension.
    For an image of size (C, H, W) and a patch size p,
    the resulting number of patches is (H/p) * (W/p).

    Example: 512x512 image, patch size=16 => 32x32=1024 patches per image.
    """
    def __init__(self, in_channels=3, embed_dim=128, patch_size=16, img_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # This turns (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        # Then we'll flatten to (B, N, embed_dim).

    def forward(self, x):
        # x shape: (B, 3, 512, 512)
        B, C, H, W = x.shape
        # Project to embeddings
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, embed_dim, N)
        # Transpose to (B, N, embed_dim)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x

# -----------------------------
# 3. Transformer Building Blocks
# -----------------------------
class MLP(nn.Module):
    """ Simple MLP block used in Transformers. """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        if not hidden_features:
            hidden_features = in_features
        if not out_features:
            out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """ Multi-head self-attention. """
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # => (3, B, num_heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # each => (B, num_heads, N, head_dim)

        # Scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v)  # (B, heads, N, head_dim)
        x = x.transpose(1, 2)  # => (B, N, heads, head_dim)
        x = x.flatten(2)       # => (B, N, D)
        x = self.proj(x)       # => (B, N, D)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """ A single Transformer encoder or decoder block. """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dropout=dropout)

    def forward(self, x):
        # Pre-norm, self-attn
        x = x + self.attn(self.norm1(x))
        # Pre-norm, MLP
        x = x + self.mlp(self.norm2(x))
        return x

# -----------------------------
# 4. ViT Encoder & Decoder
# -----------------------------
class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=128, depth=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # final layer norm
        return x

class ViTDecoder(nn.Module):
    """
    Decodes both visible + masked tokens into patch predictions.
    We'll keep it simpler than the full MAE approach (fewer layers).
    """
    def __init__(self, embed_dim=128, depth=2, num_heads=4, dropout=0.0,
                 patch_size=16, img_size=512):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Predict pixel values for each patch
        # Each patch has patch_size*patch_size*3 pixels => flatten them
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_dim = patch_size * patch_size * 3

        self.head = nn.Linear(embed_dim, self.out_dim)

    def forward(self, x):
        # x: shape (B, N, embed_dim) for ALL tokens (masked + unmasked)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Project each token to patch_size*patch_size*3
        x = self.head(x)  # (B, N, patch_dim)
        return x

# -----------------------------
# 5. Masked Autoencoder with ViT
# -----------------------------
class MaskedAutoencoderViT(nn.Module):
    """
    1. Embed all patches.
    2. Randomly mask a subset of patches, pass visible ones to the encoder.
    3. Append mask tokens for masked patches, pass full set to the decoder.
    4. Reconstruct the masked patches in pixel space.
    """
    def __init__(self, img_size=512, patch_size=16, embed_dim=128,
                 encoder_depth=4, decoder_depth=2, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Patch Embed
        self.patch_embed = PatchEmbed(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )
        self.num_patches = self.patch_embed.num_patches

        # Learnable positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Mask token: for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Encoder & Decoder
        self.encoder = ViTEncoder(embed_dim=embed_dim,
                                  depth=encoder_depth,
                                  num_heads=num_heads)
        self.decoder = ViTDecoder(embed_dim=embed_dim,
                                  depth=decoder_depth,
                                  num_heads=num_heads,
                                  patch_size=patch_size,
                                  img_size=img_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, mask_ratio=0.75):
        """
        x: (B, 3, H, W)
        mask_ratio: fraction of patches to mask
        Returns:
          (recon_loss, reconstruction, mask)
        """
        B = x.size(0)

        # 1. Patch embedding
        patches = self.patch_embed(x)  # (B, N, D)

        # 2. Add positional embeddings
        patches = patches + self.pos_embed[:, :self.num_patches, :]

        # 3. Randomly mask out a fraction of patches
        #    We'll pick a random subset of patches to be "visible"
        N = self.num_patches
        num_masked = int(mask_ratio * N)
        # We want a random permutation of [0..N-1]
        rand_indices = torch.rand(B, N, device=x.device).argsort(dim=1)
        # For each sample in the batch, pick which patches to keep vs. mask
        visible_indices = rand_indices[:, :-num_masked]
        masked_indices = rand_indices[:, -num_masked:]

        # 4. Gather the visible patches for the encoder
        patches_visible = []
        for i in range(B):
            patches_visible.append(patches[i][visible_indices[i]])
        patches_visible = torch.stack(patches_visible, dim=0)  # (B, N_visible, D)

        # 5. Encode visible patches
        encoded = self.encoder(patches_visible)  # (B, N_visible, D)

        # 6. Prepare tokens for the decoder
        # We need to re-insert (masked) tokens at their original positions
        # We'll create a new array of shape (B, N, D).
        mask_tokens = self.mask_token.expand(B, num_masked, -1)  # (B, num_masked, D)

        # Concat visible + mask tokens
        # Then we reorder them to the original patch positions
        # Start by building a placeholder
        decoder_tokens = torch.zeros_like(patches)
        # Fill in the visible tokens
        for i in range(B):
            decoder_tokens[i][visible_indices[i]] = encoded[i]
        # Fill in the mask tokens
        for i in range(B):
            decoder_tokens[i][masked_indices[i]] = mask_tokens[i]

        # 7. Pass all tokens (visible + masked) to the decoder
        decoded = self.decoder(decoder_tokens)  # (B, N, patch_size*patch_size*3)

        # 8. Compute MSE loss on the masked patches only
        # We'll compare the output vs the ground-truth patches in pixel space
        # We first need the "target" patch pixels
        # The original patch shape is (B, N, D) in embed space, but let's reconstruct from x
        # Actually easier: We'll build "true_patches" from x at the start or directly from patch_embed
        with torch.no_grad():
            # Re-run patch_embed without gradient to get GT patch embeddings in pixel domain
            # Actually, we need the original pixels of each patch, not the embedding.
            # Let's do a direct approach: chunk the image x into patches in pixel space.
            # We'll do it carefully below.
            pass

        # Instead, a simpler approach is to do it the same way the decoder does:
        # We'll create an array of shape (B, N, patch_size*patch_size*3) with the ground truth.
        true_patches = self._get_patches_as_pixels(x)  # shape (B, N, patch_dim)

        # We'll gather only the masked patches for the loss
        mask_indices_sorted = masked_indices.sort(dim=1).values  # sort for consistent gather
        # Gather from true_patches
        masked_true_patches = []
        masked_pred_patches = []
        for i in range(B):
            masked_true_patches.append(true_patches[i][mask_indices_sorted[i]])
            masked_pred_patches.append(decoded[i][mask_indices_sorted[i]])
        masked_true_patches = torch.stack(masked_true_patches, dim=0)  # (B, num_masked, patch_dim)
        masked_pred_patches = torch.stack(masked_pred_patches, dim=0)  # (B, num_masked, patch_dim)

        recon_loss = nn.functional.mse_loss(masked_pred_patches, masked_true_patches)
        return recon_loss, decoded, (visible_indices, masked_indices)

    def _get_patches_as_pixels(self, x):
        """
        Convert a batch of images (B, 3, H, W) into shape (B, N, patch_size*patch_size*3)
        in pure pixel space, matching the decoder’s output dimension.

        We'll manually unfold the image into patches. Alternatively, we could use
        torch.nn.functional.unfold, but here we'll do a straightforward approach.
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # We'll create (B, N, p*p*C) in a loop.
        # N = (H/p)*(W/p)
        patches_pixel = []
        for img in x:  # shape (3, H, W)
            # Unfold along height/width
            # shape => (C, p, p) for each patch
            # We'll do it row by row
            row_patches = []
            for row_start in range(0, H, p):
                row_slice = slice(row_start, row_start + p)
                col_patches = []
                for col_start in range(0, W, p):
                    col_slice = slice(col_start, col_start + p)
                    patch = img[:, row_slice, col_slice]  # shape (3, p, p)
                    col_patches.append(patch.flatten())  # shape (3*p*p,)
                row_patches.append(torch.stack(col_patches, dim=0))  # (#patches_in_width, 3*p*p)
            # row_patches is (H/p, W/p, 3*p*p)
            row_patches = torch.cat(row_patches, dim=0)  # => (N, 3*p*p)
            patches_pixel.append(row_patches)
        patches_pixel = torch.stack(patches_pixel, dim=0)  # (B, N, 3*p*p)
        return patches_pixel

# -----------------------------
# 6. Simple Evaluate Function
# -----------------------------
def evaluate(model, data_loader, device, mask_ratio, max_batches=None):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, images in enumerate(data_loader):
            images = images.to(device)
            loss, _, _ = model(images, mask_ratio=mask_ratio)
            total_loss += loss.item()
            count += 1
            if max_batches and (batch_idx + 1) >= max_batches:
                break
    return total_loss / count

# -----------------------------
# 7. Main Training Loop
# -----------------------------
def main():
    # Speed up GPU conv operations for fixed image-size
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 1000
    batch_size = 4
    init_mask_ratio = 0.10  # Start with 10%
    final_mask_ratio = 0.90 # End with 90%

    def get_current_mask_ratio(epoch):
        """Linearly interpolate from init_mask_ratio to final_mask_ratio."""
        if num_epochs == 1:
            return final_mask_ratio
        step = (final_mask_ratio - init_mask_ratio) / (num_epochs - 1)
        return init_mask_ratio + step * epoch

    # Create datasets & loaders
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolderDataset("Data/train/image", transform=train_transform)
    test_dataset = ImageFolderDataset("Data/test/image", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Build the MAE-ViT model
    model = MaskedAutoencoderViT(
        img_size=512,
        patch_size=16,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=4,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Determine the incremental mask ratio
        current_mask_ratio = get_current_mask_ratio(epoch)

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            loss, _, _ = model(images, mask_ratio=current_mask_ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Mask Ratio: {current_mask_ratio:.2f}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on the test set
        avg_test_loss = evaluate(model, test_loader, device, mask_ratio=current_mask_ratio)

        print(f"\n==> Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
              f"Mask Ratio: {current_mask_ratio:.2f}\n")

    # Optional: visualize reconstructions on the test set
    visualize_recon(model, test_loader, device, mask_ratio=init_mask_ratio)


def visualize_recon(model, data_loader, device, mask_ratio):
    """Visualize some reconstructions on the test set."""
    model.eval()
    images = next(iter(data_loader))
    images = images.to(device)
    
    with torch.no_grad():
        loss, decoded, (visible_idx, masked_idx) = model(images, mask_ratio=mask_ratio)

    # `decoded` has shape (B, N, patch_size*patch_size*3). We can reconstruct the full image.
    # For clarity, let's only visualize the masked patches we just reconstructed.
    # We'll "un-patchify" the entire decoded output so we can see the final image.
    B = images.size(0)
    patch_size = model.patch_size
    # We'll store the reconstructed patches in pixel space
    recon_patches = decoded  # (B, N, patch_dim)

    # "Un-patchify" (like _get_patches_as_pixels but in reverse)
    # recon_patches is (B, N, 3*patch_size*patch_size)
    recon_imgs = []
    for b in range(B):
        # Each image has model.num_patches patches => each patch is patch_size^2*3
        patches_b = recon_patches[b]  # (N, patch_dim)
        # Convert to (N, 3, p, p)
        patches_b = patches_b.reshape(-1, 3, patch_size, patch_size)
        # We need to reorder them row by row
        # N = (img_size/patch_size)^2. For 512/16 => 32 * 32 = 1024
        num_patches_per_row = model.img_size // patch_size
        rows = []
        for r in range(num_patches_per_row):
            row_patches = patches_b[r * num_patches_per_row : (r+1) * num_patches_per_row]
            rows.append(torch.cat(list(row_patches), dim=2))  # concat in width
        full_img = torch.cat(rows, dim=1)  # concat rows in height => shape (3, H, W)
        # Clip to [0,1]
        full_img = torch.clamp(full_img, 0, 1)
        recon_imgs.append(full_img)
    recon_imgs = torch.stack(recon_imgs, dim=0)  # (B, 3, H, W)

    # Move to CPU for plotting
    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    recon_np = recon_imgs.cpu().permute(0, 2, 3, 1).numpy()

    num_show = min(4, B)
    for i in range(num_show):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].imshow(images_np[i])
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(recon_np[i])
        axs[1].set_title(f"Reconstruction (mask={mask_ratio:.2f})")
        axs[1].axis('off')

        plt.show()

if __name__ == "__main__":
    main()
