import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block with residual connection"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Skip connection
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else \
                   nn.Conv2d(in_ch, out_ch, 1, stride)
                   
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + self.skip(residual)
        x = self.relu(x)
        return x

class Attention(nn.Module):
    """Simple self-attention module"""
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch//8, 1)
        self.key = nn.Conv2d(ch, ch//8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, ch, h, w = x.size()
        
        q = self.query(x).view(batch, -1, h*w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h*w)
        v = self.value(x).view(batch, -1, h*w)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, ch, h, w)
        
        return x + self.gamma * out

class AudioUNet(nn.Module):
    """Simplified UNet for audio spectrogram restoration"""
    def __init__(self, base_ch=64):
        super().__init__()
        
        # Encoder
        self.inc = ConvBlock(1, base_ch)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(base_ch, base_ch*2)
        )
        self.atten1 = Attention(base_ch*2)
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(base_ch*2, base_ch*4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(base_ch*4, base_ch*8)
        )
        self.atten2 = Attention(base_ch*8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*8, base_ch*8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.atten3 = Attention(base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out = nn.Conv2d(base_ch, 1, 1)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.atten1(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.atten2(x4)
        
        # Bottleneck
        x4 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.atten3(x)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        
        # Residual output
        out = self.out(x) + x
        
        return torch.clamp(out, 0.0, 1.0)

class SimpleParamPredictor(nn.Module):
    """Simplified parameter predictor preserving frame-wise information"""
    def __init__(self, n_params=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(128, n_params)
        
    def forward(self, energy):
        x, _ = self.lstm(energy)
        params = torch.sigmoid(self.linear(x))
        return params

class SimpleMasterNet(nn.Module):
    """Simplified MasterNet with separated audio and parameter paths"""
    def __init__(self, sr=22050, hop=256):
        super().__init__()
        self.sr = sr
        self.hop = hop
        
        # Audio restoration network
        self.audio_net = AudioUNet(base_ch=64)
        
        # Parameter prediction network
        self.param_net = SimpleParamPredictor(n_params=10)
        
    def forward(self, x):
        # Energy curve for parameter prediction
        energy = x.mean(dim=2).permute(0, 2, 1)  # [B,T,1]
        
        # Independent spectrogram restoration
        restored = self.audio_net(x)
        
        # Parameter prediction
        params = self.param_net(energy)
        
        return restored, params
