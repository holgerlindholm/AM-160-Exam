import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalConvVAE(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=64, latent_dim=128, n_past=5):
        super(ConditionalConvVAE, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_past = n_past
        
        # Encoder
        # Process the target frame (x_t+1)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=2, padding=1)
        
        # Calculate the exact dimensions after each conv layer
        # For height=91: 91 -> 46 -> 23 -> 12
        # For width=180: 180 -> 90 -> 45 -> 23
        self.encoder_h = 12
        self.encoder_w = 23
        self.flat_size = hidden_dim * 4 * self.encoder_h * self.encoder_w
        
        # FC layers for mu and logvar
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Process the conditioning sequence
        self.cond_conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.cond_conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=2, padding=1)
        self.cond_conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=2, padding=1)
        
        self.cond_flat_size = hidden_dim * 4 * self.encoder_h * self.encoder_w
        self.cond_fc = nn.Linear(self.cond_flat_size * n_past, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim * 2, self.flat_size)
        
        # Custom output padding to ensure exact dimension matching
        # We need to ensure the output sizes exactly match: 12->23->46->91 and 23->45->90->180
        self.deconv1 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=3, stride=2, padding=1, output_padding=(0, 0))
        self.deconv2 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=(0, 0))
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, input_channels, kernel_size=3, stride=2, padding=1, output_padding=(0, 1))
        
        # Add crop/pad layers to ensure exact dimension matching
        self.output_size = (91, 180)
    
    def encode(self, x, cond):
        # Encode the target frame
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # Encode each conditioning frame separately
        batch_size, n_frames, channels, height, width = cond.size()
        cond_flat_features = []
        
        for i in range(n_frames):
            frame = cond[:, i]  # Shape: [batch_size, channels, height, width]
            frame = F.relu(self.cond_conv1(frame))
            frame = F.relu(self.cond_conv2(frame))
            frame = F.relu(self.cond_conv3(frame))
            frame = frame.view(batch_size, -1)
            cond_flat_features.append(frame)
        
        # Concatenate all frames' features
        cond_combined = torch.cat(cond_flat_features, dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Get conditioning embedding
        cond_embedding = self.cond_fc(cond_combined)
        
        return mu, logvar, cond_embedding
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, cond_embedding):
        # Concatenate latent vector with conditioning embedding
        z_combined = torch.cat([z, cond_embedding], dim=1)
        
        # Decode
        x = self.fc_decoder(z_combined)
        x = x.view(-1, self.hidden_dim * 4, self.encoder_h, self.encoder_w)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        
        # Ensure the output has exact dimensions
        if x.size(-2) != self.output_size[0] or x.size(-1) != self.output_size[1]:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        x = torch.sigmoid(x)
        return x
    
    def forward(self, x, cond):
        # x: target frame [B, C, H, W]
        # cond: conditioning frames [B, n_past, C, H, W]
        
        mu, logvar, cond_embedding = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond_embedding)
        
        return recon, mu, logvar, cond_embedding
    
    def predict_next(self, condition):
        """
        Predict the next frame given the conditioning sequence.
        condition: conditioning sequence (B, n_past, C, H, W)
        """
        with torch.no_grad():
            # Encode the condition
            cond_encoded = self.encode_condition(condition)
            
            # Sample from prior (standard normal)
            z = torch.randn(condition.size(0), self.latent_dim, device=condition.device)
            
            # Decode with z and condition
            next_frame = self.decode(z, cond_encoded)
            
            return next_frame
        
    def encode_condition(self, cond):
        """
        Encode the conditioning sequence.
        cond: conditioning sequence (B, n_past, C, H, W)
        """
        batch_size, n_frames, channels, height, width = cond.size()
        cond_flat_features = []
        
        for i in range(n_frames):
            frame = cond[:, i]  # Shape: [batch_size, channels, height, width]
            frame = F.relu(self.cond_conv1(frame))
            frame = F.relu(self.cond_conv2(frame))
            frame = F.relu(self.cond_conv3(frame))
            frame = frame.view(batch_size, -1)
            cond_flat_features.append(frame)


        # Concatenate all frames' features
        cond_combined = torch.cat(cond_flat_features, dim=1)
        return self.cond_fc(cond_combined)
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # Reconstruction loss: Mean Squared Error
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss