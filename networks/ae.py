import torch





class Autoencoder(torch.nn.Module):
    def __init__(self):
        #Encoder
        self.enc = []
        self.enc.append(torch.nn.Conv2d(3, 16, 3, padding=1))
        self.enc.append(torch.nn.ReLU(inplace=True))
        self.enc.append(torch.nn.MaxPool2d(2, 2))
        self.enc.append(torch.nn.Conv2d(16, 4, 3, padding=1))
        self.enc.append(torch.nn.ReLU(inplace=True))
        self.enc.append(torch.nn.MaxPool2d(2, 2))

        self.enc = torch.nn.Sequential(*self.enc)

        #Decoder
        self.dec = []
        self.dec.append(torch.nn.ConvTranspose2d(4, 16, 2, stride=2))
        self.dec.append(torch.nn.ReLU(inplace=True))
        self.dec.append(torch.nn.ConvTranspose2d(16, 3, 2, stride=2))
        self.dec.append(torch.nn.Sigmoid())

        self.dec = torch.nn.Sequential(*self.dec)

    def forward(self, x):
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        return decoded