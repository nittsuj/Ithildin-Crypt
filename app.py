import gradio as gr

import torch

import torch.nn as nn

import torchvision.transforms as transforms

from PIL import Image



class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.conv(x)



class Down(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.mpconv = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(in_ch, out_ch)

        )

    def forward(self, x):

        return self.mpconv(x)



class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):

        super().__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:

            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)



    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]

        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,

                                          diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class UNetDualHead(nn.Module):

    def __init__(self, n_channels=3, n_classes=3, bilinear=True):

        super().__init__()

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)

        self.down2 = Down(128, 256)

        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        self.down4 = Down(512, 1024 // factor)

        

        self.up1_1 = Up(1024, 512 // factor, bilinear)

        self.up2_1 = Up(512, 256 // factor, bilinear)

        self.up3_1 = Up(256, 128 // factor, bilinear)

        self.up4_1 = Up(128, 64, bilinear)

        self.outc1 = nn.Conv2d(64, n_classes, 1)

        

        self.up1_2 = Up(1024, 512 // factor, bilinear)

        self.up2_2 = Up(512, 256 // factor, bilinear)

        self.up3_2 = Up(256, 128 // factor, bilinear)

        self.up4_2 = Up(128, 64, bilinear)

        self.outc2 = nn.Conv2d(64, n_classes, 1)



    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        

        x1_head1 = self.up1_1(x5, x4)

        x2_head1 = self.up2_1(x1_head1, x3)

        x3_head1 = self.up3_1(x2_head1, x2)

        x4_head1 = self.up4_1(x3_head1, x1)

        out1 = torch.tanh(self.outc1(x4_head1))

        

        x1_head2 = self.up1_2(x5, x4)

        x2_head2 = self.up2_2(x1_head2, x3)

        x3_head2 = self.up3_2(x2_head2, x2)

        x4_head2 = self.up4_2(x3_head2, x1)

        out2 = torch.tanh(self.outc2(x4_head2))

        

        return out1, out2



# ==========================================

# Setup Device & Load Model

# ==========================================

device = torch.device('cpu') 

model = UNetDualHead(n_channels=3, n_classes=3).to(device)



model.load_state_dict(torch.load('generator_epoch_1.pth', map_location=device))

model.eval()



# ==========================================

# Transformasi Input (Dengan Normalize)

# ==========================================

transform = transforms.Compose([

    transforms.Resize((128, 128)), 

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 

])



# ==========================================

# Fungsi Prediksi

# ==========================================

def generate_image(input_image):

    if input_image is None:

        return None

        

    input_tensor = transform(input_image).unsqueeze(0).to(device)

    

    with torch.no_grad():

        output = model(input_tensor)

        if isinstance(output, tuple):

            output = output[0]

            

    output_tensor = output.squeeze(0).cpu()

    output_tensor = output_tensor * 0.5 + 0.5 

    output_tensor = torch.clamp(output_tensor, 0, 1) 

    

    output_pil = transforms.ToPILImage()(output_tensor)

    return output_pil



# ==========================================

# UI Gradio

# ==========================================

demo = gr.Interface(

    fn=generate_image,

    inputs=gr.Image(type="pil", label="Upload Gambar Input"),

    outputs=gr.Image(type="pil", label="Hasil Generasi AI (Epoch 1)"),

    title="AI Image Generator (UNet Dual Head)",

    description="Aplikasi ini menggunakan model berbasis U-Net yang dilatih hingga Epoch 1. Sebuah studi kasus nyata mengatasi Mode Collapse & Bug Normalisasi.",

)



demo.launch(server_name="0.0.0.0", server_port=7860)