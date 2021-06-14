import torch
import models
import numpy as np
from PIL import Image, ImageFilter

# load model
model = torch.load("network.bin", map_location=torch.device('cpu'))
model.eval()

# open input image
img = torch.tensor(np.array(Image.open("in.png").convert("RGB").resize((32, 32))))
img = torch.reshape(img, (3, 32, 32))
img = img.unsqueeze(0).type(torch.FloatTensor)

# get prediction
y = model.forward(img)
y.squeeze_()
y = y.reshape(32, 32, 3)

# save prediction
a = Image.fromarray(np.uint8(np.clip(y.detach().numpy(), 0, 255)))
a = a.filter(ImageFilter.MedianFilter(size=1))
a.resize((32, 32), Image.NEAREST).save("out.png")
