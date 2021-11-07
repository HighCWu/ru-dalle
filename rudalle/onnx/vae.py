import os
import glob
import torch
import torch.nn as nn

import onnxmltools
import numpy as np
import onnxruntime as ort

from onnxmltools.utils.float16_converter import convert_float_to_float16


class Encoder(nn.Module):
    def __init__(self, vae):
        super(Encoder, self).__init__()

        self.vae = vae

    def forward(self, x):
        indices, height, width = self.vae._get_codebook_indices(x)

        return indices, height, width

class Decoder(nn.Module):
    def __init__(self, vae):
        super(Decoder, self).__init__()

        self.vae = vae

    def forward(self, img_seq, height, width):
        img = self.vae._decode(img_seq, height, width)

        return img


class VQGanGumbelVAEONNX(object):
    def __init__(self, save_dir, device):
        self.encoder = ort.InferenceSession(os.path.join(save_dir, 'encoder.onnx'))
        self.decoder = ort.InferenceSession(os.path.join(save_dir, 'decoder.onnx'))

        self.device = device
        assert 'cpu' in device or 'cuda' in device, f'device {device} is not supported'

    def get_codebook_indices(self, x):
        device = self.device
        x = ort.OrtValue.ortvalue_from_numpy(x.astype(np.float32), device, 0)
        io_binding = self.encoder.io_binding()
        io_binding.bind_ortvalue_input("x", x)
        io_binding.bind_output("indices")
        io_binding.bind_output("height")
        io_binding.bind_output("width")
        self.encoder.run_with_iobinding(io_binding)
        indices, height, width = io_binding.get_outputs()[:3]
        indices = indices.numpy()
        height = int(height.numpy())
        width = int(width.numpy())

        return indices, height, width

    def decode(self, img_seq, height, width):
        device = self.device
        img_seq = ort.OrtValue.ortvalue_from_numpy(img_seq.astype(np.int64), device, 0)
        height = ort.OrtValue.ortvalue_from_numpy(np.asarray(height).astype(np.int64), device, 0)
        width = ort.OrtValue.ortvalue_from_numpy(np.asarray(width).astype(np.int64), device, 0)
        io_binding = self.decoder.io_binding()
        io_binding.bind_ortvalue_input("img_seq", img_seq)
        io_binding.bind_ortvalue_input("height", height)
        io_binding.bind_ortvalue_input("width", width)
        io_binding.bind_output("img")
        self.decoder.run_with_iobinding(io_binding)
        img = io_binding.get_outputs()[0]
        img = img.numpy()

        return img


def exponential_(self):
    eps = 1e-10
    U = torch.rand_like(self)
    out = -torch.log(U + eps) + eps
    self[:] = out
    return self


def convert_vae(vae, save_dir):

    native_exponential_ = torch.Tensor.exponential_
    torch.Tensor.exponential_ = exponential_
    
    os.makedirs(save_dir, exist_ok=True)

    device = next(vae.parameters()).device
    img = torch.zeros(1, 3, 256, 256, dtype=torch.float32, device=device)

    def convert_encoder(save_path):
        model = Encoder(vae)
        model.eval()

        with torch.no_grad():
            indices, height, width = model(img)

        model = torch.jit.freeze(torch.jit.trace(model, (img, ))) # warning missing match because of exponential distribution
        model.eval()
        model.training = False
        torch.onnx.export(
            model=model, 
            input_names=["x"], 
            output_names=["indices", "height", "width"],
            dynamic_axes={
                "x": {
                    0: "batch_size",
                    2: "height",
                    3: "width"
                },
                "indices": {
                    0: "batch_size",
                    1: "seq_len"
                }
            },
            args=(
                img
            ), 
            opset_version=11,
            f=save_path, 
            verbose=False)

        return indices, torch.tensor(height).long(), torch.tensor(width).long()

    img_seq, height, width = convert_encoder(os.path.join(save_dir, 'encoder.onnx'))

    def convert_decoder(save_path):
        model = Decoder(vae)
        model.eval()

        with torch.no_grad():
            img = model(img_seq, height, width)

        model = torch.jit.freeze(torch.jit.trace(model, (img_seq, height, width)))
        model.eval()
        model.training = False
        torch.onnx.export(
            model=model, 
            input_names=["img_seq", "height", "width"], 
            output_names=["img"],
            dynamic_axes={
                "img_seq": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "img": {
                    0: "batch_size",
                    2: "height",
                    3: "width"
                }
            },
            args=(
                img_seq,
                height, 
                width
            ), 
            opset_version=11,
            f=save_path, 
            verbose=False)

        return img

    convert_decoder(os.path.join(save_dir, 'decoder.onnx'))

    torch.Tensor.exponential_ = native_exponential_
