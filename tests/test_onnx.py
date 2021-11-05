# -*- coding: utf-8 -*-
import torch

def test_dalle_to_onnx(small_dalle):
    small_dalle_jit = torch.jit.script(small_dalle)
