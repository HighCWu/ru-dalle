# -*- coding: utf-8 -*-
import torch
import pytest

from .test_vae import preprocess
from rudalle.onnx.dalle import convert_dalle
from rudalle.onnx.vae import convert_vae
from tempfile import TemporaryDirectory


@pytest.mark.parametrize('text', [
    'мальчик играет с оленем',
])
def test_dalle_onnx(text, sample_image, yttm_tokenizer, vae, small_dalle):
    bs = 1
    text_seq_length = small_dalle.get_param('text_seq_length')
    total_seq_length = small_dalle.get_param('total_seq_length')
    device = small_dalle.get_param('device')

    img = sample_image.copy()
    img = preprocess(img, target_image_size=256)
    images = img.repeat(bs, 1, 1, 1).to(device)

    text = text.lower().strip()
    text_input_ids = yttm_tokenizer.encode_text(text, text_seq_length=text_seq_length)
    text_input_ids = text_input_ids.unsqueeze(0).repeat(bs, 1).to(device)

    attention_mask = torch.tril(torch.ones((bs, 1, total_seq_length, total_seq_length), device=device))
    with torch.no_grad():
        image_input_ids = vae.get_codebook_indices(images)[:,:-20]
        input_ids = torch.cat((text_input_ids, image_input_ids), dim=1)
        logits, caches = small_dalle.forward(input_ids, attention_mask, [], False)

        input_ids2 = torch.cat([input_ids, input_ids[:,:1]], 1)
        logits2, caches2 = small_dalle.forward(input_ids2, attention_mask, caches, True)

    f = TemporaryDirectory()
    convert_dalle(small_dalle, f.name)
    small_dalle.set_onnx(f.name, use_fp16=True)

    _logits, _caches = small_dalle.forward(input_ids, attention_mask, [], False)
    _logits2, _caches2 = small_dalle.forward(input_ids2, attention_mask, _caches, True)

    assert (logits - _logits).pow(2).mean().item() < 1e-6
    assert (logits2 - _logits2).pow(2).mean().item() < 1e-6

    f.cleanup()


def test_vae_onnx(vae):
    device = next(vae.parameters()).device
    img = torch.zeros(1,3,128,128).to(device)
    with torch.no_grad():
        img_seq = vae.get_codebook_indices(img)
        img_rec = vae.decode(img_seq)

    f = TemporaryDirectory()
    convert_vae(vae, f.name)
    vae.set_onnx(f.name)

    img_seq2 = vae.get_codebook_indices(img)
    img_rec2 = vae.decode(img_seq)

    assert img_seq.shape == img_seq2.shape
    assert img_rec.shape == img_rec2.shape

    f.cleanup()
