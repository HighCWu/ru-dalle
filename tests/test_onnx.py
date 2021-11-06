# -*- coding: utf-8 -*-
import torch
import pytest

from .test_vae import preprocess
from rudalle.convert_to_onnx import convert_dalle
from tempfile import NamedTemporaryFile


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

    f = NamedTemporaryFile(delete=False)
    convert_dalle(small_dalle, f.name)
    small_dalle.set_onnx(f.name)

    _logits, _caches = small_dalle.onnx_forward(input_ids, attention_mask, [], False)
    _logits2, _caches2 = small_dalle.forward(input_ids2, attention_mask, _caches, True)

    assert (logits - _logits).pow(2).mean().item() < 1e-9
    assert (logits2 - _logits2).pow(2).mean().item() < 1e-9
    assert (caches - _caches).pow(2).mean().item() < 1e-9
    assert (caches2 - _caches2).pow(2).mean().item() < 1e-9

    f.close()
