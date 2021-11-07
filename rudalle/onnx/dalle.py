import os
import glob
import torch
import torch.nn as nn

import onnxmltools
import numpy as np
import onnxruntime as ort

from onnxmltools.utils.float16_converter import convert_float_to_float16


class Embedding(nn.Module):
    def __init__(self, dalle):
        super(Embedding, self).__init__()

        self.dalle = dalle

    def forward(
            self,
            input_ids,
            attention_mask
    ):
        embeddings, attention_mask = self.dalle.embedding(input_ids, attention_mask)
        return embeddings, attention_mask


class SingleTransformer(nn.Module):
    def __init__(self, dalle, i):
        super(SingleTransformer, self).__init__()

        self.layer = dalle.transformer.layers[i]
        self._mask_map = dalle.transformer._mask_map[i]

    def forward(self, hidden_states, attention_mask, caches, use_cache: bool=False):
        mask = attention_mask
        if len(self._mask_map):
            layer_mask = self._mask_map[:mask.size(2), :mask.size(3)]
            mask = torch.mul(attention_mask, layer_mask)
        hidden_states, present_caches = self.layer(
            hidden_states, mask, caches, use_cache=use_cache
        )
        return hidden_states, present_caches


class TransformersFinalNorm(nn.Module):
    def __init__(self, dalle):
        super(TransformersFinalNorm, self).__init__()

        self.dalle = dalle

    def forward(self, hidden_states):
        return self.dalle.transformer.final_layernorm(hidden_states)


class ToLogits(nn.Module):
    def __init__(self, dalle):
        super(ToLogits, self).__init__()

        self.dalle = dalle

    def forward(self, transformer_output):
        return self.dalle.to_logits(transformer_output)


class DalleONNXModel(object):
    def __init__(self, save_dir, device, use_fp16=False):
        prefix = ''
        if use_fp16:
            prefix = 'fp16_'
            if len(glob.glob(os.path.join(save_dir, prefix + '*.onnx'))) == 0:
                onnxfiles = os.listdir(save_dir)
                for file in onnxfiles:
                    if '.onnx' in file and 'fp16_' not in file:
                        onnx_model = onnxmltools.utils.load_model(os.path.join(save_dir, file))
                        onnx_model = convert_float_to_float16(onnx_model)
                        onnxmltools.utils.save_model(onnx_model, os.path.join(save_dir, prefix + file))
        self.dtype = np.float16 if use_fp16 else np.float32
        self.embedding = ort.InferenceSession(os.path.join(save_dir, prefix + 'embedding.onnx'))
        self.transformers = [
            ort.InferenceSession(path) for path in sorted(glob.glob(os.path.join(save_dir, prefix + 'transformer_*.onnx')))
        ]
        self.transformer_final_norm = ort.InferenceSession(os.path.join(save_dir, prefix + 'trans_final_norm.onnx'))
        self.to_logits = ort.InferenceSession(os.path.join(save_dir, prefix + 'to_logits.onnx'))
        self.device = device
        assert device == 'cpu' or device == 'cuda'

    def __call__(
        self,
        input_ids,
        attention_mask,
        caches=[],
        use_cache: bool=False
    ):
        device = self.device
        batch_size = input_ids.shape[0]
        input_ids = ort.OrtValue.ortvalue_from_numpy(input_ids.astype(np.int64), device, 0)
        attention_mask = ort.OrtValue.ortvalue_from_numpy(attention_mask.astype(self.dtype), device, 0)
        use_cache = ort.OrtValue.ortvalue_from_numpy(np.asarray(use_cache), device, 0)
        io_binding = self.embedding.io_binding()
        io_binding.bind_ortvalue_input("input_ids", input_ids)
        io_binding.bind_ortvalue_input("attention_mask", attention_mask)
        io_binding.bind_output("embeddings")
        io_binding.bind_output("attention_mask_out")
        self.embedding.run_with_iobinding(io_binding)
        hidden_states, attention_mask = io_binding.get_outputs()[:2]

        new_caches = []
        for i, transformer in enumerate(self.transformers):
            io_binding = transformer.io_binding()
            io_binding.bind_ortvalue_input("hidden_states", hidden_states)
            io_binding.bind_ortvalue_input("attention_mask", attention_mask)
            if len(caches) == 0:
                caches_shape = transformer.get_inputs()[2].shape
                _caches = np.zeros(
                    [caches_shape[0], batch_size, 1], 
                    dtype=self.dtype
                )
                _caches = ort.OrtValue.ortvalue_from_numpy(_caches, device, 0)
            else:
                _caches = caches[i]
            io_binding.bind_ortvalue_input("caches", _caches)
            io_binding.bind_ortvalue_input("use_cache", use_cache)
            io_binding.bind_output("hidden_states_out")
            io_binding.bind_output("present_caches")
            transformer.run_with_iobinding(io_binding)
            hidden_states, present_caches = io_binding.get_outputs()[:2]
            new_caches.append(present_caches)
        
        io_binding = self.transformer_final_norm.io_binding()
        io_binding.bind_ortvalue_input("hidden_states", hidden_states)
        io_binding.bind_output("transformer_output")
        self.transformer_final_norm.run_with_iobinding(io_binding)
        transformer_output = io_binding.get_outputs()[0]

        io_binding = self.to_logits.io_binding()
        io_binding.bind_ortvalue_input("transformer_output", transformer_output)
        io_binding.bind_output("logits")
        self.to_logits.run_with_iobinding(io_binding)
        logits = io_binding.copy_outputs_to_cpu()[0]

        return logits, new_caches


def convert_dalle(dalle, save_dir):

    assert not hasattr(dalle, 'module'), "FP16 Module is not supported"
    
    os.makedirs(save_dir, exist_ok=True)

    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    num_layers = dalle.get_param('num_layers')
    hidden_size = dalle.get_param('hidden_size')
    device = dalle.get_param('device')

    input_ids = torch.zeros(1, text_seq_length+1, dtype=torch.int64, device=device)
    attention_mask = torch.tril(torch.ones((1, 1, total_seq_length, total_seq_length), device=device))
    caches = torch.zeros(num_layers*hidden_size*4, 1, text_seq_length+1, device=device)

    def convert_embedding(save_path):
        model = Embedding(dalle)
        model.eval()
        with torch.no_grad():
            embeddings, _attention_mask = model(input_ids, attention_mask)

        model = torch.jit.freeze(torch.jit.script(model).eval())
        model.training = False
        torch.onnx.export(
            model=model, 
            input_names=["input_ids", "attention_mask"], 
            output_names=["embeddings", "attention_mask_out"],
            dynamic_axes={
                "input_ids": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "attention_mask": {
                    0: "batch_size"
                },
                "embeddings": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "attention_mask_out": {
                    0: "batch_size"
                }
            },
            args=(
                input_ids, 
                attention_mask
            ), 
            opset_version=11,
            f=save_path, 
            verbose=False)

        return embeddings, _attention_mask

    hidden_states, attention_mask = convert_embedding(os.path.join(save_dir, 'embedding.onnx'))

    def convert_single_transformer(i, save_path):
        base = hidden_size*4
        _caches = caches[i*base:(i+1)*base]
        model = SingleTransformer(dalle, i)
        model.eval()
        with torch.no_grad():
            _hidden_states, present_caches = model(hidden_states, attention_mask, _caches, True)

        model = torch.jit.freeze(torch.jit.script(model).eval())
        model.training = False
        torch.onnx.export(
            model=model, 
            input_names=["hidden_states", "attention_mask", "caches", "use_cache"], 
            output_names=["hidden_states_out", "present_caches"],
            dynamic_axes={
                "hidden_states": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "attention_mask": {
                    0: "batch_size",
                    2: "seq_len",
                    3: "seq_len"
                },
                "caches": {
                    1: "batch_size",
                    2: "pre_seq_len"
                },
                "hidden_states_out": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "present_caches": {
                    1: "batch_size",
                    2: "seq_len"
                }
            },
            args=(
                hidden_states, 
                attention_mask, 
                _caches, 
                True
            ), 
            opset_version=11,
            f=save_path, 
            verbose=False)

        return _hidden_states

    for idx in range(num_layers):
        hidden_states = convert_single_transformer(idx, os.path.join(save_dir, f'transformer_{str(idx).zfill(3)}.onnx'))

    def convert_transformer_final_norm(save_path):
        model = TransformersFinalNorm(dalle)
        model.eval()
        with torch.no_grad():
            transformer_output = model(hidden_states)
        
        model = torch.jit.freeze(torch.jit.script(model).eval())
        model.training = False
        torch.onnx.export(
            model=model, 
            input_names=["hidden_states"], 
            output_names=["transformer_output"],
            dynamic_axes={
                "hidden_states": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "transformer_output": {
                    0: "batch_size",
                    1: "seq_len"
                }
            },
            args=(
                hidden_states
            ), 
            opset_version=11,
            f=save_path, 
            verbose=False)

        return transformer_output

    transformer_output = convert_transformer_final_norm(os.path.join(save_dir, 'trans_final_norm.onnx'))

    def convert_to_logits(save_path):
        model = ToLogits(dalle)
        model.eval()
        with torch.no_grad():
            logits = model(transformer_output)
        
        model = torch.jit.freeze(torch.jit.script(model).eval())
        model.training = False
        torch.onnx.export(
            model=model, 
            input_names=["transformer_output"], 
            output_names=["logits"],
            dynamic_axes={
                "transformer_output": {
                    0: "batch_size",
                    1: "seq_len"
                },
                "logits": {
                    0: "batch_size",
                    1: "seq_len"
                }
            },
            args=(
                transformer_output
            ), 
            opset_version=11,
            f=save_path, 
            verbose=False)

        return logits

    convert_to_logits(os.path.join(save_dir, 'to_logits.onnx'))
