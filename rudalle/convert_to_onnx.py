import torch

def convert_dalle(dalle, save_path):
    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    num_layers = dalle.get_param('num_layers')
    hidden_size = dalle.get_param('hidden_size')
    device = dalle.get_param('device')

    input_ids = torch.zeros(1, 1+1, dtype=torch.int64, device=device)
    attention_mask = torch.tril(torch.ones((1, 1, total_seq_length, total_seq_length), device=device))
    caches = torch.zeros(num_layers*hidden_size*4, 1, 1+1, device=device)
    with torch.no_grad():
        dalle(input_ids, attention_mask, caches, True, True)
    dalle = torch.jit.script(dalle)
    torch.onnx.export(
        model=dalle, 
        input_names=["input_ids", "attention_mask", "caches", "use_cache", "fast_convert"], 
        output_names=["logits", "present_caches"],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "seq_len"
            },
            "attention_mask": {
                0: "batch_size"
            },
            "caches": {
                1: "batch_size",
                2: "pre_seq_len"
            },
            "logits": {
                0: "batch_size",
                1: "seq_len"
            },
            "present_caches": {
                1: "batch_size",
                2: "seq_len"
            }
        },
        args=(
            input_ids,
            attention_mask,
            caches,
            True,
            True
        ), 
        opset_version=11,
        f=save_path, 
        verbose=False)
