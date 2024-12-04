from jac import calc_jacobian
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import torch as t
import einops
import matplotlib.pyplot as plt
from datasets import load_dataset

if __name__ == '__main__':
    model = HookedTransformer.from_pretrained("gpt2-small")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    ds.set_format(type="torch", columns=["text"])

    # Batch some inputs
    batch_size = 2
    context_size = 1024

    for i, batch in zip(range(2), ds['train'].iter(batch_size=batch_size)):
        flattened = "".join([char for text in batch["text"] for char in text])
        #for j in range(len(flattened)//context_size):
        for j in range(1):
            tokens = tokenizer(
                        flattened[j * context_size : (j + 1) * context_size],
                        max_length=1024,
                        return_tensors="pt").input_ids

            jac = calc_jacobian(model.blocks[0].hook_resid_mid, model.blocks[0].hook_resid_post, model, tokens,
                                stop_idx=16)
            f = plt.imshow(einops.rearrange(jac, 
                "batch seq d_res1 d_res2 -> (seq d_res1) (batch d_res2)").cpu(), vmin=-1, vmax=1)
            plt.colorbar(f)
            plt.savefig(f"{i}-{j}.png")


