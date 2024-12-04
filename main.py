from jac import calc_jacobian
from transformer_lens import HookedTransformer
from datasets import load_dataset

if __name__ == '__main__':
    model = HookedTransformer.from_pretrained("gpt2-small")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")
