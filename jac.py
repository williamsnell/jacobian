import torch as t
import einops

from torch import Tensor
from typing import Optional
from jaxtyping import Float


def attach_jacobian_hooks(upstream: Float[Tensor, "d_upstream"],
                          downstream: Float[Tensor, "d_downstream"],
                          network: t.nn.Module,
                          stop_idx_downstream: int,
                          start_idx_downstream=0):
    """Calculate the jacobian matrix between an upstream vector and
       a downstream vector. You must run a forward pass through the model
       before the gradients tensor will be populated.

       upstream - The upstream vector.
       downstream - The downstream vector. Does not need to have the same
                    shape as `upstream`.
       network - The model that contains both `upstream` and `downstream`.

       stop_idx_downstream (required) and
       start_idx_downstream(optional)-
            The jacobian will be calculated for the downstream vector elements
            as downstream[ start_idx_downstream : stop_idx_downstream ]

       Returns: (get_jacobian(), get_upstream_vec(), remove_hooks())
    """
    n_outputs = stop_idx_downstream - start_idx_downstream

    capture = {}

    def setup_upstream_hook(module, inp, out):

        capture['upstream_vec'] = out

        out = einops.repeat(out,
                            "batch ... d_hidden -> (batch d_out) ... d_hidden",
                            d_out=n_outputs)

        # Setup a do-nothing vector to let us extract the gradients
        # of this intermediate layer.
        output_shape = out.shape

        capture['upstream_grad'] = t.zeros(output_shape, requires_grad=True, device=out.device)

        return out + capture['upstream_grad']

    def setup_downstream_hook(module, inp, out):
        # Extract the jacobian dimension we snuck
        # into the batch dimension.
        out = einops.rearrange(out,
                               "(batch d_out) ... d_hidden -> batch ... d_out d_hidden",
                               d_out=n_outputs)

        network.zero_grad()
        out[..., start_idx_downstream : stop_idx_downstream].backward(
            t.eye(n_outputs, device=out.device).repeat(*out.shape[:-2], 1, 1))

    remove_upstr_hook = upstream.register_forward_hook(setup_upstream_hook)
    remove_downstr_hook = downstream.register_forward_hook(setup_downstream_hook)

    def remove_hooks():
        remove_upstr_hook.remove()
        remove_downstr_hook.remove()

    def get_jacobian():
        if capture.get("upstream_grad") is None:
            raise RuntimeError("Gradients must be initialized by "
                               "running a forward pass through "
                               "the model before they can be  "
                               "accessed.")
        rearranged = einops.rearrange(capture['upstream_grad'].grad,
                                "(batch d_out) ... d_in -> batch ... d_out d_in",
                                d_out=n_outputs)
        return rearranged
    def get_upstream_vec():
        if capture.get("upstream_vec") is None:
            raise RuntimeError("Vectors must be initialized by "
                               "running a forward pass through "
                               "the model before they can be  "
                               "accessed.")

        return capture['upstream_vec']

    return get_jacobian, get_upstream_vec, remove_hooks

def calc_jacobian(
    upstream_vec: Float[Tensor, "d_up"],
    downstream_vec: Float[Tensor, "d_down"],
    model: t.nn.Module,
    tokens: Float[Tensor, "batch seq 1"],
    stop_idx: Optional[int] = None,
    ) -> Float[Tensor, "batch seq d_down d_up"]:
  """
      Return the jacobian,
        J = d(downstream_vec)/d(upstream_vec)

      The jacobian will be calculated across the batches and sequences
      in `tokens`.

      upstream_vec: Vector in `model` upstream (i.e. before) `downstream_vec` in
                    `model`'s forward pass.
      downstream_vec: Vector in `model`
      model: The torch neural net containing `upstream_vec` and `downstream_vec`,
             and accepting `tokens` for its forward pass.
  """
  jacs = []

  with t.set_grad_enabled(True):
      model.requires_grad_(True)

      for i in range(tokens.shape[0]):
          get_jacobian, get_upstream_vec, remove_hooks = attach_jacobian_hooks(
              upstream_vec, downstream_vec, model, model.cfg.d_model if stop_idx is None else stop_idx
              )

          # Run a forward pass through the model
          model(tokens[i: i + 1])

          jacs += [get_jacobian()]

          remove_hooks()

      model.requires_grad_(False)

  return t.cat(jacs)

