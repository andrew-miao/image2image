import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from typing import Any


class AccumulatingTrainState(TrainState):
    """A TrainState that accumulates gradients over multiple steps before
    applying them. This is an alternative to `optax.MultiSteps` that uses less
    memory because it takes `do_update` as a static argument (rather than
    computing it internally).

    Since MultiSteps effectively computes `do_update` internally and uses
    `jax.lax.cond` to select the output, it must allocate an additional buffer
    with the size of `opt_state` because it doesn't "know" at compile time
    whether it will update `opt_state` or not. Instead, this version leads to
    two compiled functions, one for each value of `do_update`, meaning it only
    allocates one buffer for `opt_state` in each case.

    This still requires an additional buffer of size `params` to store the
    accumulated gradients, but that is unavoidable."""

    grad_acc: FrozenDict[str, Any]
    n_acc: int

    def apply_gradients(self, *, grads, do_update, **kwargs):
        if do_update:
            new_state = super().apply_gradients(
                grads=jax.tree_map(
                    lambda ga, g: (ga + g) / (self.n_acc + 1), self.grad_acc, grads
                ),
                **kwargs
            )
            new_state = new_state.replace(
                grad_acc=jax.tree_map(jnp.zeros_like, self.grad_acc), n_acc=0
            )
        else:
            new_state = self.replace(
                grad_acc=jax.tree_map(jnp.add, self.grad_acc, grads),
                n_acc=self.n_acc + 1,
            )
        return new_state

    @classmethod
    def create(cls, *, params, **kwargs):
        return super().create(
            params=params,
            grad_acc=jax.tree_map(jnp.zeros_like, params),
            n_acc=0,
            **kwargs
        )