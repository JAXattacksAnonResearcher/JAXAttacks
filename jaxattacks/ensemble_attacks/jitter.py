import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack
from ..utils import mode, mode_along_axis
mode_along_0 = mode_along_axis(0)

class JitterEnsemble(EnsembleAttack):
    r"""
    Jitter in the paper 'Exploring Misclassifications of Robust Neural Networks to Enhance Adversarial Attacks'
    [https://arxiv.org/abs/2105.10304]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)
        main_rng_seed (int): random seed used for initializing the pseudo-random number generator. (Default: 42)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True, main_rng_seed=42):
        super().__init__("JitterEnsemble")
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.scale = scale
        self.std = std 
        self.random_start = random_start
        self.main_rng = jax.random.PRNGKey(main_rng_seed)
        self.supported_modes = ["default", "targeted"]

        self.create_functions()

    def main_attack_function(self, images, labels):
    
        adv_images, self.main_rng = self._call(images, labels, self.targeted, self.eps, self.alpha, self.steps, self.scale, self.std, self.random_start, self.main_rng)
        
        return adv_images

    def _call(self, images, labels, targeted, eps, alpha, steps, scale, std, random_start, main_rng):
        
        main_rng, random_start_rng, normal_rng = jax.random.split(main_rng, num=3)

        adv_images = jnp.copy(images)
        adv_images = jax.lax.cond(random_start,
                                  lambda _: adv_images + jax.random.uniform(random_start_rng, adv_images.shape, minval=-eps, maxval=eps, dtype=jnp.float32),
                                  lambda _: adv_images,
                                  operand=None)
        adv_images = jnp.clip(adv_images, 0.0, 1.0)

        x = {
            "ori_images": images,
            "adv_images": adv_images,
            "labels": labels,
            "targeted": targeted,
            "eps": eps,
            "alpha": alpha,
            "scale": scale,
            "std": std,
            "normal_rng": normal_rng,
        }

        x = jax.lax.fori_loop(0, steps, self.body_fn, x)
        adv_images = jnp.clip(x["adv_images"], 0.0, 1.0)

        return adv_images, main_rng


    def create_functions(self):
        def calculate_loss(adv_imgs, ori_imgs, lbls, scale, std, normal_rng, targeted):
            all_logits = self.get_all_logits(adv_imgs) # (model_count, batch_size, num_classes)

            all_pre = jnp.argmax(all_logits, axis=-1) # (model_count, batch_size)
            pre = mode_along_0(all_pre)
            wrong = pre != lbls # (batch_size)

            norm_z = jnp.linalg.norm(all_logits, ord=jnp.inf, axis=-1, keepdims=True) # (model_count, batch_size, 1)
            hat_z = jax.nn.softmax(scale * all_logits / norm_z) # (model_count, batch_size, num_classes)

            hat_z = jax.lax.cond(std != 0,
                                 lambda _: hat_z + std * jax.random.normal(normal_rng, hat_z.shape, dtype=jnp.float32),
                                 lambda _: hat_z, # std == 0.0
                                 operand=None)
            
            Y = jax.nn.one_hot(lbls, all_logits.shape[-1]) # (batch_size, num_classes)
            # copy onehot_labels across model_count dimension for loss calculation
            Y = jnp.broadcast_to(Y, hat_z.shape) # (model_count, batch_size, num_classes)

            cost = jnp.mean(optax.squared_error(hat_z, Y), axis=-1) # (model_count, batch_size) 
            cost = jax.lax.cond(targeted,
                                lambda _: -cost,
                                lambda _: cost,
                                operand=None)

            norm_r = jnp.max(jnp.abs(adv_imgs - ori_imgs), axis=(1, 2, 3)) # (batch_size)
            nonzero_r = norm_r != 0 # (batch_size)
            # NOTE:
            # JAX and PyTorch differ in how they handle branching operations during
            # automatic differentiation. In PyTorch, only the branch actually taken
            # during the forward pass contributes to the backward pass. In JAX,
            # however, both branches of `jnp.where` (or `lax.cond`) are traced for
            # differentiation regardless of which branch is selected.
            #
            # In this implementation, `norm_r` can be zero for some samples early in
            # the optimization. Although the forward pass intends to avoid division
            # by zero by selecting the `cost` branch when `norm_r == 0`, JAX still
            # traces the unused branch that contains `cost / norm_r`. This means the
            # autodiff system internally encounters a division by zero, producing
            # NaNs in the backward pass even though the forward values appear valid.
            #
            # To prevent this, we add a small epsilon (1e-12) to `norm_r`, ensuring
            # that the traced computation graph never contains an actual `1 / 0`
            # operation. This preserves numerical stability and avoids NaN gradients,
            # while leaving the forward behavior effectively unchanged.
            cost = jnp.where((wrong * nonzero_r)[None, :], cost / (norm_r + 1e-12)[None, :], cost) # ((batch_size), (model_count, batch_size) / (batch_size), (model_count, batch_size) )

            loss = jnp.mean(cost)
            return loss

        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)

        def body_fn(i, x):
            x["normal_rng"], hat_z_rng = jax.random.split(x["normal_rng"], num=2)

            loss_to_print, grads = self.grad_fn(
                x["adv_images"],
                x["ori_images"],
                x["labels"],
                x["scale"],
                x["std"],
                hat_z_rng,
                x["targeted"]
            )
            
            x["adv_images"] = x["adv_images"] + x["alpha"] * jnp.sign(grads)
            delta = jnp.clip(
                x["adv_images"] - x["ori_images"], -x["eps"], x["eps"]
            )
            x["adv_images"] = jnp.clip(x["ori_images"] + delta, 0.0, 1.0)

            return x

        self.body_fn = body_fn
        
    def jit_call(self):
        self._call = jax.jit(self._call)
        
        