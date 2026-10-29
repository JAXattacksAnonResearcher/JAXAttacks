import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class FGSM(Attack):
    r"""
    FGSM paper: 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 8 / 255):
        super().__init__("FGSM")
        self.eps = eps
        self.supported_modes = ["default", "targeted"]

        self.create_functions()

    def main_attack_function(self, images, labels):
    
        return self._call(images, labels, self.targeted, self.eps)
    
    def _call(self, images, labels, targeted, eps):
        
        loss_to_print, grads = self.grad_fn(images, labels)
        
        grads = jax.lax.cond(targeted,
                             lambda _: -1 * grads,
                             lambda _: grads,
                             operand=None)

        adv_images = images + eps * jnp.sign(grads)
            
        adv_images = jnp.clip(adv_images, 0.0, 1.0)
        return adv_images


    def create_functions(self):
        def calculate_loss(imgs, lbls):
            logits = self.get_logits(imgs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, lbls).mean()
            return loss

        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
    def jit_call(self):
        self._call = jax.jit(self._call)
        
        