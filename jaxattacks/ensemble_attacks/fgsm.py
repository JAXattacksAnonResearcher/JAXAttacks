import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack

class FGSMEnsemble(EnsembleAttack):
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
        super().__init__("FGSMEnsemble")
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
        # Weighted loss can also be defined by changing self.calculate_loss
        # Any operation is permissible as long as calculate_loss returns a scalar
        # Since "Gradient only defined for scalar-output functions" for JAX self.grad and self.value_and_grad
        def calculate_loss(imgs, lbls):
            logits = self.get_all_logits(imgs)

            # Note:
            # 1-) First summing individual losses and then taking gradient of total loss wrt. image
            # 2-) Firs calculating individual gradients wrt image then summing gradients
            # '1' and '2' is the same [might differ 'slightly' (in magnitude of 1e-7) because of floating point precision]
            _, all_losses = jax.lax.scan(lambda _, x: (_, optax.softmax_cross_entropy_with_integer_labels(x, lbls).mean()),
                                         0, # dummy init for carry
                                         logits)
            total_loss = jnp.mean(all_losses)
            return total_loss
            

        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
    def jit_call(self):
        self._call = jax.jit(self._call)

        