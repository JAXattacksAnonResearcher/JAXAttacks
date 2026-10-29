import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack

class RFGSMEnsemble(EnsembleAttack):
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    Distance Measure : Linf

    Arguments:
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float) : step size. (Default 2/255)
        main_rng_seed (int): random seed used for initializing the pseudo-random number generator. (Default: 42)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """
    
    def __init__(self, eps = 8 / 255, alpha = 2 / 255, main_rng_seed = 42):
        super().__init__("RFGSMEnsemble")
        self.eps = eps
        self.alpha = alpha
        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.supported_modes = ["default", "targeted"]

        self.create_functions()
        

    def main_attack_function(self, images, labels):
        
        adv_images, self.main_rng = self._call(images, labels, self.targeted, self.eps, self.alpha, self.main_rng)
        
        return adv_images
        
    def _call(self, images, labels, targeted, eps, alpha, main_rng):
        
        main_rng, normal_rng = jax.random.split(main_rng, num = 2)
        
        randomized_images = images + alpha * jnp.sign(jax.random.normal(normal_rng, images.shape))
        
        loss_to_print, grads = self.grad_fn(randomized_images, labels)
            
        grads = jax.lax.cond(targeted,
                             lambda _ : -1 * grads,
                             lambda _ : grads,
                             operand=None)
            
        adv_images = randomized_images + (eps - alpha) * jnp.sign(grads)
        adv_images = jnp.clip(adv_images, 0.0, 1.0)
        return adv_images, main_rng
        
    
    def create_functions(self):
        def calculate_loss(imgs, lbls):
            logits = self.get_all_logits(imgs)
            
            _, all_losts = jax.lax.scan(lambda _, x : (_, optax.softmax_cross_entropy_with_integer_labels(x, lbls).mean()),
                                        0, # dummy init for carry
                                        logits)
            total_lost = jnp.mean(all_losts)
            return total_lost

        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
        
        
    def jit_call(self):
        self._call = jax.jit(self._call)

