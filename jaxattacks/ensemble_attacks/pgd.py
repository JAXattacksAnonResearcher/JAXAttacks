import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack

class PGDEnsemble(EnsembleAttack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta (Default: True)
        main_rng_seed (int): random seed used for initializing the pseudo-random number generator. (Default: 42)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 8/255, alpha = 2/255, steps=10, random_start = True, main_rng_seed = 42):
        super().__init__("PGDEnsemble")
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self. main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.supported_modes = ["default", "targeted"]

        self.create_functions()
        
    def main_attack_function(self, images, labels):
    
        adv_images, self.main_rng = self._call_fori_loop(images, labels, self.targeted, self.eps, self.alpha, self.steps, self.random_start, self.main_rng)
        
        return adv_images
        
    def _call_fori_loop(self, images, labels, targeted, eps, alpha, steps, random_start, main_rng):
        
        ori_images = images
        adv_images = jnp.copy(images)
        
        main_rng, uniform_rng = jax.random.split(main_rng, num=2)
        
        adv_images = jax.lax.cond(random_start,
                                  lambda _: adv_images + jax.random.uniform(uniform_rng, adv_images.shape, minval = -eps, maxval = eps,  dtype=jnp.float32),
                                  lambda _: adv_images,
                                  operand=None)
                                  
        adv_images = jnp.clip(adv_images, 0.0, 1.0)
                                  
        x = {
            "ori_images" : ori_images,
            "adv_images" : adv_images,
            "labels": labels,
            "targeted": targeted,
            "eps" : eps,
            "alpha" : alpha,
            }
        
        x = jax.lax.fori_loop(0, steps, self.body_fn, x)
        
        adv_images = jnp.clip(x["adv_images"], 0.0, 1.0)
        return adv_images, main_rng
        
        
    def create_functions(self):
        ## Loss Function
        def calculate_loss(imgs, lbls):
            logits = self.get_all_logits(imgs)
            
            _, all_losses = jax.lax.scan(lambda _, x : (_, optax.softmax_cross_entropy_with_integer_labels(x, lbls).mean()),
                                         0, # dummy init for carry
                                         logits)
            total_loss = jnp.mean(all_losses)
            return total_loss
            
        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
        ## body_fn for jax.lax.fori_loop
        def body_fn(_, x): 
            loss_to_print, grads = self.grad_fn(x["adv_images"], x["labels"])
            
            grads = jax.lax.cond(x["targeted"],
                                 lambda _: -1*grads,
                                 lambda _: grads,
                                 operand=None)
                                 
            adv_images = x["adv_images"] + x["alpha"] * jnp.sign(grads)
            delta = jnp.clip(adv_images - x["ori_images"], -x["eps"], x["eps"])
            x["adv_images"] = jnp.clip(x["ori_images"] + delta, 0.0, 1.0)
            
            return x
            
        self.body_fn = body_fn 
        
    def jit_call(self):
        self._call_fori_loop = jax.jit(self._call_fori_loop)
            
    