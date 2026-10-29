import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class TPGD(Attack):
    r"""
    PGD based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/abs/1901.08573]

    Distance Measure : Linf

    Arguments:
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        main_rng_seed (int): random seed used for initializing the pseudo-random number generator. (Default: 42)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 8/255, alpha = 2/255, steps=10, main_rng_seed = 42):
        super().__init__("TPGD")
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.supported_modes = ["default"]

        self.create_functions()
        
        
    def main_attack_function(self, images, labels):
    
        return self._call_fori_loop(images, self.eps, self.alpha, self.steps, self. main_rng)
        
        
    def _call_fori_loop(self, images, eps, alpha, steps, main_rng):
    
        ori_images = images
        adv_images = jnp.copy(images)
        
        ori_logits = self.get_logits(ori_images)
        
        # Perturb image a little to get starting grads 
        main_rng, normal_rng = jax.random.split(main_rng, num=2)
        
        adv_images = adv_images + 0.001 * jax.random.normal(normal_rng, adv_images.shape, dtype=jnp.float32)                          
        adv_images = jnp.clip(adv_images, 0.0, 1.0)
        
        x = {
            "ori_images" : ori_images,
            "adv_images" : adv_images,
            "ori_logits" : ori_logits,
            "eps" : eps,
            "alpha" : alpha,
        }
    
        x = jax.lax.fori_loop(0, steps, self.body_fn, x)
        
        adv_images = jnp.clip(x["adv_images"], 0.0, 1.0)
        return adv_images
        
        
    def create_functions(self):
        ## Loss Function
        def calculate_loss(adv_imgs, ori_logits):
            adv_logits = self.get_logits(adv_imgs)
            loss = optax.losses.kl_divergence(jax.nn.log_softmax(adv_logits, axis=1),
                                              jax.nn.softmax(ori_logits, axis=1)).sum()
            return loss
                
        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
        
        ## body_fn for jax.lax.fori_loop
        def body_fn(_, x):
            # Grad calculation
            loss_to_print, grads = self.grad_fn(x["adv_images"], x["ori_logits"])
            
            # Image Update
            adv_images = x["adv_images"] + x["alpha"] * jnp.sign(grads)
            delta = jnp.clip(adv_images - x["ori_images"], -x["eps"], x["eps"])
            x["adv_images"] = jnp.clip(x["ori_images"] + delta, 0.0, 1.0)
            
            return x
            
        self.body_fn = body_fn
        
        
    
    def jit_call(self):
        self._call_fori_loop = jax.jit(self._call_fori_loop)
        