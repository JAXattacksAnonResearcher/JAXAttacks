import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack

class PGDL2Ensemble(EnsembleAttack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta (Default: True)
        main_rng_seed (int): random seed used for initializing the pseudo-random number generator. (Default: 42)
        eps_for_division (float): small constant added to the gradient norm to prevent division by zero during normalization. (Default: 1e-10)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 1.0, alpha = 0.2, steps=10, random_start = True, main_rng_seed = 42, eps_for_division=1e-10):
        super().__init__("PGDL2Ensemble")
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self. main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.eps_for_division = eps_for_division
        self.supported_modes = ["default", "targeted"]

        self.create_functions()
        
    def main_attack_function(self, images, labels):
        
        adv_images, self.main_rng = self._call_fori_loop(images, labels, self.targeted, self.eps, self.alpha, self.steps, self.random_start, self.main_rng, self.eps_for_division)
        
        return adv_images
        
    def _call_fori_loop(self, images, labels, targeted, eps, alpha, steps, random_start, main_rng, eps_for_division):
        
        ori_images = images
        adv_images = jnp.copy(images)
        
        main_rng, delta_normal_rng, r_uniform_rng = jax.random.split(main_rng, num=3)
        
        adv_images = jax.lax.cond(random_start,
                                  self.randomize_images, # + NORMAL
                                  self.not_randomize_images,
                                  operand=(adv_images, delta_normal_rng, r_uniform_rng, eps))
                                  
        x = {
            "ori_images" : ori_images,
            "adv_images" : adv_images,
            "labels": labels,
            "targeted": targeted,
            "eps" : eps,
            "alpha" : alpha,
            "eps_for_division": eps_for_division,
        }
        
        x = jax.lax.fori_loop(0, steps, self.body_fn, x)
        
        adv_images = jnp.clip(x["adv_images"], 0.0, 1.0)
        return adv_images, main_rng
        
        
    def create_functions(self):
        ## Loss function
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
                                 lambda _: -1 * grads, 
                                 lambda _: grads,
                                 operand=None)
                                 
            # grad normalization 
            grads_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=(1,2,3))) + x["eps_for_division"] ## jnp.linalg.vector_norm(grads, ord=2, axis=(1,2,3)) + x["eps_for_division"] # jax.__version__ == 0.5.2, Also our version seems to be faster
            grads = grads / grads_norm[:, None, None, None]
            
                    
            # image update
            adv_images = x["adv_images"] + x["alpha"] * grads
            
            delta = adv_images - x["ori_images"]
            delta_norm = jnp.sqrt(jnp.sum(delta ** 2, axis=(1,2,3))) ## jnp.linalg.vector_norm(delta, ord=2, axis=(1,2,3))
            
            factor = x["eps"] / delta_norm
            factor = jnp.minimum(factor, jnp.ones_like(delta_norm))
            
            delta = delta * factor[:, None, None, None]
            
            x["adv_images"] = jnp.clip(x["ori_images"] + delta, 0.0, 1.0)
            
            return x
            
        self.body_fn = body_fn
        
        ## Helper functions
        def randomize_images(args):
            adv_imgs, delta_n_rng, r_n_rng, eps = args
            
            delta_init = jax.random.normal(delta_n_rng, adv_imgs.shape, dtype=jnp.float32)
            
            n = jnp.sqrt(jnp.sum(delta_init ** 2, axis=(1,2,3))) ## jnp.linalg.vector_norm(delta_init, ord=2, axis=(1,2,3)) # jax.__version__ == 0.5.2, Also seems faster
            
            r_init = jax.random.uniform(r_n_rng, n.shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
            
            delta = delta_init * (r_init[:, None, None, None] / n[:, None, None, None] * eps)
            perturbed_images = jnp.clip(adv_imgs + delta, 0.0, 1.0)
            
            return perturbed_images
            
        self.randomize_images = randomize_images
            
        def not_randomize_images(args):
            adv_imgs, delta_n_rng, r_n_rng, eps = args
            
            return adv_imgs
            
        self.not_randomize_images = not_randomize_images
        
              
    def jit_call(self):
        self._call_fori_loop = jax.jit(self._call_fori_loop)
        
        