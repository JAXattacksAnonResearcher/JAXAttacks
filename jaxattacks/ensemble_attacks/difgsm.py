import jax
import jax.numpy as jnp
import optax
import jax.image
from functools import partial

from ..ensemble_attack import EnsembleAttack

class DIFGSMEnsemble(EnsembleAttack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 0.0)
        steps (int): number of iterations. (Default: 10)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 8/255, alpha = 2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False, main_rng_seed=42):
        super().__init__("DIFGSMEnsemble")
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.supported_modes = ["default", "targeted"]

        self.create_functions()
        
    def main_attack_function(self, images, labels):
    
        adv_images, self.main_rng = self._call_fori_loop(images, labels, self.targeted, self.eps, self.steps, self.decay, self.alpha, self.resize_rate, self.diversity_prob, self.random_start, self.main_rng)
        
        return adv_images
        
    def _call_fori_loop(self, images, labels, targeted, eps, steps, decay, alpha, resize_rate, diversity_prob, random_start, main_rng):
        
        ori_images = images
        adv_images = jnp.copy(images)
        
        rngs = jax.random.split(main_rng, num=2 + 4 * steps)  
        main_rng = rngs[0]
        uniform_rng = rngs[1]
        
        adv_images = jax.lax.cond(random_start,
                                  lambda x: adv_images + jax.random.uniform(uniform_rng, adv_images.shape, dtype=jnp.float32),
                                  lambda x: adv_images,
                                  operand=None)
                                  
        momentum = jnp.zeros_like(images)
        
        
        x = {
            "ori_images": ori_images, 
            "adv_images": adv_images,
            "labels": labels,
            "targeted": targeted,
            "eps": eps,
            "alpha": alpha,
            "decay": decay,
            "momentum": momentum,
        }
        
        
        for i in range(steps):
            #for_loop_rng, rnd_randint_rng, pad_top_randint_rng, pad_left_randint_rng, output_uniform_rng = jax.random.split(for_loop_rng, num=5)
            rnd_randint_rng = rngs[2 + i*steps]
            pad_top_randint_rng = rngs[3 + i*steps]
            pad_left_randint_rng = rngs[4 + i*steps]
            output_uniform_rng = rngs[5 + i*steps]
            
            x["adv_images"] = self.input_diversity(x["adv_images"], resize_rate, rnd_randint_rng, pad_top_randint_rng, pad_left_randint_rng, output_uniform_rng, diversity_prob)
            x = self.body_fn(x)
        
        
        del rngs
        
        adv_images = jnp.clip(x["adv_images"], 0.0, 1.0)
        return adv_images, main_rng
        
                
    def create_functions(self):
        ## Loss Function
        def calculate_loss(imgs, lbls):
            logits = self.get_all_logits(imgs)
            _, all_losts = jax.lax.scan(lambda _, x : (_, optax.softmax_cross_entropy_with_integer_labels(x, lbls).mean()),
                                        0, # dummy init for carry
                                        logits)
            total_lost = jnp.mean(all_losts)
            return total_lost
        
        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
        
        ## body_fn for jax.lax.fori_loop
        def body_fn(x):
            # Grad calculation
            loss_to_print, grads = self.grad_fn(x["adv_images"], x["labels"])
            
            grads = jax.lax.cond(x["targeted"],
                                 lambda _: -1*grads,
                                 lambda _: grads,
                                 operand=None)
            
            # Momentum adjustments
            grads = grads / jnp.mean(jnp.abs(grads), axis=(1,2,3), keepdims=True)
            grads = grads + x["momentum"] * x["decay"]
            x["momentum"] = grads
                         
            # Image Update
            adv_images = x["adv_images"] + x["alpha"] * jnp.sign(grads)
            delta = jnp.clip(adv_images - x["ori_images"], -x["eps"], x["eps"])
            x["adv_images"] = jnp.clip(x["ori_images"] + delta, 0.0, 1.0)
            
            return x
        
        self.body_fn = body_fn

     
        def input_diversity(x, resize_rate, rnd_randint_rng, pad_top_randint_rng, pad_left_randint_rng, output_uniform_rng, diversity_prob):
            B, H, W, C = x.shape
            img_size = H
            img_resize = jnp.int64(jnp.floor(img_size * resize_rate))
            
            img_size, img_resize = jax.lax.cond(resize_rate < 1,
                                                lambda _: (img_resize, img_size),
                                                lambda _: (img_size, img_resize),
                                                operand=None)
            
            rnd = jax.random.randint(rnd_randint_rng, (1,), img_size, img_resize, dtype=jnp.int32)
            
            resized_x = jax.image.resize(x, (B, rnd[0].item(), rnd[0].item(), C), method='bilinear')
        
            h_rem = img_resize - rnd
            w_rem = img_resize - rnd
            
            pad_top = jax.random.randint(pad_top_randint_rng, (1, ), 0, h_rem, dtype=jnp.int32)
            pad_bottom = h_rem - pad_top
            pad_left= jax.random.randint(pad_left_randint_rng, (1, ), 0, w_rem, dtype=jnp.int32)
            pad_right = h_rem - pad_left
            
            padded = jnp.pad(resized_x, 
                             ((0,0), (pad_top[0].item(), pad_bottom[0].item()), (pad_left[0].item(), pad_right[0].item()), (0,0)),
                             'constant') # pad with 0
                             
            result = jax.lax.cond(jax.random.uniform(output_uniform_rng, ()) < diversity_prob,
                                  lambda _: padded,
                                  lambda _: x,
                                  operand=None)
            
            del resized_x
            del padded
            del x
                                  
            return result
        
        self.input_diversity = input_diversity
            
        
    def jit_call(self):
      #self.body_fn = jax.jit(self.body_fn, donate_argnums=(0, 4))
      self.body_fn = jax.jit(self.body_fn)
      