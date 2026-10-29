import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 8/255, alpha = 2/255, steps=10):
        super().__init__("BIM")
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_modes = ["default", "targeted"]

        self.create_functions()

    def main_attack_function(self, images, labels):

        return self._call_fori_loop(images, labels, self.targeted, self.eps, self.alpha, self.steps)
        
    
    def _call_fori_loop(self, images, labels, targeted, eps, alpha, steps):
        # Now only for targeted            
        ori_images = images
        inter_images = jnp.copy(images)
        adv_images = jnp.copy(images)
        
        x = {"ori_images": ori_images, 
             "inter_images": inter_images,
             "labels": labels,
             "targeted": targeted,
             "eps": eps,
             "alpha": alpha,
        }
        
        x = jax.lax.fori_loop(0, steps, self.body_fn, x)
        
        adv_images = jnp.clip(x["inter_images"], 0.0, 1.0)
        return adv_images
    

    def create_functions(self):   
        ## Loss Function    
        def calculate_loss(imgs, lbls):
            logits = self.get_logits(imgs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, lbls).mean()
            return loss
        
        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
        ## body_fn for jax.lax.fori_loop
        def body_fn(_, x): 
            # i is step_number
            # x is dictionary (targeted [boolean] , inter_images [jax.numpy.array])
            
            loss_to_print, grads = self.grad_fn(x["inter_images"], x["labels"])
        
            grads = jax.lax.cond(x["targeted"],
                                 lambda _: -1 * grads,
                                 lambda _: grads,
                                 operand=None)
            
            adv_images = x["inter_images"] + x["alpha"] * jnp.sign(grads)
            a = jnp.clip(x["ori_images"] - x["eps"], min=0.0)
            b = jnp.maximum(adv_images, a) 
            c = jnp.minimum(b, x["ori_images"] + x["eps"]) 
            x["inter_images"] = jnp.clip(c, max=1.0)
            
            return x
        
        self.body_fn = body_fn
        
    def jit_call(self):
        self._call_fori_loop = jax.jit(self._call_fori_loop)

    