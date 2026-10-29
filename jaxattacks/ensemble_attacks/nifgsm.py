import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack

class NIFGSMEnsemble(EnsembleAttack):
    r"""
    NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 8/255, alpha = 2/255, decay=1.0, steps=10):
        super().__init__("NIFGSMEnsemble")
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.steps = steps
        self.supported_modes = ["default", "targeted"]

        self.create_functions()
        
    def main_attack_function(self, images, labels):
    
        return self._call_fori_loop(images, labels, self.targeted, self.eps, self.alpha, self.decay, self.steps)
        
    def _call_fori_loop(self, images, labels, targeted, eps, alpha, decay, steps):
        
        ori_images = images
        adv_images = jnp.copy(images)
        
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
        
        
        x = jax.lax.fori_loop(0, steps, self.body_fn, x)
        
        adv_images = jnp.clip(x["adv_images"], 0.0, 1.0)
        return adv_images
        
                
    def create_functions(self):
        ## Loss Function
        def calculate_loss(imgs, lbls, decay, alpha, momentum):
            logits = self.get_all_logits(imgs + decay * alpha * momentum)
            _, all_losses = jax.lax.scan(lambda _, x : (_, optax.softmax_cross_entropy_with_integer_labels(x, lbls).mean()),
                                         0, # dummy init for carry
                                         logits)
            total_loss = jnp.mean(all_losses)
            return total_loss
        
        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)
        
        
        ## body_fn for jax.lax.fori_loop
        def body_fn(_, x):
            # Grad calculation
            loss_to_print, grads = self.grad_fn(x["adv_images"], x["labels"], x["decay"], x["alpha"], x["momentum"]) 
            
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
        
    def jit_call(self):
      self._call_fori_loop = jax.jit(self._call_fori_loop)
      
