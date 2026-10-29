import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class VMIFGSM(Attack):
    r"""
    VMI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps = 8/255, alpha = 2/255, decay=1.0, steps=10, N=5, beta=3/2, main_rng_seed=42):
        super().__init__("VMIFGSM")
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.steps = steps
        self.N = N
        self.beta = beta
        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.supported_modes = ["default", "targeted"]

        self.create_functions()
        
    def main_attack_function(self, images, labels):
    
        adv_images, self.main_rng =  self._call_fori_loop(images, labels, self.targeted, self.eps, self.alpha, self.decay, self.steps, self.N, self.beta, self.main_rng)
        
        return adv_images

    def _call_fori_loop(self, images, labels, targeted, eps, alpha, decay, steps, N, beta, main_rng):
        
        main_rng, normal_rng = jax.random.split(main_rng, num=2)

        ori_images = images
        adv_images = jnp.copy(images)
        
        momentum = jnp.zeros_like(images)
        v = jnp.zeros_like(images)
        
        x = {
            "ori_images": ori_images, 
            "adv_images": adv_images,
            "labels": labels,
            "targeted": targeted,
            "eps": eps,
            "alpha": alpha,
            "decay": decay,
            "beta": beta,
            "momentum": momentum,
            "v": v,
            "normal_rng": normal_rng,
        }
        
        
        x = jax.lax.fori_loop(0, steps, self.body_fn, x)
        
        adv_images = jnp.clip(x["adv_images"], 0.0, 1.0)
        return adv_images, main_rng
        
                
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
            # Grad calculation
            loss_to_print, grads = self.grad_fn(x["adv_images"], x["labels"])
            
            adv_grads = jax.lax.cond(x["targeted"],
                                 lambda _: -1*grads,
                                 lambda _: grads,
                                 operand=None)
            
            # Momentum adjustments
            grads = (adv_grads + x["v"]) / jnp.mean(jnp.abs(adv_grads + x["v"]), axis=(1,2,3), keepdims=True)
            grads = grads + x["momentum"] * x["decay"]
            x["momentum"] = grads

            ### Calculate gradient variance

            #varince_dict
            GV_grad = jnp.zeros_like(x["adv_images"])
            v_dict = {
                "adv_images": x["adv_images"],
                "GV_grad": GV_grad,
                "labels": x["labels"],
                "targeted": x["targeted"],
                "eps": x["eps"],
                "beta": x["beta"],
                "normal_rng": x["normal_rng"],
            }

            v_dict = jax.lax.fori_loop(0, self.N, self.variance_calculation, v_dict)
            x["v"] = v_dict["GV_grad"] / self.N - adv_grads

            ### End of Calculate gradient variance

                         
            # Image Update
            adv_images = x["adv_images"] + x["alpha"] * jnp.sign(grads)
            delta = jnp.clip(adv_images - x["ori_images"], -x["eps"], x["eps"])
            x["adv_images"] = jnp.clip(x["ori_images"] + delta, 0.0, 1.0)
            
            return x
            
        self.body_fn = body_fn

        def variance_calculation(_, v_dict):
            v_dict["normal_rng"], cur_uniform_rng = jax.random.split(v_dict["normal_rng"], num=2)
            neighbor_images = v_dict["adv_images"] + jax.random.uniform(cur_uniform_rng, v_dict["adv_images"].shape, dtype=jnp.float32, minval= -v_dict["eps"] * v_dict["beta"], maxval= v_dict["eps"] * v_dict["beta"])
            
            _, neighbor_grad = self.grad_fn(neighbor_images, v_dict["labels"])
            neighbor_grad = jax.lax.cond(v_dict["targeted"],
                                            lambda _: -1*neighbor_grad,
                                            lambda _: neighbor_grad,
                                            operand=None)

            v_dict["GV_grad"] = v_dict["GV_grad"] + neighbor_grad

            return v_dict
        self.variance_calculation = variance_calculation


    def jit_call(self):
      self._call_fori_loop = jax.jit(self._call_fori_loop)
      