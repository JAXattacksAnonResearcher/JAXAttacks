import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack

class SPSAEnsemble(EnsembleAttack):
    r"""
    SPSA in the paper 'Adversarial Risk and the Dangers of Evaluating Against Weak Attacks'
    [https://arxiv.org/abs/1802.05666]

    Distance Measure : Linf

    Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        delta (float): scaling parameter of SPSA. (Default: 0.01)
        lr (float): the learning rate of the `Adam` optimizer. (Default: 0.01)
        nb_iter (int): number of iterations of the attack. (Default: 1)
        nb_sample (int): number of samples for SPSA gradient approximation. (Default: 128)
        max_batch_size (int): maximum batch size to be evaluated at once. (Default: 64)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, eps=0.3, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64, main_rng_seed=42):
        super().__init__("SPSAEnsemble")
        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.max_batch_size = max_batch_size

        self.N_times = self.nb_sample // self.max_batch_size
        self.leftover_size = self.nb_sample % self.max_batch_size

        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)

        self.supported_modes = ["default", "targeted"]

        
        self.create_functions()

    def main_attack_function(self, images, labels):

        adv_images, self.main_rng =  self._call(images, labels, self.eps, self.delta, self.lr, self.nb_iter, self.main_rng)

        return adv_images

    def _call(self, images, labels, eps, delta, lr, nb_iter, main_rng):
        main_rng, normal_rng = jax.random.split(main_rng, num=2)

        dx = jnp.zeros_like(images)
        
        opt_state = self.optimizer.init(dx)

        x = {
            "images": images,
            "labels": labels,
            "eps": eps,
            "delta": delta,
            "normal_rng": normal_rng,

            "dx": dx,
            "opt_state": opt_state
        }

        x = jax.lax.fori_loop(0, nb_iter, self.spsa_perturb_body_fn, x)

        adv_images = images + x["dx"]
        adv_images = jnp.clip(adv_images, 0.0, 1.0)

        return adv_images, main_rng

    def create_functions(self):
        def calculate_loss(imgs, trgts):
            """
            imgs.shape => (max_batch_size*B, H, W, C)
            trgts.shape => (max_batch_size*B, 1)
            """
            logits = self.get_all_logits(imgs) #.shape => (model_count, max_batch_size*B, embedding_dim)
            
            # Choose top_2 class and corresponding logits for each model and for eacg image
            top_logits, top_classes = jax.lax.top_k(logits, 2)  #.shape => (model_count, max_batch_size*B, 2)
            
            target_logits = logits[:, jnp.arange(logits.shape[1]), trgts] #.shape => (model_count, max_batch_size*B, 1)

            max_nontarget_logits = jnp.where(
                top_classes[..., 0] == trgts, top_logits[..., 1], top_logits[..., 0]
            ) #.shape => (model_count, max_batch_size*B, 1)
 
            loss = max_nontarget_logits - target_logits #.shape => (model_count, max_batch_size*B, 1)
            # To reduce the loss, any other averaging method also suffice
            loss = jnp.mean(loss, axis=0) #.shape => (max_batch_size*B, 1)

            loss = jax.lax.cond(self.targeted,
                                lambda _: loss,
                                lambda _: -loss,
                                operand=None)

            return loss 

        self.calculate_loss = calculate_loss

        ## Optimizer 
        optimizer = optax.adam(learning_rate=self.lr)
        self.optimizer = optimizer

        ## CLAMP
        def linf_clamp(dx, x, eps):
            dx_clamped = jnp.clip(dx, -eps, eps)

            x_adv = jnp.clip(x + dx_clamped, 0, 1)

            dx += x_adv - x - dx

            return dx

        self.linf_clamp = linf_clamp


        ## Helper functions
        def spsa_perturb_body_fn(_, x):
            # Calculate grad with SPSA method
            dx_grad = self.spsa_grad(x["images"] + x["dx"], x["labels"], x["delta"], x["normal_rng"])

            # Optimize the 'dx'
            updates, x["opt_state"] = self.optimizer.update(dx_grad, x["opt_state"])
            x["dx"] = optax.apply_updates(x["dx"], updates)

            # Clip 'dx' to appropiate range
            x["dx"] = self.linf_clamp(x["dx"], x["images"], x["eps"])

            return x
        
        self.spsa_perturb_body_fn = spsa_perturb_body_fn

        def spsa_grad(imgs, lbls, delta, normal_rng):
            grad = jnp.zeros_like(imgs)

            images_expanded = jnp.tile(imgs, (self.max_batch_size, 1, 1, 1, 1)) #.shape => (max_batch_size, B, H, W, C)
            labels_expanded = jnp.tile(lbls, (self.max_batch_size, 1)) #.shape => (max_batch_size, B)

            x = {
                "images_expanded": images_expanded,
                "labels_expanded": labels_expanded,
                "delta": delta,
                "grad": grad,
                "normal_rng": normal_rng,

            }

            x = jax.lax.fori_loop(0, self.N_times, self.spsa_grad_body_fn, x)

            # Update 'grad' and 'normal_rng' to feed leftover
            grad = x["grad"] 
            normal_rng = x["normal_rng"]

            ## NEED TO HANDLE LEFTOVER 
            # leftover = lo
            grad = jax.lax.cond(self.leftover_size > 0,
                                lambda _: self.spsa_grad_body_fn(0, 
                                                                 {
                                                                    "images_expanded": jnp.tile(imgs, (self.leftover_size, 1, 1, 1, 1)),
                                                                    "labels_expanded": jnp.tile(lbls, (self.leftover_size, 1)),
                                                                    "delta": delta,
                                                                    "grad": grad,
                                                                    "normal_rng": normal_rng
                                                                 })["grad"],
                                lambda _: grad,
                                operand=None)
            ## End of left over

            grad /= self.nb_sample
            return grad
        
        self.spsa_grad = spsa_grad

        def spsa_grad_body_fn(_, x):
            x["normal_rng"], cur_rng = jax.random.split(x["normal_rng"], num=2)
            # Sample random directions 
            vb = jax.random.bernoulli(cur_rng, shape=x["images_expanded"].shape) * 2 - 1 #.shape => (max_batch_size, B, H, W, C)

            # Reshape 'images', 'labels' to calculate loss
            images_reshaped = jnp.reshape(x["images_expanded"], (-1, *x["images_expanded"].shape[2:])) #.shape => (max_batch_size * B, H, W, C)
            labels_reshaped = jnp.reshape(x["labels_expanded"], (-1, *x["labels_expanded"].shape[2:]))  #.shape => (max_batch_size * B, )
            v_reshaped = jnp.reshape(vb, (-1, *vb.shape[2:])) #.shape => (max_batch_size * B, H, W, C)
            
            # loss calculation
            df = self.calculate_loss(images_reshaped + x["delta"] * v_reshaped, labels_reshaped) - self.calculate_loss(images_reshaped - x["delta"] * v_reshaped, labels_reshaped)
            df = jnp.reshape(df, (-1, 1, 1, 1)) #.shape => (max_batch_size * B, H, W, C)
            
            # Calculate grad
            grad_ = df / (2.0 * x["delta"] * v_reshaped) #.shape => (max_batch_size * B, H, W, C)
            grad_ = jnp.reshape(grad_, x["images_expanded"].shape) #.shape => (max_batch_size, B, H, W, C)
            grad_ = jnp.sum(grad_, axis=0) #.shape => (B, H, W, C)

            x["grad"] = x["grad"] + grad_
            
            return x
        
        self.spsa_grad_body_fn = spsa_grad_body_fn

    def jit_call(self):
        self._call = jax.jit(self._call)