import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class DeepFool(Attack):
    r"""
    DeepFool paper: 'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

    Arguments:
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, steps=50, overshoot=0.02):
        super().__init__("DeepFool")
        self.steps = steps
        self.overshoot = overshoot
        self.supported_modes = ["default"]

        self.create_functions()

    def main_attack_function(self, images, labels):
    
        return self._call(images, labels, self.steps, self.overshoot)
    
    def _call(self, images, labels, steps, overshoot):
    
        adv_images, target_labels = self.forward_return_target_labels(images, labels, steps, overshoot)
            
        
        return adv_images


    def create_functions(self): 
        ##
        def forward_return_target_labels(imgs, lbls, steps, overshoot):
            
            correct = jnp.full(imgs.shape[0], 1, dtype=bool)
            trgt_lbls = jnp.copy(lbls)
            
            adv_imgs = jnp.copy(imgs)
            
            wl_dict = {
                "ori_images": imgs,
                "labels": lbls,
                "target_labels": trgt_lbls,
                "steps": steps,
                "overshoot": overshoot,
                
                
                "correct": correct,
                "adv_images": jnp.asarray(adv_imgs, dtype=jnp.float32),
                "curr_step": 0,   
            }
            
            wl_dict = jax.lax.while_loop(self.wl_cond, self.while_loop, wl_dict)
            
            adv_imgs = jnp.clip(wl_dict["adv_images"], 0.0, 1.0)
            return adv_imgs, wl_dict["target_labels"]
            
        self.forward_return_target_labels = forward_return_target_labels
        
        #
        def while_loop(wl_dict):
        
            adv_images, target_labels, pred = self._forward(wl_dict["adv_images"], wl_dict["labels"], wl_dict["overshoot"])
            
            
            
            wl_dict["adv_images"] = jnp.where((pred != wl_dict["labels"])[:, None, None, None], wl_dict["adv_images"], adv_images)
            wl_dict["adv_images"] = jnp.asarray(wl_dict["adv_images"], dtype=jnp.float32)
            wl_dict["target_labels"] = jnp.where((pred != wl_dict["labels"]),  wl_dict["target_labels"], target_labels)
            
            
            wl_dict["curr_step"] = wl_dict["curr_step"] + 1
            wl_dict["correct"] = wl_dict["labels"] == pred
            
            
            return wl_dict
            
        
            
        def wl_cond(cond_dict):
            index_stop_cond = cond_dict["curr_step"] < cond_dict["steps"]
            any_true_cond = jnp.any(cond_dict["correct"])
            
            
            return any_true_cond & index_stop_cond
        
        self.while_loop = while_loop
        self.wl_cond = wl_cond
        
        #
        def _forward(adv_imgs, lbls, overshoot):
            batch_size = adv_imgs.shape[0]
            u = jnp.arange(batch_size)
            
            def f_single(im_single):
                y_single = self.get_logits(im_single[None, ...])[0]
                return y_single
            
            jac_fn = jax.jacobian(f_single)


            ws = jax.vmap(jac_fn)(adv_imgs) # [batch_size, n_classes, H, W, C]
            fs = self.get_logits(adv_imgs) # [batch_size, dim]

            ws = jax.vmap(jac_fn)(adv_imgs) # 
            fs = self.get_logits(adv_imgs)

            
            pre = jnp.argmax(fs, axis=-1) # [batch_size, ]
            
            
            f_0 = fs[u, pre] # [batch_size, ]
            w_0 = ws[u, pre, :, :, :] # [batch_size, H, W, C]
            
            
            #Try out
            f_prime = fs - f_0[:, None] # [batch_size, n_classes]
            w_prime = ws - w_0[:, None, :, :, :] # [batch_size, n_classes, H, W, C]
            
            
            value = (jnp.abs(f_prime) # value.shape => [batch_size, dim]
                    / jnp.linalg.norm(jnp.reshape(w_prime, (batch_size, w_prime.shape[1], -1)), ord=2, axis=-1))
            
            value = value.at[u, pre].set(jnp.inf) # [batch_size, n_classes]

            
            hat_L = jnp.argmin(value, axis=-1) # [batch_size, ]

            
            delta = (
                jnp.abs(f_prime[u, hat_L])[:, None, None, None]
                * w_prime[u, hat_L]
                / (jnp.linalg.norm(jnp.reshape(w_prime[u, hat_L], (batch_size, -1)), ord=2, axis=-1) **2)[:, None, None, None]
            ) # [batch_size, H, W, C]
            
            target_label = jnp.where(hat_L < lbls, hat_L, hat_L + 1)
            
            result_images = adv_imgs + ( 1 + overshoot) * delta
            result_images = jnp.clip(result_images, 0.0, 1.0)
            
            return result_images, target_label, pre
            
        self._forward = _forward
        
        
    def jit_call(self):
        self._call = jax.jit(self._call)
        
        