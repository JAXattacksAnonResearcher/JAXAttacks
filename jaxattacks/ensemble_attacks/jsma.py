import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack
from ..utils import mode, mode_along_axis
mode_along_0 = mode_along_axis(0)

class JSMAEnsemble(EnsembleAttack):
    r"""
    Jacobian Saliency Map Attack in the paper 'The Limitations of Deep Learning in Adversarial Settings'
    [https://arxiv.org/abs/1511.07528v1]

    Distance Measure : L0

    Arguments:
        theta (float): perturb length, range is either [theta, 0], [0, theta]. (Default: 1.0)
        gamma (float): highest percentage of pixels can be modified. (Default: 0.1)


    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, theta=1.0, gamma=0.1):
        super().__init__("JSMAEnsemble")
        self.theta = theta
        self.gamma = gamma
        self.supported_modes = ["default", "targeted"]

        self.create_functions()

    def main_attack_function(self, images, labels):

        return self._call(images, labels, self.targeted, self.theta, self.gamma)

    def _call(self, images, labels, targeted, theta, gamma):
        
        ori_images = jnp.copy(images)

        ### Perturbation single
        increasing = jax.lax.cond(theta > 0,
                                lambda _: True,
                                lambda _: False,
                                operand=None)
        
        B, H, W, C = ori_images.shape
        num_features = H * W * C

        max_iters = H * W * C * gamma // 2.0 + 1 # 32 * 32 * 3 * 0.1 // 2.0 + 1

        search_domain = jax.lax.cond(increasing,
                                    lambda _: ori_images < 0.99,
                                    lambda _: ori_images > 0.01,
                                    operand=None)

        search_domain = jnp.reshape(search_domain, (-1, num_features))
        all_outputs = self.get_all_logits(ori_images) # (model_count, batch_size, num_classes)
        all_current_pred = jnp.argmax(all_outputs, -1) # (model_count, batch_size, )
        current_pred = mode_along_0(all_current_pred)  # (batch_size, )

        n_classes = all_outputs.shape[-1]
        target_labels = jax.lax.cond(targeted,
                                    lambda _: labels,
                                    lambda _: (labels + 1) %n_classes,
                                    operand=None)

        iter_i = 0
        # while loop
        wl_dict = {
            "var_images": ori_images, # (batch_size, H, W, C)
            "var_labels": target_labels, # (batch_size, )
            "search_domain": search_domain, # (batch_size, H*W*C)
            "increasing": increasing,
            "num_features": num_features,

            "theta": theta,
            "current_pred": current_pred, # (batch_size, )
            "iter_i": iter_i,
            "max_iters": max_iters,
            "target_labels": target_labels,
            
        }
        
        wl_dict = jax.lax.while_loop(self.wl_cond, self.while_loop, wl_dict)
        
        adv_images = jnp.clip(wl_dict["var_images"], 0.0, 1.0)
        return adv_images



    def create_functions(self):
        def while_loop(wl_dict):
            B, H, W, C = wl_dict["var_images"].shape
            nb_features = wl_dict["search_domain"].shape[1]

            def f_single(im_single):
                # Output: [M, D]
                return self.get_all_logits(im_single[None, ...])[:, 0, :]
            jac_fn = jax.jacobian(f_single)

            jacobian_imgs = jnp.transpose(jax.vmap(jac_fn)(wl_dict["var_images"]), (1, 0, 2, 3, 4, 5)) # (model_count, batch_size, n_classes, H, W, C)
            jacobian_imgs = jnp.reshape(jacobian_imgs, (jacobian_imgs.shape[0], jacobian_imgs.shape[1], jacobian_imgs.shape[2], -1)) # (model_count, batch_size, n_classes, H*W*C)
            
            
            p1, p2 = self.saliency_map(jacobian_imgs, wl_dict["var_labels"], wl_dict["increasing"], wl_dict["search_domain"])

            u = jnp.arange(B)

            # Apply modifications
            var_sample_flatten = jnp.reshape(wl_dict["var_images"], (-1, nb_features))
            var_sample_flatten = var_sample_flatten.at[u, p1].add(wl_dict["theta"])
            var_sample_flatten = var_sample_flatten.at[u, p2].add(wl_dict["theta"])

            new_images = jnp.clip(var_sample_flatten, 0.0, 1.0)
            new_images = jnp.reshape(new_images, (B, H, W, C))
            
            # Update the images still requires 
            wl_dict["var_images"] = jnp.where(
                                    jnp.logical_and(
                                        wl_dict["current_pred"] != wl_dict["target_labels"], 
                                        jnp.any(jnp.sum(wl_dict["search_domain"], -1) != 0)
                                    )[:, None, None, None], 
                                    new_images,
                                    wl_dict["var_images"])

            wl_dict["search_domain"] = wl_dict["search_domain"].at[u, p1].set(0)
            wl_dict["search_domain"] = wl_dict["search_domain"].at[u, p2].set(0)
            
            
            all_outputs = self.get_all_logits(wl_dict["var_images"])
            all_current_pred = jnp.argmax(all_outputs, -1) 
            wl_dict["current_pred"] = mode_along_0(all_current_pred)


            wl_dict["iter_i"] = wl_dict["iter_i"] + 1
            return wl_dict

        def wl_cond(cond_dict):
            iter_cond = cond_dict["iter_i"] < cond_dict["max_iters"]
            pred_cond = jnp.any(cond_dict["current_pred"] != cond_dict["target_labels"])
            domain_cond = jnp.any(jnp.sum(cond_dict["search_domain"], -1) != 0)


            return iter_cond & pred_cond & domain_cond

        self.while_loop = while_loop
        self.wl_cond = wl_cond

        # Saliency map
        def saliency_map(jacob, trgt_lbl, inc, search_space):
            """
            Inputs: 
                jacob.shape =>  (model_count, batch_size, n_classes, H*W*C)
                trgt_lbl => (batch_size, )
                inc : boolean
                search_space => (batch_size, H*W*C)
            """
            model_count = jacob.shape[0]
            batch_size = jacob.shape[1]
            n_classes = jacob.shape[2]
            nb_features = search_space.shape[1]
            
            # saliency map
            domain = search_space == 1 # (batch_size, H*W*C)

            all_sum = jnp.sum(jacob, axis=2) # (model_count, batch_size, H*W*C)

            # Select the predicted logit for each model, each image 
            #'u = jnp.arange(batch_size)' in vanilla version
            # m_idx.shape = (model_count, 1)
            # b_idx.shape = (1, batch_size)
            m_idx = jnp.arange(model_count)[:, None]  
            b_idx = jnp.arange(batch_size)[None, :]  
            trgt_lbl = jnp.broadcast_to(trgt_lbl[None, :], (model_count, batch_size))

            target_grad = jacob[m_idx, b_idx, trgt_lbl, :]  # (model_count, batch_size, H*W*C)

            others_grad = all_sum - target_grad # (model_count, batch_size, H*W*C)

            increase_coef = jax.lax.cond(inc,
                                        lambda _: 2 * (domain == 0),
                                        lambda _: -1 * 2 * (domain == 0),
                                        operand=None)
            increase_coef = jnp.reshape(increase_coef, (batch_size, -1)) # (batch_size, H*W*C)

            
            target_tmp = jnp.copy(target_grad) # (model_count, batch_size, H*W*C) 
            target_tmp -= increase_coef[None, :, :] * jnp.max(jnp.abs(target_grad), axis=-1)[:, :, None] # (model_count, batch_size, H*W*C) 
            alpha = (jnp.reshape(target_tmp, (model_count, batch_size, 1, nb_features)) 
                    +jnp.reshape(target_tmp, (model_count, batch_size, nb_features, 1)) 
            ) # (model_count, batch_size, H*W*C, H*W*C)
            
            
            others_tmp = jnp.copy(others_grad) # (model_count, batch_size, H*W*C)
            others_tmp += increase_coef[None, :, :] * jnp.max(jnp.abs(others_grad), axis=-1)[:, :, None] # (model_count, batch_size, H*W*C)
            beta = (jnp.reshape(others_tmp, (model_count, batch_size, 1, nb_features)) 
                    +jnp.reshape(others_tmp, (model_count, batch_size, nb_features, 1)) 
            ) # (model_count, batch_size, H*W*C, H*W*C)
            
            tmp = jnp.identity(nb_features, dtype=jnp.int32) # (nb_features, nb_features)
            zero_diagonal = 1 - tmp

            mask1, mask2 = jax.lax.cond(inc,
                                        lambda _: (alpha > 0.0, beta < 0.0),
                                        lambda _: (alpha < 0.0, beta > 0.0),
                                        operand=None) # (model_count, batch_size, H*W*C, H*W*C)
            
            
            mask = (mask1 * mask2) * zero_diagonal[None, None, :, :] # (model_count, batch_size, H*W*C, H*W*C)

            all_saliency_map = (alpha * jnp.abs(beta)) * mask # (model_count, batch_size, H*W*C, H*W*C)
            saliency_map = jnp.sum(all_saliency_map, axis=0) # (batch_size, H*W*C, H*W*C)

            max_idx = jnp.argmax(jnp.reshape(saliency_map, (-1, nb_features * nb_features)), axis=-1)
            
            p = max_idx // nb_features
            q = max_idx - p * nb_features
            return p, q
            
        self.saliency_map = saliency_map


    def jit_call(self):
        self._call = jax.jit(self._call)