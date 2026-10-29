import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack
from ..utils import mode, mode_along_axis
mode_along_0 = mode_along_axis(0)

class SparseFoolEnsemble(EnsembleAttack):
    r"""
    Attack in the paper 'SparseFool: a few pixels make a big difference'
    [https://arxiv.org/abs/1811.02248]

    Distance Measure : L2

    Arguments:
        steps (int): number of steps. (Default: 10)
        lam (float): parameter for scaling DeepFool noise. (Default: 3)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, steps=3, lam=3, overshoot=0.02, deepfool_steps=50, deepfool_overshoot=0.02):
        super().__init__("SparseFoolEnsemble")
        self.steps = steps
        self.lam = lam
        self.overshoot = overshoot

        #self.deepfool = DeepFool_S(model)
        self.df_steps = deepfool_steps
        self.df_overshoot = deepfool_overshoot

        self.supported_modes = ["default"]

        self.create_functions()

        

    def main_attack_function(self, images, labels):
    
        return self._call(images, labels, self.steps, self.lam, self.overshoot, self.df_steps, self.df_overshoot)
    
    def _call(self, images, labels, steps, lam, overshoot, df_steps, df_overshoot):
    
        adv_images, target_labels = self.forward(images, labels, steps, lam, overshoot, df_steps, df_overshoot)
            
        return adv_images



    def create_functions(self): 
        ##
        def forward(imgs, lbls, steps, lam, overshoot, df_steps, df_overshoot):
            
            batch_size = imgs.shape[0]
            correct = jnp.full(imgs.shape[0], 1, dtype=bool)
            trgt_lbls = jnp.copy(lbls)

            adv_imgs = jnp.copy(imgs)

            wl_dict = {
                "ori_images": imgs,
                "labels": lbls,
                "target_labels": trgt_lbls,
                "steps": steps,
                "lam": lam,
                "overshoot": overshoot,
                "df_steps": df_steps,
                "df_overshoot": df_overshoot,

                "correct": correct,
                "adv_images": adv_imgs,
                "curr_step": 0,
            }


            wl_dict = jax.lax.while_loop(self.wl_cond, self.while_loop, wl_dict)

            adv_imgs = wl_dict["adv_images"]
            adv_imgs = jnp.clip(adv_imgs, 0.0, 1.0)

            return adv_imgs, wl_dict["target_labels"]

        self.forward = forward

        
        def while_loop(wl_dict):

            adv_images, wl_dict["target_labels"] = self.df_forward_return_target_labels(
                wl_dict["adv_images"], 
                wl_dict["labels"],
                wl_dict["df_steps"],
                wl_dict["df_overshoot"])            

            
            wl_dict["target_labels"] = wl_dict["target_labels"] - 1
            
            # Scaling deepfool perturbation
            adv_images = wl_dict["ori_images"] + wl_dict["lam"] * (adv_images - wl_dict["ori_images"])
            
            # Grad calculation
            (loss_to_print, pre) , grads = self.grad_fn(adv_images, wl_dict["labels"], wl_dict["target_labels"])
            grads = grads / jnp.linalg.norm(jnp.reshape(grads, (grads.shape[0], -1)), axis=-1)[:, None, None, None]


            # Updating adv_images with calculated grad
            adv_images = self._linear_solver(wl_dict["ori_images"], grads, adv_images, wl_dict["correct"])
            adv_images = wl_dict["ori_images"] + (1 + wl_dict["overshoot"]) * (adv_images - wl_dict["ori_images"])
            adv_images = jnp.clip(adv_images, 0.0, 1.0)            
            
            # Update the images that are still correctly labeled in a batch
            wl_dict["adv_images"] = jnp.where(wl_dict["correct"][:, None, None, None], adv_images, wl_dict["adv_images"])

            # Check which images still needs updates
            all_fs = self.get_all_logits(wl_dict["adv_images"]) # (batch_size, n_classes)
            all_pre = jnp.argmax(all_fs, axis=-1) # (batch_size, )
            pre = mode_along_0(all_pre)
            wl_dict["correct"] = wl_dict["labels"] == pre
            wl_dict["curr_step"] = wl_dict["curr_step"] + 1

            return wl_dict

        def wl_cond(cond_dict):
            index_stop_cond = cond_dict["curr_step"] < cond_dict["steps"]
            any_true_cond = jnp.any(cond_dict["correct"])
            
            return any_true_cond & index_stop_cond

        self.while_loop = while_loop
        self.wl_cond = wl_cond

        def loss_func(adv_imgs, lbls, trgt_lbls):
            """
            fs = self.get_logits(prms, adv_imgs) # (batch_size, n_classes)
            pre = jnp.argmax(fs, axis=-1) # (batch_size, )

            pre = jnp.where(pre == lbls, trgt_lbls, pre) # (batch_size, )
            u = jnp.arange(pre.shape[0]) # (batch_size, )

            cost = fs[u, pre] - fs[u, lbls] # (batch_size, )
            cost = jnp.sum(cost) # (1, )

            return (cost, pre)
            """

            all_fs = self.get_all_logits(adv_imgs) # (model_count, batch_size, n_classes)
            all_pre = jnp.argmax(all_fs, axis=-1) # (model_count, batch_size)

            all_pre = jnp.where(all_pre == lbls[None, :], trgt_lbls, all_pre) # (model_count, batch_size)

            #
            model_count = all_fs.shape[0]
            batch_size = all_fs.shape[1]
            n_classes = all_fs.shape[2]
            #
            m_idx = jnp.arange(model_count)[:, None]
            b_idx = jnp.arange(batch_size)[None, :]
            u = jnp.arange(batch_size) # (batch_size, )

            cost = all_fs[m_idx, b_idx, all_pre] - all_fs[:, u, lbls] # (model_count, batch_size)
            cost = jnp.sum(cost)

            return (cost, mode_along_0(all_pre))




        self.loss_func = loss_func
        self.grad_fn = jax.value_and_grad(self.loss_func, has_aux=True)

        def _linear_solver(x_0, coord_vec, boundary_point, dont_care_flag):
            batch_size = x_0.shape[0]

            plane_normal = jnp.reshape(coord_vec, (batch_size, -1))
            plane_point = jnp.reshape(boundary_point, (batch_size, -1))

            x_i = jnp.copy(x_0)

            f_k = jnp.sum(plane_normal * (jnp.reshape(x_0, (batch_size, -1)) - plane_point), -1) #  jnp.dot(plane_normal, jnp.reshape(x_0, (batch_size, -1)) - plane_point)
            sign_true = jnp.sign(f_k)

            beta = 0.001 * sign_true
            current_sign = sign_true

            wl_dict = {
                "plane_normal": plane_normal, # (batch_size, H*W*C)
                "plane_point": plane_point, # (batch_size, H*W*C)
                "f_k": f_k, # (batch_size, )
                "beta": beta, #  (batch_size, )
                "coord_vec": coord_vec, # (batch_size, H, W, C)
                "x_i": x_i, # (batch_size, H, W, C)
                "batch_size": batch_size,
                "current_sign": current_sign, # (batch_size, )
                "sign_true": sign_true,
                "dont_care_flag": dont_care_flag,
            }

            wl_dict = jax.lax.while_loop(self.ln_wl_cond, self.ln_while_loop, wl_dict)
            
            x_i = wl_dict["x_i"]

            return x_i

        self._linear_solver = _linear_solver
        
        
        def ln_while_loop(wl_dict):
            batch_size = wl_dict["x_i"].shape[0]
            
            wl_dict["f_k"] = jnp.sum(wl_dict["plane_normal"] * ( jnp.reshape(wl_dict["x_i"], (batch_size, -1)) - wl_dict["plane_point"])  , axis=-1) + wl_dict["beta"]

            pert = (
                    jnp.abs(wl_dict["f_k"]) 
                    / jnp.max(
                        jnp.reshape(
                            jnp.abs(wl_dict["coord_vec"]), (batch_size, -1)
                            ), axis = -1
                        )
                    )

            ## Creating a mask to select the most effective pixels in the image
            mask = jnp.zeros_like(wl_dict["coord_vec"])

            # Build per-sample mask: one hot at the largest-abs entry for each sample
            abs_flat = jnp.reshape(jnp.abs(wl_dict["coord_vec"]), (batch_size, -1))   # (batch, N)
            max_idx = jnp.argmax(abs_flat, axis=-1)                    # (batch,)
            mask_flat = jax.nn.one_hot(max_idx, abs_flat.shape[1])     # (batch, N)
            mask = jnp.reshape(mask_flat, wl_dict["coord_vec"].shape).astype(wl_dict["coord_vec"].dtype)  # (batch, H, W, C)

            # Update
            r_i = jnp.clip(pert, min=1e-4)[:, None, None, None] * mask * jnp.sign(wl_dict["coord_vec"]) 

            
            # Which images still needs update in a batch
            keep_update_flag = (
                jnp.logical_and(wl_dict["current_sign"] == wl_dict["sign_true"], wl_dict["dont_care_flag"]) &
                jnp.logical_and(jnp.any(wl_dict["coord_vec"] != 0, axis=(1, 2, 3)), wl_dict["dont_care_flag"])
            )

            x_i = wl_dict["x_i"] + r_i
            x_i = jnp.clip(x_i, 0.0, 1.0)

            # Updating the images that still requires update
            wl_dict["x_i"] = jnp.where(keep_update_flag[:, None, None, None], x_i, wl_dict["x_i"])


            wl_dict["f_k"] = jnp.sum(wl_dict["plane_normal"] * ( jnp.reshape(wl_dict["x_i"], (batch_size, -1)) - wl_dict["plane_point"])  , axis=-1)

            # Updating stop conditions for 'while loop'
            wl_dict["current_sign"] = jnp.sign(wl_dict["f_k"])
            wl_dict["coord_vec"] = jnp.where(r_i != 0, 0, wl_dict["coord_vec"])

            return wl_dict

        def ln_wl_cond(cond_dict):
            sign_flag = jnp.any(jnp.logical_and(cond_dict["current_sign"] == cond_dict["sign_true"], cond_dict["dont_care_flag"]))
            vec_flag = jnp.any(jnp.logical_and(jnp.any(cond_dict["coord_vec"] != 0, axis=(1, 2, 3)), cond_dict["dont_care_flag"])) 
    
            return sign_flag & vec_flag

        self.ln_while_loop = ln_while_loop
        self.ln_wl_cond = ln_wl_cond

        ##DEEPFOOL
        ##
        def df_forward_return_target_labels(imgs, lbls, steps, overshoot):
            
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
                "adv_images": adv_imgs,
                "curr_step": 0,   
            }
            
            wl_dict = jax.lax.while_loop(self.df_wl_cond, self.df_while_loop, wl_dict)
            
            adv_imgs = jnp.clip(wl_dict["adv_images"], 0.0, 1.0)
            return adv_imgs, wl_dict["target_labels"]
            
        self.df_forward_return_target_labels = df_forward_return_target_labels
        
        #
        def df_while_loop(wl_dict):
        
            adv_images, target_labels, pred = self._df_forward(wl_dict["adv_images"], wl_dict["labels"], wl_dict["overshoot"])
            
            
            
            wl_dict["adv_images"] = jnp.where((pred != wl_dict["labels"])[:, None, None, None], wl_dict["adv_images"], adv_images)
            wl_dict["target_labels"] = jnp.where((pred != wl_dict["labels"]),  wl_dict["target_labels"], target_labels)
            
            
            wl_dict["curr_step"] = wl_dict["curr_step"] + 1
            wl_dict["correct"] = wl_dict["labels"] == pred
            
            
            return wl_dict
            
        
            
        def df_wl_cond(cond_dict):
            index_stop_cond = cond_dict["curr_step"] < cond_dict["steps"]
            any_true_cond = jnp.any(cond_dict["correct"])
            
            
            return any_true_cond & index_stop_cond
        
        self.df_while_loop = df_while_loop
        self.df_wl_cond = df_wl_cond
        
        #
        def _df_forward(adv_imgs, lbls, overshoot):
            
            def f_single(im_single):
                # Output: [M, D]
                return self.get_all_logits(im_single[None, ...])[:, 0, :]
            jac_fn = jax.jacobian(f_single)


            all_ws = jnp.transpose(jax.vmap(jac_fn)(adv_imgs), (1, 0, 2, 3, 4, 5)) # [model_count, batch_size, n_classes, H, W, C]
            all_fs = self.get_all_logits(adv_imgs) # [model_count, batch_size, n_classes]

            
            all_pre = jnp.argmax(all_fs, axis=-1) # [model_count, batch_size]
            
            
            # Select the predicted logit for each model, each image 
            #'u = jnp.arange(batch_size)' in vanilla version
            model_count = all_fs.shape[0]
            batch_size = all_fs.shape[1]
            n_classes = all_fs.shape[2]
            # m_idx.shape = (model_count, 1)
            # b_idx.shape = (1, batch_size)
            m_idx = jnp.arange(model_count)[:, None]  
            b_idx = jnp.arange(batch_size)[None, :]   


            all_f_0 = all_fs[m_idx, b_idx, all_pre] # (model_count, batch_size)
            all_w0 = all_ws[m_idx, b_idx, all_pre, :, :, :]  # (model_count, batch_size, H, W, C)

            
            
            #Try out
            f_prime = all_fs - all_f_0[:, : , None] # (model_count, batch_size, n_classes)
            w_prime = all_ws - all_w0[:, :, None, :, :, :] # [model_count, batch_size, num_classes, H, W, C]
            
            
            all_value = (jnp.abs(f_prime) # all_value.shape => (model_count, batch_size, n_classes)
                        / jnp.linalg.norm(jnp.reshape(w_prime, (model_count, batch_size, n_classes, -1)), ord=2, axis=-1))
            
            all_value = all_value.at[m_idx, b_idx, all_pre].set(jnp.inf)

            
            all_hat_L = jnp.argmin(all_value, axis=-1) # (model_count, batch_size)
            
            all_delta = (
                jnp.abs(f_prime[m_idx, b_idx, all_hat_L])[:, :, None, None, None]
                * w_prime[m_idx, b_idx, all_hat_L]
                / (jnp.linalg.norm(jnp.reshape(w_prime[m_idx, b_idx, all_hat_L], (model_count, batch_size, -1)), ord=2, axis=-1) **2)[:, :, None, None, None]
            ) # (model_count, batch_size, H, W, C)

            delta = jnp.mean(all_delta, axis=0)
            
            result_images = adv_imgs + ( 1 + overshoot) * delta
            result_images = jnp.clip(result_images, 0.0, 1.0)


            pre = mode_along_0(all_pre)
            hat_L = mode_along_0(all_hat_L)
            target_label = jnp.where(hat_L < lbls, hat_L, hat_L + 1)
            
            return result_images, target_label, pre
            
        self._df_forward = _df_forward


    def jit_call(self):
        self._call = jax.jit(self._call)

