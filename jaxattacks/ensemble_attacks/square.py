import jax
import jax.numpy as jnp
import optax
import math

from ..ensemble_attack import EnsembleAttack
from ..utils import mode_along_axis
mode_along_0 = mode_along_axis(0)

class SquareEnsemble(EnsembleAttack):
    r"""
    Square Attack in the paper 'Square Attack: a query-efficient black-box adversarial attack via random search'
    [https://arxiv.org/abs/1912.00049]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        n_queries (int): max number of queries (each restart). (Default: 5000)
        n_restarts (int): number of random restarts. (Default: 1)
        p_init (float): parameter to control size of squares. (Default: 0.8)
        loss (str): loss function optimized ['margin', 'ce'] (Default: 'margin')
        resc_schedule (bool): adapt schedule of p to n_queries (Default: True)
        seed (int): random seed for the starting point. (Default: 0)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self,  
                 norm="Linf", 
                 eps = 8/255, 
                 S = 32,
                 n_queries = 100,
                 n_restarts=1,
                 p_init = 0.8,
                 loss="margin",
                 resc_schedule=True,
                 main_rng_seed=42,
                 targeted=False
                 ):
        super().__init__("SquareEnsemble")
        self.norm = norm
        self.eps = eps 
        self.S = S
        self.n_queries = n_queries
        self.n_restarts = n_restarts
        self.p_init = p_init
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.supported_modes = ["default", "targeted"]
        self.targeted = targeted

        self.create_functions()

        self.eta_patch = self.eta(self.S)

    def main_attack_function(self, images, labels):
        
        adv_images, self.main_rng = self._call(images, labels, self.targeted, self.eps, self.n_queries, self.n_restarts, self.p_init, self.rescale_schedule, self.main_rng)
        
        
        return adv_images
        
    def _call(self, images, labels, targeted, eps, n_queries, n_restarts, p_init, rescale_schedule, main_rng):
    
        main_rng, random_gen_rng = jax.random.split(main_rng, num=2)
        
        self.B, self.H, self.W, self.C = images.shape
        
        adv_images = self.perturb(images, labels, targeted, eps, n_queries, n_restarts, p_init, rescale_schedule, random_gen_rng)
        
        return adv_images, main_rng

        
    def create_functions(self):
        def perturb(images, labels, targeted, eps, n_queries, n_restarts, p_init, rescale_schedule, random_gen_rng):
            acc = jax.lax.cond(targeted,
                               lambda _: mode_along_0(jnp.argmax(self.get_all_logits(images), axis = -1)) != labels,
                               lambda _: mode_along_0(jnp.argmax(self.get_all_logits(images), axis = -1)) == labels,
                               operand=None)   
            
            ip_dict = {
                "ori_images": images,
                "adv": jnp.copy(images),
                "labels": labels,
                "eps": eps,
                "p_init": p_init,
                "random_gen_rng": random_gen_rng,
                "rescale_schedule": rescale_schedule,
                "n_queries": n_queries,
                "targeted": targeted,
                
                "acc": acc,
                
            }
            
            ip_dict = jax.lax.fori_loop(0, n_restarts, self.inner_perturb, ip_dict)
            
            adv_images = jnp.clip(ip_dict["adv"], 0.0, 1.0)
            
            return adv_images
        
        self.perturb = perturb
        
        
        def inner_perturb(_, ip_dict):
            ind_to_fool = jnp.where(ip_dict["acc"], jnp.arange(ip_dict["ori_images"].shape[0]), -1)
            
            images_to_fool = jnp.where((ind_to_fool >= 0)[:, None, None, None], ip_dict["ori_images"], 0)
            labels_to_fool = jnp.where((ind_to_fool >= 0), ip_dict["labels"], -1)
            
            ##### attack_single_run
            adv_curr = self.attack_single_run(images_to_fool, labels_to_fool,
                                                   ip_dict["targeted"],
                                                   ip_dict["eps"],
                                                   ip_dict["n_queries"],
                                                   ip_dict["p_init"],
                                                   ip_dict["rescale_schedule"],
                                                   ip_dict["random_gen_rng"])
            #####
            
            # Check if images are perturbed
            acc_curr = jax.lax.cond(ip_dict["targeted"],
                                    lambda _: mode_along_0(jnp.argmax(self.get_all_logits(adv_curr), axis = -1)) != labels_to_fool,
                                    lambda _: mode_along_0(jnp.argmax(self.get_all_logits(adv_curr), axis = -1)) == labels_to_fool,
                                    operand=None)  
            
            # Choose the images that are sucessfully perturbed
            ind_curr = jnp.where(acc_curr == 0, jnp.arange(acc_curr.shape[0]), -1) 
            
            
            # Choose new images
            new_to_fool = jnp.where((ind_to_fool >= 0) & (ind_curr >= 0), jnp.arange(ind_curr.shape[0]), -1)
            
            ip_dict["acc"] = jnp.where(new_to_fool >= 0, False, ip_dict["acc"])
            ip_dict["adv"] = jnp.where((new_to_fool >= 0)[:, None, None, None], adv_curr, ip_dict["adv"])
            
            return ip_dict
            
        self.inner_perturb = inner_perturb
            
            
        # Attack Single Run            
        ## Linf
        def attack_single_run(images_to_fool, labels_to_fool, targeted, eps, n_queries, p_init, rescale_schedule, random_gen_rng):
            adv = jnp.copy(images_to_fool)
            B, H, W, C = images_to_fool.shape
            n_features = self.H * self.W * self.C
            n_ex_total = self.B
            
            _, t_uniform_rng, wl_rng = jax.random.split(random_gen_rng, num=3)
            
            #Initial horizontal stripes perturbation
            x_best = self.init_patch_insertion(images_to_fool, eps, t_uniform_rng)
            
            margin_min, loss_min = self.margin_and_loss(x_best, labels_to_fool, targeted)
            n_queries_lst = jnp.ones(images_to_fool.shape[0])
            
            wl_dict = {
                "p_init": p_init,
                "n_queries": n_queries,
                "rescale_schedule": rescale_schedule,
                "eps": eps,
                "ind_succ": (margin_min <= 0.0), 
                "idx_to_fool": labels_to_fool,
                "targeted": targeted,
                
            
                "x": images_to_fool,
                "y": labels_to_fool,
                "x_best": x_best,
                "n_queries_lst": n_queries_lst,
                "H": H,
                "W": W,
                "C": C,
                "n_features": n_features,
                "n_ex_total": n_ex_total,
                "wl_rng": wl_rng,
                "i_iter": 0.0,
                "margin_min": margin_min,
                "loss_min": loss_min,
            }
            
            wl_dict = jax.lax.while_loop(self.asr_wl_cond, self.asr_wl, wl_dict)
            
            return wl_dict["x_best"]
            
            
        self.attack_single_run = attack_single_run
        
        
        
            
        def asr_while_loop(wl_dict):        
            wl_dict["idx_to_fool"] = jnp.where(wl_dict["margin_min"] > 0.0, jnp.arange(wl_dict["margin_min"].shape[0]), -1)
              
            x_curr = wl_dict["x"]
            x_best_curr = wl_dict["x_best"]
            y_curr = wl_dict["y"]
            
            
            
            wl_dict["wl_rng"], vh_rng = jax.random.split(wl_dict["wl_rng"], num=2)
            
            ## L2 - Linf patch insertion
            x_new = self.x_new_update(x_best_curr, x_curr, vh_rng, wl_dict["eps"], wl_dict["margin_min"])
            
            margin, loss = self.margin_and_loss(x_new, y_curr, wl_dict["targeted"])
            
            
            # CHECK IMPROVEMENTS and UPDATE SETS            
            idx_improved = jnp.where(loss < wl_dict["loss_min"], jnp.arange(loss.shape[0]), -1)
            
            wl_dict["loss_min"] = jnp.where((idx_improved >= 0) & (wl_dict["idx_to_fool"] >= 0), loss, wl_dict["loss_min"]) 
            
            
            idx_miscl = jnp.where(margin <= 0.0, jnp.arange(margin.shape[0]), -1)
            idx_improved_af = jnp.maximum(idx_improved, idx_miscl)
            
            
            
            wl_dict["margin_min"] = jnp.where((idx_improved_af >= 0) & (wl_dict["idx_to_fool"] >= 0), 
                                               margin, 
                                               wl_dict["margin_min"]) 
            
            
            wl_dict["x_best"] = jnp.where(((idx_improved_af >= 0) & (wl_dict["idx_to_fool"] >= 0))[:, None, None, None],
                                          x_new,
                                          x_best_curr)
            
            
            wl_dict["n_queries_lst"] = jnp.where(wl_dict["idx_to_fool"] >= 0, 
                                                 wl_dict["n_queries_lst"] + 1, 
                                                 wl_dict["n_queries_lst"])
            
            wl_dict["i_iter"] = wl_dict["i_iter"] + 1.0
            
            wl_dict["ind_succ"] = wl_dict["margin_min"] <= 0.0
            
            return wl_dict
            
        
        
        def asr_wl_cond(cond_dict):
            state_loop_terminate = cond_dict["i_iter"] < cond_dict["n_queries"]
            ind_succ_terminate = jnp.sum(cond_dict["ind_succ"]) != cond_dict["n_ex_total"]
            count_of_to_fool_terminate = jnp.any(cond_dict["idx_to_fool"] >= 0)
            return state_loop_terminate & ind_succ_terminate & count_of_to_fool_terminate
        
        
        self.asr_wl = asr_while_loop
        self.asr_wl_cond = asr_wl_cond
        

        ## Patch inserttion ##
        def init_linf_patch_insertion(images_to_fool, eps, t_uniform_rng):
            #Initial horizontal stripes perturbation
            x_best = jnp.clip(images_to_fool + eps * self.random_choice([images_to_fool.shape[0], 1, self.W, self.C], t_uniform_rng), 0.0, 1.0)

            return x_best

        def init_l2_patch_insertion(images_to_fool, eps, t_uniform_rng):
            eta_rngs = jax.random.split(t_uniform_rng, self.B)
            delta_init = jnp.zeros_like(images_to_fool)
            delta_init = self.batched_apply_eta_tiled(delta_init, eta_rngs, self.eta_patch, self.S, self.H, self.W, (self.H - self.S * (self.H // self.S))//2) # (H,W,C)
            
            #Initial horizontal stripes perturbation
            x_best = jnp.clip(images_to_fool + eps * self.normalize_delta(delta_init), 0.0, 1.0)            

            return x_best


        if self.norm.lower() == "linf":
            self.init_patch_insertion = init_linf_patch_insertion
        elif self.norm.lower() == "l2":
            self.init_patch_insertion = init_l2_patch_insertion 
        else:
            print("Norm should be either 'ce' or 'margin'")
            exit(0)

        
        def x_new_update_linf(x_best_curr, x_curr, vh_rng, eps, margin_min):
            patch_rngs = jax.random.split(vh_rng, self.B)
            
            new_deltas, _ = jax.vmap(lambda rng: self.insert_random_patch(eps, rng))(patch_rngs)
            x_new = jnp.where((margin_min > 0.0)[:, None, None, None], x_best_curr + new_deltas, x_best_curr)
            x_new = jnp.minimum(
                        jnp.maximum(x_new, x_curr - eps),
                        x_curr + eps)
            x_new = jnp.clip(x_new, 0.0, 1.0)

            return x_new

        def x_new_update_l2(x_best_curr, x_curr, vh_rng, eps, margin_min):
            patch_rngs = jax.random.split(vh_rng, self.B)
            delta_curr = x_best_curr - x_curr

            new_deltas, _ = self.batched_l2_patches(patch_rngs, x_best_curr, x_curr, delta_curr, eps)
            
            x_new = x_curr + new_deltas * eps
            x_new = jnp.clip(x_new, 0.0, 1.0)

            return x_new


        if self.norm.lower() == "linf":
            self.x_new_update = x_new_update_linf
        elif self.norm.lower() == "l2":
            self.x_new_update = x_new_update_l2
        else:
            print("Norm should be either 'ce' or 'margin'")
            exit(0)

        
        
        ### HELPER Functions      
        def insert_random_patch(eps, rng):
            rng, rng_vh, rng_vw, rng_sign = jax.random.split(rng, 4)
        
            # random upper-left corner
            vh = self.random_int(rng_vh, 0, self.H - self.S)[0]
            vw = self.random_int(rng_vw, 0, self.W - self.S)[0]
        
            # --------  PyTorch-style sign pattern  --------
            # one sign per colour channel, broadcast over the square
            sign_c = self.random_choice((self.C,), rng_sign)          # shape (C,)
            patch   = (2.0 * eps) * sign_c.reshape(1, 1, self.C)     # (1,1,C)
            patch   = jnp.broadcast_to(patch, (self.S, self.S, self.C))        # (s,s,C)
        
            # place the patch
            delta = jnp.zeros((self.H, self.W, self.C), dtype=jnp.float32)
            delta = jax.lax.dynamic_update_slice(delta, patch,
                                                 (jnp.squeeze(jnp.int64(vh)), jnp.squeeze(jnp.int64(vw)), 0))   # (H,W,C)
            return delta, rng
    
        self.insert_random_patch = insert_random_patch
        
        
        def random_int(t_uniform_rng, low=0, high=1, shape=[1]):
            t = low + (high - low) * jax.random.uniform(t_uniform_rng, shape)
            t = jnp.int32(t)
            return t
            
        self.random_int = random_int
        
        def random_choice(shape, t_uniform_rng):
            t = 2 * jax.random.uniform(t_uniform_rng, shape, dtype=jnp.float32) -1
            return jnp.sign(t)
        
        self.random_choice = random_choice
        
        # loss function
        def margin_and_loss_ce(x, y, targeted):
            
            all_logits = self.get_all_logits(x) # M, B, D
            
            _, all_losses = jax.lax.scan(lambda _, x: (_, optax.softmax_cross_entropy_with_integer_labels(x, y)),
                                         0, # dummy init for carry
                                         all_logits)
            xent = jnp.mean(all_losses, axis=0)
            
            u = jnp.arange(x.shape[0])
            y_corr = all_logits[:, u, y]            
            all_logits = all_logits.at[:, u, y].set(-jnp.inf)
            y_others = jnp.max(all_logits, axis=-1)
            
            margin = jax.lax.cond(targeted,
                                  lambda _: jnp.mean(y_others - y_corr, axis=0),
                                  lambda _: jnp.mean(y_corr - y_others, axis=0),
                                  operand=None)
                                  
            loss = jax.lax.cond(targeted,
                                lambda _: xent,
                                lambda _: -1.0 * xent,
                                operand=None)
            
            return margin, loss
            
            
        def margin_and_loss_margin(x, y, targeted):
            
            all_logits = self.get_all_logits(x) # M, B, D
            u = jnp.arange(x.shape[0])
            y_corr = all_logits[:, u, y]            
            all_logits = all_logits.at[:, u, y].set(-jnp.inf)
            y_others = jnp.max(all_logits, axis=-1)
            
            margin = jax.lax.cond(targeted,
                                  lambda _: jnp.mean(y_others - y_corr, axis=0),
                                  lambda _: jnp.mean(y_corr - y_others, axis=0),
                                  operand=None)
                                  
            loss = jax.lax.cond(targeted,
                                lambda _: jnp.mean(y_others - y_corr, axis=0),
                                lambda _: jnp.mean(y_corr - y_others, axis=0),
                                operand=None)
            
            return margin, loss
        
            
        if self.loss == "ce":
            self.margin_and_loss = margin_and_loss_ce
        elif self.loss == "margin":
            self.margin_and_loss = margin_and_loss_margin
        else:
            print("Norm should be either 'ce' or 'margin'")
            exit(0)      

        ## FUNCTIONS FOR L2 NORM ATTACK ##
        # eta function
        def eta_rectangles(x, y):
            # same as torch.zeros([x, y])
            delta = jnp.zeros((x, y), dtype=jnp.float32)

            x_c = x // 2 + 1
            y_c = y // 2 + 1

            counter2_x = x_c - 1
            counter2_y = y_c - 1

            for counter in range(max(x_c, y_c)):
                w = 2 * counter + 1

                # clipped boundaries
                x0 = max(counter2_x, 0)
                y0 = max(counter2_y, 0)
                x1 = min(counter2_x + w, x)
                y1 = min(counter2_y + w, y)

                # create patch
                val = 1.0 / (counter + 1) ** 2
                patch = jnp.full((x1 - x0, y1 - y0), val, dtype=jnp.float32)

                # delta[x0:x1, y0:y1] += patch
                delta = delta.at[x0:x1, y0:y1].add(patch)

                # update counters
                counter2_x -= 1
                counter2_y -= 1

            # normalize
            norm = jnp.sqrt(jnp.sum(delta ** 2))
            delta = delta / norm

            return delta


        def eta_jax(s, key=None):
            delta = jnp.zeros((s, s), dtype=jnp.float32)

            top = eta_rectangles(s // 2, s)
            bottom = eta_rectangles(s - s // 2, s)

            delta = delta.at[: s // 2].set(top)
            delta = delta.at[s // 2 :].set(-bottom)

            # normalize
            norm = jnp.sqrt(jnp.sum(delta ** 2))
            delta = delta / norm

            return delta

        self.eta = eta_jax

        def apply_eta_tiled(delta_init, eta_rng, eta_s, s, h, w, sp_init):
            # total number of tiles vertically and horizontally
            Hn = h // s
            Wn = w // s
            total_tiles = Hn * Wn

            def body(idx, x):
                # compute i,j from flat idx
                i = idx // Wn
                j = idx %  Wn

                vh = sp_init + i * s
                vw = sp_init + j * s

                #jax.debug.print("vh = {}", vh)
                #jax.debug.print("vw = {}", vw)
                x["eta_rng"], transpose_rng, random_choice_rng = jax.random.split(x["eta_rng"], num=3)
                eta_s_curr = jax.lax.cond(
                    jax.random.uniform(transpose_rng) > 0.5,
                    lambda _: eta_s,
                    lambda _: jnp.transpose(eta_s),
                    operand=None
                )

                # broadcast eta(s) to batch+channel
                # eta_s: (s, s)
                # rand_block: (N, C, 1, 1)
                patch = eta_s_curr[:, :, None] * self.random_choice((1, 1, self.C), random_choice_rng) # * random_array_jax  # (N, C, s, s)

                # update slice in a jittable way
                x["delta"] = jax.lax.dynamic_update_slice(x["delta"], patch, (vh, vw, 0))
                return x

            out_dict = jax.lax.fori_loop(0, total_tiles, body, {"delta": delta_init, "eta_rng": eta_rng})
            delta_out = out_dict["delta"]
            return delta_out

        self.apply_eta_tiled = apply_eta_tiled
        self.batched_apply_eta_tiled = jax.vmap(
            self.apply_eta_tiled,
            in_axes=(0, 0, None, None, None, None, None) 
        )

        def normalize_delta(x):
            t = jnp.sqrt(
                jnp.sum(
                    jnp.reshape((x ** 2), (x.shape[0], -1)),
                    axis=-1
                )
            ) # (B, )

            out = x / (jnp.reshape(t, (-1, 1, 1, 1)) + 1e-12)

            return out

        self.normalize_delta = normalize_delta

        def normalize_delta_unbatched(x):
            # Compute L2 norm over all elements
            t = jnp.sqrt(jnp.sum(x**2))

            # Normalize
            out = x / (t + 1e-12)

            return out

        self.normalize_delta_unbatched = normalize_delta_unbatched

        def l2_norm(x):
            t = jnp.sqrt(
                jnp.sum(
                    jnp.reshape((x ** 2), (x.shape[0], -1)),
                    axis=-1
                )
            ) # (B, )

            out = jnp.reshape(t, (-1, 1, 1, 1)) 

            return out

        self.l2_norm = l2_norm

        def l2_norm_unbatched(x):
            return jnp.sqrt(jnp.sum(x**2))

        self.l2_norm_unbatched = l2_norm_unbatched

        def l2_patches(rng, x_best_curr, x_curr, delta_curr, eps):
            rng, rng_vh, rng_vw, rng_sign = jax.random.split(rng, 4)


            vh = self.random_int(rng_vh, 0, self.H - self.S)[0]
            vw = self.random_int(rng_vw, 0, self.W - self.S)[0]

            new_deltas_mask = jnp.zeros_like(x_curr)
            new_deltas_mask = jax.lax.dynamic_update_slice(
                new_deltas_mask, 
                jnp.ones((self.S, self.S, self.C), dtype=new_deltas_mask.dtype), 
                (jnp.squeeze(jnp.int64(vh)), jnp.squeeze(jnp.int64(vw)), 0)
            )

            norms_window_1 = jnp.sqrt(
                jnp.sum(
                    jax.lax.dynamic_slice(
                        delta_curr,
                        (jnp.squeeze(jnp.int64(vh)), jnp.squeeze(jnp.int64(vw)), 0),
                        (self.S, self.S, self.C)
                    )**2,
                    axis=(-3, -2), keepdims=True
                )
            )

            rng, rng_vh, rng_vw, rng_sign = jax.random.split(rng, 4)

            vh2 = self.random_int(rng_vh, 0, self.H - self.S)[0]
            vw2 = self.random_int(rng_vw, 0, self.W - self.S)[0]

            new_deltas_mask2 = jnp.zeros_like(x_curr)
            new_deltas_mask2 = jax.lax.dynamic_update_slice(
                new_deltas_mask2, 
                jnp.ones((self.S, self.S, self.C), dtype=new_deltas_mask2.dtype), 
                (jnp.squeeze(jnp.int64(vh2)), jnp.squeeze(jnp.int64(vw2)), 0)
            )

            norms_image = self.l2_norm_unbatched(x_best_curr - x_curr)
            mask_image = jnp.maximum(new_deltas_mask, new_deltas_mask2)
            norms_windows = self.l2_norm_unbatched(delta_curr * mask_image)

            new_deltas = jnp.ones([self.S, self.S, self.C])
            new_deltas = (
                new_deltas 
                * jnp.reshape(self.eta_patch, (self.S, self.S, 1))
                * self.random_choice((self.C, ), rng_sign)[None, None, :]
            )
            old_deltas = (jax.lax.dynamic_slice(
                delta_curr,
                (jnp.squeeze(jnp.int64(vh)), jnp.squeeze(jnp.int64(vw)), 0),
                (self.S, self.S, self.C)
                )
                / (
                    1e-12 + norms_window_1
                )
            )

            new_deltas += old_deltas
            new_deltas = (
                new_deltas
                / (
                    1e-12
                    + jnp.sqrt(jnp.sum(new_deltas ** 2, axis=(-3, -2), keepdims=True))
                )
                * jnp.sqrt(
                    jnp.maximum(
                        (eps * jnp.ones_like(new_deltas)) **2 - norms_image **2,
                        jnp.zeros_like(new_deltas)
                    )
                    / self.C
                    + norms_windows ** 2
                )
            )
            delta_curr = jax.lax.dynamic_update_slice(
                delta_curr, 
                jnp.zeros((self.S, self.S, self.C), dtype=new_deltas_mask2.dtype), 
                (jnp.squeeze(jnp.int64(vh2)), jnp.squeeze(jnp.int64(vw2)), 0)
            )
            delta_curr = jax.lax.dynamic_update_slice(
                delta_curr, 
                new_deltas.astype(jnp.float32), 
                (jnp.squeeze(jnp.int64(vh)), jnp.squeeze(jnp.int64(vw)), 0)
            )

            delta_curr = self.normalize_delta_unbatched(delta_curr)

            return delta_curr, rng

        self.l2_patches = l2_patches
        self.batched_l2_patches = jax.vmap(
            self.l2_patches,
            in_axes=(0, 0, 0, 0, None)   # last arg eps is NOT batched
        )        
                 
    
    def jit_call(self):
        self.perturb = jax.jit(self.perturb)