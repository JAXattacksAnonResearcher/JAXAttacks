import jax
import jax.numpy as jnp
import optax
import math

from ..attack import Attack

class FAB(Attack):
    r"""
    Fast Adaptive Boundary Attack in the paper 'Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack'
    [https://arxiv.org/abs/1907.02044]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2, L1

    Arguments:
        norm (str) : Lp-norm to minimize. ['Linf', 'L2', 'L1'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        main_rng_seed (int): random seed used for initializing the pseudo-random number generator. (Default: 42)
        alpha_max (float): alpha_max. (Default: 0.1)
        eta (float): overshooting. (Default: 1.05)
        beta (float): backward step. (Default: 0.9)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self,  
                 norm="Linf", 
                 eps = 8/255, 
                 steps=10,
                 n_restarts=1,
                 main_rng_seed=42,
                 alpha_max=0.1,
                 eta=1.05,
                 beta=0.9):
        super().__init__("FAB")
        self.norm = norm
        self.eps = eps 
        self.steps = steps
        self.n_restarts = n_restarts
        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.supported_modes = ["default"]

        self.create_functions()
        
    
    def main_attack_function(self, images, labels):
    
        adv_images, self.main_rng = self._call(images, labels, self.eps, self.steps, self.n_restarts, self.main_rng, self.alpha_max, self.eta, self.beta)
        
        return adv_images
        
        
    def _call(self, images, labels, eps, steps, n_restarts, main_rng, alpha_max, eta, beta):
        
        main_rng, random_gen_rng = jax.random.split(main_rng, num=2)
        
        adv_images = self.perturb(images, labels, eps, steps, n_restarts, random_gen_rng, alpha_max, eta, beta)
        
        adv_images = jnp.clip(adv_images, 0.0, 1.0)
        
        return adv_images, main_rng
        
        
    def create_functions(self):
        def perturb(images, labels, eps, steps, n_restarts, random_gen_rng, alpha_max, eta, beta): 
            ori_images = images
            adv_images = jnp.copy(ori_images)
            
            # Check if model predictions are 'True'
            acc = jnp.argmax(self.get_logits(ori_images), axis=1) == labels  
            
            perturb_fori_dict = {
                "adv": jnp.float32(adv_images),
                "acc": acc,
                
                "ori_images": ori_images,
                "labels": labels,
                "eps": eps,
                "steps": steps,
                "n_restarts": n_restarts,
                "random_gen_rng": random_gen_rng,
                "alpha_max": alpha_max,
                "eta": eta,
                "beta": beta,
            }
            
            perturb_fori_dict = jax.lax.fori_loop(0, n_restarts, self.perturb_fori_loop, perturb_fori_dict)
            
            adv_images = jnp.clip(perturb_fori_dict["adv"], 0.0, 1.0)
            
            return adv_images
        
        self.perturb = perturb
        
        def perturb_fori_loop(_, perturb_fori_dict):
            # Select 'True' labeled images (we don't need to perturb others)
            ind_to_fool =  jnp.where(perturb_fori_dict["acc"], jnp.arange(perturb_fori_dict["ori_images"].shape[0]), -1)
            
            # Select images and labels to send acc_curr 
            images_to_fool = jnp.where((ind_to_fool >= 0)[:, None, None, None], perturb_fori_dict["ori_images"], 0)
            labels_to_fool = jnp.where((ind_to_fool >= 0), perturb_fori_dict["labels"], -1) 
            
            ###
            # IF targeted
        
            # else
            adv_curr = attack_single_run_prep(images_to_fool, labels_to_fool,
                                              perturb_fori_dict["eps"], 
                                              perturb_fori_dict["beta"], 
                                              perturb_fori_dict["eta"],  
                                              perturb_fori_dict["alpha_max"],  
                                              perturb_fori_dict["random_gen_rng"]) 
            ####
            
            acc_curr = jnp.argmax(self.get_logits(adv_curr), axis=1) == labels_to_fool  
            
            res = self.l_normalizer(images_to_fool, adv_curr)
            
            acc_curr = jnp.maximum(acc_curr, res > perturb_fori_dict["eps"])
            
            ind_curr = jnp.where(acc_curr == 0, jnp.arange(acc_curr.shape[0]), -1)
            
            new_to_fool =jnp.where((ind_curr >= 0) & (ind_to_fool >= 0), jnp.arange(ind_curr.shape[0]), -1)
        
            # Update 'acc' with 'new_to_fool' (we were able to flip the accuracy/prediction, so update 'acc')
            perturb_fori_dict["acc"] = jnp.where(new_to_fool >= 0, False, perturb_fori_dict["acc"])
            # Update 'adv' (images) with 'new_to_fool' (clean images in 'adv' will be replaced with succesfully perturbed images)
            perturb_fori_dict["adv"] = jnp.float32(jnp.where((new_to_fool >= 0)[:, None, None, None], adv_curr, perturb_fori_dict["adv"]))
            
            return perturb_fori_dict
            
        self.perturb_fori_loop  = perturb_fori_loop
                
        
        def attack_single_run_prep(imgs, lbls, eps, beta, eta, alpha_max, random_gen_rng):
            
            pred = jnp.argmax(self.get_logits(imgs), axis=1)
            
            acc = pred == lbls
            correctly_classified = jnp.sum(acc)
            
            # runs the attack only on correctly classified points
            ind_to_fool =  jnp.where(acc, jnp.arange(acc.shape[0]), -1)
            
            # Select images and labels to send acc_curr 
            images_to_fool = jnp.where((ind_to_fool >= 0)[:, None, None, None], imgs, 0) # im2
            labels_to_fool = jnp.where((ind_to_fool >= 0), lbls, -1) # la2 
            
            batch_size = images_to_fool.shape[0]
            u1 = jnp.arange(batch_size)
            
            adv = jnp.copy(images_to_fool) 
            adv_c = jnp.copy(imgs) 
            
            res2 = 1e10 * jnp.ones([batch_size])
            res_c = jnp.zeros([batch_size])
            
            x1 = jnp.copy(images_to_fool) 
            x0 = jnp.reshape(jnp.copy(images_to_fool), (batch_size, -1)) 
            
            # Stick to the original implementation:
            # https://github.com/fra31/auto-attack
            x1 = jax.lax.cond(False,
                              self.l_randomizer,
                              lambda args: jnp.float32(args[1]),
                              operand=(images_to_fool, x1, res2, eps, random_gen_rng)) 
                              
            single_run_step_dict = {
                "u1": u1,
                "res2": res2,
                "x0": x0,
                "x1": x1,
                
                "batch_size": batch_size,
                "adv": jnp.float32(adv),
                "images_to_fool": images_to_fool,
                "labels_to_fool": labels_to_fool,
                
                "alpha_max": alpha_max,
                "eta": eta,
                "beta": beta,
            }
            
            
            single_run_step_dict = jax.lax.fori_loop(0, self.steps, self.single_run_step, single_run_step_dict) 
            
            res2_succ = single_run_step_dict["res2"] < 1e10
            
            res_c = res_c.at[pred].set(single_run_step_dict["res2"] * res2_succ * + 1e10 * (1-res2_succ))
            
            ind_succ = jnp.where(res2_succ, jnp.arange(res2_succ.shape[0]), -1)
            
            selection = res2_succ & acc
            adv_succ = jnp.where(res2_succ[:, None, None, None], single_run_step_dict["adv"], adv_c)
            adv_c = jnp.where(selection[:, None, None, None], adv_succ, adv_c)
 
            return adv_c 
            
        
        self.attack_single_run_prep = attack_single_run_prep
            
        def single_run_step(srs_counter, srs_dict):
            
            df, dg = get_diff_logits_grads_batch(srs_dict["x1"], srs_dict["labels_to_fool"])#get_diff_logits_grads_batch(srs_dict["x1"], srs_dict["labels_to_fool"])
            
            dist1 = self.distance_l(df, dg)
            
            # batch_size = single_run_step["x1"].shape[0]
            
            dist1_sorted_ind = jnp.argmin(dist1, axis=1)
            dg2 = dg[srs_dict["u1"], dist1_sorted_ind]
            b = -df[srs_dict["u1"], dist1_sorted_ind] + jnp.sum(jnp.reshape((dg2 * srs_dict["x1"]),  (srs_dict["x1"].shape[0], -1)), -1)
            w = jnp.reshape(dg2, (srs_dict["x1"].shape[0], -1))
            
            
            d3 = self.projection_l(
                jnp.concatenate((jnp.reshape(srs_dict["x1"], (srs_dict["x1"].shape[0], -1)), srs_dict["x0"]), 0),
                jnp.concatenate((w, w), 0),
                jnp.concatenate((b, b), 0))
            
            
            d1 = jnp.reshape(d3[:srs_dict["x1"].shape[0]], srs_dict["x1"].shape)
            d2 = jnp.reshape(d3[-srs_dict["x1"].shape[0]:], srs_dict["x1"].shape)
            
            
            a0 = self.a0_l(d3, len(srs_dict["x1"].shape) - 1)
            a0 = jnp.maximum(a0, 1e-8 * jnp.ones_like(a0))
            a1 = a0[:srs_dict["x1"].shape[0]]
            a2 = a0[srs_dict["x1"].shape[0]:]
            
            
            alpha = jnp.minimum(
                jnp.maximum(
                    a1 / (a1 + a2),
                    jnp.zeros_like(a1)),
                srs_dict["alpha_max"] * jnp.ones_like(a1) 
            )
            
            
            srs_dict["x1"] = jnp.clip(
                ((srs_dict["x1"] + srs_dict["eta"] * d1) * (1 - alpha) 
                 + (srs_dict["images_to_fool"] + d2 * srs_dict["eta"]) * alpha), 
                 0.0,
                 1.0
            )
            
            
            # get_predicted_label
            logits = self.get_logits(srs_dict["x1"])
            prediction = jnp.argmax(logits, -1) 
            
            is_adv = prediction != srs_dict["labels_to_fool"]
            
            ### IF jnp.sum(is_adv) > 0
            
            ind_adv = jnp.where(is_adv != 0, True, False)
            
            t = self.t_l(srs_dict["x1"], srs_dict["images_to_fool"], ind_adv)
            
            srs_dict["adv"] = jnp.where(ind_adv[:, None, None, None], 
                            srs_dict["x1"] * jnp.reshape(t < srs_dict["res2"], (-1, 1, 1, 1)) + srs_dict["adv"] * jnp.reshape(t >= srs_dict["res2"], (-1, 1, 1, 1)), #  [-1, *[1] * ndims]
                            srs_dict["adv"])
                            
            srs_dict["res2"] = jnp.where(ind_adv,
                             t * (t < srs_dict["res2"]) + srs_dict["res2"] * (t >= srs_dict["res2"]),
                             srs_dict["res2"])
                             
            srs_dict["x1"] = jnp.where(ind_adv[:, None, None, None],
                           srs_dict["images_to_fool"] + (srs_dict["x1"] - srs_dict["images_to_fool"]) * srs_dict["beta"],
                           srs_dict["x1"])
            ###

            srs_dict["x1"] = srs_dict["x1"].astype(jnp.float32)
            srs_dict["adv"] = srs_dict["adv"].astype(jnp.float32)
            
            return srs_dict
            
        self.single_run_step = single_run_step
        
        ## HELPER FUNCTIONS    
        # Calculate 't'        
        def t_linf(x1, im2, ind_adv):
            t = jnp.abs(
                    jnp.reshape(
                        x1 - im2,
                        (x1.shape[0], -1)))
            # We are only interested in ones that we can not change the label
            t = jnp.where(ind_adv[:, None], t, 0)
            t = jnp.max(t, axis=1)
            
            return t
            
        def t_l2(x1, im2, ind_adv):
            t = jnp.reshape(
                    jnp.square(x1 - im2),
                    (x1.shape[0], -1))
            # We are only interested in ones that we can not change the label
            t = jnp.where(ind_adv[:, None], t, 0)
            t = jnp.sqrt(jnp.sum(t, axis=-1))
            
            return t
            
        def t_l1(x1, im2, ind_adv):
            t = jnp.reshape(
                    jnp.abs(x1- im2),
                    (x1.shape[0], -1))
            # We are only interested in ones that we can not change the label
            t = jnp.where(ind_adv[:, None], t, 0)
            t = jnp.sum(t, axis=-1)
            
            return t

 
        if self.norm == "Linf":
            self.t_l = t_linf
        elif self.norm == "L2":
            self.t_l = t_l2
        elif self.norm == "L1":
            self.t_l = t_l1
        else:
            print("Norm should be either 'Linf' or 'L2' or 'L1'")
            exit(0)               
                          
        
        # logits operations
        def get_diff_logits_grads_batch(imgs, la):
            def f_single(im_single):
                y_single = self.get_logits(im_single[None, ...])[0]
                return y_single  # shape [num_classes]
        
            g2 = jax.vmap(jax.jacobian(f_single))(imgs) # shape [batch_size, num_classes, C, H, W]
        

            y2 = self.get_logits(imgs)
            df = y2 - y2[jnp.arange(imgs.shape[0]), la][:, None]
            dg = g2 - g2[jnp.arange(imgs.shape[0]), la][:, None]
            
            df = df.at[jnp.arange(imgs.shape[0]), la].set(1e10)
            
            return df, dg
          
        
        # A0 calculation
        def a0_linf(d3, ndims):
            a0 = (
                jnp.reshape(
                    jnp.max(
                        jnp.abs(d3),
                        axis=1, keepdims=True),
                    (-1, *[1] * ndims)) 
            )
            
            return a0
        
        def a0_l2(d3, ndims):
            a0 = (
                jnp.reshape(
                    jnp.sqrt(
                        jnp.sum(
                            jnp.square(d3),
                            axis=1, keepdims=True)),
                    (-1, *[1] * ndims)) 
            )
            
            return a0
        
        def a0_l1(d3, ndims):
            a0 = (
                jnp.reshape(
                    jnp.sum(
                        jnp.abs(d3),
                        axis=1, keepdims=True),
                    (-1, *[1] * ndims)) 
            )
            
            return a0
            
                        
        if self.norm == "Linf":
            self.a0_l = a0_linf
        elif self.norm == "L2":
            self.a0_l = a0_l2
        elif self.norm == "L1":
            self.a0_l = a0_l1
        else:
            print("Norm should be either 'Linf' or 'L2' or 'L1'")
            exit(0) 
        
        
        # Distance Calculation
        def distance_linf(df, dg):
            dist1 = jnp.abs(df) / (
                1e-12
                + jnp.sum(
                    jnp.reshape(
                        jnp.abs(dg), 
                        (dg.shape[0], dg.shape[1], -1))
                    , axis=-1)
            )
            
            return dist1
            
        def distance_l2(df, dg):
            dist1 = jnp.abs(df) / (
                1e-12
                + jnp.sqrt(
                    jnp.sum(
                        jnp.reshape(
                            jnp.square(dg),
                            (dg.shape[0], dg.shape[1], -1)),
                        axis=-1)
                    )
            )
            
            return dist1
        
        def distance_l1(df, dg):
            dist1= jnp.abs(df) / (
                1e-12
                + jnp.max(
                    jnp.reshape(
                        jnp.abs(dg),
                        (df.shape[0], df.shape[1], -1)),
                    axis=2)
            )
            
            return dist1
        
        if self.norm == "Linf":
            self.distance_l = distance_linf
        elif self.norm == "L2":
            self.distance_l = distance_l2
        elif self.norm == "L1":
            self.distance_l = distance_l1
        else:
            print("Norm should be either 'Linf' or 'L2' or 'L1'")
            exit(0)             
        
            
        # Projection
        
        def projection_l2(ptp, whp, bhp):
            # ptp = points_to_project
            # whp = w_hyper_plane
            # bhp = b_heyper_plane
            c = jnp.sum((whp * ptp), axis=1) - bhp
            sign = 2 * (c >= 0) - 1
            whp = sign[:, None] * whp
            c = sign * c
        
        
            r = jnp.clip(jnp.maximum(ptp / whp, (ptp-1) / whp), -1e12, 1e12)
            r = jnp.where(jnp.abs(whp) < 1e-8, 1e12, r)
            r = jnp.where(r == -1e12, 1e12, r)
            r_sorted_ind = jnp.argsort(r, axis=1)
            r_sorted = jnp.take_along_axis(r, r_sorted_ind, axis=1)
        
            r_sorted2 = jnp.pad(r_sorted[:, 1:], ((0, 0), (0, 1)))
            r_sorted = jnp.where(r_sorted == 1e12, 0, r_sorted)
            r_sorted2 = jnp.where(r_sorted2 == 1e12, 0, r_sorted2)
        
            # r_sorted_ind is different from PyTorch 
            # which affects rest of computation and result in order of 1e-3 difference
        
            w3s = jnp.take_along_axis(whp ** 2, r_sorted_ind, axis=1)
            w5 = jnp.sum(w3s, axis=1, keepdims=True)
            ws = w5 - jnp.cumsum(w3s, axis=1)
            d = - (r*whp)
            d = d * (jnp.abs(whp) > 1e-8)
        
            s = jnp.concatenate(
              (-w5 * r_sorted[:, 0:1], jnp.cumsum((-r_sorted2 + r_sorted) * ws, axis=1) - w5 * r_sorted[:, 0:1]),
              axis=1
            )  
        
            c4 = s[:,0] + c < 0
            c3 = jnp.sum(d * whp, axis = 1) + c > 0
            c2 = ~(c4 | c3)
        
            lb = binary_search_loop_l2(s, c, ws, r_sorted, whp, c2)
        
            d = jax.lax.cond(jnp.any(c4),
                         c4_any_l2,
                         lambda x: x[-1],
                         operand=(c, c4, whp, w5, d))
        
        
            d = jax.lax.cond(jnp.any(c2),
                         c2_any_l2,
                         lambda x: x[-1],
                         operand=(c, c2, whp, ws, r, r_sorted, lb, s, d))
        
            return d * jnp.float32(jnp.abs(whp) > 1e-8)
            
        
        def projection_linf(ptp, whp, bhp):
            # ptp = points_to_project
            # whp = w_hyper_plane
            # bhp = b_heyper_plane
            sign = 2 * (jnp.sum(whp * ptp, 1) - bhp >= 0) - 1
            whp = whp * jnp.expand_dims(sign, axis=1)
            bhp = bhp * sign
        
            a = whp < 0
            d = (a - ptp) * (whp != 0)
        
            p = a - ptp * (2 * a - 1)
            p_sorted_ind = jnp.argsort(p, axis=1)
        
            bhp = bhp - jnp.sum(whp * ptp, axis=1)
            b0 = jnp.sum(whp * d, axis=1)
        
            p_sorted_ind2 = jnp.flip(p_sorted_ind, (1, )) 
            ws = jnp.take_along_axis(whp, p_sorted_ind2, 1)
            bs2 = -ws * jnp.take_along_axis(d, p_sorted_ind2, 1)
        
            s = jnp.cumsum(jnp.abs(ws), axis=1)
            sb = jnp.cumsum(bs2, axis=1) + b0[:, None]
        
            b2 = sb[:, -1] - s[:, -1] * jnp.squeeze(jnp.take_along_axis(p, p_sorted_ind[:, 0:1], axis=1), 1)
            c_l = bhp - b2 > 0
            c2 = (bhp - b0 > 0) & (~c_l)
        
            lb, ub = binary_search_loop_linf(p_sorted_ind, sb, s, p, bhp, whp, b2, c_l, c2)
            lb = jnp.int64(lb)
        
            d = jax.lax.cond(jnp.any(c_l),
                             c_l_any,
                             lambda x: jnp.float32(x[-1]),
                             operand=(bhp, c_l, sb, s, a, d))
        
            batch_idx = jnp.arange(c_l.shape[0])
        
            lmbd_opt = jnp.clip(
                (bhp[batch_idx] - sb[batch_idx, lb]) / (-s[batch_idx, lb]),
                min=0.0
            )
            d = jnp.where(c2[:, None],
                          jnp.minimum(lmbd_opt[:, None], d) * a + jnp.maximum(-lmbd_opt[:, None], d) * (1-a),
                          d)
        
            return d * (whp != 0)
            
        
        def projection_l1(ptp, whp, bhp):
            # ptp = points_to_project
            # whp = w_hyper_plane
            # bhp = b_heyper_plane
        
            c = jnp.sum(whp * ptp, axis=1) - bhp
            ind2 = 2 * (c >= 0) -1
        
            whp = whp * ind2[:, None]
            c = c * ind2
        
            r = jnp.clip(jnp.abs(1 / whp), None, 1e12)
            r_sorted_ind = jnp.argsort(r, axis=1)
            r_sorted_rank = jnp.argsort(r_sorted_ind)
        
        
            c6 = (whp < 0)
            d = (-ptp + c6) * (whp != 0)
            ds = jnp.take_along_axis(jnp.minimum(-whp *ptp, whp * (1 - ptp)), r_sorted_ind, axis=1)
            ds2 = jnp.concatenate((jnp.expand_dims(c, -1), ds), 1)
            s = jnp.cumsum(ds2, axis=1)
        
            c2 = s[:, -1] < 0
        
            lb = binary_search_loop_l1(s, c2, max_iter=None)
            lb2 = lb.astype(jnp.int32)
        
            d = jax.lax.cond(jnp.any(c2),
                         c2_any_l1,
                         lambda x: x[-1],
                         operand=(c2, lb2, s, lb, whp, r_sorted_ind, r_sorted_rank, d))
        
            return d * (jnp.abs(whp) > 1e-8)
            
            
        if self.norm == "Linf":
            self.projection_l = projection_linf
        elif self.norm == "L2":
            self.projection_l = projection_l2
        elif self.norm == "L1":
            self.projection_l = projection_l1
        else:
            print("Norm should be either 'Linf' or 'L2' or 'L1'")
            exit(0)


        #Random start
        
        def linf_randomizer(arguments):
            (images_to_fool, x1, res2, eps, random_gen_rng) = arguments 
            
            t = 2 * jax.random.uniform(random_gen_rng, x1.shape) - 1
            
            res = (
                    images_to_fool
                    + (
                        jnp.reshape(
                            jnp.minimum(
                                res2,
                                eps * jnp.ones_like(res2)),
                            (-1, 1, 1, 1)) 
                        )
                    * t
                    / (
                        jnp.reshape(
                            jnp.max(
                                jnp.abs(jnp.reshape(t, (t.shape[0], -1))),
                                axis=1, keepdims=True),
                            (-1, 1, 1, 1))
                      )
                    * 0.5
                    )
            
            res = jnp.clip(res, 0.0, 1.0)
            res = jnp.float32(res)
            return res
            
        def l2_randomizer(arguments):
            (images_to_fool, x1, res2, eps, random_gen_rng) = arguments 
            
            t = jax.random.normal(random_gen_rng, x1.shape)
            
            res = (
                   images_to_fool
                   + (
                       jnp.reshape(
                           jnp.minimum(res2, eps * jnp.ones_like(res2)),
                           (-1, 1, 1, 1))
                     )
                   * t
                   / (
                       jnp.reshape(
                           jnp.sqrt(jnp.sum(
                                       jnp.reshape(
                                           jnp.square(t),
                                           (t.shape[0], -1)),
                                       axis=-1)),
                           (t.shape[0], 1, 1, 1))
                     )
                   * 0.5
                   )
            
            res = jnp.clip(res, 0.0, 1.0)
            res = jnp.float32(res)
            return res
                               
                                       
        def l1_randomizer(arguments):
            (images_to_fool, x1, res2, eps, random_gen_rng) = arguments 
            t = jax.random.normal(random_gen_rng, x1.shape)
            
            res = (
                    images_to_fool
                    + (
                        jnp.reshape(
                            jnp.minimum(res2, eps * jnp.ones_like(res2)),
                            (-1, 1, 1, 1))
                      )
                    * t
                    / (
                        jnp.reshape(
                            jnp.sum(
                                jnp.reshape(
                                    jnp.abs(t),
                                    (t.shape[0], -1)),
                                axis=-1), 
                            (t.shape[0], 1, 1, 1))
                        )
                    * 0.5
                    )
            
            res = jnp.clip(res, 0.0, 1.0)
            res = jnp.float32(res)
            return res
                                
                   
        if self.norm == "Linf":
            self.l_randomizer = linf_randomizer
        elif self.norm == "L2":
            self.l_randomizer = l2_randomizer
        elif self.norm == "L1":
            self.l_randomizer = l1_randomizer
        else:
            print("Norm should be either 'Linf' or 'L2' or 'L1'")
            exit(0)                   
            

        # Normalizer
        
        def linf_normalizer(x_to_fool, adv_curr):
            
            res = jnp.max(
                      jnp.reshape(
                          jnp.abs(x_to_fool - adv_curr), 
                          (x_to_fool.shape[0], -1)),
                      axis=-1)
            
            return res
            
        def l2_normalizer(x_to_fool, adv_curr):
            
            res = jnp.sqrt(
                      jnp.sum(
                          jnp.reshape(
                              jnp.square(x_to_fool - adv_curr),
                              (x_to_fool.shape[0], -1)),
                          axis = -1)
                      )
            
            return res
            
        def l1_normalizer(x_to_fool, adv_curr):
            
            res = jnp.sum(
                      jnp.reshape(
                          jnp.abs(x_to_fool - adv_curr),
                          (x_to_fool.shape[0], -1)),
                          axis=-1)
            
            return res
        
        if self.norm == "Linf":
            self.l_normalizer = linf_normalizer
        elif self.norm == "L2":
            self.l_normalizer = l2_normalizer
        elif self.norm == "L1":
            self.l_normalizer = l1_normalizer
        else:
            print("Norm should be either 'Linf' or 'L2' or 'L1'")
            exit(0)
                              
        ## OTher helpers:
        def binary_search_loop_l2(s, c, ws, rs, whp, c2, max_iter=None):
            B, D = whp.shape
            idxs = jnp.arange(B)
        
            # Avoid dynamic shapes: keep all arrays full-sized, mask with c2
            lb = jnp.zeros(B)
            ub = jnp.full((B,), D - 1)
            nitermax = math.ceil(math.log2(D)) if max_iter is None else max_iter
        
            def body_fun(i, state):
                lb, ub = state
                mid = jnp.floor((lb + ub) / 2)
                mid_idx = mid.astype(jnp.int32)
        
                # gather s[batch, mid_idx]
                rows = jnp.arange(B)
                sm = s[jnp.arange(B), mid_idx]
                c_val = sm + c
        
                # if c_val > 0 and c2 is active, then update bounds
                cond = (c_val > 0) & c2
                lb_new = jnp.where(cond, mid, lb)
                ub_new = jnp.where(cond, ub, mid)
                return (lb_new, ub_new)
        
            lb, ub = jax.lax.fori_loop(0, nitermax, body_fun, (lb, ub))
        
            return lb.astype(jnp.int32)
        
        def c4_any_l2(operand):
            c, c4, whp, w5, d = operand
            d = jnp.where(c4[:, None],  (- 1* c / jnp.squeeze(w5, -1))[:, None] * whp, d)
        
            return d
        
        def c2_any_l2(operand):
            c, c2, whp, ws, r, r_sorted, lb, s, d = operand
        
            batch_idx = jnp.arange(c2.shape[0])
        
            alpha = (s[batch_idx, lb] + c[batch_idx]) / ws[batch_idx, lb] + r_sorted[batch_idx, lb]
            alpha = jnp.where(ws[batch_idx, lb] == 0, 0, alpha)
        
            c5 = alpha[:, None] > r
            d = jnp.where(c2[:, None], d * c5 - alpha[:, None] * whp * (1-c5), d)
        
            return d
                
                
        def binary_search_loop_linf(indp, sb, s, p, bhp, whp, b2, c_l, c2, max_iter=None):
            B, D = whp.shape
        
            lb = jnp.zeros(B)
            ub = jnp.full((B,), D - 1)
            nitermax = math.ceil(math.log2(D)) if max_iter is None else max_iter
        
            def body_fun(i, state):
                lb, ub = state
                mid = jnp.floor((lb + ub) / 2)
                mid_idx = mid.astype(jnp.int32)
        
                gather_idx = indp.shape[1] - 1 - mid_idx[:, None]  # shape [B, 1]
                indcurr = jnp.take_along_axis(indp, gather_idx, axis=1).squeeze(1)  # shape [B]
        
                sb_val = jnp.take_along_axis(sb, mid_idx[:, None], axis=1).squeeze(1)
                s_val = jnp.take_along_axis(s, mid_idx[:, None], axis=1).squeeze(1)
                p_val = jnp.take_along_axis(p, indcurr[:, None], axis=1).squeeze(1)
        
                b2_val = sb_val - s_val * p_val
                c_val = bhp - b2_val
        
                # Only update if c2 is active; otherwise keep bounds
                cond = (c_val > 0) & c2
        
                lb_new = jnp.where(cond, mid, lb)
                ub_new = jnp.where(cond, ub, mid)
        
                return (lb_new, ub_new)
        
            lb, ub = jax.lax.fori_loop(0, nitermax, body_fun, (lb, ub))
        
            return lb.astype(jnp.int32), ub.astype(jnp.int32)
        
        def c_l_any(operand):
            bhp, c_l, sb, s, a, d = operand
        
            lmbd_opt = jnp.clip(
                (bhp - sb[:, -1]) / (-s[:, -1]),
                0.0
            )
            d = jnp.where(c_l[:, None], (2 * a -1) * lmbd_opt[:, None], d)

            d = jnp.float32(d)

            return d
            
        def binary_search_loop_l1(s, c2, max_iter=None):
            B, D = s.shape
        
            lb = jnp.zeros(B)
            ub = jnp.full((B,), D - 1)
            nitermax = math.ceil(math.log2(D)) if max_iter is None else max_iter
        
            def body_fun(i, state):
                lb, ub = state
                mid = jnp.floor((lb + ub) / 2)
                mid_idx = mid.astype(jnp.int32)
        
                sm = s[jnp.arange(B), mid_idx]
        
                cond = (sm > 0) & c2
                lb_new = jnp.where(cond, mid, lb)
                ub_new = jnp.where(cond, ub, mid)
                return (lb_new, ub_new)
        
            lb, ub = jax.lax.fori_loop(0, nitermax, body_fun, (lb, ub))
        
            return lb
        
        def c2_any_l1(operand):
            c2, lb2, s, lb, whp, r_sorted_ind, r_sorted_rank, d = operand
        
            r_sorted_ind = jnp.squeeze(jnp.take_along_axis(r_sorted_ind, lb2[:, None], axis=1))
            u = jnp.expand_dims(jnp.arange(whp.shape[0]), 1)
            u2 = jnp.expand_dims(jnp.arange(whp.shape[1]), 0)
        
            batch_idx = jnp.arange(c2.shape[0])
        
            alpha = -s[batch_idx, lb2] / whp[batch_idx, r_sorted_ind]
            c5 = u2 < jnp.expand_dims(lb, -1)
            u3 = c5[u[: c5.shape[0]], r_sorted_rank[batch_idx]]
        
        
            d = jnp.where(c2[:, None], d * u3, d)
            d = d.at[batch_idx, r_sorted_ind].set(alpha) 
        
            return d
            
            
    def jit_call(self):
        self.perturb = jax.jit(self.perturb)
            