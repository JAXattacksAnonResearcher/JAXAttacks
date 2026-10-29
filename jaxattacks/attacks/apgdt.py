import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class APGDT(Attack):
    r"""
    APGD in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    Targeted attack for every wrong classes.
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        main_rng_seed (int): random seed used for initializing the pseudo-random number generator. (Default: 42)
        loss (str): loss function optimized. ['dlr'] (Default: 'dlr')
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)
        n_classes (int): MAXIMUM number of classes that will be used to generate adversarial attack. (Default: 10)
        

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - target_labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self,  
                 norm="Linf", 
                 eps = 8/255, 
                 steps=10,
                 n_restarts=1,
                 main_rng_seed = 42,
                 loss="dlr",
                 rho=0.75,
                 verbose=False,
                 n_classes=10):
        super().__init__("APGDT")
        self.norm = norm
        self.eps = eps
        self.steps = steps
        self.n_restarts = n_restarts
        self.main_rng = jax.random.PRNGKey(seed=main_rng_seed)
        self.loss = loss
        self.thr_decr = rho
        self.verbose = verbose
        self.n_target_classes = n_classes - 1
        self.supported_modes = ["targeted"]

        self.create_functions()
        
    
    def main_attack_function(self, images, labels):
    
        adv_images, self.main_rng = self._call(images, labels, self.eps, self.steps, self.n_target_classes, self.n_restarts, self.main_rng, self.thr_decr)
        
        return adv_images
        
        
    def _call(self, images, labels, eps, steps, n_target_classes, n_restarts, main_rng, thr_decr):
        
        main_rng, random_gen_rng = jax.random.split(main_rng, num=2)
        
        adv_images = self.perturb(images, labels, eps, steps, n_target_classes, n_restarts, random_gen_rng, thr_decr)
        
        return adv_images, main_rng
        
        
    def create_functions(self):
        def perturb(images, labels, eps, steps, n_target_classes, n_restarts, random_gen_rng, thr_decr): 
            ori_images = images
            adv_images = jnp.copy(ori_images)
            
            # Check if model predictions are 'True'
            acc = jnp.argmax(self.get_logits(ori_images), axis=1) == labels  
            
            perturb_fori_dict = {
                # Calculated this time
                "adv": jnp.float32(adv_images),
                "acc": acc,
                "nth_rank_target": 0, # dummy ( updated in for loop below)
                
                # Already available
                "ori_images": ori_images,
                "labels": labels,
                "eps": eps,
                "steps": steps,
                "n_restarts": n_restarts,
                "random_gen_rng": random_gen_rng,
                "thr_decr": thr_decr,
                "n_restarts": n_restarts,
            }

            perturb_fori_dict = jax.lax.fori_loop(2, n_target_classes+2, self.perturb_nth_rank_loop, perturb_fori_dict)
            
            adv_images = jnp.clip(perturb_fori_dict["adv"], 0.0, 1.0)
            
            return adv_images
        
        self.perturb = perturb
        
        def perturb_nth_rank_loop(nth_rank_target, perturb_fori_dict):
            perturb_fori_dict["nth_rank_target"] = nth_rank_target
            perturb_fori_dict = jax.lax.fori_loop(0, perturb_fori_dict["n_restarts"], self.perturb_fori_loop, perturb_fori_dict)
            
            return perturb_fori_dict
        
        self.perturb_nth_rank_loop = perturb_nth_rank_loop
            
        def perturb_fori_loop(counter, perturb_fori_dict):
            # Select 'True' labeled images (we don't need to perturb others)
            ind_to_fool =  jnp.where(perturb_fori_dict["acc"], jnp.arange(perturb_fori_dict["ori_images"].shape[0]), -1)
            
            # Select images and labels to send acc_curr 
            images_to_fool = jnp.where((ind_to_fool >= 0)[:, None, None, None], perturb_fori_dict["ori_images"], 0)
            labels_to_fool = jnp.where((ind_to_fool >= 0), perturb_fori_dict["labels"], -1) 
            
            # New 'acc' is coming from 'attack_single_run()'
            (best_curr, acc_curr, loss_curr, adv_curr) = self.attack_single_run_prep(images_to_fool, labels_to_fool, perturb_fori_dict["nth_rank_target"],
                                                                                     perturb_fori_dict["eps"], 
                                                                                     perturb_fori_dict["steps"], 
                                                                                     perturb_fori_dict["random_gen_rng"], 
                                                                                     perturb_fori_dict["thr_decr"])
            
            # Select 'False' labeled in 'acc_curr' ('attack_single_run()' succesfully modified them)
            ind_curr = jnp.where((acc_curr == 0), jnp.arange(acc_curr.shape[0]), -1)
            
            # Select the ones with 
            # 'True' labeled at first step ('acc')
            # 'False' labeled after ('acc_curr')
            # (Got the indices of images that were truely labeled at first and modified to fool the model)
            new_to_fool =jnp.where((ind_curr >= 0) & (ind_to_fool >= 0), jnp.arange(ind_curr.shape[0]), -1)
            
            # Update 'acc' with 'new_to_fool' (we were able to flip the accuracy/prediction, so update 'acc')
            perturb_fori_dict["acc"] = jnp.where(new_to_fool >= 0, False, perturb_fori_dict["acc"])
            # Update 'adv' (images) with 'new_to_fool' (clean images in 'adv' will be replaced with succesfully perturbed images)
            perturb_fori_dict["adv"] = jnp.float32(jnp.where((new_to_fool >= 0)[:, None, None, None], adv_curr, perturb_fori_dict["adv"]))
            
            return perturb_fori_dict
            
        self.perturb_fori_loop = perturb_fori_loop    
    
        ## Helper Functions        
        def attack_single_run_prep(images_to_fool, labels_to_fool, nth_rank_target, eps, steps, random_gen_rng, thr_decr):
            # No need for reshaping since whole batch is passed
            
            # steps parameters initialized
            steps_2, steps_min, size_decr = (
                                              jnp.maximum(jnp.array([0.22 * steps], int), jnp.array([1], int)), 
                                              jnp.maximum(jnp.array([0.06 * steps], int), jnp.array([1], int)), 
                                              jnp.maximum(jnp.array([0.03 * steps], int), jnp.array([1], int))
                                            )
                                            
            
            
            # Randomly perturb image a little
            x_adv = self.l_step(images_to_fool, eps, random_gen_rng) 
            x_adv = jnp.clip(x_adv, 0.0, 1.0)
            
            
            x_best = jnp.copy(x_adv) 
            x_best_adv = jnp.copy(x_adv) 
            
            
            ## Variables to trace the adversarial image generation process 
            loss_steps = jnp.zeros([self.steps, images_to_fool.shape[0]]) 
            loss_best_steps = jnp.zeros([self.steps +1, images_to_fool.shape[0]])
            acc_steps = jnp.zeros_like(loss_best_steps)
            
            ###
            itf_logits = self.get_logits(images_to_fool)
            itf_logits_sorted_ind = jnp.argsort(itf_logits, axis=-1)
            target_labels = itf_logits_sorted_ind[:, -nth_rank_target]
            #target_labels = jnp.argsort(itf_logits, axis=-1)[:, -nth_rank_target]
            ###
            
            
            # Grad calculation
            # There is no need for eot_iter since we have just one model
            # It will be added in 'APGDEnsemble' version
            (loss, (loss_indiv, logits)), grads = self.grad_fn(x_adv, labels_to_fool, target_labels)
            
            grads_best = jnp.copy(grads) 
            
            
            acc = jnp.argmax(logits, axis=1) == labels_to_fool 
            acc_steps = acc_steps.at[0].set(acc)
            loss_best = jnp.copy(loss_indiv)
            
            
            step_size = eps * jnp.ones([images_to_fool.shape[0], 1, 1, 1]) * 2 
            
            x_adv_old = jnp.copy(x_adv)
            
            k = steps_2 
            u = jnp.arange(images_to_fool.shape[0])
            counter3 = 0
            
            
            loss_best_last_check = jnp.copy(loss_best)
            reduced_last_check = jnp.zeros(loss_best.shape) == jnp.zeros(loss_best.shape)
            
            
            fori_loop_dict = {
                "images_to_fool": images_to_fool,
                "labels_to_fool": labels_to_fool,
                "eps": eps,
                "step_size": step_size,
                "thr_decr": thr_decr,
                "target_labels": target_labels,
                
                ## Calculated in this function
                # Fori_loop
                "x_adv": x_adv,
                "x_adv_old": x_adv_old,
                "x_best" : x_best,
                "x_best_adv": x_best_adv,
                "acc": acc,
                "acc_steps": acc_steps,
                "loss_steps": loss_steps,
                "loss_best_steps": loss_best_steps,
                "loss_best": loss_best,
                "grads": grads,
                "grads_best": grads_best,
                "counter3": counter3,
                "k": k,
                
                # First if 
                "reduced_last_check" : reduced_last_check,
                "loss_best_last_check": loss_best_last_check,
                "size_decr": size_decr, 
                "steps_min": steps_min, 
                
                #Second if
                "u" : u,
                "step_size": step_size,
                
                # NEW IMPLMENTATION (Comment when run first implementation) (causes around 20 second slowness, but makes it much more readable)
                "fl_oscillation" : jnp.zeros_like(labels_to_fool, dtype=jnp.bool_),
                "i": -1,
            }
            
            fori_loop_dict = jax.lax.fori_loop(0, steps, self.attack_single_run_body_fn, fori_loop_dict)
            
            return fori_loop_dict["x_best"], fori_loop_dict["acc"], fori_loop_dict["loss_best"], fori_loop_dict["x_best_adv"]
            
            
        def attack_single_run_body_fn(i, fori_loop_dict):
            # Take the difference between old batch and new batch
            grads2 = fori_loop_dict["x_adv"] - fori_loop_dict["x_adv_old"]
            # Save current batch
            fori_loop_dict["x_adv_old"] = jnp.copy(fori_loop_dict["x_adv"])
            
            a = jax.lax.cond(i > 0, lambda _: 0.75, lambda _: 1.0, operand=None) 
            
            # Update the current batch
            fori_loop_dict["x_adv"] = self.l_update_step(fori_loop_dict["images_to_fool"], 
                                                         fori_loop_dict["x_adv"], 
                                                         fori_loop_dict["grads"], grads2,
                                                         fori_loop_dict["eps"], 
                                                         fori_loop_dict["step_size"], 
                                                         a)
            # Calculate the gradients 
            (loss, (loss_indiv, logits)), fori_loop_dict["grads"] = self.grad_fn(fori_loop_dict["x_adv"], fori_loop_dict["labels_to_fool"], fori_loop_dict["target_labels"])
            
            # Calculate the predictions made with current batch
            # Update relevant variables
            curr_acc = jnp.argmax(logits, axis=1) == fori_loop_dict["labels_to_fool"]
            fori_loop_dict["acc"] = jnp.minimum(fori_loop_dict["acc"], curr_acc) ##
            fori_loop_dict["acc_steps"] = fori_loop_dict["acc_steps"].at[i+1].set(fori_loop_dict["acc"])
            
            # Choose the images that fooled the model 
            # Save them to 'x_best_adv'
            fori_loop_dict["x_best_adv"] = jnp.where(jnp.squeeze(curr_acc==0).reshape(-1, 1, 1, 1), fori_loop_dict["x_adv"], fori_loop_dict["x_best_adv"]) 
            
            
            # Save the losses to "loss_steps"
            #y1 = jnp.copy(loss_indiv)
            fori_loop_dict["loss_steps"] = fori_loop_dict["loss_steps"].at[i].set(loss_indiv) # y1
            
            # Compare current loss with best losses
            ind_curr_loss_better = loss_indiv > fori_loop_dict["loss_best"] # y1
            
            # Update the best variables according to current loss
            fori_loop_dict["loss_best"] = jnp.where(ind_curr_loss_better, loss_indiv, fori_loop_dict["loss_best"]) # y1
            fori_loop_dict["x_best"] = jnp.where(ind_curr_loss_better[:, None, None, None], fori_loop_dict["x_adv"], fori_loop_dict["x_best"])
            fori_loop_dict["grads_best"] = jnp.where(ind_curr_loss_better[:, None, None, None], fori_loop_dict["grads"], fori_loop_dict["grads_best"])
        
            
            # Save the current best losses     
            fori_loop_dict["loss_best_steps"] = fori_loop_dict["loss_best_steps"].at[i+1].set(fori_loop_dict["loss_best"])
            
            fori_loop_dict["counter3"] += 1
            
            
            
            # if counter3 == k:
            
            fori_loop_dict["i"] = i
            fori_loop_dict = jax.lax.cond(fori_loop_dict["counter3"] == fori_loop_dict["k"][0], 
                                         attack_single_run_first_if_fn,
                                         lambda x: x,
                                         operand=fori_loop_dict)  
            
                                         
            return fori_loop_dict
            
        self.attack_single_run_body_fn = attack_single_run_body_fn
            
        
        def attack_single_run_first_if_fn(first_if_dict):
            # Check oscillation
            fl_oscillation = self.check_oscillation(first_if_dict["loss_steps"],
                                                    first_if_dict["i"],
                                                    first_if_dict["k"][0],
                                                    first_if_dict["loss_best"],
                                                    first_if_dict["thr_decr"])
                                                    
            fl_reduce_no_impr = (~first_if_dict["reduced_last_check"]) * (first_if_dict["loss_best_last_check"] >= first_if_dict["loss_best"])
            
            fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
            
            first_if_dict["reduced_last_check"] = jnp.copy(fl_oscillation) 
            first_if_dict["loss_best_last_check"] = jnp.copy(first_if_dict["loss_best"]) 
            
            
            
            # if  jnp.sum(fl_oscillation) > 0:
            
            first_if_dict["fl_oscillation"] = fl_oscillation
            first_if_dict = jax.lax.cond(jnp.sum(fl_oscillation) > 0,
                                          attack_single_run_second_if_fn,
                                          lambda x: x,
                                          operand=first_if_dict)
            
            # Update counter3 and k accordingly                       
            first_if_dict["counter3"] = 0
            first_if_dict["k"] = jnp.maximum(first_if_dict["k"] - first_if_dict["size_decr"], first_if_dict["steps_min"])
            
            
            return first_if_dict
            
        self.attack_single_run_prep = attack_single_run_prep
        
        
        def attack_single_run_second_if_fn(second_if_dict):
            selection = jnp.where(second_if_dict["fl_oscillation"], second_if_dict["u"], -1)
            second_if_dict["step_size"] = jnp.where((selection >= 0)[:, None, None, None], second_if_dict["step_size"] / 2, second_if_dict["step_size"])
            
            
            second_if_dict["x_adv"] = jnp.where(second_if_dict["fl_oscillation"][:, None, None, None], second_if_dict["x_best"], second_if_dict["x_adv"])
            second_if_dict["grads"] = jnp.where(second_if_dict["fl_oscillation"][:, None, None, None], second_if_dict["grads_best"], second_if_dict["grads"])
            
            return second_if_dict
             
                
        ## Oscillation   
        def check_oscillation(x, j, k, y5, k3=0.5):            
            def body_fun(counter5, t):
                t = t + (x[j - counter5] > x[j - counter5 - 1])
                return t
        
            # initialize t
            t = jnp.zeros(x.shape[1])
            
            # use lax.fori_loop instead of Python for loop
            t = jax.lax.fori_loop(0, k, body_fun, t)
        
            return t <= k * k3 * jnp.ones(t.shape)
           
        self.check_oscillation = check_oscillation
        
                
            
            
        ## Linf and L2 in for loop
        def linf_update_adv(x, x_adv, grads, grads2, eps, step_size, a):
        
            x_adv_1 = x_adv + step_size * jnp.sign(grads)
            
            x_adv_1 = jnp.clip(jnp.minimum(jnp.maximum(x_adv_1, x - eps), x + eps),
                               0.0, 
                               1.0)
            
            x_adv_1 = jnp.clip(
                          jnp.minimum(
                              jnp.maximum(
                                  x_adv + (x_adv_1 - x_adv) * a + grads2 * (1 - a),
                                  x - eps),
                              x + eps),
                          0.0,
                          1.0)
            
            return x_adv_1
                               
        def l2_update_adv(x, x_adv, grads, grads2, eps, step_size, a):
        
            x_adv_1 = x_adv + step_size * grads / (jnp.sqrt(jnp.sum(jnp.square(grads), axis=(1,2,3), keepdims=True)) + 1e-12)
            
            x_adv_1 = jnp.clip(
                          x 
                          + (x_adv_1 - x) 
                          / (jnp.sqrt(jnp.sum(jnp.square(x_adv_1 - x), axis=(1,2,3), keepdims=True)) + 1e-12)
                          * jnp.minimum(
                              eps * jnp.ones(x.shape),
                              jnp.sqrt(jnp.sum(jnp.square(x_adv_1 - x), axis=(1,2,3), keepdims=True))),
                          0.0,
                          1.0)
            
            x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grads2 * (1-a)
            
            x_adv_1 = jnp.clip(
                          x 
                          + (x_adv_1 - x)
                          / (jnp.sqrt(jnp.sum(jnp.square(x_adv_1 - x), axis=(1,2,3), keepdims=True)) + 1e-12)
                          * jnp.minimum(
                              eps * jnp.ones(x.shape),
                              (jnp.sqrt(jnp.sum(jnp.square(x_adv_1 - x), axis=(1,2,3), keepdims=True)) + 1e-12)),
                          0.0,
                          1.0)
                          
            return x_adv_1
                              
        
        if self.norm == "Linf":
            self.l_update_step = linf_update_adv
        elif self.norm == "L2":
            self.l_update_step = l2_update_adv
        else:
            print("Norm should be either 'Linf' or 'L2'")
            exit(0)
        
        
        
        ## Loss Functions
        """
        def ce_loss(imgs, prms, lbls):
            logits = self.get_logits(prms, imgs)
            loss_indiv = optax.softmax_cross_entropy_with_integer_labels(logits, lbls)
            loss = jnp.sum(loss_indiv)
            return loss, (loss_indiv, logits)
        """
            
        def dlr_loss(imgs, lbls, trgt_lbls):
            logits = self.get_logits(imgs)
            
            logits_sorted_ind = jnp.argsort(logits, axis=1)
            logits_sorted = jnp.take_along_axis(logits, logits_sorted_ind, axis=1)
            
            ###
            loss_indiv = -((logits[jnp.arange(imgs.shape[0]), lbls] 
                            - logits[jnp.arange(imgs.shape[0]), trgt_lbls]) 
                            / 
                            (logits_sorted[:, -1] 
                             - 0.5 * logits_sorted[:, -3]
                             - 0.5 * logits_sorted[:, -4] + 1e-12))
            ###
            loss = jnp.sum(loss_indiv)
            
            return loss, (loss_indiv, logits)
            
        
        if self.loss == "dlr":
            self.calculate_loss = dlr_loss
        else:
            print("Loss should be 'dlr'")
            exit(0)
        self.grad_fn = jax.value_and_grad(self.calculate_loss, has_aux=True) # ((loss, (loss_indiv logits)), grads)
            
            
        ## Linf and L2 step
        def linf_step(x, eps, uniform_rng):
            t = 2 * jax.random.uniform(uniform_rng, shape=x.shape, dtype=jnp.float32) - 1 # basically between -1 and 1
            
            x_adv = x + eps * jnp.ones([x.shape[0], 1, 1, 1]) * t / jnp.reshape(jnp.max(jnp.abs(jnp.reshape(t, [t.shape[0], -1])), axis=1, keepdims=True), [-1, 1, 1, 1])
            
            return x_adv
            
        def l2_step(x, eps, uniform_rng):
            
            t = jax.random.normal(uniform_rng, shape=x.shape, dtype=jnp.float32)
            
            x_adv = x + eps * jnp.ones([x.shape[0], 1, 1, 1]) * t / (jnp.sqrt(jnp.sum(jnp.square(t), axis=(1,2,3), keepdims=True)) + 1e-12)  ##### SOLVE it
            
            return x_adv
            
        
        if self.norm == "Linf":
            self.l_step = linf_step
        elif self.norm == "L2":
            self.l_step = l2_step
        else:
            print("Norm should be either 'Linf' or 'L2'")
            exit(0)
        
        
            
            
    def jit_call(self):
        self.perturb = jax.jit(self.perturb)
            