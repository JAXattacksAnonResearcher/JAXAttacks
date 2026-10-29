import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class EAD(Attack):
    r"""
    EAD attack in the paper 'EAD: Elastic-Net Attacks to Deep Neural Networks'
    [https://arxiv.org/abs/1709.04114]

    Distance Measure : L1 and L2

    Arguments:
        kappa (float): how strong the adversarial example should be (also written as 'confidence'). (Default: 0)
        lr (float): larger values converge faster to less accurate results. (Default: 0.01)
        binary_search_steps (int): number of times to adjust the constant with binary search. (Default: 9)
        max_iterations (int): number of iterations to perform gradient descent. (Default: 100)
        abort_early (bool): if we stop improving, abort gradient descent early. (Default: True)
        initial_const (float): the initial constant c to pick as a first guess. (Default: 0.001)
        beta (float): hyperparameter trading off L2 minimization for L1 minimization. (Default: 0.001)


    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, kappa=0, lr=0.01, binary_search_steps=9, max_iterations=100, abort_early=True, initial_const=0.001, beta=0.001, check_every=10, rule="EN"):
        super().__init__(f"EAD{rule.upper()}")
        self.kappa = kappa
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.beta = beta
        self.check_every = check_every
        self.rule = rule
        # The last iteration (if we run many steps) repeat the search once. 
        self.repeat = binary_search_steps >= 10 

        self.supported_modes = ["default", "targeted"]

        self.create_functions()

    
    def main_attack_function(self, images, labels):

        return self._call(images, labels, self.kappa, self.lr, self.binary_search_steps, self.max_iterations, self.abort_early, self.initial_const, self.beta, self.check_every, self.repeat, self.targeted)

    def _call(self, images, labels, kappa, lr, binary_search_steps, max_iterations, abort_early, initial_const, beta, check_every, repeat, targeted):
        
        # forward
        batch_size = images.shape[0]

        outputs = self.get_logits(images) # (batch_size, num_classes)
        
        lower_bound = jnp.zeros((batch_size, ))
        const = jnp.ones((batch_size, )) * initial_const
        upper_bound = jnp.ones((batch_size, )) * 1e10

        final_adv_images = jnp.copy(images)
        y_one_hot = jnp.identity(outputs.shape[1])[labels]

        o_bestl1 = jnp.ones((batch_size, )) * 1e10
        o_bestscore = jnp.ones((batch_size, )) * -1

        x_k = jnp.copy(images)
        y_k = jnp.copy(x_k)

        # forward for loopss
        x = {
            "images": images, #
            "x_k": x_k, # 
            "y_k": y_k, #
            "labels": labels, #
            "y_one_hot": y_one_hot, #
            "lr": lr, #
            "kappa": kappa, #
            "beta": beta, #
            "check_every": check_every, #
            "max_iterations": max_iterations, # 
            "binary_search_steps": binary_search_steps, #
            "abort_early": abort_early, # 
            "repeat": repeat,

            "lower_bound": lower_bound,
            "const": const,
            "upper_bound": upper_bound,

            "final_adv_images": final_adv_images, #
            "o_bestl1": o_bestl1, # 
            "o_bestscore": o_bestscore, #
            
            "targeted": targeted, #

            ## For inner while loop 
            "global_step": 0,
            "bestl1": jnp.ones((batch_size, )) * 1e10,
            "bestscore": jnp.ones((batch_size, )) * -1,
            "prevloss": 1e6,
            
            "cur_lr": lr,
            "iter_i": 0,
            "loss": 0,
            "break_flag" : jnp.array(True, dtype=bool),

        }

        x = jax.lax.fori_loop(0, binary_search_steps, self._call_fori_body_fn, x)
        
        final_adv_images = x["final_adv_images"]
        final_adv_images = jnp.clip(final_adv_images, 0.0, 1.0)

        return final_adv_images 

    def create_functions(self):
        def _call_fori_body_fn(outer_step, fori_dict):
            batch_size = fori_dict["images"].shape[0]
            
            const = jax.lax.cond(fori_dict["repeat"] & (outer_step == fori_dict["binary_search_steps"] -1),
                                lambda _: fori_dict["upper_bound"],
                                lambda _: fori_dict["const"],
                                operand=None)

            fori_dict["global_step"] = 0
            fori_dict["bestl1"] = jnp.ones((batch_size, )) * 1e10
            fori_dict["bestscore"] = jnp.ones((batch_size, )) * -1
            fori_dict["prevloss"] = 1e6
            fori_dict["const"] = const
            fori_dict["iter_i"] = 0
            fori_dict["loss"] = 0.0
            fori_dict["cur_lr"] = fori_dict["lr"]
            fori_dict["break_flag"] = jnp.array(True, dtype=bool)

            # While loop (inner loop)
            
            fori_dict = jax.lax.while_loop(self.wl_cond, self.inner_while_loop, fori_dict)

            upper_bound, lower_bound, const = self.adjust_constant(
                fori_dict["labels"],
                fori_dict["bestscore"],
                fori_dict["const"],
                fori_dict["upper_bound"],
                fori_dict["lower_bound"],
                fori_dict["kappa"],
                fori_dict["targeted"]
            )

            fori_dict["upper_bound"] = upper_bound
            fori_dict["lower_bound"] = lower_bound
            fori_dict["const"] = const
            

            return fori_dict

        self._call_fori_body_fn = _call_fori_body_fn

        # Inner loop
        def inner_while_loop(wl_dict):
            
            loss_to_print, grads = self.grad_fn(
                wl_dict['y_k'], 
                wl_dict['images'], 
                wl_dict['y_one_hot'], 
                wl_dict['const'], 
                wl_dict['kappa'], 
                wl_dict['targeted'] 
            )

            # Gradient step
            wl_dict['global_step'] += 1
            wl_dict['y_k'] -= grads * wl_dict['cur_lr']

            # Polynomial decau of learning rate
            wl_dict['cur_lr'] = (
                wl_dict['lr'] * (1 - wl_dict['global_step'] / wl_dict['max_iterations']) ** 0.5
            )

            wl_dict['x_k'], wl_dict['y_k'] = self.FISTA(
                wl_dict['images'], 
                wl_dict['x_k'],
                wl_dict['y_k'],
                wl_dict['global_step'],
                wl_dict['beta']
            )

            # Without grad part
            output = self.get_logits(wl_dict['x_k']) # (batch_size, num_classes)
            L2_loss_value = self.L2_loss(wl_dict['x_k'], wl_dict['images'])
            L1_loss_value = self.L1_loss(wl_dict['x_k'], wl_dict['images'])


            wl_dict['loss'] = self.EAD_loss_w_L1(output, 
                wl_dict['y_one_hot'],
                L1_loss_value, 
                L2_loss_value, 
                wl_dict['const'], 
                wl_dict['kappa'], 
                wl_dict['beta'],
                wl_dict['targeted']
            )

            cost = self.cost_for_rule(L1_loss_value, L2_loss_value, wl_dict['beta'])

            #Here above
            bestl1, bestscore, o_bestl1, o_bestscore, final_adv_images = self.adjust_best_result(
                wl_dict['x_k'],
                wl_dict['labels'],
                output,
                cost,
                wl_dict['bestl1'],
                wl_dict['bestscore'],
                wl_dict['o_bestl1'],
                wl_dict['o_bestscore'],
                wl_dict['final_adv_images'],
                wl_dict['kappa'],
                wl_dict['targeted']
            )

            wl_dict['bestl1'] = bestl1 # reinit
            wl_dict['bestscore'] = bestscore # reinit
            wl_dict['o_bestl1'] = o_bestl1 # prev
            wl_dict['o_bestscore'] = o_bestscore # prev
            wl_dict['final_adv_images'] = final_adv_images # prev

            ### IMPLEMENTATION DIFFERENCE (in abort_early=True)
            ### The Python implementations have 'for loop' instead of 'while loop' in implementation
            ### However, jax.laf.fori_loop does not have break option. 
            ### So to be able to implement break logic 'while_loop' is used.
            ### However while loop only checks at the beginning of the loop.
            ### So, it does not work like a regular break in for loop.
            ### Due to the this issue, final_adv_images and other variables are updated once more in JAX implementation.
            ### This results in difference between PyTorch and JAX implementation.
            ### To be able to match exactly with PyTorch implementation 'prev_final_adv_images' like variables can be defined.
            ### With this vairable all the changes can be reverted back to compansate one extra step issue in JAX implementation.
            ### However this requires to revert back all of the variables updated in this while block.
            ### The code is already complicated right now. So I don't see it is as a necessity.
            ### Since the implementation results are already very close to each other.
            ### So adding prev functionality will complicate the repo just to close gap for around estimated 1e-3 or 1e-4 value.
            ### Hence the code is now left as it is. 
            
            # early_abort condition
            wl_dict['break_flag'] = jnp.logical_not(
                jnp.logical_and(
                    wl_dict['abort_early'],
                    jnp.logical_and(
                        wl_dict['iter_i'] % (wl_dict['max_iterations'] // wl_dict['check_every']) == 0,
                        wl_dict['loss'] > wl_dict['prevloss'] * 0.999999
                    )
                )
            )
            update_flag = jnp.logical_and(
                wl_dict['abort_early'],
                wl_dict['iter_i'] % (wl_dict['max_iterations'] // wl_dict['check_every']) == 0
            )
            wl_dict['prevloss'] = jax.lax.cond(update_flag,
                                               lambda _: wl_dict['loss'],
                                               lambda _: wl_dict['prevloss'],
                                               operand=None)

            wl_dict['iter_i'] = wl_dict['iter_i'] + 1
            return wl_dict

        def wl_cond(cond_dict):
            iter_flag = cond_dict["iter_i"] < cond_dict["max_iterations"]         

            return iter_flag & cond_dict['break_flag']

        self.inner_while_loop = inner_while_loop
        self.wl_cond = wl_cond

        ## HELPER FUNCTIONS
        
        # L2_Loss
        def L2_loss(x1, x2):
            x1 = jnp.reshape(x1, (x1.shape[0], -1))
            x2 = jnp.reshape(x2, (x1.shape[0], -1))
            loss = jnp.sum((x1 - x2)**2, axis=1)
            return loss

        def L1_loss(x1, x2):
            x1 = jnp.reshape(x1, (x1.shape[0], -1))
            x2 = jnp.reshape(x2, (x1.shape[0], -1)) 
            loss = jnp.sum(jnp.abs(x1 - x2), axis=1)
            return loss
        
        self.L1_loss = L1_loss
        self.L2_loss = L2_loss

        # EAD Loss
        def EAD_loss_wo_L1(output, one_hot_labels, L2_loss_value, const, kappa, targeted):
            # one_hot_labels.shape => (batch_size, num_classes)
            # output.shape => (batch_size, num_classes)

            other = jnp.max(
                (1 - one_hot_labels) * output - (one_hot_labels * 1e4), axis=-1
            ) # (batch_size, )

            real = jnp.max(one_hot_labels * output, axis=-1) # (batch_size, )
            
            F_loss = jax.lax.cond(targeted,
                                  lambda _: jnp.clip((other - real), min=-kappa), 
                                  lambda _: jnp.clip((real - other), min=-kappa),
                                  operand=None)

            loss = jnp.sum(const * F_loss) + jnp.sum(L2_loss_value)

            return loss
        
        def EAD_loss_w_L1(output, one_hot_labels, L1_loss_value, L2_loss_value, const, kappa, beta, targeted):
            # one_hot_labels.shape => (batch_size, num_classes)
            # output.shape => (batch_size, num_classes)

            other = jnp.max(
                (1 - one_hot_labels) * output - (one_hot_labels * 1e4), axis=-1
            ) # (batch_size, )

            real = jnp.max(one_hot_labels * output, axis=-1) # (batch_size, )
            
            F_loss = jax.lax.cond(targeted,
                                  lambda _: jnp.clip((other - real), min=-kappa), 
                                  lambda _: jnp.clip((real - other), min=-kappa),
                                  operand=None)

            loss = jnp.sum(const * F_loss) + jnp.sum(L2_loss_value) + jnp.sum(beta * L1_loss_value)

            return loss
        
        self.EAD_loss_wo_L1 = EAD_loss_wo_L1
        self.EAD_loss_w_L1 = EAD_loss_w_L1
        
        # Total loss and gradient function for y_k
        def calculate_loss(y_k, images, y_one_hot, const, kappa, targeted):
            
            output = self.get_logits(y_k)
            L2_loss_value = self.L2_loss(y_k, images)

            cost = self.EAD_loss_wo_L1(output, y_one_hot, L2_loss_value, const, kappa, targeted)

            return cost

        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)

        # FISTA
        def FISTA(images, x_k, y_k, global_step, beta): 

            zt = global_step / (global_step + 3)

            upper = jnp.clip(y_k - beta, max=1)
            lower = jnp.clip(y_k + beta, min=0)

            diff = y_k - images
            cond1 = diff > beta
            cond2 = jnp.abs(diff) <= beta
            cond3 = diff < - beta

            new_x_k = (cond1 * upper) + (cond2  * images) + (cond3 * lower)
            y_k = new_x_k + (zt * (new_x_k - x_k))

            return new_x_k, y_k

        self.FISTA = FISTA

        # cost calculate for 'rules' of EAD algorithm

        def cost_for_en_rule(L1_loss_value, L2_loss_value, beta):
            return L2_loss_value + (L1_loss_value * beta)

        def cost_for_l1_rule(L1_loss_value, L2_loss_value, beta):
            return L1_loss_value

        if self.rule.lower() == "en":
            self.cost_for_rule = cost_for_en_rule
        elif self.rule.lower() == "l1":
            self.cost_for_rule = cost_for_l1_rule
        else:
            print("Norm should be either 'en' or 'l1'")
            exit(0)

        # adjust_best_result
        def adjust_best_result(adv_img, labels, output, cost, bestl1, bestscore, o_bestl1, o_bestscore, final_adv_images, kappa, targeted):
            # output.shape => (batch_size, num_classes)

            output_label = jnp.argmax(output, axis=-1) # (batch_size, )
            
            mask = (cost < bestl1) & self.compare(output, labels, kappa, targeted)

            bestl1 = jnp.where(mask, cost, bestl1)
            bestscore = jnp.where(mask, output_label, bestscore)

            mask = (cost < o_bestl1) & self.compare(output, labels, kappa, targeted)
            
            o_bestl1 = jnp.where(mask, cost, o_bestl1)
            o_bestscore = jnp.where(mask, output_label, o_bestscore)
            final_adv_images = jnp.where(mask[:, None, None, None], adv_img, final_adv_images)

            return bestl1, bestscore, o_bestl1, o_bestscore, final_adv_images

        def compare(output, labels, kappa, targeted):
            # output.shape => (batch_size, num_classes)

            batch_size = output.shape[0]
            u = jnp.arange(batch_size)
            output = jax.lax.cond(targeted,
                                  lambda _: output.at[u, labels].add(-kappa),
                                  lambda _: output.at[u, labels].add(kappa),
                                  operand=None)
                
            output = jnp.argmax(output, axis=-1) # (batch_size, )

            result = jax.lax.cond(targeted, 
                                  lambda _: output == labels,
                                  lambda _: output != labels,
                                  operand=None)

            return result

        self.adjust_best_result = adjust_best_result
        self.compare = compare

        # adjust constant
        def adjust_constant(labels, bestscore, const, upper_bound, lower_bound, kappa, targeted):
            mask = (self.compare_constant(bestscore, labels, targeted)) & (bestscore != -1)
            upper_bound = jnp.where(mask, jnp.minimum(upper_bound, const), upper_bound) # upper_bound[mask]
            lower_bound = jnp.where(~mask, jnp.maximum(lower_bound, const), lower_bound) # lower_bound[~mask]

            mask = upper_bound < 1e9
            const = jnp.where(mask, (lower_bound + upper_bound)/2, 10*const)

            return upper_bound, lower_bound, const

        def compare_constant(output, labels, targeted):
            result = jax.lax.cond(targeted, 
                                  lambda _: output == labels,
                                  lambda _: output != labels,
                                  operand=None)

            return result

        self.adjust_constant = adjust_constant
        self.compare_constant = compare_constant
        

    def jit_call(self):
        self._call = jax.jit(self._call)