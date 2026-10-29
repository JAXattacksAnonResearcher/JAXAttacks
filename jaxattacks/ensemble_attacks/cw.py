import jax
import jax.numpy as jnp
import optax

from ..ensemble_attack import EnsembleAttack
from ..utils import mode

class CWEnsemble(EnsembleAttack):
    r"""
    CW paper: 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        c (float): c in the paper. parameter for box-constraint. (Default: 1)
            :math:'minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))'
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
        check_every_step (int) : step interval to check for early stopping. If set to between 1 and steps (both inclusive), the algorithm will evaluate whether to stop every 'check_every_step' steps. If value is not specified or given smaller than 0, then algorithm will check 10 times. If value is bigger than step, it will throw error.

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """
    
    def __init__(self, c = 1, kappa = 0, steps = 50, lr = 0.01, check_every_step = 0):
        super().__init__("CWEnsemble")
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        assert check_every_step <= steps, f"check_every_step ({check_every_step}) should be smaller than steps ({steps})"
        self.checkpoint_count = 10 if check_every_step <= 0 else steps // check_every_step # default checkpoint frequency is 10 times 
        self.supported_modes = ["default", "targeted"]
        
        self.create_functions()
        
    def main_attack_function(self, images, labels):
        
        return self._call_while_loop(images, labels, self.targeted, self.c, self.kappa, self.steps, self.lr, self.checkpoint_count)
        
    def _call_while_loop(self, images, labels, targeted, c, kappa, steps, lr, checkpoint_count):
        
        ori_images = images
        w = self.inverse_tanh_space(images)
        batch_size = images.shape[0]
        
        best_adv_images = jnp.copy(images)
        best_L2 = 1e10 * jnp.ones(images.shape[0])
        prev_cost = 1e10
        
        opt_state = self.optimizer.init(w)   
           
        cur_step = 0
        cost = 0.0
        current_L2 = jnp.array([0.0] * batch_size, dtype=jnp.float32)
        
        x = {"ori_images": ori_images,
             "w": w,
             "best_adv_images": best_adv_images,
             "best_L2": best_L2,
             "current_L2": current_L2,
             "prev_cost": prev_cost,
             "cost": cost,
             "opt_state": opt_state,
             "cur_step": cur_step,
             "labels": labels,
             "targeted": targeted,
             "c": c,
             "kappa": kappa,
             "steps": steps,
             "lr": lr,
             "checkpoint_count": checkpoint_count,
        }
             
        x = jax.lax.while_loop(self.cond_fn, self.body_fn, x)
        best_adv_images = jnp.clip(x["best_adv_images"], 0.0, 1.0)
        
        return best_adv_images       
        
        
            
    def create_functions(self):
        ## Loss Function
        def calculate_loss(w, imgs, lbls, c, kappa, targeted): 
            # Get adversarial images
            adv_images = self.tanh_space(w)
            
            # Calculate loss
            current_L2 = jnp.sum(optax.losses.squared_error(jnp.reshape(imgs, (imgs.shape[0], -1)),
                                                            jnp.reshape(adv_images, (imgs.shape[0], -1))), axis=1)
            L2_loss = jnp.sum(current_L2)
            
            # Multiple logits coming from models
            # We pass each logit to 'f' function and get mean f_loss
            outputs = self.get_all_logits(adv_images)
            
            _, all_f_losses = jax.lax.scan(lambda _, x: (_, jnp.sum(self.f(x, lbls, targeted, kappa))),
                                           0, # dummy init for carry
                                           outputs)
            f_loss = jnp.mean(all_f_losses)       
            
            cost = L2_loss + c * f_loss
            return cost, (outputs, current_L2, adv_images)
            
        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss, has_aux=True) # ((loss_value, auxillary_data), gradients)
        
        ## Optimizer 
        optimizer = optax.adam(learning_rate=self.lr)
        self.optimizer = optimizer
        
        ## cond_fn and body_fn for 'jax.lax.while_loop'
        def cond_fn(x):
            mod_step = jnp.mod(x["cur_step"], jnp.maximum(jnp.floor_divide(x["steps"], x["checkpoint_count"]), 1))
            cond1 = jnp.logical_and(mod_step == 0, x["cost"] > x["prev_cost"])
            cond2 = (x["cur_step"] + 1) == x["steps"]       
                
            
            return jnp.logical_not(jnp.logical_or(cond1, cond2))
            
        self.cond_fn = cond_fn
        
            
        def body_fn(x):
            
            (x["cost"], (outputs, x["current_L2"], adv_images)), grads = self.grad_fn(x["w"], x["ori_images"], x["labels"], x["c"], x["kappa"], x["targeted"])
            x["current_L2"] = jnp.float32(x["current_L2"])
            
            ## Optimizer logic
            updates, x["opt_state"] = self.optimizer.update(grads, x["opt_state"])
            x["w"] = optax.apply_updates(x["w"], updates)
            
            # Update adversarial images
            
            # Mutliple predictions comming from different models
            # We take the softmax predictions of these models
            # Then majority voting to get final predictions
            # If scores are the same we chose the first label so order your models from best to worst
            preds =  jnp.argmax(outputs, axis=-1)
            pred = mode(preds)
            condition = jax.lax.cond(x["targeted"],
                                     lambda _: pred == x["labels"], 
                                     lambda _: pred != x["labels"],
                                     operand=None)
            
            # Filter out images
            mask = condition * (x["best_L2"] > x["current_L2"])
            x["best_L2"] = jnp.where(mask, x["current_L2"], x["best_L2"]) # x["best_L2"] = mask * x["current_L2"] + (1 - mask) * x["best_L2"] # #
            
            
            mask = jnp.reshape(mask, [-1] + [1] * (jnp.ndim(x["ori_images"])-1)) 
            x["best_adv_images"] = jnp.where(mask, adv_images, x["best_adv_images"]) # x["best_adv_images"] = mask * adv_images + (1 - mask) * x["best_adv_images"]


            # Update prev_cost if necessary
            mod_step = jnp.mod(x["cur_step"], jnp.maximum(jnp.floor_divide(x["steps"], x["checkpoint_count"]), 1))
            cond1 = jnp.logical_and(mod_step == 0, x["cost"] <= x["prev_cost"])
            
            
            x["prev_cost"] = jax.lax.cond(cond1,
                                          lambda _: x["cost"],
                                          lambda _: x["prev_cost"],
                                          operand=None)
             
            # Update cur_step          
            x["cur_step"] = x["cur_step"] + 1
        
            return x 
            
        self.body_fn = body_fn
    
        ## Helper functions
        def inverse_tanh_space(x):
            return jnp.arctanh(jnp.clip(x * 2 -1, min=-1, max=1))
            
        self.inverse_tanh_space = inverse_tanh_space
        
            
        def tanh_space(x):
            return 0.5 * (jnp.tanh(x) + 1)
            
        self.tanh_space = tanh_space
        
        
        def f(outputs, labels, targeted, kappa):
            one_hot_labels = jnp.eye(outputs.shape[1])[labels]
            
            # find the max logit other than the target class
            other = jnp.max((1 - one_hot_labels) * outputs, axis=1) 
            
            # get the target class's logit
            real = jnp.max(one_hot_labels * outputs, axis=1) 
            
            result = jax.lax.cond(targeted,
                         lambda _: jnp.clip((other - real), min=-kappa),
                         lambda _: jnp.clip((real - other), min=-kappa),
                         operand=None)
            
            return result
            
        self.f = f
        
    
    def jit_call(self):
        self._call_while_loop = jax.jit(self._call_while_loop)
            
         