import jax
import jax.numpy as jnp
import optax

from ..attack import Attack

class PIFGSM(Attack):
    r"""
    PIFGSM in the paper 'Patch-wise Attack for Fooling Deep Neural Network'
    [https://arxiv.org/abs/2007.06765]

    Distance Measure : Linf

    Arguments:
        max_epsilon (float): maximum size of adversarial perturbation. (Default: 16/255)
        num_iter_set (float): number of iterations. (Default: 10)
        momentum (float): momentum. (Default: 1.0)
        amplification (float): to amplifythe step size. (Default: 10.0)

    Shape:
        - images: :math:`(N, H, W, C)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, H, W, C)`.
    """

    def __init__(self, max_epsilon=16/255, num_iter_set=10, momentum=1.0, amplification=10.0):
        super().__init__("PIFGSM")
        self.max_epsilon = max_epsilon
        self.num_iter_set = num_iter_set
        self.momentum = momentum
        self.amplification = amplification
        self.supported_modes = ["default"]

        self.create_functions()

    def main_attack_function(self, images, labels):

        return self._call(images, labels, self.max_epsilon, self.num_iter_set, self.momentum, self.amplification)
    
    def _call(self, images, labels, max_epsilon, num_iter_set, momentum, amplification):        
        
        adv_images = self.graph(images, labels, max_epsilon, num_iter_set, momentum, amplification)

        adv_images = jnp.clip(adv_images, 0.0, 1.0)
        return adv_images


    def create_functions(self):
        def calculate_loss(imgs, lbls):
            logits = self.get_logits(imgs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, lbls).mean()
            return loss

        self.calculate_loss = calculate_loss
        self.grad_fn = jax.value_and_grad(self.calculate_loss)

        ## 
        def project_noise(images, P_kern, groups_number=3, stride=(1, 1)):
            # images.shape =  batch_size, H, W, C
            # JAX kernel: (kH, kW, in_c_per_group, out_c)
            padding_size = P_kern.shape[0]//2
            padding = [(padding_size, padding_size), (padding_size, padding_size)]

            convolved = jax.lax.conv_general_dilated(
                lhs=jnp.float32(images),
                rhs=P_kern,
                window_strides=stride,
                padding=padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=groups_number
            )

            return convolved
            

        self.project_noise = project_noise

        def project_kern(kern_size):
            kern = jnp.ones((kern_size, kern_size), dtype=jnp.float32) / (kern_size ** 2 - 1)
            kern = kern.at[kern_size // 2, kern_size // 2].set(0.0) # (kH, kW)

            # Stack per group (like 3-channel groups)
            stack_kern = jnp.stack([kern, kern, kern], axis=2)  # (kH, kW, 3)
            stack_kern = jnp.expand_dims(stack_kern, 2)  # (kH, kW, 1, 3)

            return stack_kern

        self.project_kern = project_kern

        

        def graph(images, labels, max_epsilon, num_iter_set, momentum, amplification):
            adv_images = jnp.copy(images)
            eps = max_epsilon
            num_iter = num_iter_set
            alpha = eps / num_iter
            alpha_beta = alpha * amplification
            gamma = alpha_beta
            P_kern = self.project_kern(3)

            amplification = jnp.zeros_like(images, dtype=jnp.float32)

            # For loop
            x = {
                "ori_images": images,
                "adv_images": adv_images,
                "labels": labels,
                "alpha_beta": alpha_beta,
                "amplification": amplification,
                "eps": eps,
                "gamma": gamma,
                "P_kern": P_kern,
            }

            x = jax.lax.fori_loop(0, num_iter, self.graph_for, x)
            adv_images = x["adv_images"]

            return adv_images

        self.graph = graph
                    
        def graph_for(_, fori_dict):
            loss_to_print, noise = self.grad_fn(fori_dict["adv_images"], fori_dict["labels"])

            fori_dict["amplification"] += fori_dict["alpha_beta"] * jnp.sign(noise)
            cut_noise = jnp.clip(jnp.abs(fori_dict["amplification"]) - fori_dict["eps"], 0.0, 10000.0) * jnp.sign(fori_dict["amplification"])

            projection = fori_dict["gamma"] * jnp.sign(
                self.project_noise(cut_noise, fori_dict["P_kern"])
            )
            fori_dict["amplification"] += projection
            fori_dict["amplification"] = jnp.float32(fori_dict["amplification"])

            fori_dict["adv_images"] = fori_dict["adv_images"] + fori_dict["alpha_beta"] * jnp.sign(noise) + projection
            delta = jnp.clip(fori_dict["adv_images"] - fori_dict["ori_images"], -fori_dict["eps"], fori_dict["eps"] )
            fori_dict["adv_images"] = jnp.clip(fori_dict["ori_images"] + delta, 0.0, 1.0)

            return fori_dict

        self.graph_for = graph_for

        
    def jit_call(self):
        self._call = jax.jit(self._call)
        
