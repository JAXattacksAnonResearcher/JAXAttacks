import jax
import jax.numpy as jnp


class EnsembleAttack:
    r"""
    Base class for all attacks:
        For now, it just accept models.
    
    """
    def __init__(self, name):
        self.name = name

        # Normalization
        self.normalization_used = None
        self._normalization_applied = False

        # Attack modes
        self.supported_modes = ["default"] # Default mode is untargeted
        self.attack_mode = "default"
        self.targeted = False
        self._target_map_function = None
    
    def __call__(self, inputs, labels):
        if self._normalization_applied is True:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs = self.main_attack_function(inputs, labels)

            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.main_attack_function(inputs, labels)
        
        return adv_inputs

    def main_attack_function(self, inputs, labels):
        raise NotImplementedError

    def get_all_logits(self, inputs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        
        logits_list = [fn(inputs) for fn in self.forward_fns]    
        return jnp.stack(logits_list, axis=0)

    def get_mode(self):
        return self.attack_mode

    def set_mode_default(self):
        self.attack_mode = "default"
        self.targeted = False
        print("Attack mode is changed to 'default'")
    
    def set_mode_targeted(self):
        if "targeted" in self.supported_modes:
            self.attack_mode = "targeted"
            self.targeted = True
            print("Attack mode is changed to 'targeted'")
        else:
            print("Targeted does not exist in supported modes")

    def set_normalization_used(self, mean, std):
        r"""
        Use it in case if your model uses normalized image data and your dataloader is normalizing the images.
        
        Ex:
        > val_dataset = torchvision.datasets.ImageFolder(..., transform=transform_func()) # Assume transform func normalize data
        >
        > ... Init attack model and call 'set_normalization_used()'
        > 
        > out = model.apply(model_variables, img_batch, train=False, mutable=False)
        > adv_batch = attack_model(img_batch, lbl_batch)
        > adv_logits = model.apply(model_variables, adv_batch, train=False, mutable=False)
        """
        self._set_normalization_applied(True)
        
        n_channels = len(mean)
        mean = jnp.array(mean)[None, None, :] # Assuming input shape [Batch size, Height, Width, Channel]
        std = jnp.array(std)[None, None, :]
        
        self.normalization_used = {"mean": mean, "std": std}

        def normalize(inputs):
            return (inputs - mean) / std

        self.normalize = jax.jit(normalize)

        def inverse_normalize(inputs):
            return (inputs * std) + mean

        self.inverse_normalize = jax.jit(inverse_normalize)

    def normalization_done_by_user(self, flag=False):
        """
        Use it in case, if your dataloader does not normalize the data
        but you explicitly normalize it before giving to model or attack_model
        
        Ex:
        > val_dataset = torchvision.datasets.ImageFolder(..., transform=transform_func()) # Assume transform func does not have normalization
        >
        > ... Init attack model and call 'set_normalization_used()' and 'normalization_done_by_user()' 
        > 
        > out = model.apply(model_variables, attack_model.normalize(img_batch), train=False, mutable=False)
        > adv_batch = attack_model(img_batch, lbl_batch)
        > adv_logits = model.apply(model_variables, attack_model.normalize(adv_batch), train=False, mutable=False)
        """
        if flag:
            self._set_normalization_applied(False)
        else:
            self._set_normalization_applied(True)
        
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag
    
    def set_forwards(self, forward_fns):
        self.forward_fns = forward_fns
        self.models_count = len(forward_fns)

    def set_models_name(self, models_name):
        self.models_name = models_name