from tqdm import tqdm
import argparse

import torch
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial

import jax
import jax.numpy as jnp

import flax
import flaxmodels

from jaxattacks import FGSM, BIM, CW, RFGSM, PGD, PGDL2, MIFGSM, TPGD, APGD, APGDT, FAB, FABTargeted, FABMulti, Square, DeepFool, DIFGSM, NIFGSM, SINIFGSM, VMIFGSM, SPSA, SparseFool, EAD, PIFGSM, PIFGSMPP, Jitter, VNIFGSM, JSMA

from utils import transform_func_normalize_numpy, transform_func_numpy, numpy_collate, save_images

# Arguments
parser = argparse.ArgumentParser(description='Attack Test')

parser.add_argument("--jit-model", action="store_true", help="Enable jitted model evaluation")
parser.add_argument("--jit-attack", action="store_true", help="Enable jitted attack")
parser.add_argument("--resnet", type=int, default=50,
                        choices=[18, 34, 50, 101, 152])
parser.add_argument("--dataset-folder", type=str)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--attack", type=str, default="FGSM")
parser.add_argument("--epsilon", type=float, default = 0)

args = parser.parse_args()
args.epsilon = args.epsilon / 255

ATTACKS = {
    "FGSM": FGSM,
    "PGD": PGD,
    "CW": CW,
}

# Raw Accuracy 
def get_attack_imagenet_jax_resnet50_accuracy():

    print("Attack evaluate")
    jax.config.update("jax_enable_x64", True)
    
    ## Model load
    main_rng = jax.random.PRNGKey(seed=42)
    resnet_map = {
        18: flaxmodels.ResNet18,
        34: flaxmodels.ResNet34,
        50: flaxmodels.ResNet50,
        101: flaxmodels.ResNet101,
        152: flaxmodels.ResNet152,
    }
    model = resnet_map[args.resnet](
        output='logits',
        pretrained='imagenet',
        normalize=False,
        ckpt_dir="./flax_ckpts"
    )
    main_rng, resnet_rng = jax.random.split(main_rng, 2)
    model_variables = model.init(resnet_rng, jnp.zeros((1, 224, 224, 3)))

    # JIT Model
    @partial(jax.jit, static_argnames=["train", "mutable"])
    def model_apply_jitted(prms, imgs, train, mutable):
        return model.apply(jax.lax.stop_gradient(prms), imgs, train=train, mutable=mutable)
        
    # Attack Init
    attack_model = ATTACKS[args.attack](eps=args.epsilon)
    attack_model.set_forward(
        lambda x: model.apply(jax.lax.stop_gradient(model_variables), x, train=False, mutable=False)
    )
    attack_model.set_normalization_used([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # JIT Attack
    if args.jit_attack:
        attack_model.jit_call()
    
        
    # Init Dataset
    val_dataset = torchvision.datasets.ImageFolder(args.dataset_folder, transform=transform_func_normalize_numpy())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=numpy_collate)
    
    
    exmp_imgs_batch, exmp_lbls_batch = next(iter(val_loader))
    print(f"Image batch shape = {exmp_imgs_batch.shape}")
    print(f"Labels batch shape = {exmp_lbls_batch.shape}")

    ## Model running
    gt_correct = 0
    adv_correct = 0
    total = 0 

    print(f"Attack name = {attack_model.name}")    
    print("Start Evaluating...")
    
    
    for img_batch, lbl_batch in tqdm(val_loader):
            
        ## Ground truth
        if args.jit_model:
            out = model_apply_jitted(model_variables, img_batch, False, False)
        else:
            out = model.apply(model_variables, img_batch, train=False, mutable=False)
        pred = jnp.argmax(out, axis=1)

        ## Adversarial
        adv_batch = attack_model(img_batch, lbl_batch)
            
        if args.jit_model:
            adv_logits = model_apply_jitted(model_variables, adv_batch, False, False)
        else:
            adv_logits = model.apply(model_variables, adv_batch, train=False, mutable=False)
            
        
        adv_pred = jnp.argmax(adv_logits, axis=1)

        ## Statistics
        gt_correct += jnp.sum(pred == lbl_batch)
        adv_correct += jnp.sum(adv_pred == lbl_batch)
        total += pred.shape[0]
        
        #print(f'GT  Validation score : {(float(gt_correct)/float(total))*100}')
        #print(f"Predicted label = {pred}")
        #print(f'Adv Validation score : {(float(adv_correct)/float(total))*100}')
        #print(f"Predicted label = {adv_pred}")
        #save_images(attack_model.inverse_normalize(img_batch)[0], attack_model.inverse_normalize(adv_batch)[0], "JAXAttack_Comparison")
        #exit(0)

    print(f'[JAX ResNet50]')
    print(f'GT  Validation score : {(float(gt_correct)/float(total))*100}')
    print(f'Adv Validation score : {(float(adv_correct)/float(total))*100}')



if __name__ == "__main__":
    get_attack_imagenet_jax_resnet50_accuracy()
