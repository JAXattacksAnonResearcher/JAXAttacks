# JAXAttacks

**JAXAttacks** is a JAX-native library for generating adversarial examples, designed specifically for JAX-based vision models.

It leverages `jax.jit` and `jax.lax` primitives to enable **fully compiled adversarial attack pipelines**, eliminating Python-side overhead and improving execution efficiency on GPUs/TPUs.

---

## 🚀 Why JAXAttacks?

Most existing adversarial attack libraries (e.g., Foolbox, ART, CleverHans):

- Are not fully JAX-native  
- Rely on Python control flow or NumPy operations  
- Cannot fully benefit from JAX transformations like `jit`
- Have limited attack coverage  

**JAXAttacks is built from the ground up for JAX**, using:

- `jax.lax.fori_loop` for iterative attacks  
- `jax.lax.cond` for conditional logic  
- Fully traceable, JIT-compatible implementations  

This enables **end-to-end XLA compilation** of attack pipelines.

---

## ⚡ Performance Highlights

Based on benchmarking across multiple models and batch sizes:

- **+10% faster** than Foolbox for multi-step attacks (e.g., PGD-10)  
- **+30–40% faster** for FGSM at large batch sizes  
- **Up to 3–4× faster** for small batch sizes  

Performance gains come from eliminating Python overhead and enabling compiler-level optimization.

---

## 📦 Installation

Currently not available via `pip`.

```bash
git clone https://github.com/JAXattacksAnonResearcher/JAXAttacks.git
```

---

## 🧪 Quick Example

```python
import jax
import jaxattacks

attack = jaxattacks.FGSM(eps=8.0/255)

attack.set_forward(
    lambda x: model.apply(jax.lax.stop_gradient(model_variables), x)
)

adv_images = attack(images, labels)
```

**Note:**  
If your model expects normalized inputs, call:

```python
attack.set_normalization_used(mean, std)
```

---

## 🧠 Supported Attacks

### Gradient-Based
- FGSM, BIM (I-FGSM), PGD  
- MI-FGSM and variants  

### Optimization-Based
- CW  
- EAD  
- DeepFool

### AutoAttack Components
- APGD, APGDT  
- FAB  
- Square Attack  

### Sparse Attacks
- JSMA 
- SparseFool  

### BlackBox 
- SPSA 
- Square Attack  

> Total: **20+ attacks / 27 variants**

---

## 🏗️ Design Principles

JAXAttacks is designed as a **compilation-aware attack library**:

- ✅ JAX-native implementations (no Python loops in core logic)  
- ✅ JIT-compatible attack pipelines  
- ✅ Static-shape friendly design  
- ✅ Modular and extensible architecture  
- ✅ Forward-function injection (framework-agnostic)  

The API is inspired by the transition from `Flax.linen` to `Flax.nnx`, combining **modularity with JAX compatibility**.

---

## ⚠️ Limitations

- **Static shape constraint (JIT)**  
  Dynamic-shape attacks (e.g., random patch-based methods) are not fully supported.

- **No dynamic batch shrinking**  
  Iterative attacks use masking instead of removing converged samples, which may introduce extra computation.

These are largely **framework-level limitations** and open areas for improvement.

---

## 🔨 Requirements

PyTorch's data loader and torchvision transforms are used to process data. Hence, they are not core requirement for our library but necessary for example script.

**Core Requirements**
- Python >= 3.12  
- JAX >= 0.7.0  
- Optax >= 0.2.6  

**Demo / Examples (optional)**
- PyTorch (for dataloaders) >= 2.8.0
- torchvision  >= 0.23.0
- matplotlib  >= 3.10.8
- tqdm >= 4.67.3
- flax >= 0.11.2
- flaxmodels >= 0.1.3

---

## 🧪 Getting Started

Example script:

```
python imagenet_jaxattack_test.py --resnet 18 --attack FGSM --epsilon 8.0 --dataset-folder /datasets/ImageNet/ILSVRC/Data/CLS-LOC/val/ --jit-model --jit-attack
```

This demonstrates adversarial attack generation on ResNet18 model on ImageNet validation dataset using JAXAttacks FGSM (eps 8/255) implementation jitted.

---

## 🤝 Contributing

Contributions are welcome, especially for:

- New attack implementations  
- Performance optimizations  
- Dynamic-shape support  

---

## 📌 Notes

- This is an active development version  
- API may change before official release  
- `pip` package support is planned  
