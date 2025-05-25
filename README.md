# 🔁 Equivariant Neural Networks (ENNs)

A modular PyTorch library for building neural networks that respect the symmetries of data — using principles from group theory, Lie groups, and gauge theory. Supports Group Equivariant CNNs (G-CNNs), Lie group-equivariant models (SO(3), SE(3)), and Gauge Equivariant CNNs for applications in computer vision, molecular modeling, and physics-informed learning.

---

## 📌 Features

- ✅ **Group Equivariant CNNs (G-CNNs)** — for rotation and reflection equivariance in 2D images
- ✅ **Lie Group-equivariant networks** — for continuous symmetries like 3D rotations and rigid body motions
- ✅ **Gauge Equivariant CNNs** — for data on curved manifolds or with local symmetries
- 📚 Educational Jupyter notebooks and visualizations
- 🧪 Benchmarks on datasets like RotMNIST, QM9, CIFAR10, and 3D point clouds

---

## 🧠 Why Equivariance?

Many datasets have inherent symmetries — like images rotated in space, molecules invariant to atom order, or physical systems that follow conservation laws. Standard neural networks often ignore these, requiring more data and learning redundant patterns. ENNs build **symmetry directly into the architecture**, improving efficiency, generalization, and interpretability.

---

## 📂 Project Structure
<pre>
equivariant-nn/
├── equivariant_nn/
│   ├── layers/
│   │   ├── gcnn_layers.py         # G-CNNs
│   │   ├── se3_layers.py          # Lie group equivariant layers
│   │   ├── gauge_layers.py        # Gauge equivariant layers
│   ├── models/
│   │   ├── mnist_rot_model.py
│   │   ├── qm9_model.py
│   │   └── shape_model.py
│   ├── utils/
│   │   ├── group_ops.py
│   │   └── data_utils.py
├── experiments/
│   ├── train_mnist_rot.py
│   ├── train_qm9.py
├── notebooks/
│   ├── 01_gcnn_mnist_tutorial.ipynb
│   ├── 02_se3_equivariance_qm9.ipynb
│   └── 03_gauge_equivariance_on_sphere.ipynb
├── data/                         # Scripts or links to datasets
├── tests/                        # Unit tests
├── requirements.txt
├── setup.py
└── README.md
</pre>
1. Clone the repo
<pre>
git clone https://github.com/your-username/equivariant-nn.git
cd equivariant-nn
</pre>
3. Install dependencies
<pre>
  pip install -r requirements.txt
</pre>
4. Run a demo (e.g., RotMNIST)
<pre>
  python experiments/train_mnist_rot.py
</pre>
I'm adding a few of my notes on group theory just to provide a mathematical foundation over what we are doing apart from the codes, but it is highly recommended repo is viewed by people who already have a good understanding of group theory. Although CS grads would anyway find their way out even if they don't know the mathematical background of this repo.
[📘 View Group Theory Notes (PDF)](GCNNs.pdf)
(Really Sorry for the first page being a bit shaky)
