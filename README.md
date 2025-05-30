# 🌸 Iris Flower Classification using Flux.jl

This project implements a neural network classifier to predict the species of Iris flowers using the famous **Iris dataset**, written in **Julia** with the [Flux.jl](https://fluxml.ai/) deep learning library.

It covers the complete machine learning pipeline: data preprocessing, model definition, training, evaluation, and visualization of results including accuracy trends and a confusion matrix.

---

## 📊 Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) contains 150 samples with the following features:

* Sepal length
* Sepal width
* Petal length
* Petal width
  Each sample is labeled with one of three species: *Iris-setosa*, *Iris-versicolor*, or *Iris-virginica*.

---

## 🧠 Model Architecture

The classifier is a fully connected neural network:

```julia
model = Chain(
    Dense(4, 16, relu),
    Dense(16, 8, relu),
    Dense(8, 3),
    softmax
)
```

* Optimizer: `Adam` with custom learning rate (0.01)
* Loss: `crossentropy`
* Activation: `ReLU` for hidden layers, `softmax` for output

---

## ⚙️ Workflow

1. **Load and preprocess the data** using `CSV.jl` and `DataFrames.jl`
2. **Normalize** input features column-wise (z-score)
3. **Encode** class labels using one-hot encoding
4. **Train** a 3-layer neural network using Flux
5. **Evaluate** on a held-out test set
6. **Visualize**:

   * Training Loss
   * Training Accuracy
   * Confusion Matrix

---

## 📈 Results

* Final test accuracy is printed at the end of training.
* Visualization files saved in the `results/` directory with timestamps:

  * `confusion_matrix_<timestamp>.png`
  * `loss_<timestamp>.png`
  * `accuracy_<timestamp>.png`

<p align="center">
  <img src="results/example_confusion_matrix.png" width="400" alt="Confusion Matrix">
</p>

---

## 🛠 Technologies Used

* **Julia**
* [Flux.jl](https://fluxml.ai/) – Neural networks
* [CSV.jl](https://github.com/JuliaData/CSV.jl) – Data loading
* [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) – Data manipulation
* [Plots.jl](http://docs.juliaplots.org/latest/) – Visualization
* [MLUtils.jl](https://github.com/FluxML/MLUtils.jl) – Data splitting/shuffling

---

## 🚀 Getting Started

### Requirements

* Julia ≥ 1.6
* Install dependencies:

```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Flux", "MLUtils", "Statistics", "Random", "Plots"])
```

### Run the code

Make sure `iris.csv` is in the same directory. Then simply execute:

```julia
include("iris_classification.jl")
```

---

## 📁 Project Structure

```
.
├── iris.csv                   # Input dataset
├── iris_classification.jl     # Main Julia script
└── results/
    ├── confusion_matrix_<timestamp>.png
    ├── loss_<timestamp>.png
    └── accuracy_<timestamp>.png
```

---

## 📌 License

This project is open source under the MIT License.

