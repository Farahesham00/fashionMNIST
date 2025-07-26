
# 🧥 FashionMNIST Classification - Linear, Non-Linear, and CNN Models

This project demonstrates and compares the performance of **three different PyTorch models** on the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist):  
1. `model_0`: Linear Model  
2. `model_1`: Non-Linear Model (MLP)  
3. `model_2`: Convolutional Neural Network (CNN)

---

## 📦 Dataset

- **FashionMNIST**: A dataset of 28×28 grayscale images across 10 clothing categories.
- Downloaded automatically using `torchvision.datasets.FashionMNIST`
- Normalized and transformed into tensors using `ToTensor()`.

---

## 🧠 Model Architectures

### 🧪 `model_0`: Linear Model
A basic linear classifier using only fully connected layers.

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)
```

### 🔁 `model_1`: Non-Linear MLP
Adds a hidden layer and ReLU activations to introduce non-linearity.

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU()
)
```

### 🧠 `model_2`: CNN (TinyVGG-style)
A convolutional architecture inspired by TinyVGG. Uses two convolutional blocks followed by a fully connected classifier.

```python
Conv2D → ReLU → Conv2D → ReLU → MaxPool  
→ Conv2D → ReLU → Conv2D → ReLU → MaxPool  
→ Flatten → Linear
```

---

## 🕓 Training Time Comparison

We record and compare the training time of all three models to evaluate their performance-efficiency tradeoff:

| Model     | Parameters | Accuracy | Training Time |
|-----------|------------|----------|----------------|
| model_0   | Linear     | Low      | Very Fast      |
| model_1   | MLP        | Medium   | Fast           |
| model_2   | CNN        | High     | Slower         |

---

## 📊 Evaluation

All models are trained on the same training loop and evaluated on test data.

Evaluation steps include:

- 📉 **Accuracy Calculation** on training and test datasets  
- 🔮 **Random Predictions** using the best model on unseen test images  
- 🧠 **Confusion Matrix** for visual analysis of model performance across classes  
- 💾 **Model Saving/Loading**: Best model (`model_2`) is saved and reloaded using:

```python
torch.save(best_model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

---

## 🔍 Sample Code Snippet - Flatten Layer

Example of flattening an image before passing it to a linear model:

```python
flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)
print(f"Before: {x.shape}, After: {output.shape}")
```

---

## 📁 File Structure (Example)
```
fashionmnist_comparison/
├── models/
│   ├── model_0.py       # Linear model definition
│   ├── model_1.py       # Non-linear MLP definition
│   ├── model_2.py       # CNN definition
├── utils.py             # Helper functions (accuracy, timer, confusion matrix)
├── train.py             # Shared training loop
├── evaluate.py          # Prediction and evaluation (random images + confusion matrix)
├── model.pth            # Saved best performing model (CNN)
├── README.md            # This file
```

---

## ✅ Requirements

Install required libraries:
```bash
pip install torch torchvision matplotlib
```

---

## ✨ Features Summary

✔️ Three models (Linear, MLP, CNN)  
✔️ Performance and training time comparison  
✔️ Best model prediction visualization  
✔️ Confusion matrix analysis  
✔️ Model checkpoint saving and reloading

---

## 🙌 Credits

- Code inspired by **[@sentdex](https://www.youtube.com/@sentdex)** PyTorch tutorials.
- Dataset by Zalando Research ([FashionMNIST](https://github.com/zalandoresearch/fashion-mnist))

---

## 🏁 Conclusion

This project is an excellent introduction to deep learning with PyTorch. It illustrates how model complexity impacts accuracy and training time, and how evaluation tools like confusion matrices help in understanding classifier behavior.
