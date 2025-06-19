 Predictive Coding Neural Network for MNIST Classification

 🧠 Objective  
This project implements a minimalist predictive coding neural network to classify handwritten digits from the MNIST dataset. It simulates biologically inspired computation by using prediction errors across network layers, latent state refinement, and lateral connections to encode spatial structure in the input.

---

🔬 Theoretical Foundations  
Predictive coding is a neuro-inspired approach where higher layers attempt to predict the activity of lower layers. Learning occurs through **prediction errors**—the mismatch between actual and predicted signals—used to fine-tune the model's internal representations and parameters.

**Key Concepts:**  
- *Top-down prediction:* Each layer predicts the activity of the layer below.  
- *Bottom-up error signaling:* Discrepancies are passed upward as error signals.  
- *Latent updates:* Internal states are iteratively adjusted to minimize error.  
- *Weight updates:* Weights are updated via Hebbian or gradient-based learning.  
- *Lateral connections:* Neurons in the same layer interact to promote local consistency.

---

 🏗️ Model Architecture  
- **Input Layer:** 784 units (flattened 28×28 grayscale image)  
- **Hidden Layer:** 256 units (abstract latent representation)  
- **Output Layer:** 10 units (digit class probabilities)

**Special Features:**  
- **Error computation:** `error = actual - predicted` at each layer  
- **Latent refinement:** Activation states are optimized iteratively  
- **Hebbian-style learning:** Local weight updates driven by error signals  
- **Lateral interactions:** Spatial smoothing via learned neighborhood influences

---

## ⚙️ Implementation Workflow

| Step | Action |
|------|--------|
| 1    | Load and normalize MNIST data; flatten images |
| 2    | Initialize weights and lateral matrices |
| 3    | Forward pass with hierarchical prediction |
| 4    | Compute prediction errors per layer |
| 5    | Refine latent states iteratively |
| 6    | Update weights based on accumulated errors |
| 7    | Apply lateral interactions to reinforce local features |
| 8    | Train over several epochs on mini-batches |

---

## 📉 Results and Observations  

| Epoch | Accuracy |
|-------|----------|
| 1–7   | 9.87%    |

**Observations:**  
- Accuracy remained near 10%, equivalent to random guessing  
- **Potential issues:**  
  - Weak learning signals due to vanishing gradients or poor weight initialization  
  - Output layer lacked clear class supervision  
  - Use of softmax-based prediction error instead of cross-entropy may limit discrimination  
  - Lateral connections had minimal influence in current setup

---

## 🧪 Conclusion  
Despite limited classification accuracy, this project effectively demonstrates the essential principles of predictive coding. With enhancements—such as better error modeling, learning rate tuning, and stronger label feedback—this framework holds promise for bridging neuroscience and machine learning paradigms.

