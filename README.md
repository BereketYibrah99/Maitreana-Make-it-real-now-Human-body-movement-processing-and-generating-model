# Maitreana

*"Make it real now."*

**Maitreana** is an experimental AI project designed to process and generate human body language movements based on input angular position sequences. 

The project is currently in a prototype stage, working at **algorithm level**. The long-term vision is bold: enabling AI to generate **full character movements and body language** for entire movie scenes, driven by a story prompt — moving beyond traditional frame-by-frame animation.

---

## Project Status

🚀 Prototype stage  
⚠️ Not an API or packaged app yet — algorithm-level prototype  
✅ Early tests successful on one type of body language  

---

## How It Works

### Input

- Sequence of **89-dimensional vectors** representing angular positions.
- The first **4 dimensions** represent **hand part angular positions** (skeleton parts of the hand), not including wrist or fingers.
- **89 vectors** represent a sequence over time — the movement.

### Preprocessing Pipeline

#### 1️⃣ First pass:

- Apply **Discrete Sine Transform (DST)** to each of the 89 input vectors.
- Concatenate the transformed vectors.

#### 2️⃣ Second pass:

- Apply **DST again** to the concatenated vector.
- Feed the result into the model.

### Model

- **Variational Autoencoder (VAE)**  
- Built using **Python** + **TensorFlow**

### Output Pipeline

#### Post-processing:

- Apply **Inverse DST (IDST)** twice:
    - First on the model output.
    - Then again, in the reverse order of preprocessing.

- The final result is a **clear output sequence** of angular positions, representing generated body language movement.

---

## Usage

⚠️ Prototype code — not yet an API or packaged module.

### Requirements

- TensorFlow
- NumPy
- SciPy


