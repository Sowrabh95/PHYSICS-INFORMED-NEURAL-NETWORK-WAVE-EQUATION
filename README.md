# PHYSICS-INFORMED-NEURAL-NETWORK-WAVE-EQUATION
1D WAVE EQUATION USING PHYSICS-INFORMED NEURAL  NETWORKS (PINNS)

# 1D Wave Equation using Physics-Informed Neural Networks (PINNs)

This repository demonstrates solving the **1D wave equation** using **Physics-Informed Neural Networks (PINNs)** in Julia. The approach combines deep learning with physical laws to approximate solutions of partial differential equations (PDEs) while respecting boundary and initial conditions.

---

## 📖 Problem Statement
We solve the wave equation:



\[
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}, \quad c=1
\]



- **Domain:** \(t \in [0,1], x \in [0,1]\)  
- **Boundary Conditions:** Fixed ends at \(x=0\) and \(x=1\) with zero displacement  
- **Initial Condition:** Parabolic displacement profile with zero initial velocity  

---

## ⚙️ Implementation Details
- **Packages Used:** NeuralPDE.jl, Lux.jl, Optimization.jl, OptimizationOptimJL.jl, ModelingToolkit.jl  
- **Neural Network Setup:**
  - Input: \((t, x)\)  
  - Hidden layers: 2 layers × 16 neurons (sigmoid activation)  
  - Output: \(u(t,x)\)  
- **Training:** BFGS optimizer with residual-based loss function  
- **Evaluation:** PINN predictions compared against Fourier series analytic solution  

---

## 📊 Results
- Maximum absolute error: **0.0125**  
- PINN predictions closely match the analytic solution  
- Error localized in regions with sharp changes  

Generated plots:
- `analytic.png` → Analytic solution  
- `predict.png` → PINN prediction  
- `error.png` → Error distribution  

Data exports:
- `analytic.csv`  
- `predict.csv`  

---

## 🚀 Getting Started

### Prerequisites
- Julia (≥ 1.9 recommended)
- Install required packages:
  ```julia
  using Pkg
  Pkg.add(["NeuralPDE", "Lux", "Optimization", "OptimizationOptimJL", "ModelingToolkit"])

Running the Code

Clone the repository:
git clone https://github.com/your-username/1D-wave-PINN.git
cd 1D-wave-PINN


Run the main script:
include("wave_pinn.jl")

