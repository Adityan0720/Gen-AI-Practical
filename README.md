# **Generative AI Lab Experiments**

## **RAGHUVEER SINGH**
- **Roll No:** 2301420038
- **Email:** 2301420038@krmu.edu.in
- **Program:** B.Tech CSE (Data Science)
- **Semester:** 6

---

## **📋 Project Overview**
This repository contains a comprehensive suite of experiments conducted for the Generative AI Lab. It serves as a practical exploration of generative models, ranging from foundational neural networks to advanced architectures like GANs, VAEs, and transformer-based LLMs. The repository culminates in a final integrated project focusing on parameter estimation.

---

## **📐 Architecture & Workflow Pipeline**

```mermaid
graph TD
    subgraph Data & Setup
        A[Dataset Acquisition] --> B[Data Preprocessing & EDA]
    end
    
    subgraph Generative Image Models
        B --> C[Exp 1-3: Autoencoders & VAEs]
        C --> D[Exp 4-5: Generative Adversarial Networks]
    end
    
    subgraph Generative Text Models
        B --> E[Exp 6-8: Transformers & Fine-Tuning]
        E --> F[Exp 9-10: Prompt Engineering & RAG]
    end
    
    subgraph Culmination
        D --> G[project_parameter_estimation.py]
        F --> G
        G --> H((Final Analysis & Deployment))
    end
```

---

## **📂 Repository Structure & Reference Map**

| File | Topic & Focus | Description |
|------|--------------|-------------|
| **`experiment1.py`** | Environment Setup & EDA | Initial data loading, exploratory data analysis, and tensor operations. |
| **`experiment2.py`** | Multi-Layer Perceptrons | Building foundational neural networks for generative baselines. |
| **`experiment3.py`** | Autoencoders (AEs) | Dimensionality reduction and basic image reconstruction techniques. |
| **`experiment4.py`** | Variational Autoencoders (VAEs) | Probabilistic latent space modeling for content generation. |
| **`experiment5.py`** | Intro to GANs | Training basic Generative Adversarial Networks (Generator vs Discriminator). |
| **`experiment6.py`** | Advanced GANs (DCGANs) | Deep Convolutional GANs for high-resolution image synthesis. |
| **`experiment7.py`** | Sequential Data & RNNs | Text generation baselines using Recurrent Neural Networks. |
| **`experiment8.py`** | Transformer Architectures | Implementing attention mechanisms and utilizing pre-trained LLMs. |
| **`experiment9.py`** | Advanced Prompt Engineering | Techniques like Few-Shot, Chain-of-Thought (CoT), & Instruction Tuning. |
| **`experiment10.py`** | RAG (Retrieval-Augmented Gen) | Integrating external knowledge bases into generative text workflows. |
| **`project_parameter_estimation.py`** | **Final Consolidated Project** | **End-to-end integration, final evaluation metrics, and comprehensive analysis.** |

---

## **🛠️ Setup & Deployment Instructions**

To deploy and test these experiments locally, follow the steps below:

**1. Clone the repository & navigate to the workspace:**
```bash
git clone <repo-url>
cd Gen-AI
```

**2. Create and activate a Virtual Environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install Dependencies:**
*(Assuming standard data science and generative AI prerequisites)*
```bash
pip install torch torchvision transformers datasets diffusers pandas matplotlib scikit-learn numpy scipy seaborn
```

**4. Execute the experiments:**
You can run any experiment from the terminal like so:
```bash
python experiment1.py
```

---

## **🚀 Usage Guidelines**
- Execute the experiments sequentially (1 through 10) to accurately track the progression from foundational concepts to advanced generative models.
- Each script (`experiment<N>.py`) is robust and self-contained; verify that correct environments and requirements are active before running.
- Access the `project_parameter_estimation.py` for the comprehensive evaluation, overarching metrics, and overall conclusion of the complete pipeline.
