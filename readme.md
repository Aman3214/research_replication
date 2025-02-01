# 🚀 NeRF Replication Project

This repository contains the **NeRF (Neural Radiance Fields)** model replication based on the original research paper. The implementation combines insights from the paper and additional guidance from online resources, including YouTube tutorials.

## 📂 File Structure

### 1️⃣ nerf/

- 📌 Contains the **vanilla NeRF implementation** replicated from the research paper.
- 🏋️‍♂️ Includes the **training code** used for learning scene representations.
- 💾 Stores the **weights** from the last training epoch for model evaluation.

### 2️⃣ novel_view/

- 🎥 Contains **generated novel views** from the **synthetic LEGO dataset**.
- 🗂️ The novel views are stored as **pickle files**:
  - 🏗️ `training.pkl` – Contains novel views used during training.
  - 🧪 `testing.pkl` – Contains novel views used for evaluation.

### 3️⃣ requirements.txt

- 📜 Lists all **additional dependencies** that must be installed before running the code.
- ⚙️ Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

## 🚀 Getting Started

1️⃣ **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/research_replication.git
   cd research_replication
   ```
2️⃣ **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ **Run the training script** (if applicable):
   ```bash
   python nerf/train.py
   ```
4️⃣ **Generate novel views** using the trained model.

## 📜 References

- 📖 **Original NeRF Paper:** [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

