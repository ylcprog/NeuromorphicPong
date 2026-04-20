

<h1>NeuromorphicPong</h1>
<p>Simulate advanced cognitive functions and evaluate executive control through a Nengo-powered neuromorphic AI playing a dynamic Pong game.</p>

<p>
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

---

## The Strategic "Why" (Overview)

> The inherent complexity of assessing human executive functions, coupled with the limitations of traditional computational models, often hinders a deeper understanding of cognitive processes and the development of biologically plausible AI. Current approaches frequently lack the dynamic, real-time adaptability needed for comprehensive behavioral and neural analysis, making it challenging to model nuanced cognitive behaviors.

NeuromorphicPong addresses this by providing a sophisticated Python framework that integrates Nengo to build and run neuromorphic models within a modified Pong game environment. This project offers a unique, dynamic platform for researchers and AI developers to evaluate executive functions such as attention, decision-making, and working memory, paving the way for more nuanced cognitive research and the development of truly brain-inspired AI.

---

## Key Features

*   🧠 **Biologically Plausible Modeling**: Construct and simulate brain-like neural networks using Nengo to mimic cognitive processes.
*   🎮 **Dynamic Modified Pong Environment**: Utilize a familiar yet adaptable game for real-time interaction and cognitive stress testing.
*   🔬 **Executive Function Evaluation**: Directly assess attention, decision-making, and working memory performance within a controlled game setting.
*   🛠️ **Nengo Integration**: Leverage a powerful, open-source framework for building large-scale brain models with spiking neurons.
*   🐍 **Pythonic & Extensible**: A clean, modular Python codebase designed for easy understanding, modification, and extension for diverse research applications.
*   📈 **Performance Analysis Ready**: Generate rich data for evaluating model efficacy, behavioral strategies, and cognitive performance metrics.

---

## Technical Architecture

NeuromorphicPong is built on a robust Python foundation, leveraging specialized libraries for neuromorphic computation and model management.

| Technology | Purpose                                | Key Benefit                                         |
| :--------- | :------------------------------------- | :-------------------------------------------------- |
| Python     | Primary Development Language           | High readability, extensive libraries, rapid prototyping. |
| Nengo      | Neuromorphic Computing Framework       | Simulates brain-like networks, biologically plausible models. |
| Keras/H5   | Model Serialization & Management       | Efficient storage and loading of pre-trained neural models. |
| pip/venv   | Dependency & Environment Management    | Isolated project dependencies, reproducible builds. |

### Directory Structure

```
📁 NeuromorphicPong/
├── 📄 README.md
├── 📁 Utilities/
│   └── # (Contains helper scripts, game logic, etc.)
├── 📄 nengo_attention_model.py
├── 📄 reconstructed_model_LIF_simple_velocity_one_layer.h5
└── 📄 requirements.txt
```

---

## Operational Setup

### Prerequisites

Ensure you have the following installed on your system:

*   **Python**: Version 3.8 or higher.
*   **pip**: Python package installer (usually comes with Python).
*   **git**: For cloning the repository.

### Installation

Follow these steps to get NeuromorphicPong up and running on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/NeuromorphicPong.git
    cd NeuromorphicPong
    ```

2.  **Create and activate a virtual environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    With your virtual environment active, install all required packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Neuromorphic Model (Example):**
    You can typically start the main model or a demo using:
    ```bash
    python nengo_attention_model.py
    ```
    *Refer to the `nengo_attention_model.py` file for specific execution instructions or command-line arguments.*

### Environment Configuration

This project utilizes `reconstructed_model_LIF_simple_velocity_one_layer.h5` as a pre-trained or reconstructed model asset. This file serves as a key configuration for the neuromorphic model's architecture and weights. No additional `.env` files are typically required, as model parameters are either hardcoded or loaded directly from this `.h5` file.

---

### License

This project is licensed under the **MIT License**.

You are free to:

*   **Use** - Run the software for any purpose, including commercial.
*   **Study** - Inspect the source code to understand how it works.
*   **Modify** - Adapt the software to your needs.
*   **Distribute** - Share copies of the software.

Provided that you include the original copyright and license notice in any copy of the software or substantial portions of it. For the full license text, please refer to the `LICENSE` file in the root of this repository.
