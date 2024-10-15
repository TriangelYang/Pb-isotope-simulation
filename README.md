
# Pb-Isotope-Simulation

## Steps to Set Up the Environment

1. **Create a new Conda environment**

   ```bash
   conda create --name simulation_env python=3.8
   ```

2. **Activate the environment**

   ```bash
   conda activate simulation_env
   ```

3. **Install required packages**

   - Install PyTorch (ensure you have GPU support with CUDA if available):
   
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

   - Install other essential libraries:

     ```bash
     pip install numpy tqdm
     ```

## Running the Simulation

```bash
python montecario.py
```
