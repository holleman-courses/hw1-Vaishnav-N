[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HjPz605o)
# tinyml_hw1
Template and test script for HW1

## Development setup

### Virtual environment (recommended)

1. **Create the virtual environment** (one time):
   ```powershell
   cd hw1-Vaishnav-N-main
   python -m venv .venv
   ```

2. **Activate the virtual environment** (each new terminal):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If you use Command Prompt instead: `\.venv\Scripts\activate.bat`

3. **Install dependencies** (one time after creating the venv):
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the homework script**:
   ```powershell
   python hw1_complete.py
   ```

5. **Run the tests**:
   ```powershell
   pytest hw1_test.py -v
   ```
