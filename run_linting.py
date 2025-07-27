import subprocess
import sys
import os

venv_python = os.path.join(r"C:\Users\Psycho_Ind\Desktop\CodeCraft Projects Task\CODECRAFT_GA_01", ".venv", "Scripts", "python.exe")
venv_pip = os.path.join(r"C:\Users\Psycho_Ind\Desktop\CodeCraft Projects Task\CODECRAFT_GA_01", ".venv", "Scripts", "pip.exe")

def run_command(command, description):
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return e.returncode
    except FileNotFoundError:
        print(f"Error: Command not found. Make sure {command[0]} is in your PATH or specify full path.")
        return 1

# Install black
print("Attempting to install black...")
install_black_command = [venv_pip, "install", "black"]
if run_command(install_black_command, "Install black") != 0:
    print("Failed to install black. Exiting.")
    sys.exit(1)

# Run black
print("Attempting to run black...")
run_black_command = [venv_python, "-m", "black", "--line-length", "79", "src", "scripts"]
if run_command(run_black_command, "Run black") != 0:
    print("Failed to run black. Exiting.")
    sys.exit(1)

# Run flake8
print("Attempting to run flake8...")
run_flake8_command = [venv_python, "-m", "flake8", "src", "scripts"]
if run_command(run_flake8_command, "Run flake8") != 0:
    print("Flake8 found issues. Please review the output.")
    sys.exit(1)

print("Linting complete.")