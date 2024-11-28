# vamr-project
Repository for VAMR course project at UZH.

# How to run the docker container in Visual Studio Code
1. Install docker in your computer (e.g., Docker engine).
2. Install the Docker extension in VS Code.
3. Open the repository folder in VS Code.
4. Press Ctrl+Shift+P y select "Dev Container: Rebuild Container".

# To update the .devcontainer folder
To track changes to the folder, run:
git update-index --no-assume-unchanged .devcontainer/

To once again stop tracking changes, please run:
git update-index --assume-unchanged .devcontainer/
