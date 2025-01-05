# vamr-project
Repository for VAMR course project at UZH.

# Running with conda environment
```
conda create -n vamr-proj
conda activate vamr-proj
```
The conda environment used for development can be found in the file 'environment.yml'.


# Running with Docker container in Visual Studio Code
1. Install docker in your computer (e.g., Docker engine).
2. Install the Docker extension in VS Code.
3. Open the repository folder in VS Code.
4. Press Ctrl+Shift+P y select "Dev Container: Rebuild Container".

## Add X11 visualization to the Docker container
```
xhost +local:docker
```


# Run the VO algorithm
To run the corresponding Python script, simply execute these lines:
```
cd src
python3 run.py
```

To change the evaluated dataset (available options: 'kitti', 'malaga', 'parking'), use the --dataset argument. For example, to switch to the Malaga dataset, simply run:
```
python3 run.py --dataset malaga
```

You can also enable the debug mode as:
```
python3 run.py --debug
```

To visualize the candidate features, simply run:
```
python3 run.py --visualize_candidates
```

In addition, you can enable ground-truth initialization as follows:
```
python3 run.py --gt_init
```

Lastly, you can generate a video from the trajectory as follows:
```
python3 run.py --save
```

# Download datasets
Run the following bash script to download the required datasets and unzip them:
```
bash data/download_dataset.sh
```


# To update the .devcontainer folder
To track changes to the folder, run:
```
git update-index --no-assume-unchanged .devcontainer/
```

To once again stop tracking changes, please run:
```
git update-index --assume-unchanged .devcontainer/
```