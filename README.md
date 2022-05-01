# cue: a deep learning framework for SV calling and genotyping

### Installation

#### Setup a Python virtual environment (recommended)

1. Create the virtual environment (in the env directory): 
```$> python3.7 -m venv env```

2. Activate the environment: 
```$> source env/bin/activate```

3. Install all the required packages (in the virtual environment):
```$> pip --no-cache-dir install -r install/requirements.txt```

### Execution

Prior to running the code, setup a Python virtual environment (as described below) 
and set the PYTHONPATH as follows: ```export PYTHONPATH=${PYTHONPATH}:${PATH_TO_REPO}```

The ```engine``` directory contains the following key scripts to train/evaluate the model 
and generate image datasets:

* ```generate.py```: creates an annotated image dataset from alignments (BAM file(s))
* ```train.py```: trains a deep learning model (currently, this is a stacked hourglass network architecture) 
to detect SV keypoints in images
* ```call.py```: calls structural variants given a pre-trained model and an input BAM file 
(can be executed on multiple GPUs or CPUs)
* ```view.py```: plots images annotated with SVs from a VCF/BEDPE file given genome alignments (BAM format);
can be used to visualize model predictions or ground truth SVs 

Each script accepts as input one or multiple YAML config files, 
which encode a variety of parameters. Template config files are provided 
in the ```config``` directory.

The recommended workflow for launching a new experiment:
1. Create a new folder under the ```experiments``` directory or an internal subdirectory.
2. Place YAML config file(s) in this folder (either copying the template or a previous config).
3. Populate the YAML config files with the parameters specific to this experiment.
4. Execute the appropriate ```engine``` script providing the path to the newly configured YAML file(s).
The engine scripts will automatically create auxiliary directories with results in the experiments folder 
where the config YAML files are located.
