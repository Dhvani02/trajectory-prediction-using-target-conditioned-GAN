
# Trajectory Prediction in Autonomous Vehicles
The goal of the project is to predict future trajectories of autonomous vehicles in dense traffic environments using GANs.

## Setup
Use virtual env (used python 3.6.8). Install dependencies with 
```
pip3 install -r requirements.txt
```
Add virtual env to jupyter notebook
```
python -m ipykernel install --user --name=your_virtual_env
jupyter notebook --port=your_port_number
```
Remember to activate environment to install libraries
```
pynev shell your_virtual_env
```

### Notebook setup
Leverage the following code snippet to use the src directory in notebooks
```python
import sys
# This should navigate to the repository root
sys.path.append('../')
%reload_ext autoreload
%autoreload 2
```

## Project Structure
Most idea pulled from [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#directory-structure)

### app.py
Main command line interface to run each step of the ML pipeline
 - Create dataset compatible to the code by running
   ```python 
   python app.py 
   ```
   This will create data in a csv file containing the trajectory information with the form <frame_id> <person_id> <x_cood> <y_cood>.
  
### data
The code is compatible with Lyft Level 5 motion prediction dataset.

### notebook
Should contains experiments with GAN, ResNet, Clustering (ensemble)

### metrics
Any output of metrics (eg. average displacement error, .etc) should be saved under here as a csv
### src
All code. Here's the breakdown per folder within src.
#### conf
Config file (can include model configurations, or train test data configurations, etc.)
#### data
All code related to data preprocessing to create a trainable data 
#### train
All code related to modeling and traning

## Pipeline
### Target location prediction
Combine visual scene information with historical trajectory to give a probability distribution of target locations. This will help determine the possible directions in which the trajectory should be predicted.
### Generator
This will create trajectories similar to ground truth, conditioned on the predicted target location. Thus, the input to the generator will also include the predicted goal location, with the help of this the problem of mode collapse (drawback of other papers using GAN) should dissapear.
### Discriminator
This will have access to the ground truth trajectories, and it will take the input from the generator and try to predict if it is real or fake.
### Ensemble
The idea of an ensemble is to take the multiple models like ResNet, LSTM, GAN to predict multi-modal trajectories, then use the outputs from all the models to predict the final n trajectories. This can be done by using clustering algorithms like k-means or GMMs to cluster the output from every model, then compute the means in various clusters to give a final outcome.








