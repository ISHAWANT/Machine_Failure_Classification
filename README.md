# Machine Failure Classification Project

### About Project:

The goal of the Machine Failure Prediction project is to precisely forecast a machine's likelihood of failure based on its operational parameters. In predictive maintenance, when minimizing machine downtime is essential to guarantee optimal resource usage, this prediction is extremely important. The issue is approached as a classification problem and is resolved via feature engineering, thorough exploratory data analysis, and reliable classification modeling.

### About Dataset: 
A variety of sensor readings and situations that are critical to predicting machine failure are included in the dataset for this competition. A machine learning model that was first trained on machine failure predictions produces the training and test datasets. Though the feature distributions are comparable, it's crucial to remember that they differ from the original data. This raises the competition's level of difficulty and complexity.

The following features are included in the dataset:

- ID: Every record in the dataset has a unique ID assigned to it. It facilitates the indexing and reference of every single record.

- Product Id: An identifying number combined with the Type variable

- Type: The machine type for which the readings are recorded is indicated by this. Knowing the kind of equipment you have can help you understand the kinds of operations it does and how likely it is that something will go wrong.

- Air temperature [K]: This is the machine's surrounding ambient temperature expressed in Kelvin. Given that machines may respond differently depending on the ambient temperature, it could be a significant factor.

- Process temperature [K]: This is the temperature, expressed in Kelvin, of the process that the machine is operating in. The machine may overheat during some operations, which raises the possibility of failure.

- Rotational speed (rpm): This is the machine's operating speed. Rotations per minute, or rpm, are used to measure it. Increasing the speed may result in more wear and tear.

- Torque [Nm]: The force that rotates a machine is measured in terms of torque [Nm]. High torque could be a sign of a heavy load on the equipment, which raises the possibility of failure.

- Tool wear [min]: This parameter shows the level of deterioration the machine has experienced. High tool wear may be a sign that the machine needs maintenance; this is measured in minutes.

- Machine Failure: The goal variable, which is a binary indicator indicating whether the machine failed (1) or not (0), is machine failure.

In addition to these, there are other failure modes captured in the dataset:

- TWF: Tool wear failure. This indicates whether the machine failed due to tool wear.
- HDF: Heat dissipation failure. This indicates whether the machine failed due to an inability to dissipate heat.
- PWF: Power failure. This indicates whether the machine failed due to a power problem.
- OSF: Overstrain failure. This indicates whether the machine failed due to being overstrained.
- RNF: Random failure. This indicates whether the machine failed due to a random, unspecified issue.

### Methodology : 
```
Data Ingestion 

Data Validation

Data Transformation

Model Training

Model Evaluation 

Model Pusher
```
### Project Setup
Create a conda eviroment
```
conda create -p venv python==3.9 -y 
```
Activate your enviroment
```
conda activate venv/
```
Install the requirements
```
pip install -r requirements.txt
```
Run the app.py file 
```
python app.py
```



