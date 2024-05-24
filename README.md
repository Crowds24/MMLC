# MMLC
## Requirements
To install requirements:
```setup
python==3.8
numpy==1.23.2
pandas==1.3.1
scikit-learn==0.24.2
tensorflow==2.11.2
```
Hardware resources for this paper：
Apple M1 Pro
## Structure
Introduce the role of each package
```
aggregationMethod -- aggregation method
data -- training data、 fill data and redundant data
fillData -- data filling method
getOracleWorker -- get oracle worker and do truth inference
train -- model train and save model
until -- toolkit: get redundant data, scatter plots, and more
upperLimit -- get the upper limit of oracle worker ability
```
## Clone
Clone data needs to use git lfs. Because of the size of the uploaded training data, git lfs is used to upload it.
## Result
### truth inference
True inference can be directly performed through the trained model. For example, under the getOracleWorker package of music, you can load the trained model (musicModel_origin) in the train package and run the following command to obtain the results.
```setup
python getOracleWorker.py
```
### train
Train the model through the train method under the trian package.Train the model by training the train method under the trian package, and then load the corresponding model in the getOracleWorker package method for true value inference.
In train package：
```setup
python train.py
```
In getOracleWorker package：
```setup
python getOracleWorker.py
```
### fillData 
Data can be filled under the fillData package, and the filled data can be used to train the model. The process is the same as in the train chapter.
```setup
python fill.py
```
### redundancy
In the until package, data can be divided according to the redundancy value.the divided data can be used to train the model. The process is the same as in the train chapter.
```setup
python redundancy.py
```
### upperLimit
Get the upper limit of oracle worker capabilities by running the method in upperLImit
```setup
python trainModel.py
```
