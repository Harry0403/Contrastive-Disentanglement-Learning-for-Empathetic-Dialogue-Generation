# Contrastive-Disentanglement-Learning-for-Empathetic-Dialogue-Generation

## Installation
```
pip install -r requirements.txt
```
## Training
go to the directory "source" then do
```
python3 train_multi.py
```

## Model weight
The pre-trained is provide in [Weight](https://drive.google.com/drive/folders/1n684i_F2ioNvaFFe6A4xDAVDQOz9hZmM?usp=sharing).
Make sure the weight show be put in the right direction.
Please create a directory```My_model_pth``` inside the directory ```Empathetic_Dialogue``` or the directory ```NYCUKA```.
Then you can run the evaluation.

## Evaluation
go outside the directory "source" then do 
```
python3 eval_multi_continuous.py
```

## Sampleing
Make sure to run the evluation first.
You can create the directory what you want to put the output file inside and then you can run the file ```example.ipynb```
