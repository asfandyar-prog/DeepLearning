


Q1: following steps should take for enhancing the accuracy:

Right learning rate: means that you should use standard learning rate to avoid overfiting.

Better Data: the more examples the model sees, the better it learns

More layers in model 
Dropout 
Batch Normalization.


Q2:These are just two sides of the same coin:
Error Rate = 1 − Accuracy
So if accuracy is 92%, error rate is 8%. You plot both over epochs (training rounds) to see how your model improves over time. You want accuracy going up and error going down , ideally for both training and validation data.

Q3:

Large batch size → you get a good average view of the terrain → you take confident, steady steps → smooth curve
	•	Small batch size → you only feel a tiny patch of ground each step → your direction keeps changing → jagged/noisy curve
	•	High learning rate → your steps are too big → you keep jumping over the lowest point → oscillating curve
	•	Low learning rate → tiny steps → very slow but smooth descent
The curve shape is basically telling you how confidently and consistently your model is learning.
