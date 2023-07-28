This is my first NLP project,using python.

* `note` is my markdown.

* `baseline1` is the **traditional baseline** of the project,running on the Baidu AI Studio([relative link](https://aistudio.baidu.com/aistudio/projectdetail/6522950?sUid=377372&shared=1&ts=1689827255213)),and this is the local version.

* `NLP_baseline` is a series of baseline,transmitting different classifiers including the **Logistic Regression**,the **Support Vector Machine** and the **Random Forest Classifier**. Based on the classifiers above,fine-tune the parameters with `parameter_tuning.py` `baseline_tuning.py`. 

  According to the score given by the platform,**the fine-tuned Logistic Regression model**(AKA fine-tuned baseline) performs best up to now,reaching 0.99401.

  The official provides another dataset: `testB.csv` on 24th，July. The dataset remove the column `Keywords`. Thus, I update `baseline2` into `baseline3` to fix the dataset

* `NLP_upper` is the upper project,using the BERT model from transformers to solve the classify-problem.

  ~~Regretfully, my local environment couldn't support the project(my poor GTX1650 4GB).~~

  SOLUTION: Run the project on Ali Cloud(not success yet)<---*It's still a good solution*

  ~~**However,this project has run for 26 epochs before I stopped the interpreter and the score was unsatisfactory**~~.<---*maybe overfitting*
  
  Set the epoch=10,and the model works well,accuracy reaching 0.9850.<---*for task 1*
  
  The latest version of `NLP_upper` is a **complete version**. It uses the BERT model to solve **two tasks** compared with **only one** in last version. The result is quite good but a bit late :).
  
* `NLP_chatGLM` is the project using the LLM,leveraging chatGLM in the case of the stability of the connection. However,**using API may casuse the problem that the input including sensitive words stops the program**,emphasizing the essence of training the LLM locally.

> To be continued...