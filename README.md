This is my first NLP project,using python.

* `note` is my markdown.

* `baseline1` is the **traditional baseline** of the project,running on the Baidu AI Studio([relative link](https://aistudio.baidu.com/aistudio/projectdetail/6522950?sUid=377372&shared=1&ts=1689827255213)),and this is the local version.

* `NLP_baseline` is a series of baseline,transmitting different classifiers including the **Logistic Regression**,the **Support Vector Machine** and the **Random Forest Classifier**. Based on the classifiers above,fine-tune the parameters with `parameter_tuning.py` `baseline_tuning.py`. 

  According to the score given by the platform,**the fine-tuned Logistic Regression model**(AKA fine-tuned baseline) performs best up to now,reaching 0.99401.

* `NLP_upper` is the upper project,using the BERT model from transformers. Regretfully, my local environment couldn't support the project(~~my poor GTX1650 12GB~~).

  SOLUTION: Run the project on Ali Cloud(not success yet)

  **However,this project has run for 26 epoches before I stopped the interpreter and the score was unsatisfactory.**

  

> To be continued...