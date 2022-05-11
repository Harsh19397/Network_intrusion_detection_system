# Network intrusion detection system using Machine learning
1.	Introduction

Intrusion detection system (IDS): An Intrusion detection system is a monitoring system that monitors the malicious activities in the network and generates alert whenever they detect any. 
An Intrusion detection system can be of 2 types:
Host-Based IDS (HIDS): A host-based IDS system is designed to protect the network from internal and external threats. Such an IDS system may have the ability to monitor to and from the machine, observe running processes, and inspect the system’s logs. 
Network-Based IDS (NIDS): A network-based IDS solution is designed to monitor an entire protected network. It has visibility into all traffic flowing through the network and makes determinations based upon packet metadata and contents. 
Detection methods used by IDS
•	Signature Detection: Signature-based IDS solutions use fingerprints of known threats to identify them. Once malware or other malicious content has been identified, a signature is generated and added to the list used by the IDS solution to test incoming content. 
•	Anomaly Detection: Anomaly-based IDS solutions build a model of the “normal” behavior of the protected system. All future behavior is compared to this model, and any anomalies are labeled as potential threats and generate alerts. 
•	Hybrid Detection: A hybrid IDS uses both signature-based and anomaly-based detection. 
2.	Business problem statement

With the advancement in internet technologies, the number of cyber-attacks has been growing exponentially due to several vulnerabilities in the network. So, the need to bring in the advanced Network Intrusion Detection System (NIDS) is immense. These intrusion detection systems are used to help protect and secure the resources present in the network infrastructure. There is a requirement for efficient intrusion models to analyze and asses both present and future network attacks. 
Classical intrusion detection system (IDs) and firewalls suffers from the drawback of continuously updating the databases for threats. With the advancements in machine learning and deep learning we aim to build more robust systems with higher detection and lower false alarm rates (FAR).
3.	Business constraint
1.	Since dataset is highly imbalanced high accuracy can sometimes cannot be used to measure the performance of our model. Hence, we should also have high AUC and F1 scores.
2.	Network intrusion detection should not take hours and block the user's computer. It should finish in a few seconds or a minute.
4.	Machine learning problem formulation

4.1	Mapping the real-world data to Machine Learning

4.1.1	Type of ML problem
Here our objective is to detect and classify whether incoming traffic to the network is malicious or not. Since we need to make a classification between two classes (0=normal and 1 =abnormal), this problem is a binary classification-based machine learning problem.
4.1.2	Performance metric
It has been seen in the past that the general detection rate of Intrusion detection system has been low with a high False Alarm Rate, accuracy and False Positive Rate are good performance indicators.
It has also been seen that dataset we will be working on is highly imbalanced and even a dumb model can have high accuracy that predicts the majority class. Hence, we would also that AUC-ROC and F1 scores as key performance indicators.
4.1.3	ML Objectives
Objective: Our objective is to predict whether the incoming network traffic is normal or abnormal. Label 0 stands for normal traffic, whereas 1 stand for abnormal or malicious traffic.

4.2	Data
4.2.1	Data Overview
For this case study we are using UNSW-NB15 dataset. This dataset has been taken from here. This dataset contains 49 features which are described in UNSW-NB15_features.csv file. The argus, Bro-IDS tools and 12 algorithms which were developed for creating these 49 features from the raw traffic in pcap files (100GB).
There are in total 2.54 million records which are stored in 4 CSV files. There is also a ground truth table and a table which contains attack category and subcategory. From these 4 CSV files, a training dataset and test dataset were created which contain around 175,321 points and 82,332 points respectively. 
4.2.2	Dataset column analysis
This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. There is a feature called label which is of binary type. It states 0 for normal traffic and 1 for malicious traffic.
This dataset has 49 features. Column wise description of attributes used in UNSW-NB15 dataset are shown below in the table below.
 
There are 9 types of attacks in the dataset as shown in the table below:
 
4	Research section (At least 5 or just explain everything that is relevant)

1.	https://www.hindawi.com/journals/cin/2021/5557577/
Observations
a.	In this research paper after doing exploratory data analysis, it is shown that the dataset is highly imbalanced and even a dumb model can create a classifier with very high accuracy. Hence, it is necessary to first deal with the data imbalance problem.
b.	Here, the technique used to deal with this imbalance problem is called Synthetic Minority Oversampling technique (SMOTE). SMOTE is a method where we create new examples from the minority class. This is a type of data augmentation for tabular data which is very effective. SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
c.	Once the dataset imbalance problem was dealt with, it was seen that creating and training complex models on unimportant features won’t help us in getting good results and hence the important features are selected from the given set of features for the models to be trained on those set of features.
d.	These important features were selected by Gini Impurity criterion using Extremely Randomized Trees Classifier (Extra Trees Classifier).
e.	 After that, a pretrained extreme learning machine (ELM) model was used for detecting the attacks separately, “One-Versus-All” as a binary classifier for each of them. Finally, the ELM classifier outputs become the inputs to a fully connected layer in order to learn from all their combinations, followed by a logistic regression layer to make soft decisions for all classes.
f.	Finally, the performance of the model was evaluated in terms of accuracy, False Alarm Rate, Receiver Operating Characteristics and Precision-Recall Curves.
Takeaways
a.	The dataset we are using is highly imbalanced and we need to balance it using some technique. The method used in this research paper SMOTE is also and effective one. We can use SMOTE to deal with the data imbalance problem before creating and training classifiers.
b.	Training on all the features does not yield good results and hence we need to select important set of features from the given set of 49 features. Here in the paper Gini Impurity criterion using Extremely Randomized Trees Classifier (Extra Trees Classifier) is used which also gives good results.
c.	While modeling we can use combinations of models as used in this paper to get high accuracy and low false alarm rate.
d.	Along with Accuracy also calculate AUC-ROC, precision and recall for getting better understanding of the performance of the model.


2.	https://ieeexplore.ieee.org/abstract/document/9001867/metrics#metrics
Observations
a.	In this research paper the technique used for Modeling is XGBoost. Since XGBoost uses only numeric values, only those features that are of integer and float types are included.
b.	Binary features were converted to numeric types and all the categorical features were removed.
c.	Then the features in which the maximum Information gain calculated in XGBoost were observed were selected. Hence a subset of 23 features were obtained and then XGBoost classifier was trained on it.
Takeaways
a.	Here the key takeaway is that we can use Information gain method calculated in XGBoost classifier for selecting the important features subset from the given set of features.

3.	https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3784406

Observations

a.	In this research paper, we observe that there are null values observed for the service feature in our UNSW-NB15 dataset used. In the data cleaning and preprocessing steps these null values found are handled and the data set is cleaned.
b.	After handling the null-values we then convert the categorical data into numeric form by using the label encoder. Then they used one hot encoder to break the relation between the values obtained through label encoder.
c.	Then the authors of this paper also discuss various feature selection methods used for selecting the relevant features. They categorize these features under filter, wrapper and hybrid categories. They also discuss the merits and demerits of these methods used. Then finally they use Chi square method under the filter category for selecting the important features to finally train their models.
d.	Then the classifiers are categorized under two categories namely Lazy/ Memory based learner and Eager/ Global based learners. KNN and SGD are picked from Lazy learners and Logistic Regression, Random Forest Classifier and Naïve Bayes classifiers are picked from Eager learners’ category to train on the training data.
e.	Then in the end Accuracy, Recall, Precision, F1-Score and MSE are calculated and compared for these models.
f.	It is observed that Random Forest Classifier produces the best results.
Takeaways
a.	The first takeaway from this paper is that we need to handle the null values from the service feature and also check for null values from the other features and handle them as well if found any.
b.	Secondly, in most of the research papers label encoder followed by one hot encoding is used for converting the categorical features, hence we would use the same.
c.	As mentioned in the paper for this problem chi square method produces features that in modelling stage produces results with very high accuracy, recall, precision and low False positive rate. Hence, we will also try with chi square method for selecting the important features.
d.	Along with this research paper and others, it has been observed that Random Forest Classifier produces very good results. Hence, we will also train Random Forest models.

4.	https://www.tandfonline.com/doi/full/10.1080/19393555.2015.1125974?scroll=top&needAccess=true
Observations
a.	In this research paper the goal of the authors is to evaluate the complexity of the training and test dataset.
b.	Firstly, there are different types of features in the data set so the first step taken in this paper is to convert all the features to numeric types. They perform one hot encoding on all the multi class and binary class features.
c.	Then since the attributes are having a large scale between min and max, therefore it is extremely difficult to estimate their variance. So, to handle this they standardize the dataset.
d.	They apply the KS test, skewness and kurtosis to compare the distributions of training and test set. Feature correlations and feature correlations with class labels is also calculated to understand which features contribute more.
e.	Naïve Bayes, Decision trees, Artificial Neural networks, Logistic Regression and EM Clustering has been used executed on the training and testing sets to asses the complexity in terms of accuracy and False alarm rates.
Takeaways 
a.	This tells us that the data have a large scale between min and max, therefore before applying the Machine Learning models, we need to standardize the dataset.
b.	During performing the exploratory data analysis, we can also calculate feature correlation with the class labels to see which features can contribute most to the classification techniques and understand the dataset better. 
c.	As seen in this paper as well, Naïve Bayes and Decision trees-based classifiers are performing well in classification task. Hence, we will also try these models in our implementation.

5.	https://www.elastic.co/guide/en/ecs/master/ecs-network.html
Understanding
a.	Feature Engineering is the process of coming up with new features from the given raw data to make machine learning work well on new tasks. 
b.	Here in this link, we have a list of network attributes. Network attributes are the details about the network activity associated with an event.
c.	Using the network fields provided in the link above we can come up with 2 new features which we can use for our problem statements. These new features are:
i.	Network bytes: These are the total number of bytes transferred in both directions. This can be obtained by making a sum of sbytes and dbytes from our dataset. Hence, we will create a new feature as total_bytes which will contain the sum of sbytes and dbytes.
ii.	Network packets: These are the total number of packets transferred in both directions. This can be obtained by making a sum of Spkts and Dpks from our dataset. Hence, we will create a new feature as total_packets which will contain the sum of spkts and dpkts.
d.	These new features which are created by the process of Feature engineering helps in reaping better results.
e.	When feature engineering activities are done correctly, the resulting dataset is optimal and contains all of the important factors that affect the business problem. As a result of these datasets, the most accurate predictive models and the most useful insights are produced.

6.	One hot encoding
a.	Categorical data is are variables which stores label values rather than numerical values. These labels generally have a length of fixed set. Each label represents a different value. 
b.	The problem with categorical data is that many Machine learning algorithms cannot operate on these categorical data and needs the data to be in the numeric format. To convert these categorical data to numeric form we have many such techniques, one such techniques is One hot encoding.
c.	One hot encoding creates a new binary feature for each possible category and assigns a value of 1 to the feature of each sample that corresponds to its original category.
d.	Advantages of One hot encoding:
i.	One hot encoding makes the categorical data useful for machine learning algorithms.
ii.	One hot encoding ensures that machine learning does not consider higher numbers are more important.
e.	Disadvantages
i.	The disadvantage that one hot encoding has is that in case of high cardinality the dimensions blow up and you have to deal with the curse of dimensionality.
f.	For our problem we will have to one hot encode all the categorical features present in our dataset. 

7.	Bagging
a.	Bagging aggregating also called as machine learning ensemble technique is commonly used for reducing variance and producing results with higher accuracy for statistical classification tasks.
i.	Random Forest Classifier: Random Forest, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.
b.	As seen in most of the research papers studied for this problem statement, had Random Forest, Decision Trees or stack of some decision tress perform with very high accuracy, ROC-AUC Score, Precision, recall and very low False Alarm rate. Hence, we will try out with Random Forest or some combinations of Decision trees to come up with classifiers which performs well on the key performance indices discussed above.
 
8.	Boosting
a.	Boosting is an ensemble learning method that combines a set of weak learners into a strong learner to minimize training errors. In boosting, a random sample of data is selected, fitted with a model and then trained sequentially—that is, each model tries to compensate for the weaknesses of its predecessor.
b.	We have many boosting techniques but as seen in the research papers studied for this problem statements, we can try training our data on two major boosting techniques namely XGBoost and Light weight Gradient Boosted Decision Trees.
i.	XGBoost /LGBM: XGBoost is a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models. 
c.	Techniques like LGBM produces results faster as they are more optimized implementation of vanilla Gradient Boosting techniques. 
5	First Cut Solution

Based on all the research that I have done by reading the research papers and concepts required, I will be taking the below mentioned steps:
a.	The dataset has a big problem of class imbalance. Count of 0 (Normal label) in target variable is very high as compared to the 1 (abnormal label). Hence, either using SMOTE or some other technique I will have to deal with the class imbalance problem.
b.	Secondly, I will have cleaned the data by handling the null values present in the service feature and also perform one hot encoding on all the categorical features for our dataset. 
c.	In the Exploratory Data Analysis, as seen in the research its important to study the correlation of features among them selves and also with the target variable.
d.	Then as studied in feature engineering I will create features like total_bytes and total_packets in our data. I will try to come up with some other network features as well.
e.	Next for feature selection we have studied a few well performing techniques like using Gini impurity, information gain, chi square method etc. for selecting the feature. I will try and experiment with a few to get important features and train the final models to see what performs the best on the performance metrics.
f.	For performance metrics as seen in most of the places I will define functions for Accuracy, False positive rate, precision, recall and ROC_AUC score. 
g.	For modelling as seen in most of the research papers Support Vector Machines, Naïve Bayes, Logistic Regression, Decision Trees, Random Forest and XGBoost produces the best results. Hence, I would try to train these classifiers on our train dataset.
h.	Finally for hyper parameter tuning I would use cross validation technique with Grid or Random Search.


