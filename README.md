# Risk-Analysis-on-Loans-of-Lending-club
Identify the business problem 
	Lending Club is a Peer-to-Peer lending company that utilizes a group of private investors to fund loan requests. 
	The evaluation of the credit-worthiness of their borrowers is based on assigning them a grade and a subgrade.
	Investors have the opportunity to choose which borrowers they will fund, and the percentage of funding that they will cover based on the credit grade of Lending Club
	The business problem is that the grade system is not comprehensive enough for the investors to assess these borrowers to make a smart business decision. We will develop a model to identify new borrowers that would likely default on their loans.

Data Description
Lending Club provided recent 5 years of historical data (2012-2015). These datasets contained information of the borrower’s past credit history and Lending Club loan information consisted of over 800,000 records.  Here we chose the typical dataset 2014 (over 150,000 records) very similar to the new loan-list, which was sufficient for our team to conduct   analysis models.

Variables in the dataset provided much more information than just grade level which can help us to gauge their effect upon the success or failure of a borrower for their loans.

Data Preparing and Processing
	check the train data and test data such as 'intrate'
![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/intrate.png)
	Removed columns that obviously had no relation to the analysis in question (E.g. Applicant ID etc.)
	Removed columns that had bad quality data (i.e. missing values in observations, unintelligible values etc such as inqfi', 'ilutil' etc.)
	Removed columns that had identical relationships to the analysis in question (E.g. funded_amnt is  always the same as loanamt)
check the corrolation 
 ![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/cor.png)
find the corrolated features
 ![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/'numsats'.png)
  ![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/loanamnt.png)
	Established derived columns from existing columns to facilitate model analysis (E.g. addrstate and addrstate frequency etc.)
![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/allstate.png)
	Converted continuous variables to range of values to enhance interpretation of results (E.g. mthssincerecentrevoldelinq, etc.)

Data Mining and Model Build
We use xgboost data models to predict the results and utilized multiple assessment parameters to determine the most successful model, to include: Confusion Matrix (Accuracy, Sensitivity and Specificity) Receiver Operating Characteristic (ROC)
1 We train the data with simple model
![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/roc1.png)

2 We get the best parameters by using BayesianOptimization
![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/roc2.png)

3 We get and prediction and get the top-10 important features
![alt text](https://github.com/shenbingdy/Risk-Analysis-on-Loans-of-Lending-club/blob/master/data/importance.png)

Conclusion:


Software environments:

The program needs to be run in a python 3 environments. (Python 3 installation: https://www.python.org/downloads/)

Meanwhile, the following libraries were used and some of them need to be installed: (1)	Numpy and Pandas (2)	Scikit-learn (3) xgboost 

The Py folder contains all the code:

