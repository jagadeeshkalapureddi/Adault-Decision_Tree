# Adult-Decision_Tree

A popular example is the adult income dataset that involves predicting personal income levels as above or below $50,000 per year based on personal details such as relationship and education level. There are many more cases of incomes less than $50K than above $50K, although the skew is not severe.

The dataset provides 14 input variables that are a mixture of categorical, ordinal, and numerical data types. The complete list of variables is as follows:

Age,"\n"
Workclass,
Final Weight,
Education,
Education Number of Years,
Marital-status,
Occupation,
Relationship,
Race,
Sex,
Capital-gain,
Capital-loss,
Hours-per-week,
Native-country and Income

The dataset contains missing values that are marked with a question mark character (?).

There are a total of 48,842 rows of data, and 3,620 with missing values, leaving 45,222 complete rows.

There are two class values ‘>50K‘ and ‘<=50K‘, meaning it is a binary classification task. The classes are imbalanced, with a skew toward the ‘<=50K‘ class label.

‘>50K’: majority class, approximately 25%.
‘<=50K’: minority class, approximately 75%.
Given that the class imbalance is not severe and that both class labels are equally important, it is common to use classification accuracy or classification error to report model performance on this dataset.

Using predefined train and test sets, reported good classification error is approximately 14 percent or a classification accuracy of about 86 percent. This might provide a target to aim for when working on this dataset.
