import pandas as pd # For data manipulation.
from flaml import AutoML # from ML model training.
from sklearn.model_selection import train_test_split # For spliting Data. 
from sklearn.metrics import confusion_matrix # For MLmodel evelution.
from sklearn.preprocessing import LabelEncoder # For Encoding string data.
import joblib # For saving trained MLmodel.

# Loading Dataset.
data = "C:/Users/meeth/Downloads/diabetes_data.csv"
df = pd.read_csv(data,sep=';')

# Encoding string gender data to 1-male and 0-Female
enc = LabelEncoder()
df['gender'] = enc.fit_transform(df['gender'])

# Creating feature and lable datasets
X = df[list(df.columns)].drop('class',axis=1)
y = df['class']
# Spliting dataset into training and testing. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ML Model making.
automl = AutoML()
automl_settings = {
    "time_budget": 10,
    "metric": 'log_loss',
    "task": 'classification',
}
model = automl.fit(X_train=X_train, y_train=y_train,**automl_settings)
pred = automl.predict(X_test)
df_pred = pd.DataFrame(pred,y_test)

# Confusion matrics for model evaluation.
cm = confusion_matrix(y_true=y_test,y_pred=pred)
print(cm)
#Saving trained ML Model.
joblib.dump(automl, 'C:/Users/meeth/Desktop/model_local.pkl',)
