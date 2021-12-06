from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

file = open('covid_dataset.pkl', 'rb'); 
checkpoint = pickle.load(file); 
file.close(); 

data = pd.read_pickle("covid_dataset.pkl")

X_train, y_train, X_val, y_val, X_test = checkpoint["X_train"], checkpoint["y_train_log_pos_cases"], checkpoint["X_val"], checkpoint["y_val_log_pos_cases"], checkpoint["X_test"] 
LR = LinearRegression()  
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

