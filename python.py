import pandas as pd
data = {
    "units_sold":[100,200,300,400,500,600,700,800],
    "revenue":[0,20,40,60,80,100,120,140]
}
df = pd.DataFrame(data)
# print(df)

X = df[['units_sold']] #input 
Y = df['revenue'] #output

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train ,Y_test = train_test_split(
    X,Y ,test_size=0.6, random_state=42
)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,Y_train)

predictions = model.predict(X_test)
print(predictions)                                       
print(Y_test.values)

from sklearn.metrics import mean_absolute_error
print("mae is ",mean_absolute_error(Y_test,predictions))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test,predictions))

from sklearn.metrics import mean_squared_error
import numpy as np
mse=mean_squared_error(Y_test,predictions)
rmse=np.sqrt(mse)
print("mse score",mse)
print("rmse score",rmse)

from sklearn.metrics import r2_score
r2=r2_score(Y_test,predictions)
print("r2 score ",r2)



