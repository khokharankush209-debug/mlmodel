
import pandas as pd
import numpy as np

data={
    "unit":[320,280,190,360,420,140,300,220,130,390,210,340],
    "price":[120,150,110,125,160,90,115,105,385,180,110,175],
    "revenue":[38400,42000,20900,45000,67200,12600,34500,23100,11050,70200,23100,59500]
}
df=pd.DataFrame(data)
print(df)

x=df[['unit','price']]
y=df['revenue']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(predictions)


from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,predictions))

print(y_test.values)

from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test,predictions)
print("mse",mse)
import numpy as np
rmse = np.sqrt(mse)
print("rmse",rmse)
print(y_test.values)

from sklearn.metrics import r2_score

check = r2_score(y_test, predictions)
print("R2 score:", check)
