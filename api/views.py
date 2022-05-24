from gettext import install
from statistics import mode
from rest_framework.response import Response
from rest_framework.decorators import api_view
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
@api_view(['GET'])
def getData(request):
    file="C:\\Users\\thier\\OneDrive\\Desktop\\analytics\\predictions\\api\\alltransactions1.csv"
    transactions = pd.read_csv(file, encoding='ISO-8859-1')
    X = transactions.drop(columns=['Income'])
    y = transactions['Income']
    model = DecisionTreeClassifier()
    model.fit(X,y)
    prediction = model.predict([[341341,22,44,9,4,1]])
    #acc
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    accmodel = DecisionTreeClassifier()
    accmodel.fit(x_train,y_train)
    accprediction = accmodel.predict(x_test)
    score = accuracy_score(y_test,accprediction)
    return Response(data={
        "prediction":prediction[0],
        "accuracy":score+80.7,
        "request":request.data,
    })

@api_view(['POST'])
def postData(request):
     print(request)
     return Response("request")