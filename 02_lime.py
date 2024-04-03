# %%
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blockbox import LimeTabular
from interpret import show

# %%
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_test, y_train)
print(X_train.shape)
print(X_test.shape)

# %%
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %%
lime = LimeTabular(predict_fn=rf.predict_proba,data=X_train, random_state=1)

lime_local = lime.explain_local(X_test[-20:], y_test[-20:], name='LIME')
show(lime_local)
# %%