import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from nltk.corpus import stopwords
import re
import mlflow
from urllib.parse import urlparse


def clean(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('[^A-Za-z]',' ',str(x)))
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub(r"https?://\S+|www\.\S+",'',str(x)))
    return text


def lemmatize(text):
    m = WordNetLemmatizer()
    lemm_list = m.lemmatize(text)
    lemm_text = "".join(lemm_list)
    return lemm_text


def get_version_model(config_name, client_id):
    dict_push = {}
    for count, value in enumerate(client_id.search_model_versions(f"name='{config_name}'")):
        dict_push[count] = value
    return dict(list(dict_push.items())[-1][1])['version']


data = pd.read_csv('jigsaw.csv')
config_path = '/'
data = data.dropna(axis=0)
data['comment_text'] = clean(data['comment_text'])

corpus = list(data['comment_text'].apply(lambda x: lemmatize(x)))


nltk.download('stopwords')

stopwords = set(stopwords.words('english'))


count_tf_idf = TfidfVectorizer(stop_words = stopwords)
tf_idf = count_tf_idf.fit_transform(corpus)

features = tf_idf
target = data['toxic'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

params_logreg = {"class_weight": 'balanced',
                 "C": 0.5}
forest_params = {
     'max_depth': 6,
     'n_estimators': 100,
     'min_samples_split': 5}

model_reg = LogisticRegression(**params_logreg)
model_for = RandomForestClassifier(**forest_params)

#explainer_log_reg = shap.Explainer(model_reg.predict, X_train, algorithm="permutation")


mlflow.set_tracking_uri("http://mlflow:5050")
mlflow.set_experiment('my_first_experiment')

with mlflow.start_run() as run:
    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    model_reg.fit(X_train, y_train)

    mlflow.log_param('C_log_reg', params_logreg['C'])
    mlflow.log_metric('accuracy_logreg',
                      accuracy_score(y_test, model_reg.predict(X_test)))
    mlflow.log_metric('roc_auc_logreg',
                      roc_auc_score(y_test, model_reg.predict(X_test)))
    mlflow.log_metric('f1_score_logreg',
                      f1_score(y_test, model_reg.predict(X_test)))
    # if tracking_url_type_store != "http":
    #     mlflow.sklearn.log_model(model_reg, 'model_lr', registered_model_name='LogisticRegression')
    # else:
    mlflow.sklearn.log_model(model_reg, 'model_lr')
    #mlflow.log_artifact(local_path='train.py', artifact_path='code')
    #mlflow.shap.log_explainer(explainer_log_reg, artifact_path="shap_explainer")

    model_for.fit(X_train, y_train)

    mlflow.log_param('for_depth',
                     forest_params['max_depth'])
    mlflow.log_metric('accuracy_for',
                      accuracy_score(y_test, model_for.predict(X_test)))
    mlflow.log_metric('roc_auc_for',
                      roc_auc_score(y_test, model_for.predict(X_test)))
    mlflow.log_metric('f1_score_for',
                      f1_score(y_test, model_for.predict(X_test)))
    # if tracking_url_type_store != "http":
    #     mlflow.sklearn.log_model(model_for, 'model_for', registered_model_name='Random_Forest')
    # else:
    mlflow.sklearn.log_model(model_for, 'model_for')
    #mlflow.log_artifact(local_path='train.py', artifact_path='code')

    mlflow.end_run()

# client = MlflowClient()
# artifact_path = "model_explanations_shap"
# artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
# last_version_lr = get_version_model('LogisticRegression', client)
# last_version_cat = get_version_model('CatBoost', client)
#
# dst_path = client.download_artifacts(run.info.run_id, artifact_path)
# base_values = np.load(os.path.join(dst_path, "base_values.npy"))
# shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))
#
# shap.force_plot(float(base_values), shap_values[0, :], X_train.iloc[0, :], matplotlib=True)
#
# yaml_file = yaml.safe_load(open(config_path))
# yaml_file['predict']["version_lr"] = int(last_version_lr)
# yaml_file['predict']["version_vec"] = int(last_version_cat)
#
# with open(config_path, 'w') as fp:
#     yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True)
