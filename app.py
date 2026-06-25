from flask import Flask, render_template, request, session
from scoring_engine import (score_lifestyle, score_social,
                            score_cognitive, score_depression,
                            score_anxiety)
import pandas as pd
import numpy as np
import pickle
import os
import gspread
from google.oauth2.service_account import Credentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'wellness2024'

# ==========================================
# GOOGLE SHEETS SETUP
# ==========================================
def get_sheet():
    try:
        import json
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds_json = os.environ.get(
            'GOOGLE_CREDENTIALS')
        if creds_json:
            creds_dict = json.loads(creds_json)
            creds = Credentials.from_service_account_info(
                creds_dict, scopes=scopes)
        else:
            creds = Credentials.from_service_account_file(
                'credentials.json', scopes=scopes)

        client = gspread.authorize(creds)
        sheet = client.open_by_key(
            '1FyP87jAYveLKsCkB7vkdpgTH1aiP2brbes5yCCn4IXQ'
        ).sheet1
        return sheet
    except Exception as e:
        print(f"⚠️ Google Sheets error: {e}")
        return None

# ==========================================
# LOAD ML MODELS
# ==========================================
def load_models():
    models = {}
    try:
        with open('models/kmeans.pkl', 'rb') as f:
            models['kmeans'] = pickle.load(f)
        with open('models/dt.pkl', 'rb') as f:
            models['dt'] = pickle.load(f)
        with open('models/knn.pkl', 'rb') as f:
            models['knn'] = pickle.load(f)
        with open('models/svm.pkl', 'rb') as f:
            models['svm'] = pickle.load(f)
        with open('models/lr.pkl', 'rb') as f:
            models['lr'] = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        with open('models/cluster_name_map.pkl',
                  'rb') as f:
            models['cluster_map'] = pickle.load(f)
        print("✅ ML Models loaded!")
    except Exception as e:
        print(f"⚠️ Models not loaded: {e}")
    return models

models = load_models()

# ==========================================
# RETRAIN MODELS
# ==========================================
def retrain_models(df):
    try:
        if len(df) < 30:
            print("⚠️ Not enough data to retrain")
            return

        features = ['sleep', 'exercise',
                    'screentime', 'sunlight',
                    'meal_regularity', 'enjoyable',
                    'loneliness', 'understood',
                    'rumination', 'phq_total',
                    'gad_total']

        X = df[features]
        y = df['risk_level']

        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)

        new_kmeans = KMeans(
            n_clusters=3, random_state=42)
        new_kmeans.fit(X_scaled)

        new_dt = DecisionTreeClassifier(
            max_depth=4, random_state=42)
        new_dt.fit(X_scaled, y)

        new_knn = KNeighborsClassifier(
            n_neighbors=5)
        new_knn.fit(X_scaled, y)

        new_svm = SVC(kernel='rbf', C=10,
                      gamma='scale',
                      random_state=42,
                      probability=True)
        new_svm.fit(X_scaled, y)

        new_lr = LinearRegression()
        new_lr.fit(
            df[['phq_total', 'sleep',
                'loneliness', 'rumination']],
            df['gad_total'])

        os.makedirs('models', exist_ok=True)
        with open('models/kmeans.pkl', 'wb') as f:
            pickle.dump(new_kmeans, f)
        with open('models/dt.pkl', 'wb') as f:
            pickle.dump(new_dt, f)
        with open('models/knn.pkl', 'wb') as f:
            pickle.dump(new_knn, f)
        with open('models/svm.pkl', 'wb') as f:
            pickle.dump(new_svm, f)
        with open('models/lr.pkl', 'wb') as f:
            pickle.dump(new_lr, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(new_scaler, f)

        models['kmeans'] = new_kmeans
        models['dt'] = new_dt
        models['knn'] = new_knn
        models['svm'] = new_svm
        models['lr'] = new_lr
        models['scaler'] = new_scaler

        print(f"✅ Models retrained with "
              f"{len(df)} real users!")

    except Exception as e:
        print(f"⚠️ Retrain error: {e}")

# ==========================================
# SAVE LOCAL (Fallback)
# ==========================================
def save_local(new_row):
    try:
        columns = ['sleep', 'exercise',
                   'screentime', 'sunlight',
                   'meal_regularity', 'enjoyable',
                   'loneliness', 'understood',
                   'rumination', 'phq_total',
                   'gad_total', 'risk_level']
        filepath = 'data/wellness_data.csv'
        os.makedirs('data', exist_ok=True)

        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            new_df = pd.DataFrame(
                [new_row], columns=columns)
            df = pd.concat(
                [df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(
                [new_row], columns=columns)

        df.to_csv(filepath, index=False)
        print(f"✅ Saved locally! "
              f"Total: {len(df)} rows")
    except Exception as e:
        print(f"⚠️ Local save error: {e}")

# ==========================================
# SAVE REAL USER DATA
# ==========================================
def save_user_data(raw_inputs, phq_score,
                   gad_score, risk_level):
    try:
        sunlight_map = {'always': 3, 'sometimes': 2,
                        'rarely': 1, 'never': 0}
        meal_map = {'always': 3, 'usually': 2,
                    'sometimes': 1, 'rarely': 0}
        enjoyable_map = {'daily': 3, 'sometimes': 2,
                         'rarely': 1, 'never': 0}

        new_row = [
            float(raw_inputs.get('sleep', 8)),
            int(raw_inputs.get('exercise', 3)),
            float(raw_inputs.get('screentime', 3)),
            sunlight_map.get(raw_inputs.get(
                'sunlight', 'sometimes'), 2),
            meal_map.get(raw_inputs.get(
                'meal_regularity', 'usually'), 2),
            enjoyable_map.get(raw_inputs.get(
                'enjoyable', 'sometimes'), 2),
            int(raw_inputs.get('loneliness', 1)),
            int(raw_inputs.get('understood', 1)),
            int(raw_inputs.get('rumination', 1)),
            phq_score,
            gad_score,
            risk_level
        ]

        # Google Sheets mein save karo
        sheet = get_sheet()
        if sheet:
            sheet.append_row(new_row)
            total = len(sheet.get_all_values())
            print(f"✅ Data saved to Google Sheets!"
                  f" Total rows: {total}")

            # CSV bhi update karo
            os.makedirs('data', exist_ok=True)
            all_data = sheet.get_all_records()
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(
                    'data/wellness_data.csv',
                    index=False)

            # Har 50 users pe retrain
            if total % 50 == 0 and total >= 51:
                print("🔄 Retraining ML models...")
                df_retrain = pd.read_csv(
                    'data/wellness_data.csv')
                retrain_models(df_retrain)
        else:
            save_local(new_row)

    except Exception as e:
        print(f"⚠️ Save error: {e}")
        save_local(new_row)

# ==========================================
# ML PREDICTION FUNCTION
# ==========================================
def get_ml_insights(lifestyle_results,
                    social_results,
                    cognitive_results,
                    depression_result,
                    anxiety_result, lang='en'):
    try:
        raw = session.get('raw_inputs', {})

        sleep_val = float(raw.get('sleep', 8))
        exercise_val = int(raw.get('exercise', 3))
        screentime_val = float(
            raw.get('screentime', 3))

        sunlight_map = {'always': 3, 'sometimes': 2,
                        'rarely': 1, 'never': 0}
        meal_map = {'always': 3, 'usually': 2,
                    'sometimes': 1, 'rarely': 0}
        enjoyable_map = {'daily': 3, 'sometimes': 2,
                         'rarely': 1, 'never': 0}

        sunlight_val = sunlight_map.get(
            raw.get('sunlight', 'sometimes'), 2)
        meal_val = meal_map.get(
            raw.get('meal_regularity', 'usually'), 2)
        enjoyable_val = enjoyable_map.get(
            raw.get('enjoyable', 'sometimes'), 2)
        loneliness_val = int(
            raw.get('loneliness', 1))
        understood_val = int(
            raw.get('understood', 1))
        rumination_val = int(
            raw.get('rumination', 1))

        phq_val = depression_result['total_score'] \
                  if depression_result else 5
        gad_val = anxiety_result['total_score'] \
                  if anxiety_result else 5

        features = np.array([[
            sleep_val, exercise_val,
            screentime_val, sunlight_val,
            meal_val, enjoyable_val,
            loneliness_val, understood_val,
            rumination_val, phq_val, gad_val
        ]])

        features_scaled = models['scaler'].transform(
            features)

        insights = {}

        # K-Means
        cluster = int(models['kmeans'].predict(
            features_scaled)[0])
        cluster_name = models['cluster_map'].get(
            cluster, 'Medium Risk Group')
        insights['cluster'] = cluster_name

        # Decision Tree
        risk_pred = int(models['dt'].predict(
            features_scaled)[0])
        risk_labels = {
            0: ('🟢 Low Risk', 'low'),
            1: ('🟡 Medium Risk', 'medium'),
            2: ('🔴 High Risk', 'high')
        }
        insights['dt_risk'] = risk_labels[risk_pred]

        # KNN
        distances, indices = \
            models['knn'].kneighbors(
                features_scaled, n_neighbors=5)
        df = pd.read_csv('data/wellness_data.csv')
        similar_users = df.iloc[indices[0]][[
            'phq_total', 'gad_total',
            'sleep', 'loneliness']].to_dict(
            'records')
        insights['similar_users'] = similar_users

        # SVM
        svm_proba = models['svm'].predict_proba(
            features_scaled)[0]
        svm_pred = int(models['svm'].predict(
            features_scaled)[0])
        confidence = round(max(svm_proba) * 100, 1)
        insights['svm_risk'] = risk_labels[svm_pred]
        insights['svm_confidence'] = confidence

        # Linear Regression
        lr_input = np.array([[
            phq_val, sleep_val,
            loneliness_val, rumination_val]])
        predicted_gad = round(float(
            models['lr'].predict(lr_input)[0]), 1)

        lr_input_no_rum = np.array([[
            phq_val, sleep_val,
            loneliness_val, 0]])
        predicted_gad_no_rum = round(float(
            models['lr'].predict(
                lr_input_no_rum)[0]), 1)

        rum_impact = round(
            predicted_gad - predicted_gad_no_rum, 1)
        insights['lr_gad'] = predicted_gad
        insights['rum_impact'] = rum_impact

        return insights

    except Exception as e:
        print(f"ML Error: {e}")
        return None

# ==========================================
# HOME
# ==========================================
@app.route('/')
def home():
    session.clear()
    return render_template('home.html')

# ==========================================
# LIFESTYLE
# ==========================================
@app.route('/lifestyle', methods=['GET', 'POST'])
def lifestyle():
    if request.method == 'POST':
        lang = request.form.get('lang', 'en')
        sleep = float(request.form.get('sleep', 7))
        exercise = int(
            request.form.get('exercise', 3))
        screentime = float(
            request.form.get('screentime', 3))
        sunlight = request.form.get(
            'sunlight', 'rarely')
        meal_regularity = request.form.get(
            'meal_regularity', 'sometimes')
        enjoyable = request.form.get(
            'enjoyable', 'rarely')

        results = score_lifestyle(
            sleep, exercise, screentime,
            sunlight, meal_regularity,
            enjoyable, lang)

        raw = session.get('raw_inputs', {})
        raw.update({
            'sleep': sleep,
            'exercise': exercise,
            'screentime': screentime,
            'sunlight': sunlight,
            'meal_regularity': meal_regularity,
            'enjoyable': enjoyable
        })
        session['raw_inputs'] = raw
        session['lang'] = lang
        session['lifestyle_done'] = True
        session['lifestyle_results'] = results

        return render_template('lifestyle.html',
                               results=results,
                               lang=lang,
                               submitted=True)

    lang = request.args.get('lang', 'en')
    return render_template('lifestyle.html',
                           lang=lang,
                           submitted=False)

# ==========================================
# SOCIAL
# ==========================================
@app.route('/social', methods=['GET', 'POST'])
def social():
    lang = session.get('lang', 'en')

    if request.method == 'POST':
        loneliness = int(
            request.form.get('loneliness', 0))
        understood = int(
            request.form.get('understood', 0))

        results = score_social(
            loneliness, understood, lang)

        raw = session.get('raw_inputs', {})
        raw.update({
            'loneliness': loneliness,
            'understood': understood
        })
        session['raw_inputs'] = raw
        session['social_done'] = True
        session['social_results'] = results

        return render_template('social.html',
                               results=results,
                               lang=lang,
                               submitted=True)

    return render_template('social.html',
                           lang=lang,
                           submitted=False)

# ==========================================
# COGNITIVE
# ==========================================
@app.route('/cognitive', methods=['GET', 'POST'])
def cognitive():
    lang = session.get('lang', 'en')

    if request.method == 'POST':
        rumination = int(
            request.form.get('rumination', 0))
        results = score_cognitive(rumination, lang)

        raw = session.get('raw_inputs', {})
        raw.update({'rumination': rumination})
        session['raw_inputs'] = raw
        session['cognitive_done'] = True
        session['cognitive_results'] = results

        return render_template('cognitive.html',
                               results=results,
                               lang=lang,
                               submitted=True)

    return render_template('cognitive.html',
                           lang=lang,
                           submitted=False)

# ==========================================
# DEPRESSION
# ==========================================
@app.route('/depression', methods=['GET', 'POST'])
def depression():
    lang = session.get('lang', 'en')

    if request.method == 'POST':
        answers = []
        for i in range(1, 10):
            ans = int(request.form.get(f'q{i}', 0))
            answers.append(ans)

        result = score_depression(answers, lang)
        session['depression_done'] = True
        session['depression_result'] = result

        return render_template('depression.html',
                               result=result,
                               lang=lang,
                               submitted=True)

    return render_template('depression.html',
                           lang=lang,
                           submitted=False)

# ==========================================
# ANXIETY
# ==========================================
@app.route('/anxiety', methods=['GET', 'POST'])
def anxiety():
    lang = session.get('lang', 'en')

    if request.method == 'POST':
        answers = []
        for i in range(1, 8):
            ans = int(request.form.get(f'q{i}', 0))
            answers.append(ans)

        result = score_anxiety(answers, lang)
        session['anxiety_done'] = True
        session['anxiety_result'] = result

        return render_template('anxiety.html',
                               result=result,
                               lang=lang,
                               submitted=True)

    return render_template('anxiety.html',
                           lang=lang,
                           submitted=False)

# ==========================================
# REPORT
# ==========================================
@app.route('/report')
def report():
    lang = session.get('lang', 'en')

    lifestyle = session.get('lifestyle_results', [])
    social = session.get('social_results', [])
    cognitive = session.get('cognitive_results', [])
    depression = session.get(
        'depression_result', None)
    anxiety = session.get('anxiety_result', None)

    # Overall risk
    warnings = 0
    for item in lifestyle + social + cognitive:
        if '⚠️' in item.get('status', ''):
            warnings += 1

    if depression:
        if depression['severity'] in [
                'moderate', 'moderately_severe',
                'severe']:
            warnings += 2

    if anxiety:
        if anxiety['severity'] in [
                'moderate', 'severe']:
            warnings += 2

    if warnings == 0:
        overall = ("🟢 Good",
            "Aapki overall wellness achhi hai!"
            if lang == 'hi' else
            "Your overall wellness looks good!")
        risk_num = 0
    elif warnings <= 2:
        overall = ("🟡 Moderate",
            "Kuch areas pe dhyan dene ki "
            "zaroorat hai."
            if lang == 'hi' else
            "Some areas need your attention.")
        risk_num = 1
    else:
        overall = ("🔴 Needs Attention",
            "Kai areas mein improvement ki "
            "gunjaish hai."
            if lang == 'hi' else
            "Several areas need improvement.")
        risk_num = 2

    # Real user data save karo
    raw = session.get('raw_inputs', {})
    phq_score = depression['total_score'] \
                if depression else 5
    gad_score = anxiety['total_score'] \
                if anxiety else 5
    save_user_data(raw, phq_score,
                   gad_score, risk_num)

    # ML Insights
    ml_insights = get_ml_insights(
        lifestyle, social, cognitive,
        depression, anxiety, lang)

    return render_template('report.html',
                           lang=lang,
                           lifestyle=lifestyle,
                           social=social,
                           cognitive=cognitive,
                           depression=depression,
                           anxiety=anxiety,
                           overall=overall,
                           ml_insights=ml_insights)

if __name__ == '__main__':
    app.run(debug=True)