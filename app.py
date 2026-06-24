from flask import Flask, render_template, request, session
from scoring_engine import (score_lifestyle, score_social,
                            score_cognitive, score_depression,
                            score_anxiety)
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = 'wellness2024'

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
        with open('models/cluster_name_map.pkl', 'rb') as f:
            models['cluster_map'] = pickle.load(f)
        print("✅ ML Models loaded!")
    except Exception as e:
        print(f"⚠️ Models not loaded: {e}")
    return models

models = load_models()

# ==========================================
# ML PREDICTION FUNCTION
# ==========================================
def get_ml_insights(lifestyle_results, social_results,
                    cognitive_results, depression_result,
                    anxiety_result, lang='en'):
    try:
        # Extract values from results
        sleep_val = 8
        exercise_val = 3
        screentime_val = 3
        sunlight_val = 2
        meal_val = 2
        enjoyable_val = 2
        loneliness_val = 1
        understood_val = 1
        rumination_val = 1
        phq_val = depression_result['total_score'] \
                  if depression_result else 5
        gad_val = anxiety_result['total_score'] \
                  if anxiety_result else 5

        # Session se values lo
        raw = session.get('raw_inputs', {})
        sleep_val = float(raw.get('sleep', 8))
        exercise_val = int(raw.get('exercise', 3))
        screentime_val = float(raw.get('screentime', 3))
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
        loneliness_val = int(raw.get('loneliness', 1))
        understood_val = int(raw.get('understood', 1))
        rumination_val = int(raw.get('rumination', 1))

        # Feature vector banao
        features = np.array([[
            sleep_val, exercise_val, screentime_val,
            sunlight_val, meal_val, enjoyable_val,
            loneliness_val, understood_val,
            rumination_val, phq_val, gad_val
        ]])

        # Scale karo
        features_scaled = models['scaler'].transform(
            features)

        insights = {}

        # K-Means — Risk Cluster
        cluster = int(models['kmeans'].predict(
            features_scaled)[0])
        cluster_name = models['cluster_map'].get(
            cluster, 'Medium Risk Group')
        insights['cluster'] = cluster_name

        # Decision Tree — Risk Level
        risk_pred = int(models['dt'].predict(
            features_scaled)[0])
        risk_labels = {
            0: ('🟢 Low Risk', 'low'),
            1: ('🟡 Medium Risk', 'medium'),
            2: ('🔴 High Risk', 'high')
        }
        insights['dt_risk'] = risk_labels[risk_pred]

        # KNN — Similar Users
        distances, indices = models['knn'].kneighbors(
            features_scaled, n_neighbors=5)
        df = pd.read_csv('data/wellness_data.csv')
        similar_users = df.iloc[indices[0]][[
            'phq_total', 'gad_total',
            'sleep', 'loneliness']].to_dict('records')
        insights['similar_users'] = similar_users

        # SVM — Confidence
        svm_proba = models['svm'].predict_proba(
            features_scaled)[0]
        svm_pred = int(models['svm'].predict(
            features_scaled)[0])
        confidence = round(max(svm_proba) * 100, 1)
        insights['svm_risk'] = risk_labels[svm_pred]
        insights['svm_confidence'] = confidence

        # Linear Regression — Rumination Impact
        lr_input = np.array([[
            phq_val, sleep_val,
            loneliness_val, rumination_val]])
        predicted_gad = round(float(
            models['lr'].predict(lr_input)[0]), 1)

        # Agar rumination 0 hota toh kya hota?
        lr_input_no_rum = np.array([[
            phq_val, sleep_val, loneliness_val, 0]])
        predicted_gad_no_rum = round(float(
            models['lr'].predict(lr_input_no_rum)[0]), 1)

        rum_impact = round(
            predicted_gad - predicted_gad_no_rum, 1)
        insights['lr_gad'] = predicted_gad
        insights['rum_impact'] = rum_impact

        return insights

    except Exception as e:
        print(f"ML Error: {e}")
        return None

# ==========================================
# HOME PAGE
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
        exercise = int(request.form.get('exercise', 3))
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

        # Raw inputs save karo session mein
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
# REPORT — ML Insights ke saath!
# ==========================================
@app.route('/report')
def report():
    lang = session.get('lang', 'en')

    lifestyle = session.get('lifestyle_results', [])
    social = session.get('social_results', [])
    cognitive = session.get('cognitive_results', [])
    depression = session.get('depression_result', None)
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
            "Aapki overall wellness achhi hai!" \
            if lang == 'hi' else \
            "Your overall wellness looks good!")
    elif warnings <= 2:
        overall = ("🟡 Moderate",
            "Kuch areas pe dhyan dene ki zaroorat hai." \
            if lang == 'hi' else \
            "Some areas need your attention.")
    else:
        overall = ("🔴 Needs Attention",
            "Kai areas mein improvement ki gunjaish hai." \
            if lang == 'hi' else \
            "Several areas need improvement.")

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