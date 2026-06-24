import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 200  # 200 synthetic users

# ==========================================
# LIFESTYLE FEATURES
# ==========================================
sleep = np.random.choice([4, 5, 8, 10],
        n, p=[0.15, 0.30, 0.40, 0.15])

exercise = np.random.choice([0, 1, 3, 6],
           n, p=[0.20, 0.30, 0.35, 0.15])

screentime = np.random.choice([1, 3, 5, 7],
             n, p=[0.15, 0.30, 0.35, 0.20])

sunlight = np.random.choice(
           ['always', 'sometimes', 'rarely', 'never'],
           n, p=[0.20, 0.30, 0.30, 0.20])

meal_regularity = np.random.choice(
                  ['always', 'usually',
                   'sometimes', 'rarely'],
                  n, p=[0.25, 0.30, 0.30, 0.15])

enjoyable = np.random.choice(
            ['daily', 'sometimes', 'rarely', 'never'],
            n, p=[0.20, 0.35, 0.30, 0.15])

# ==========================================
# SOCIAL FEATURES
# ==========================================
loneliness = np.random.choice([0, 1, 2, 3],
             n, p=[0.20, 0.30, 0.30, 0.20])

understood = np.random.choice([0, 1, 2, 3],
             n, p=[0.25, 0.30, 0.25, 0.20])

# ==========================================
# COGNITIVE FEATURES
# ==========================================
rumination = np.random.choice([0, 1, 2, 3],
             n, p=[0.20, 0.30, 0.30, 0.20])

# ==========================================
# PHQ-9 SCORES (9 questions, each 0-3)
# ==========================================
phq_scores = []
for i in range(n):
    # Higher sleep problems = higher PHQ score
    base = 0
    if sleep[i] < 7: base += 2
    if exercise[i] < 3: base += 1
    if loneliness[i] >= 2: base += 2
    if rumination[i] >= 2: base += 3

    # Generate 9 PHQ answers
    phq = []
    for q in range(9):
        score = min(3, max(0,
                np.random.randint(0, 3) +
                (1 if base > 4 else 0)))
        phq.append(score)
    phq_scores.append(sum(phq))

phq_total = np.array(phq_scores)

# ==========================================
# GAD-7 SCORES (7 questions, each 0-3)
# ==========================================
gad_scores = []
for i in range(n):
    base = 0
    if screentime[i] >= 4: base += 1
    if rumination[i] >= 2: base += 2
    if loneliness[i] >= 2: base += 1

    gad = []
    for q in range(7):
        score = min(3, max(0,
                np.random.randint(0, 3) +
                (1 if base > 2 else 0)))
        gad.append(score)
    gad_scores.append(sum(gad))

gad_total = np.array(gad_scores)

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================
sunlight_map = {'always': 3, 'sometimes': 2,
                'rarely': 1, 'never': 0}
meal_map = {'always': 3, 'usually': 2,
            'sometimes': 1, 'rarely': 0}
enjoyable_map = {'daily': 3, 'sometimes': 2,
                 'rarely': 1, 'never': 0}

sunlight_enc = np.array([sunlight_map[s]
               for s in sunlight])
meal_enc = np.array([meal_map[m]
           for m in meal_regularity])
enjoyable_enc = np.array([enjoyable_map[e]
                for e in enjoyable])

# ==========================================
# OVERALL RISK LABEL
# ==========================================
def get_risk(phq, gad, lone, rum):
    score = 0
    if phq >= 15: score += 3
    elif phq >= 10: score += 2
    elif phq >= 5: score += 1
    if gad >= 15: score += 3
    elif gad >= 10: score += 2
    elif gad >= 5: score += 1
    if lone >= 2: score += 1
    if rum >= 2: score += 1
    if score <= 2: return 0    # Low
    elif score <= 5: return 1  # Medium
    else: return 2             # High           # High

risk = np.array([get_risk(phq_total[i],
        gad_total[i], loneliness[i],
        rumination[i]) for i in range(n)])

# ==========================================
# CREATE DATAFRAME
# ==========================================
df = pd.DataFrame({
    'sleep': sleep,
    'exercise': exercise,
    'screentime': screentime,
    'sunlight': sunlight_enc,
    'meal_regularity': meal_enc,
    'enjoyable': enjoyable_enc,
    'loneliness': loneliness,
    'understood': understood,
    'rumination': rumination,
    'phq_total': phq_total,
    'gad_total': gad_total,
    'risk_level': risk
})

# Save karo
os.makedirs('data', exist_ok=True)
df.to_csv('data/wellness_data.csv', index=False)

print("✅ Synthetic dataset created!")
print(f"Total users: {len(df)}")
print(f"\nRisk distribution:")
print(f"Low Risk (0):    {(risk==0).sum()} users")
print(f"Medium Risk (1): {(risk==1).sum()} users")
print(f"High Risk (2):   {(risk==2).sum()} users")
print(f"\nAvg PHQ-9 score: {phq_total.mean():.1f}")
print(f"Avg GAD-7 score: {gad_total.mean():.1f}")
print("\nFirst 5 rows:")
print(df.head())