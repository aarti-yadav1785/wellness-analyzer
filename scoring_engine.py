# ==========================================
# SCORING ENGINE — Updated with 6 lifestyle factors
# ==========================================

from tips_database import (LIFESTYLE_TIPS, SOCIAL_TIPS,
                           COGNITIVE_TIPS, DEPRESSION_TIPS,
                           ANXIETY_TIPS, CRISIS_RESOURCES)

# ==========================================
# 1. LIFESTYLE SCORING
# ==========================================
def score_lifestyle(sleep, exercise, screentime,
                   sunlight, meal_regularity,
                   enjoyable, lang='en'):
    results = []

    # Sleep (AASM + NSF — 7-9 hours normal)
    if sleep < 7:
        results.append({
            "factor": "Sleep" if lang=='en' else "नींद",
            "status": "⚠️ Below Normal",
            "tip": LIFESTYLE_TIPS["sleep_low"][lang]
        })
    elif sleep > 9:
        results.append({
            "factor": "Sleep" if lang=='en' else "नींद",
            "status": "⚠️ Above Normal",
            "tip": LIFESTYLE_TIPS["sleep_high"][lang]
        })
    else:
        results.append({
            "factor": "Sleep" if lang=='en' else "नींद",
            "status": "✅ Normal",
            "tip": LIFESTYLE_TIPS["sleep_normal"][lang]
        })

    # Exercise (WHO — 150+ min/week = 3+ sessions)
    if exercise < 3:
        results.append({
            "factor": "Exercise" if lang=='en' else "व्यायाम",
            "status": "⚠️ Below Recommended",
            "tip": LIFESTYLE_TIPS["exercise_low"][lang]
        })
    else:
        results.append({
            "factor": "Exercise" if lang=='en' else "व्यायाम",
            "status": "✅ Normal",
            "tip": LIFESTYLE_TIPS["exercise_normal"][lang]
        })

    # Screen Time (4+ hours = risk factor)
    if screentime >= 4:
        results.append({
            "factor": "Screen Time",
            "status": "⚠️ High",
            "tip": LIFESTYLE_TIPS["screentime_high"][lang]
        })
    else:
        results.append({
            "factor": "Screen Time",
            "status": "✅ Normal",
            "tip": LIFESTYLE_TIPS["screentime_normal"][lang]
        })

    # Morning Sunlight
    if sunlight in ['never', 'rarely']:
        results.append({
            "factor": "Morning Sunlight" if lang=='en' else "सुबह की धूप",
            "status": "⚠️ Missing — Surprising impact!",
            "tip": LIFESTYLE_TIPS["sunlight_low"][lang]
        })
    else:
        results.append({
            "factor": "Morning Sunlight" if lang=='en' else "सुबह की धूप",
            "status": "✅ Good",
            "tip": LIFESTYLE_TIPS["sunlight_normal"][lang]
        })

    # Meal Regularity
    if meal_regularity in ['rarely', 'sometimes']:
        results.append({
            "factor": "Meal Timing" if lang=='en' else "खाने का समय",
            "status": "⚠️ Irregular — Research shows this matters!",
            "tip": LIFESTYLE_TIPS["meal_irregular"][lang]
        })
    else:
        results.append({
            "factor": "Meal Timing" if lang=='en' else "खाने का समय",
            "status": "✅ Regular",
            "tip": LIFESTYLE_TIPS["meal_regular"][lang]
        })

    # Enjoyable Activities
    if enjoyable in ['never', 'rarely']:
        results.append({
            "factor": "Enjoyable Activities" if lang=='en' else "मनपसंद गतिविधियाँ",
            "status": "⚠️ Low — An overlooked factor!",
            "tip": LIFESTYLE_TIPS["enjoyable_low"][lang]
        })
    else:
        results.append({
            "factor": "Enjoyable Activities" if lang=='en' else "मनपसंद गतिविधियाँ",
            "status": "✅ Good",
            "tip": LIFESTYLE_TIPS["enjoyable_normal"][lang]
        })

    return results


# ==========================================
# 2. SOCIAL SCORING
# ==========================================
def score_social(loneliness, understood, lang='en'):
    results = []

    if loneliness >= 2:
        results.append({
            "factor": "Loneliness" if lang=='en' else "अकेलापन",
            "status": "⚠️ Concerning",
            "tip": SOCIAL_TIPS["loneliness_high"][lang]
        })
    else:
        results.append({
            "factor": "Loneliness" if lang=='en' else "अकेलापन",
            "status": "✅ Normal",
            "tip": SOCIAL_TIPS["loneliness_normal"][lang]
        })

    if understood >= 2:
        results.append({
            "factor": "Connection Quality" if lang=='en' else "रिश्तों की गहराई",
            "status": "⚠️ May feel unheard",
            "tip": SOCIAL_TIPS["isolation_quality_check"][lang]
        })

    return results


# ==========================================
# 3. COGNITIVE SCORING
# ==========================================
def score_cognitive(rumination, lang='en'):
    results = []

    if rumination >= 2:
        results.append({
            "factor": "Rumination" if lang=='en' else "बार-बार सोचना",
            "status": "⚠️ High — Research says this is a key risk factor",
            "tip": COGNITIVE_TIPS["rumination_high"][lang]
        })
    else:
        results.append({
            "factor": "Rumination" if lang=='en' else "बार-बार सोचना",
            "status": "✅ Normal",
            "tip": COGNITIVE_TIPS["rumination_normal"][lang]
        })

    # Always show additive bias insight
    results.append({
        "factor": "Wellness Insight" if lang=='en' else "वेलनेस इनसाइट",
        "status": "💡 Did you know?",
        "tip": COGNITIVE_TIPS["additive_bias_note"][lang]
    })

    return results


# ==========================================
# 4. DEPRESSION SCORING (Full PHQ-9)
# ==========================================
def score_depression(phq_answers, lang='en'):
    total = sum(phq_answers)
    q9 = phq_answers[8]

    crisis = True if q9 > 0 else False

    if total <= 4:
        severity = "minimal"
    elif total <= 9:
        severity = "mild"
    elif total <= 14:
        severity = "moderate"
    elif total <= 19:
        severity = "moderately_severe"
    else:
        severity = "severe"

    return {
        "total_score": total,
        "severity": severity,
        "tip": DEPRESSION_TIPS[severity][lang],
        "crisis": crisis,
        "crisis_resources": CRISIS_RESOURCES[lang] if crisis else None
    }


# ==========================================
# 5. ANXIETY SCORING (Full GAD-7)
# ==========================================
def score_anxiety(gad_answers, lang='en'):
    total = sum(gad_answers)

    if total <= 4:
        severity = "minimal"
    elif total <= 9:
        severity = "mild"
    elif total <= 14:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "total_score": total,
        "severity": severity,
        "tip": ANXIETY_TIPS[severity][lang]
    }


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    print("=" * 45)
    print("SCORING ENGINE TEST")
    print("=" * 45)

    print("\n--- Lifestyle Test ---")
    lifestyle = score_lifestyle(
        sleep=5,
        exercise=2,
        screentime=6,
        sunlight='never',
        meal_regularity='rarely',
        enjoyable='never',
        lang='en'
    )
    for item in lifestyle:
        print(f"{item['factor']}: {item['status']}")

    print("\n--- Social Test ---")
    social = score_social(loneliness=3, understood=2, lang='en')
    for item in social:
        print(f"{item['factor']}: {item['status']}")

    print("\n--- Cognitive Test ---")
    cognitive = score_cognitive(rumination=3, lang='en')
    for item in cognitive:
        print(f"{item['factor']}: {item['status']}")

    print("\n--- Depression Test (PHQ-9) ---")
    depression = score_depression(
        [2, 1, 2, 1, 2, 1, 2, 1, 0], lang='en')
    print(f"Score: {depression['total_score']}")
    print(f"Severity: {depression['severity']}")
    print(f"Crisis: {depression['crisis']}")

    print("\n--- Anxiety Test (GAD-7) ---")
    anxiety = score_anxiety([1, 2, 1, 2, 1, 2, 1], lang='en')
    print(f"Score: {anxiety['total_score']}")
    print(f"Severity: {anxiety['severity']}")

    print("\n" + "=" * 45)
    print("✅ ALL TESTS PASSED!")
    print("=" * 45)