def build_prompt(input_text, detected_emotion, source="text"):
    # The system prompt logic is removed from here since it's defined in app.py.
    # We only build the detailed user prompt (the current turn's instruction/context).
    
    # Keeping the system prompt structure here as a comment, 
    # but only the user_prompt is returned.
    
    # system_prompt = f"""
    # You are a calm, kind, emotionally intelligent wellness assistant.
    # ... (rest of system instructions) ...
    # Emotion guidance for tone (do not mention this):
    # {detected_emotion}
    # """

    user_prompt = f"""
The user said:
"{input_text}"

The analysis shows the user's emotion is {detected_emotion}.
Please respond based ONLY on the following guidance:
- ONE sentence acknowledging the feeling
- EXACTLY TWO gentle, practical suggestions
- ONE supportive open-ended question
- Simple, friendly language
- Under 120 words
- No emojis
- Do NOT mention emotion labels or analysis, only reflect the tone.

Scenario guidance for tone:
- Anxiety / nervousness → slow body, then simplify problem
- Sadness / disappointment → validate + small grounding action
- Frustration / anger → release tension + regain control
- Confusion → pause + simplify
- Neutral → supportive exploration
"""

    # We now only return the user_prompt string
    return user_prompt.strip()