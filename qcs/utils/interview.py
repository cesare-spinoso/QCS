INTERVIEW_FILE_NAME_TO_ID_DICT: dict[str, int] = {
    "Participant_1_Interview": 1,
    "Participant_2_Interview": 2,
    "Participant_3_Interview": 3,
    "Participant_4_Interview": 4,
    "Participant_5_Interview": 5,
    "Participant_6_Interview": 6,
    "Participant_7_Interview": 7,
    "Participant_8_Interview": 8,
    "Participant_9_Interview": 9,
    "Participant_10_Interview": 10,
    "Participant_11_Interview": 11,
    "Participant_12_Interview": 12,
    "Participant_13_Interview": 13,
    "Participant_14_Interview": 14,
    "Participant_16_Interview": 16,
}

INTERVIEW_ID_TO_FILE_NAME_DICT: dict[int, str] = {
    v: k for k, v in INTERVIEW_FILE_NAME_TO_ID_DICT.items()
}

INTERVIEW_IDS = sorted(INTERVIEW_ID_TO_FILE_NAME_DICT.keys())

QUESTIONS: dict[int, str] = {
    1: "How do you think cardiovascular diseases are generally described and understood by the public?",
    2: "Have you ever been informed about your cardiovascular health? How?",
    3: "What are the most frequent and important decisions you face related to your cardiovascular health?",
    4: "What is your usual role in making decisions about your cardiovascular health and preventing CVD?",
    5: "What do you think are some challenges and needs in preventing and managing cardiovascular diseases (from your perspective as a woman at risk of CVD)?",
    6: "Have you ever considered a decision support system to help you in any decisions related to your cardiovascular health?",
    7: "What are your thoughts on using digital technology (e.g., mobile apps, AI systems/robots) to make decisions in relation to your cardiovascular health?",
    8: "How would you like us to design and develop this Xi-Care tool that is useful, helpful and effective for women at risk of CVD (e.g. no risks to users)?",
    9: "On a scale of 1 to 5 with 1 being not at all interested and 5 being very interested, what is your level of interest in monitoring tools that track health data over time?",
    10: "On a scale of 1 to 5 with 1 being not at all interested and 5 being very interested, what is your level of interest in a step-count feature?",
    11: "On a scale of 1 to 5 with 1 being not at all interested and 5 being very interested, what is your level of interest in a weight tracking feature?",
    12: "On a scale of 1 to 5 with 1 being not at all interested and 5 being very interested, what is your level of interest in educational modules on cardiovascular health?",
    13: "On a scale of 1 to 5 with 1 being not at all interested and 5 being very interested, what is your level of interest in guided exercise activities?",
    14: "On a scale of 1 to 5 with 1 being not at all interested and 5 being very interested, what is your level of interest in diet recommendations?",
    15: "Would you like to be able to follow your progress and receive push-notifications through the Xi-Care tool?",
    16: "How difficult/easy do you think it will be for you to integrate the Xi-Care tool into your daily life?",
    17: "To what extent would you trust or rely on the Xi-Care tool to make assessments about your cardiovascular health and prevent and manage CVD?",
    18: "Do you foresee any challenges with integrating the Xi-Care tool in your daily life in terms of ethics? Could you please describe these challenges?",
    19: "In terms of transparency, how important is it for you to be able to understand how the Xi-Care tool works?",
    20: "Is there something else you'd like to add about ethical aspects in regards to the Xi-Care tool that will be empowered by AI (Justice; Non-maleficence; Autonomy; Beneficence; Explicability/Transparency)?",
}

NUMBER_OF_QUESTIONS = len(QUESTIONS)
