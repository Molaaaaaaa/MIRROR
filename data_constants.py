"""
Data-dependent constants for MIRROR-Violence (MIRROR-V) dataset.

These values must match the category names and response labels in the raw CSV data.
Korean text represents actual survey response labels - do not modify unless 
the underlying dataset changes.

For use with other datasets, create a new constants file with matching labels.
"""

# RESPONSE SCALE MAPPING
# Maps Korean survey response text to numeric values for analysis
RESPONSE_SCALE = {
    # Agreement scale (4-point Likert)
    '전혀 그렇지 않다': 1,      # Not at all
    '그렇지 않은 편이다': 2,    # Somewhat disagree
    '그런 편이다': 3,           # Somewhat agree
    '매우 그렇다': 4,           # Strongly agree
    
    # Frequency scale - count based
    '전혀 없다': 1,             # Never
    '1~2번': 2,                 # 1-2 times
    '3~5번': 3,                 # 3-5 times
    '6번 이상': 4,              # 6+ times
    
    # Frequency scale - behavior
    '전혀 하지 않는다': 1,      # Never do
    '거의 하지 않는다': 2,      # Rarely do
    '가끔 한다': 3,             # Sometimes do
    '자주 한다': 4,             # Often do
    
    # Time duration scale
    '전혀 안함': 0,             # Not at all
    '30분 미만': 1,             # Less than 30 min
    '30분 ~ 1시간 미만': 2,     # 30 min - 1 hour
    '1시간 ~ 2시간 미만': 3,    # 1 - 2 hours
    '2시간 ~ 3시간 미만': 4,    # 2 - 3 hours
    '3시간 ~ 4시간 미만': 5,    # 3 - 4 hours
    '4시간 이상~': 6,           # 4+ hours
    
    # Well-being scale
    '아주 불행한 사람이다': 1,  # Very unhappy
    '불행한 사람이다': 2,       # Unhappy
    '행복한 사람이다': 3,       # Happy
    '아주 행복한 사람이다': 4,  # Very happy
    
    # Sleep quality scale
    '전혀 못 잔다': 1,          # Cannot sleep at all
    '못 자는 편이다': 2,        # Sleep poorly
    '잘 자는 편이다': 3,        # Sleep well
    '매우 잘 잔다': 4,          # Sleep very well
    
    # Count scale
    '없다': 0,                  # None
    '1~2회': 1,                 # 1-2 times
    '3~4회': 2,                 # 3-4 times
    '5회 이상': 3,              # 5+ times
    
    # Weekly frequency scale
    '1주일에 1번': 5,           # Once a week
    '1주일에 여러번': 6,        # Several times a week
    '한 달에 1번': 4,           # Once a month
    '한 달에 2~3번': 4.5,       # 2-3 times a month
    '1년에 1~2번': 2,           # 1-2 times a year
}

# OPTION TYPE DETECTION KEYWORDS
# Used to classify question types based on response option text
FREQUENCY_KEYWORDS = [
    '없다',      # none/never
    '1주일',     # week
    '한 달',     # month
    '1년',       # year
    '번',        # times (counter)
    '회',        # times (counter)
    '빈도',      # frequency
    '미만',      # less than
    '이상',      # more than
    '시간',      # hour/time
]

AGREEMENT_KEYWORDS = [
    '그렇다',        # agree
    '그렇지 않다',   # disagree
    '매우',          # very/strongly
    '전혀',          # not at all
    '편이다',        # tend to
]

# Response level inference keywords
LOW_LEVEL_KEYWORDS = [
    '전혀',      # not at all
    '없다',      # none
    '않는다',    # do not
    '않다',      # not
]

HIGH_LEVEL_KEYWORDS = [
    '매우',          # very
    '자주',          # often
    '항상',          # always
    '많이',          # a lot
    '그런 편이다',   # tend to agree
    '그렇다',        # agree
]

# NEGATIVE BEHAVIOR DETECTION KEYWORDS
# Used to identify questions related to negative behavioral indicators
NEGATIVE_KEYWORDS = [
    '무기력',        # lethargy
    '우울',          # depression
    '공격성',        # aggression
    '위축',          # withdrawal
    '비행',          # delinquency
    '거부',          # rejection
    '강요',          # coercion
    '비일관성',      # inconsistency
    '방해',          # interference
    '화가',          # anger
    '울 때',         # crying
    '싸우',          # fighting
    '트집',          # nitpicking
    '불행',          # unhappiness
    '외롭',          # loneliness
    '힘들',          # difficulty
    '쓸모없',        # uselessness
    '실패',          # failure
    '못하게',        # prevention
    '관심이 없',     # lack of interest
    '충돌',          # conflict
    '없다고 느끼',   # feeling of lack
]

# CSV COLUMN NAMES
# Column headers used in MIRROR-V dataset CSV files
CSV_COLUMNS = {
    # Question column names (priority order)
    'question': ['설문 문항', '문항'],           # Survey question, Question
    
    # Answer column names (priority order)
    'answer': ['응답 내용', '답변', '응답'],     # Response content, Answer, Response
    
    # Fallback keyword for finding answer columns
    'answer_keyword': '답',                      # Answer

    # Filter keywords for answer columns
    'answer_filter_exclude': ['코드', '자', '년도'],  # code, person, year
    
    # Filter keywords for invalid responses
    'invalid_response': ['중1', '패널'],         # Middle school 1st year, Panel
}

# DELINQUENCY DETECTION
# Keyword for identifying delinquency-related categories
DELINQUENCY_KEYWORD = "현실비행"                 # Real-world delinquency

# INPUT VARIABLE NAMES
# Demographic and background variable names in the dataset
INPUT_VARIABLE_NAMES = {
    'school_region': '시/도(학교 기준)',         # Province/City (school-based)
    'city_size': '도시규모(학교 기준)',          # City size (school-based)
    'residence_region': '시/도(거주지 기준)',    # Province/City (residence-based)
    'birth_year': '생년',                        # Birth year
    'gender': '성별',                            # Gender
    'siblings': '형제자매 수(본인포함)',         # Number of siblings (including self)
}

# For flexible matching in data loading
INPUT_VARIABLE_KEYWORDS = {
    'gender': '성별',           # Gender
    'birth_year': '생년',       # Birth year
    'region': ['거주지', '시/도'],  # Residence, Province/City
}

# CATEGORY NAMES
# Survey category names used in MIRROR-V dataset
CATEGORY_AGGRESSION = "공격성"                           # Aggression
CATEGORY_SCHOOL_VIOLENCE = "학교 폭력"                   # School Violence
CATEGORY_SOCIAL_WITHDRAWAL = "사회적 위축"               # Social Withdrawal
CATEGORY_DEPRESSION = "우울"                             # Depression
CATEGORY_ATTENTION = "주의집중"                          # Attention/Concentration
CATEGORY_FRIENDSHIP = "친구관계"                         # Friendship
CATEGORY_SELF_ESTEEM = "자아존중감"                      # Self-esteem
CATEGORY_LIFE_SATISFACTION = "삶의 만족도"               # Life Satisfaction
CATEGORY_HAPPINESS = "행복감"                            # Happiness
CATEGORY_DELINQUENCY = "현실비행 경험 유무 및 빈도"      # Real-world Delinquency

# Target categories for prediction
# These are now EXCLUDED from prediction to predict remaining 187 items
ORIGINAL_TARGET_CATEGORIES = [
    CATEGORY_AGGRESSION,
    CATEGORY_SCHOOL_VIOLENCE,
]

# Categories to EXCLUDE from prediction
PREDICTION_EXCLUDED_CATEGORIES = []

# Target categories for prediction
TARGET_CATEGORIES = [
    CATEGORY_AGGRESSION,
    CATEGORY_SCHOOL_VIOLENCE,
    CATEGORY_DELINQUENCY,
]

# Categories to exclude from historical data (--exclude-target option)
EXCLUDED_CATEGORIES = [
    CATEGORY_AGGRESSION,
    CATEGORY_SCHOOL_VIOLENCE,
    CATEGORY_DELINQUENCY,
]

# Partial exclusion (--exclude-partial option, includes aggression)
EXCLUDED_CATEGORIES_PARTIAL = [
    CATEGORY_SCHOOL_VIOLENCE,
    CATEGORY_DELINQUENCY,
]

# Negative behavior category keywords (for schema filtering)
NEGATIVE_BEHAVIOR_KEYWORDS = [
    "공격성",    # Aggression
    "위축",      # Withdrawal
    "우울",      # Depression
    "주의",      # Attention
    "신체",      # Physical
    "무기력",    # Lethargy
]

# Related category priority (for prediction reference)
RELATED_CATEGORY_PRIORITY = [
    CATEGORY_AGGRESSION,
    CATEGORY_SOCIAL_WITHDRAWAL,
    CATEGORY_DEPRESSION,
    CATEGORY_ATTENTION,
    CATEGORY_FRIENDSHIP,
]

# Categories for cross-validation analysis
WELLBEING_CATEGORIES = [
    CATEGORY_HAPPINESS,
    CATEGORY_DEPRESSION,
    CATEGORY_SELF_ESTEEM,
    CATEGORY_LIFE_SATISFACTION,
]

# TARGET DELINQUENCY ITEMS
# Specific question keywords for delinquency prediction targets
TARGET_DELINQUENCY_ITEMS = [
    "술 마시기",                          # Drinking alcohol
    "심한 욕설과 폭언",                     # Severe cursing and verbal abuse
    "무단결석",                          # Truancy
    "다른 사람 심하게 놀리거나 조롱하기",   # Severely teasing or mocking others
]

# Questions to EXCLUDE from prediction (specific items)
EXCLUDED_QUESTIONS = [
    "담배 피우기",  # Smoking - exclude from 25 target items
]

# DEFAULT VALUES
# Fallback values when data is missing
DEFAULT_OPTIONS = {
    "1": "그렇다",    # Yes/Agree
    "2": "아니다",    # No/Disagree
}

# SAMPLE CATEGORIES FOR TESTING/DISPLAY
# Used in main blocks for demonstration
SAMPLE_CATEGORIES_BEHAVIOR = [
    CATEGORY_AGGRESSION,
    CATEGORY_SCHOOL_VIOLENCE,
    CATEGORY_SOCIAL_WITHDRAWAL,
    CATEGORY_DEPRESSION,
    CATEGORY_SELF_ESTEEM,
]

SAMPLE_CATEGORIES_SIMILARITY = [
    CATEGORY_SCHOOL_VIOLENCE,
    CATEGORY_AGGRESSION,
    CATEGORY_SOCIAL_WITHDRAWAL,
    CATEGORY_DEPRESSION,
]