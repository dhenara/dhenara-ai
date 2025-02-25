"""
Default settings for Dhenara-AI.
All settings can be overridden by creating a settings file and setting DHENARA_SETTINGS_MODULE.
"""

# Request /Content Settings
ENABLE_PROMPT_VALIDATION = True  # Validate promots before api-calls. Might have issues at the moment

# Response Content Settings
ENABLE_USAGE_TRACKING = True
ENABLE_COST_TRACKING = True  # When set,ENABLE_USAGE_TRACKING  will be always true

# TODO:  Implement below settings
API_TIMEOUT = 30
MAX_RETRIES = 3

# Debug Settings
DEBUG = False
ENABLE_DETAILED_LOGGING = False
