"""
Default settings for Dhenara-AI.
All settings can be overridden by creating a settings file and setting DHENARA_SETTINGS_MODULE.
"""

#  ---- Request /Content Settings
"""
Validate promots before api-calls. Might have issues at the moment
"""
ENABLE_PROMPT_VALIDATION = True

# ----  Response Content Settings

"""Enable Usage Tracking"""
ENABLE_USAGE_TRACKING = True

"""
Enable cost data creation in response.
When set,ENABLE_USAGE_TRACKING  will be always true
"""
ENABLE_COST_TRACKING = True

"""
 Consolidate streaming responses, and returns a consolidated response at the end
"""
ENABLE_STREAMING_CONSOLIDATION = True

# TODO:  Implement below settings
API_TIMEOUT = 30
MAX_RETRIES = 3

# Debug Settings
DEBUG = False
ENABLE_DETAILED_LOGGING = False
