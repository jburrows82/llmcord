"""
Configuration defaults and template values for LLMCord.
"""
# Alternative Search Query Generation Defaults
ALT_SEARCH_SECTION_KEY = "alternative_search_query_generation"
ALT_SEARCH_ENABLED_KEY = "enabled"
ALT_SEARCH_PROMPT_KEY = "search_query_generation_prompt_template"
DEFAULT_ALT_SEARCH_PROMPT_TEMPLATE = """
<task>
Analyze the latest query to determine if it requires web search. Consider the chat history for context.
</task>
<criteria>
Web search IS required when the query asks about:
- Current events, news, or recent developments
- Real-time information (prices, weather, stock data)
- Specific facts that may have changed recently
- Information about new products, services, or updates
- People, places, or organizations where current status matters
Web search is NOT required when the query asks about:
- General knowledge or established facts
- Conceptual explanations or definitions
- Personal opinions or advice
- Mathematical calculations or logic problems
- Analysis of provided information
</criteria>
<instructions>
1. Analyze the latest query in the context of the chat history
2. Determine if web search is required based on the criteria above
3. If web search is required, generate specific search queries that would find the needed information
4. For queries with multiple distinct subjects, create separate search queries for each
5. Return your response in the exact JSON format shown in the examples
</instructions>
<examples>
<example>
<chat_history>
User: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...
</chat_history>
<latest_query>Can you explain deep learning?</latest_query>
<output>
{"web_search_required": false}
</output>
</example>
<example>
<chat_history>
User: I'm interested in electric vehicles
Assistant: Electric vehicles are becoming increasingly popular...
</chat_history>
<latest_query>What are Tesla's latest Model 3 prices and what new features did Apple announce for iPhone 15?</latest_query>
<output>
{
    "web_search_required": true,
    "search_queries": [
    "Tesla Model 3 prices 2024",
    "Apple iPhone 15 new features announcement"
    ]
}
</output>
</example>
<example>
<chat_history>
User: Tell me about climate change
Assistant: Climate change refers to long-term shifts in global temperatures...
</chat_history>
<latest_query>What were the key outcomes of the latest UN climate summit?</latest_query>
<output>
{
    "web_search_required": true,
    "search_queries": ["UN climate summit latest outcomes 2024"]
}
</output>
</example>
</examples>
<chat_history>
{chat_history}
</chat_history>
<latest_query>
{latest_query}
</latest_query>
<output_format>
Return ONLY valid JSON in one of these formats:
- If no search needed: {"web_search_required": false}
- If search needed: {"web_search_required": true, "search_queries": ["query1", "query2", ...]}
</output_format>
""".strip()
# Default System Prompts
DEFAULT_GROUNDING_SYSTEM_PROMPT = """
You are an expert at analyzing user queries and conversation history to determine the most effective web search queries that will help answer the user's latest request or continue the conversation meaningfully.
Based on the provided conversation history (especially the last user message), output a list of concise and targeted search queries.
Focus on identifying key entities, concepts, questions, or current events mentioned that would benefit from fresh information from the web.
If the user's query is a direct question, formulate search queries that would find the answer.
If the user is discussing a topic, formulate queries that would find recent developments, facts, or relevant discussions.
Do not generate more than 5 search queries.
Output only the search queries, each on a new line. Do not add any other text, preamble, or explanation.
""".strip()
DEFAULT_FALLBACK_MODEL_SYSTEM_PROMPT = """
You are a helpful AI assistant. Provide accurate, helpful, and concise responses to user queries.
""".strip()
DEFAULT_PROMPT_ENHANCER_SYSTEM_PROMPT = """
You are an expert prompt engineer. Your task is to refine a user's input to make it a more effective prompt for a large language model.
Follow the provided prompt design strategies and guides to improve the user's original prompt.
Output *only* the improved prompt, without any preamble, explanation, or markdown formatting.
""".strip()
# Numeric Defaults
DEFAULT_MAX_MESSAGE_NODES = 500
DEFAULT_EDIT_DELAY_SECONDS = 1.0
DEFAULT_MAX_MESSAGES = 25
DEFAULT_MAX_IMAGE_FILES = 5
DEFAULT_SEARXNG_NUM_RESULTS = 5
# Gemini Defaults
DEFAULT_GEMINI_USE_THINKING_BUDGET = False
DEFAULT_GEMINI_THINKING_BUDGET_VALUE = 8192
DEFAULT_GEMINI_SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}
# Grounding Model Parameter Defaults
DEFAULT_GROUNDING_MODEL_TEMPERATURE = 0.7
DEFAULT_GROUNDING_MODEL_TOP_K = 40
DEFAULT_GROUNDING_MODEL_TOP_P = 0.9
DEFAULT_GROUNDING_MODEL_USE_THINKING_BUDGET = False
DEFAULT_GROUNDING_MODEL_THINKING_BUDGET_VALUE = 8192
# Web Content API Defaults
DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED = False
DEFAULT_WEB_CONTENT_EXTRACTION_API_URL = ""
DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS = 10
DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL = 3600
DEFAULT_HTTP_CLIENT_USE_HTTP2 = True
# URL Extractor Defaults
DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR = "jina"
DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR = "crawl4ai"
# Jina Defaults
DEFAULT_JINA_ENGINE_MODE = "fast"
DEFAULT_JINA_WAIT_FOR_SELECTOR = ""
DEFAULT_JINA_TIMEOUT = 30
# Crawl4AI Defaults
DEFAULT_CRAWL4AI_CACHE_MODE = "enabled"
# Table Rendering Defaults
DEFAULT_AUTO_RENDER_MARKDOWN_TABLES = True
# Text Limits Defaults
DEFAULT_MAX_TEXT_KEY = "default_max_text"
MODEL_SPECIFIC_MAX_TEXT_KEY = "model_specific_max_text"
DEFAULT_MAX_TEXT_SAFETY_MARGIN = 5000  # Fixed amount (not percentage) to match original behavior
DEFAULT_MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN = 1000
DEFAULT_MAX_TEXT_VALUE = 128000  # Match original default of 128k tokens
# Rate Limiting Defaults
DEFAULT_RATE_LIMIT_COOLDOWN_HOURS = 24
# Discord Defaults
DEFAULT_STATUS_MESSAGE = "github.com/jakobdylanc/llmcord"
DEFAULT_ALLOW_DMS = True
DEFAULT_USE_PLAIN_RESPONSES = False
# Output Sharing Defaults
DEFAULT_OUTPUT_SHARING_CONFIG = {
    "textis_enabled": False,
    "url_shortener_enabled": False,
    "url_shortener_service": "tinyurl",
    "cleanup_on_shutdown": True,
} 