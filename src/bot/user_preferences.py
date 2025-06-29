import json
import os
from typing import Dict, Any, Optional
import aiofiles
import aiofiles.os as aio_os
from ..core.constants import (
    USER_SYSTEM_PROMPTS_FILENAME,
    USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME,
    USER_MODEL_PREFS_FILENAME,
# Global preference dictionaries
user_model_preferences: Dict[int, str] = {}
user_system_prompt_preferences: Dict[int, Optional[str]] = {}
user_gemini_thinking_budget_preferences: Dict[int, bool] = {}
async def _load_user_preferences(filename: str) -> Dict[int, Any]:
    """Loads user preferences from a JSON file asynchronously."""
    if not await aio_os.path.exists(filename):
        return {}
    pass
    try:
        async with aiofiles.open(filename, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
            return {int(k): v for k, v in data.items()}
    except (json.JSONDecodeError, IOError) as e:
        return {}
    except Exception as e:
        return {}
async def _save_user_preferences(filename: str, data: Dict[int, Any]):
    """Saves user preferences to a JSON file asynchronously."""
    try:
        directory = os.path.dirname(filename)
        if directory and not await aio_os.path.exists(directory):
            await aio_os.makedirs(directory)
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=4))
    except (IOError, Exception) as e:
        pass
async def load_all_preferences():
    """Loads all user preferences from their respective files."""
    global user_model_preferences, user_system_prompt_preferences, user_gemini_thinking_budget_preferences
    loaded_model_prefs = await _load_user_preferences(USER_MODEL_PREFS_FILENAME)
    if loaded_model_prefs:
        user_model_preferences.update(loaded_model_prefs)
    loaded_system_prompts = await _load_user_preferences(USER_SYSTEM_PROMPTS_FILENAME)
    if loaded_system_prompts:
        user_system_prompt_preferences.update(loaded_system_prompts)
    loaded_gemini_prefs = await _load_user_preferences(USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME)
    if loaded_gemini_prefs:
        user_gemini_thinking_budget_preferences.update(loaded_gemini_prefs)
def get_user_model_preference(user_id: int, default_model: str) -> str:
    """Gets the user's model preference or the default."""
    return user_model_preferences.get(user_id, default_model)
def get_user_system_prompt_preference(user_id: int, default_prompt: Optional[str]) -> Optional[str]:
    """Gets the user's system prompt preference."""
    user_specific_prompt = user_system_prompt_preferences.get(user_id)
    if user_specific_prompt is None and user_id in user_system_prompt_preferences:
        return default_prompt
    elif user_specific_prompt is not None:
        return user_specific_prompt
    else:
        return default_prompt
def get_user_gemini_thinking_budget_preference(user_id: int, default_enabled: bool) -> bool:
    """Gets the user's preference for using the Gemini thinking budget."""
    return user_gemini_thinking_budget_preferences.get(user_id, default_enabled)
async def save_model_preference(user_id: int, model_full_name: str):
    """Save user's model preference."""
    user_model_preferences[user_id] = model_full_name
    await _save_user_preferences(USER_MODEL_PREFS_FILENAME, user_model_preferences)
async def save_system_prompt_preference(user_id: int, prompt: Optional[str]):
    """Save user's system prompt preference."""
    user_system_prompt_preferences[user_id] = prompt
    await _save_user_preferences(USER_SYSTEM_PROMPTS_FILENAME, user_system_prompt_preferences)
async def save_gemini_thinking_preference(user_id: int, enabled: bool):
    """Save user's Gemini thinking budget preference."""
    user_gemini_thinking_budget_preferences[user_id] = enabled
    await _save_user_preferences(USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME, user_gemini_thinking_budget_preferences) 