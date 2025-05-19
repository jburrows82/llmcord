from datetime import datetime as dt, timezone
from typing import Optional

from .constants import PROVIDERS_SUPPORTING_USERNAMES


def prepare_system_prompt(
    is_gemini: bool, provider: str, base_prompt_text: Optional[str]
) -> Optional[str]:
    """Constructs the system prompt string with dynamic elements based on the provided base text."""
    if not base_prompt_text:
        return None

    now_utc = dt.now(timezone.utc)
    hour_12 = now_utc.strftime("%I").lstrip("0")
    minute = now_utc.strftime("%M")
    am_pm = now_utc.strftime("%p")
    time_str = f"{hour_12}:{minute}\u202f{am_pm}"  # Narrow no-break space
    date_str = now_utc.strftime("%A, %B %d, %Y")
    current_datetime_str = (
        f"current date and time: {date_str} {time_str} Coordinated Universal Time (UTC)"
    )

    system_prompt_extras = [current_datetime_str]
    if not is_gemini and provider in PROVIDERS_SUPPORTING_USERNAMES:
        system_prompt_extras.append(
            "User's names are their Discord IDs and should be typed as '<@ID>'."
        )

    return "\n".join([base_prompt_text] + system_prompt_extras)
