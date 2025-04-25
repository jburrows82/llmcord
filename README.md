<h1 align="center">
  llmcord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/jakobdylanc/llmcord/assets/38699060/789d49fe-ef5c-470e-b60e-48ac03057443" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

## Features

### Reply-based chat system
Just @ the bot **or include the phrase `at ai` (case-insensitive) anywhere in your message** to start a conversation. Reply to the bot's response to continue the chat. The bot mention (`@BotName` or `at ai`) is automatically removed before the query is sent to the LLM (e.g., `hello at ai` becomes `hello`). Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot **or include `at ai`** while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot **or include `at ai`**. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot **or include `at ai`** inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

### Choose any LLM
llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [Google Gemini API](https://ai.google.dev/docs) (via `google-genai`)
- [xAI API](https://docs.x.ai/docs/models)
- [Mistral API](https://docs.mistral.ai/getting-started/models/models_overview)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter API](https://openrouter.ai/models)

Or run a local model with:
- [Ollama](https://ollama.com)
- [LM Studio](https://lmstudio.ai)
- [vLLM](https://github.com/vllm-project/vllm)

...Or use any other OpenAI compatible API server.

- **Multiple API Key Support:** Configure multiple API keys for each LLM provider (see `config.yaml`).
- **Automatic Key Rotation & Retry:** The bot randomly rotates between available keys for each request. If an error occurs (including during response streaming), it automatically retries with another key until the request succeeds.

### YouTube Content Extraction
- Include YouTube URLs directly in your query.
- The bot automatically extracts the video's **title**, **description**, **channel name**, **transcript** (using `youtube-transcript-api`), and up to **10 top comments** (using YouTube Data API v3).
- This extracted information is appended to your original query before being sent to the LLM, providing rich context for summarization, analysis, or question-answering based on the video content.
- Handles multiple YouTube URLs within the same message concurrently.
- **Requires a YouTube Data API v3 key** configured in `config.yaml`.

### Reddit Content Extraction
- Include Reddit submission URLs directly in your query (e.g., `https://www.reddit.com/r/.../comments/.../`).
- The bot automatically extracts the submission's **title**, **self-text** (if any), and up to **10 top-level comments** using `asyncpraw`.
- This extracted information is appended to your original query before being sent to the LLM, providing context for summarization, analysis, or question-answering based on the Reddit post.
- Handles multiple Reddit URLs within the same message concurrently.
- **Requires Reddit API credentials** (client ID, client secret, user agent) configured in `config.yaml`. You can obtain these by creating a 'script' app on Reddit's [app preferences page](https://www.reddit.com/prefs/apps).

### General URL Content Extraction
- Include most HTTP/HTTPS URLs directly in your query (excluding YouTube and Reddit, which have specialized handling).
- The bot automatically fetches the content of the URL using `httpx` and parses the main textual content using `BeautifulSoup4`.
- Extracted text (limited length) is appended to your original query before being sent to the LLM, providing context for summarization, analysis, or question-answering based on the web page content.
- Handles multiple URLs within the same message concurrently.
- Basic error handling included (timeouts, non-HTML content, fetch errors).

### SerpAPI Google Lens Integration
- Trigger image analysis by starting your query with `googlelens` (case-insensitive) after mentioning the bot or using `at ai` (e.g., `@BotName googlelens what is this?` or `at ai googlelens identify this object`).
- Requires image(s) to be attached to the message.
- The bot uses the attached image(s) as input for SerpAPI's Google Lens API.
- Fetches visual matches and related content for each image concurrently.
- Appends the formatted SerpAPI results to your original query before sending it to the LLM, providing visual context for analysis or identification.
- Handles multiple image attachments in a single query.
- **Requires SerpAPI API keys** configured in `config.yaml`.
- **Multiple API Key Support:** Configure multiple SerpAPI keys in `config.yaml`.
- **Automatic Key Rotation & Retry:** The bot randomly rotates between available keys. If a request fails (e.g., rate limit), it automatically retries with another key.

### Robust Rate Limit Handling
- **Persistent Cooldown:** Rate-limited API keys (LLM providers & SerpAPI) are automatically detected and put on a 24-hour cooldown to prevent reuse.
- **SQLite Database:** Cooldown information is stored in separate SQLite databases per service (in the `ratelimit_dbs/` folder), ensuring the cooldown persists even if the bot restarts.
- **Automatic Reset:** Databases are automatically reset every 24 hours (tracked via `last_reset_timestamp.txt`). Additionally, if *all* keys for a specific service become rate-limited, that service's database is reset immediately to allow retries.
- **Silent Retries:** Error messages are only sent to Discord if *all* configured keys for a service have been tried and failed for a single request.

### Gemini Grounding with Sources
- When using a compatible Gemini model (e.g., `gemini-2.0-flash`), the bot can automatically use Google Search to ground its responses with up-to-date information.
- If a response was enhanced by grounding, a "Show Sources" button will appear below the message.
- Clicking "Show Sources" reveals the search queries the model used and the web sources it consulted to generate the response.

### And more:
- Supports image attachments when using a vision model (like gpt-4.1, claude-3, gemini-flash, llama-4, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable personality (aka system prompt)
- User identity aware (OpenAI API and xAI API only)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded, or "⚠️ Could not fetch all YouTube data" if YouTube API calls fail)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- 1 Python file, ~1700 lines of code

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile.<br />**Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br />(Default: `100,000`) |
| **max_images** | The maximum number of image attachments allowed in a single message.<br />**Only applicable when using a vision model.**<br />(Default: `5`) |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br />(Default: `25`) |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often.<br />**Also disables streamed responses, warning messages, and the 'Show Sources' button.**<br />(Default: `false`) |
| **allow_dms** | Set to `false` to disable direct message access.<br />(Default: `true`) |
| **permissions** | Configure permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`.<br />**Leave `allowed_ids` empty to allow ALL.**<br />**Role and channel permissions do not affect DMs.**<br />**You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control channel permissions in groups.** |

### YouTube Data API v3 settings:

| Setting | Description |
| --- | --- |
| **youtube_api_key** | **Required for YouTube URL processing.** Get an API key from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials). Ensure the **YouTube Data API v3** is enabled for your project. This key is used to fetch video details (title, description, channel) and comments. Transcripts are fetched using `youtube-transcript-api` and do not require this key. *(Note: Currently only supports a single key).* |

### Reddit API settings:

| Setting | Description |
| --- | --- |
| **reddit_client_id** | **Required for Reddit URL processing.** Your Reddit script app's client ID. |
| **reddit_client_secret** | **Required for Reddit URL processing.** Your Reddit script app's client secret. |
| **reddit_user_agent** | **Required for Reddit URL processing.** A unique user agent string (e.g., `discord:my-llm-bot:v1.0 (by u/your_reddit_username)`). |

### SerpAPI settings:

| Setting | Description |
| --- | --- |
| **serpapi_api_keys** | **Required for Google Lens image processing.** A **list** of API keys from [SerpApi](https://serpapi.com/manage-api-key). The bot will rotate through these keys and retry if one fails (e.g., due to rate limits). |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use. For OpenAI compatible APIs, provide a `base_url` and a **list** of `api_keys`. For Google Gemini, just provide a **list** of `api_keys`. For keyless providers (like Ollama), provide an empty list `[]` for `api_keys`.<br />**Gemini uses the `google-genai` library, others use OpenAI compatible APIs.**<br />The bot rotates through keys and retries on failure. |
| **model** | Set to `<provider name>/<model name>`, e.g:<br />-`openai/gpt-4.1`<br />-`google/gemini-2.0-flash`<br />-`ollama/llama3.3`<br />-`openrouter/anthropic/claude-3.7-sonnet` |
| **extra_api_parameters** | Extra API parameters for your LLM. Add more entries as needed.<br />**Refer to your provider's documentation for supported API parameters.**<br />(Default: `max_tokens=4096, temperature=1.0` for OpenAI compatible)<br />(Gemini uses parameters like `max_output_tokens`, `temperature`, `top_p`, `top_k`) |
| **system_prompt** | Write anything you want to customize the bot's behavior!<br />**Leave blank for no system prompt.** |

3. Install requirements:
   ```bash
   python -m pip install -U -r requirements.txt
   ```
   *(Note: This now includes `youtube-transcript-api`, `google-api-python-client`, `asyncpraw`, `beautifulsoup4`, and `google-search-results`)*

4. Run the bot:

   **No Docker:**
   ```bash
   python llmcord.py
   ```

   **With Docker:**
   ```bash
   docker compose up
   ```

## Notes

- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)

- Only models from OpenAI API and xAI API are "user identity aware" because only they support the "name" parameter in the message object. Hopefully more providers support this in the future.

- Gemini safety settings are currently hardcoded to `BLOCK_NONE` for all categories.

- YouTube Data API has usage quotas. Fetching details and comments for many videos may consume your quota quickly. Check the [Google Cloud Console](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas) for details.

- Reddit API also has rate limits. Processing many Reddit URLs quickly might lead to temporary throttling.

- General URL fetching uses `httpx` and `BeautifulSoup4`. It might fail on complex JavaScript-heavy sites or sites with strong anti-scraping measures. Content extraction focuses on main text areas and might miss some information or include unwanted elements.

- SerpAPI Google Lens usage counts towards your SerpAPI plan's search credits. The bot rotates keys and handles rate limits across your provided keys.

- **Rate Limit Handling:** The bot uses SQLite databases (in the `ratelimit_dbs/` folder) to track rate-limited API keys (LLM & SerpAPI). Keys are put on a 24-hour cooldown. This cooldown state persists even if the bot restarts (tracked via `last_reset_timestamp.txt`). If all keys for a service become rate-limited, the cooldown is reset for that service. Error messages are only sent to Discord if all keys fail for a request. Ensure `ratelimit_dbs/` and `last_reset_timestamp.txt` are added to your `.gitignore` if you manage your deployment with git.

- PRs are welcome :)

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
  </picture>
</a>
