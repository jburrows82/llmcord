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
- [Jan](https://jan.ai)
- [Oobabooga's WebUI](https://github.com/oobabooga/text-generation-webui) (via its OpenAI compatible API)

...Or use any other OpenAI compatible API server.

- **Multiple API Key Support:** Configure multiple API keys for each LLM provider (see `config.yaml`).
- **Automatic Key Rotation & Retry:** The bot randomly rotates between available keys for each request. If an error occurs (including during response streaming), it automatically retries with another key until the request succeeds.
- **User-Specific Preferences:** Users can set their preferred model using the `/model` command (see "Slash Commands" below).

### Slash Commands
llmcord offers slash commands for user-specific customizations:
- **`/model <provider/model_name>`**: Set your preferred LLM provider and model (e.g., `openai/gpt-4.1`, `google/gemini-2.5-flash-preview-04-17`). This preference is saved and used for your future messages.
- **`/systemprompt <prompt_text | "reset">`**: Set a custom system prompt for your interactions with the bot. Use `reset` to revert to the default system prompt defined in `config.yaml`.
- **`/setgeminithinking <True|False>`**: Enable or disable the `thinkingBudget` parameter for Gemini models for your interactions. This can potentially improve response quality for complex queries but may increase latency. The actual budget value is set globally in `config.yaml`.
- **`/help`**: Displays a help message listing all available commands and their usage.
User preferences for these commands are saved locally in JSON files (e.g., `user_model_prefs.json`).

### YouTube Content Extraction
- Include YouTube URLs directly in your query.
- The bot automatically extracts the video's **title**, **description**, **channel name**, **transcript** (using `youtube-transcript-api`), and up to **10 top comments** (using YouTube Data API v3).
- This extracted information is appended to your original query before being sent to the LLM, providing rich context for summarization, analysis, or question-answering based on the video content.
- Handles multiple YouTube URLs within the same message concurrently.
- **Requires a YouTube Data API v3 key** configured in `config.yaml`.
- **Optional Proxy Support:** Configure proxies in `config.yaml` to help avoid YouTube IP blocks that can cause transcript fetching errors (like `ParseError`). See `config-example.yaml`.

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

### Google Lens Integration (via SerpAPI)
- Trigger image analysis by starting your query with `googlelens` (case-insensitive) after mentioning the bot or using `at ai` (e.g., `@BotName googlelens what is this?` or `at ai googlelens identify this object`).
- Requires image(s) to be attached to the message.
- The bot uses **SerpAPI's Google Lens API** to fetch visual matches and related content for each image concurrently.
- **Requires SerpAPI API keys** configured in `config.yaml`.
- **Multiple API Key Support:** Configure multiple SerpAPI keys in `config.yaml`.
- **Automatic Key Rotation & Retry:** The bot randomly rotates between available keys. If a request fails (e.g., rate limit), it automatically retries with another key.
- Appends the formatted results to your original query before sending it to the LLM, providing visual context for analysis or identification.
- Handles multiple image attachments in a single query.

### Advanced Query Handling & Context Enhancement
llmcord employs several strategies to enrich the context provided to the LLM:
- **Deep Search Keyword**: Including `deepsearch` or `deepersearch` (case-insensitive) in your query automatically attempts to switch to a pre-configured powerful model (default: `x-ai/grok-3`) for that specific request, if the model and its API keys are configured.
- **Vision Model Fallback**: If your message includes images but your currently selected model doesn't support vision, the bot will automatically switch to a pre-configured fallback vision model (default: `google/gemini-2.5-flash-preview-04-17`) for that request, if available. A warning will inform you of the switch.
- **SearXNG Integration for Grounding**: For non-Gemini/non-Grok models, if your query doesn't contain URLs, the bot can perform a pre-step:
    1. A Gemini model (`gemini-2.5-flash-preview-04-17` by default) analyzes your query and conversation history to generate relevant web search queries.
    2. These queries are run against your configured SearXNG instance.
    3. Content from the top SearXNG result URLs (YouTube, Reddit, general web pages) is fetched, processed (with a configurable length limit for general web content via `searxng_url_content_max_length`), and formatted.
    4. This fetched web context is then provided to your chosen LLM along with your original query.
    This feature requires `searxng_base_url` and API keys for the grounding Gemini model to be configured.
- **PDF Attachment Processing**:
    - When using Gemini models (with vision/file capabilities enabled), PDF attachments are sent directly to the model.
    - For other models, text is extracted from PDF attachments using `PyMuPDF` and appended to your query.
- **Image URL Processing**: If you include direct URLs to images (e.g., `https://example.com/image.png`) in your message text, the bot will attempt to download these images and treat them as if they were attached directly, making them available to vision models or Google Lens.

### Robust Rate Limit Handling
- **Persistent Cooldown:** Rate-limited API keys (LLM providers & SerpAPI) are automatically detected and put on a 24-hour cooldown to prevent reuse.
- **SQLite Database:** Cooldown information is stored in separate SQLite databases per service (in the `ratelimit_dbs/` folder), ensuring the cooldown persists even if the bot restarts.
- **Automatic Reset:** Databases are automatically reset every 24 hours (tracked via `last_reset_timestamp.txt`). Additionally, if *all* keys for a specific service become rate-limited, that service's database is reset immediately to allow retries.
- **Silent Retries:** Error messages are only sent to Discord if *all* configured keys for a service have been tried and failed for a single request.

### Interactive Response & Sources
- **Gemini Grounding**: When using a compatible Gemini model (e.g., `gemini-2.5-flash-preview-04-17`), the bot can automatically use Google Search to ground its responses with up-to-date information.
- **Persistent Action Buttons**: If a response was enhanced by grounding or if there's a response text, action buttons will appear below the message. These buttons are persistent and do not time out.
    - **"Show sources" Button**: If grounding was used, this button reveals the search queries the model used and the web sources it consulted. The sources are displayed in embeds, split into multiple messages if necessary.
    - **"Get response as text file" Button**: Allows you to download the bot's full response as a `.txt` file.
    - **"View output properly (especially tables)" Button**: If enabled in `config.yaml`, this button shares the LLM's full Markdown response via a temporary public ngrok URL. The content is rendered as an HTML page using a local Grip server, making complex Markdown (like tables) easier to read.

### And more:
- **Slash Commands**: `/model`, `/systemprompt`, `/setgeminithinking`, `/help` for user-specific preferences and help (see "Slash Commands" section).
- Supports image attachments when using a vision model (like `gpt-4.1`, `claude-3`, `gemini-2.5-flash-preview-04-17`, etc.).
- Supports text file attachments (.txt, .py, .c, etc.) and PDF attachments (see "Advanced Query Handling").
- Customizable personality (default system prompt in `config.yaml`, user-overridable via `/systemprompt`).
- User identity aware (OpenAI API and xAI API only, sends user's Discord ID as `name`).
- Streamed responses (turns green when complete, automatically splits into separate messages when too long).
- **Imgur URL Resending**: If the LLM includes Imgur URLs under a specific header in its response, the bot automatically resends these URLs as separate messages to ensure they embed properly.
- **Configurable OpenAI Vision**: Vision capabilities for OpenAI models can be disabled via `disable_vision: true` in the provider config.
- **Configurable Gemini Thinking Budget**: The `thinkingBudget` for Gemini models can be enabled and its value set globally in `config.yaml`, and users can toggle its use for their sessions via `/setgeminithinking`.
- Hot reloading config (you can change settings without restarting the bot).
- Displays helpful warnings when appropriate (e.g., message limits, content fetching issues, model fallbacks).
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls.
- Fully asynchronous.
- Modular Python codebase.

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   cd llmcord # Navigate into the cloned directory
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile.<br />**Max 128 characters.** |
| **max_text** | The maximum total number of **tokens** (counted using `tiktoken`) for the entire chat history sent to the LLM. This includes text from all messages in the context, attached text files, and extracted text from PDFs (for non-Gemini models). If this limit is exceeded, the oldest messages (preferring user/model pairs) are removed from the history until the total token count is within the limit. If removing all prior history is still not enough, the latest user query itself will be truncated.<br />(Default: `2000`) |
| **max_images** | The maximum number of image attachments allowed in a single message.<br />**Only applicable when using a vision model or Google Lens.**<br />(Default: `5`) |
| **max_messages** | The maximum number of messages allowed in a reply chain. This acts as an initial cap before token-based truncation with `max_text` occurs.<br />(Default: `25`) |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often.<br />**Also disables streamed responses, warning messages, and the 'Show sources' button.**<br />(Default: `false`) |
| **allow_dms** | Set to `false` to disable direct message access.<br />(Default: `true`) |
| **permissions** | Configure permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`.<br />**Leave `allowed_ids` empty to allow ALL.**<br />**Role and channel permissions do not affect DMs.**<br />**You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control channel permissions in groups (including for threads within those categories/channels).** |

### YouTube Data API v3 settings:

| Setting | Description |
| --- | --- |
| **youtube_api_key** | **Required for YouTube URL processing.** Get an API key from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials). Ensure the **YouTube Data API v3** is enabled for your project. This key is used to fetch video details (title, description, channel) and comments. Transcripts are fetched using `youtube-transcript-api` (which has its own proxy settings) and do not require this key. *(Note: Currently only supports a single key).* |

### Reddit API settings:

| Setting | Description |
| --- | --- |
| **reddit_client_id** | **Required for Reddit URL processing.** Your Reddit script app's client ID. |
| **reddit_client_secret** | **Required for Reddit URL processing.** Your Reddit script app's client secret. |
| **reddit_user_agent** | **Required for Reddit URL processing.** A unique user agent string (e.g., `discord:my-llm-bot:v1.0 (by u/your_reddit_username)`). |

### Proxy settings (for youtube-transcript-api):

| Setting | Description |
| --- | --- |
| **proxy_config** | **Optional:** Configure a proxy for `youtube-transcript-api` to potentially bypass YouTube IP blocks that cause transcript fetching errors (like `ParseError: no element found`). See `config-example.yaml` for setup details (Webshare recommended). |

### SerpAPI settings:

| Setting | Description |
| --- | --- |
| **serpapi_api_keys** | **Required for Google Lens.** A **list** of API keys from [SerpApi](https://serpapi.com/manage-api-key). The bot will rotate through these keys and retry if one fails (e.g., due to rate limits). |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use. For OpenAI compatible APIs, provide a `base_url` and a **list** of `api_keys`. For Google Gemini, just provide a **list** of `api_keys`. For keyless providers (like Ollama, Jan), provide an empty list `[]` for `api_keys`.<br />**Gemini uses the `google-genai` library, others use OpenAI compatible APIs.**<br />The bot rotates through keys and retries on failure.<br />For OpenAI, you can add `disable_vision: true` to prevent image processing even if the model supports it. |
| **model** | Set the default model to `<provider name>/<model name>`, e.g:<br />-`openai/gpt-4.1`<br />-`google/gemini-2.5-flash-preview-04-17`<br />-`ollama/llama3`<br />-`openrouter/anthropic/claude-3.5-sonnet`<br />Users can override this with the `/model` command. |
| **searxng_base_url** | **Optional, for SearXNG grounding.** The base URL of your SearXNG instance (e.g., `http://localhost:18088`). Required if you want non-Gemini/non-Grok models to have web search capabilities. |
| **searxng_url_content_max_length** | **Optional, for SearXNG grounding.** Maximum character length for text extracted from each URL fetched via SearXNG results. (Default: `20000`) |
| **grounding_system_prompt** | **Optional, for SearXNG grounding.** The system prompt used for the Gemini model that generates search queries for SearXNG. |
| **gemini_use_thinking_budget** | **Optional, for Gemini models.** Set to `true` to enable the `thinkingBudget` parameter by default for Gemini models. Users can override with `/setgeminithinking`. (Default: `false`) |
| **gemini_thinking_budget_value** | **Optional, for Gemini models.** The actual budget value (0-24576) to use if `gemini_use_thinking_budget` is enabled. (Default: `0`) |
| **extra_api_parameters** | Extra API parameters for the selected LLM's provider. Add more entries as needed.<br />**Refer to your provider's documentation for supported API parameters.**<br />(Default: `max_tokens=4096, temperature=1.0` for OpenAI compatible)<br />(Gemini uses parameters like `max_output_tokens`, `temperature`, `top_p`, `top_k`) |
| **system_prompt** | The default system prompt to customize the bot's behavior. Users can set their own with `/systemprompt`.<br />**Leave blank for no default system prompt.** |

### Output Sharing Settings (for "View output properly" button):

| Setting | Description |
| --- | --- |
| **output_sharing.ngrok_enabled** | Set to `true` to enable the "View output properly" button, which uses ngrok to share rendered Markdown. (Default: `false`) |
| **output_sharing.ngrok_authtoken** | Your ngrok authtoken. Optional but recommended for stable ngrok usage. Find it on your [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken). |
| **output_sharing.grip_port** | The local port number for the Python HTTP server (formerly Grip server) that renders the Markdown to HTML. (Default: `6419`) |
| **output_sharing.ngrok_static_domain** | **Optional:** If you have a reserved static domain with ngrok (requires a paid plan), you can specify it here (e.g., `your-cool-domain.ngrok-free.app`). Otherwise, a random ngrok subdomain will be used. |
| **output_sharing.cleanup_on_shutdown** | Set to `false` to keep the generated HTML files in the `shared_html/` directory across bot restarts. If `true`, the directory is cleaned up when the bot shuts down. (Default: `true`) |
| **output_sharing.url_shortener_enabled** | Set to `true` to shorten the ngrok URL using a service like TinyURL. (Default: `false`) |
| **output_sharing.url_shortener_service** | The URL shortener service to use. Currently supported: `"tinyurl"`. (Default: `"tinyurl"`) |

### User Preferences:
- User-specific settings for model, system prompt, and Gemini thinking budget are stored in JSON files at the root of the project (e.g., `user_model_prefs.json`, `user_system_prompts.json`, `user_gemini_thinking_budget_prefs.json`). Consider adding these to your `.gitignore` if you manage your deployment with git.

3. Install requirements:
   ```bash
    # Ensure you are in the 'llmcord' directory (the one containing requirements.txt)
    python -m pip install -U -r requirements.txt
    ```
   *(Note: This includes `youtube-transcript-api`, `google-api-python-client`, `asyncpraw`, `beautifulsoup4`, `google-search-results` for SerpAPI, `PyMuPDF` for PDF text extraction, and `tiktoken` for token counting.)*

4. Run the bot:

   **No Docker:**
   ```bash
   # Ensure you are in the 'llmcord' directory (the parent of 'llmcord_app')
   python -m llmcord_app.main
   ```
   *(Note: You must use `python -m llmcord_app.main` because the code is now structured as a Python package (`llmcord_app`) and this command tells Python to run the `main.py` file within that package.)*

   **With Docker:**
   ```bash
   docker compose up
    ```

## Notes

- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)

- Only models from OpenAI API and xAI API are "user identity aware" because only they support the "name" parameter in the message object. Hopefully more providers support this in the future.

- Gemini safety settings are currently hardcoded to `BLOCK_NONE` for all categories.

- YouTube Data API has usage quotas. Fetching details and comments for many videos may consume your quota quickly. Check the [Google Cloud Console](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas) for details.
- YouTube transcript fetching (via `youtube-transcript-api`) can be blocked by YouTube, especially when running from cloud IPs. This often results in a `ParseError: no element found` error. Configure the optional `proxy_config` in `config.yaml` (using Webshare residential proxies is recommended) to mitigate this.

- Reddit API also has rate limits. Processing many Reddit URLs quickly might lead to temporary throttling.

- General URL fetching uses `httpx` and `BeautifulSoup4`. It might fail on complex JavaScript-heavy sites or sites with strong anti-scraping measures. Content extraction focuses on main text areas and might miss some information or include unwanted elements. Content from general URLs fetched via SearXNG grounding has a configurable length limit (`searxng_url_content_max_length`).

- Google Lens processing uses SerpAPI. Ensure you have valid SerpAPI keys configured.

- PDF text extraction for non-Gemini models is done using `PyMuPDF`.

- **Rate Limit Handling & User Preferences:** The bot uses SQLite databases (in the `ratelimit_dbs/` folder) and JSON files (e.g., `user_model_prefs.json`) at the project root. Ensure these (and `last_reset_timestamp.txt`) are added to your `.gitignore` if you manage your deployment with git.

- PRs are welcome :)

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
  </picture>
</a>
