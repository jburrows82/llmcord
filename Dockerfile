FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U -r requirements.txt

# Install Playwright browsers (specifically Chrome in this case)
# This adds significant size to the image
RUN python -m playwright install chrome --with-deps

# Copy the rest of the application code
COPY . .

# Command to run the bot using the new entry point via the module flag
CMD ["python", "-m", "llmcord_app.main"]