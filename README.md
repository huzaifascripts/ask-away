# Ask Away

A small AI chat app. The page is plain HTML, CSS and JavaScript, a Flask server sits behind it, and free models from OpenRouter and Groq do the answering.

I built it as a learning project, mainly to figure out how a chat app works from end to end: streaming replies, handling a flaky free API, and keeping keys out of the browser.

**Live demo:** https://ask-away-alpha.vercel.app/

## Why there's a backend at all

The first version called the AI API straight from the browser. That works, but it means the API key sits in the page, and anyone can open DevTools, copy it, and burn through your quota.

So now the browser only talks to `/api/chat` on this server, and the server is the only thing that knows the keys.

## What it does

- **Streams answers** as they're generated, so you see text appear instead of waiting for the whole reply.
- **Falls back between models.** You give it a list, and it tries them in order. If one errors out or takes more than 15 seconds to start writing, it moves on to the next. If a provider's key is rejected or its daily free limit is used up, it skips that provider's remaining models.
- **Doesn't trust the browser.** Anyone can hit the API with curl, so the server checks every message: allowed roles only, non-empty text, a length cap, and a cap on request size. The system prompt is set on the server and can't be overridden from the client.
- **Rate limits per visitor** (15 messages per hour by default) so one person can't use up the shared free quota.
- **Only sends recent history** (the last 10 messages) to keep requests cheap and fast.
- **Optional background info.** If a `knowledge.txt` file exists next to `app.py`, its contents are added to the system prompt. If it's missing, the app just runs without it.

## Running it locally

You need Python 3 and a free API key from [OpenRouter](https://openrouter.ai/keys) and/or [Groq](https://console.groq.com/keys).

```bash
git clone https://github.com/huzaifascripts/ask-away.git
cd ask-away

# optional but recommended: use a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Create your own `.env` from the example file:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Open `.env`, paste in your key, then start the server:

```bash
python app.py
```

Go to http://127.0.0.1:5000 and say hi.

## Configuration

Everything is set through environment variables (locally, that means your `.env` file).

| Variable | Required | What it does |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | One of the two keys | Key for OpenRouter models |
| `GROQ_API_KEY` | One of the two keys | Key for Groq models |
| `AI_MODELS` | No | Comma-separated `provider:model` list, tried in order. Defaults to `openrouter/free` |
| `OPENROUTER_MODEL` | No | Older single-model setting, only used if `AI_MODELS` isn't set |
| `TRUSTED_PROXY_HOPS` | No | How many proxies sit in front of the app on your host. Defaults to `1` |
| `FLASK_DEBUG` | No | Set to `1` for Flask's debug mode. Local use only |

An `AI_MODELS` example:

```
AI_MODELS=groq:openai/gpt-oss-120b,openrouter/free
```

An entry without a provider prefix is treated as an OpenRouter model. Free model names change often, so check the model lists on OpenRouter and Groq if one stops working.

If a key is missing, the app logs a warning and skips the models that need it. If no keys are set at all, it refuses to start.

## Project layout

```
app.py            the Flask server (API route, validation, rate limit, model fallback)
public/           the frontend; index.html is served at /
knowledge.txt     optional extra info added to the system prompt
requirements.txt  Python dependencies
Procfile          start command for Heroku-style hosts
.env.example      template for your own .env
```

## Deploying

The `Procfile` runs the app with gunicorn:

```
web: gunicorn app:app --workers 1 --threads 8 --timeout 120 --bind :$PORT
```

A few things worth knowing before you put it online:

- **Keep it at one worker.** The rate limiter stores its counts in memory, so with several workers each one would count separately. It also resets whenever the server restarts.
- **Set `TRUSTED_PROXY_HOPS` correctly.** The rate limit is based on the visitor's IP, which comes from the proxy headers your host adds. If this number is wrong, the limit can be bypassed or can lump everyone together.
- **Serverless hosts** (Vercel, for example) don't share memory between instances, so the rate limit there is best-effort rather than strict.
- **Add your keys in the host's environment variable settings.** Don't commit them.

## Good to know

- Nothing is stored on the server. The browser sends the recent conversation with each request.
- The free models have their own limits and can be slow or busy at times. That's what the fallback list is for.
- Anything you put in `knowledge.txt` ends up in the system prompt, and if the file is in a public repo, it's public too. Only put in things you're happy to share.
- `.env` is listed in `.gitignore`. Check that it never gets committed, and if a key ever leaks, revoke it and make a new one.
