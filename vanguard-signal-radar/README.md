# Vanguard // Signal Radar v2.4

B2B buyer-intelligence & outbound engine. It finds hardware/OEM buyers on the open web, reads their homepages, and uses Groq (Llama 3.3 70B) to draft buying signals, target roles, deal estimates and 3-sentence "sniper" cold emails. Results export as an Instantly / Manyreach-ready CSV.

## Stack

Next.js 16 (App Router) · TypeScript · Tailwind CSS v4 · lucide-react · cheerio · Groq API. No paid search API.

## Run it

```bash
npm install
cp .env.example .env.local   # then paste a free key from https://console.groq.com/keys
npm run dev                  # http://localhost:3000
```

## How a sweep works

`POST /api/sweep` streams NDJSON events, so the terminal drawer updates live.

1. **Search**: queries `https://html.duckduckgo.com/html/?q=` with browser headers and parses organic results with cheerio. Ads, directories, media, social and marketplace domains are dropped, and results are deduped per domain.
2. **Crawl**: fetches candidate homepages concurrently (4s timeout each, 1.5 MB cap, public hosts only). It keeps the top 5 that respond and extracts title, meta description and clean body text.
3. **Analyse**: sends the scraped text plus your offer to Groq `llama-3.3-70b-versatile` with `response_format: json_object`. The output is validated and normalised: fit score clamped to 86–98, EUR values, lowercase subject lines.
4. **Failsafe**: if the Groq key is missing, DuckDuckGo rate-limits, every site times out, or Groq errors, the route falls back to the 8-buyer instant cache. The UI flags this as "Failsafe cache" and shows the reason. The "Cached Instant Demo" toggle returns the cache directly in about 1.5s.

The companies in the instant cache (`src/lib/demo-cache.ts`) are **fictional** demo data. Live mode returns real companies.

## Inspector & export

- **Enrich Direct Buyer Email** shows the likely corporate email patterns for the domain and links to a LinkedIn people search and an X-ray search for that company + role. Enter the contact's name and the address is resolved from the pattern you select. These addresses are pattern guesses; verify them before sending.
- The subject and body are editable. **Copy Email** copies the subject plus body. **Push to Outbound Campaign** queues the lead.
- **Export Campaign CSV** writes `email, first_name, last_name, company_name, website, job_title, subject, personalization, …`. Use `{{subject}}` and `{{personalization}}` as custom variables in Instantly or Manyreach.

Keyboard: `j`/`k` or the arrow keys move between leads in the inspector, and `Esc` closes it.
