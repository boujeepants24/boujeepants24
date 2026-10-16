import type { Lead } from "./types";
import type { ScrapedSite } from "./scraper";

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
export const GROQ_MODEL = "llama-3.3-70b-versatile";

const SYSTEM_PROMPT = `You are an elite B2B industrial & enterprise sales engineer who writes outbound for hardware supply, manufacturing and logistics vendors.
You receive raw text scraped from prospect company homepages and the seller's offer. For EACH company, return precise buyer intelligence.

Return ONLY a JSON object of this exact shape:
{"leads":[{
  "index": <integer, the company index given to you>,
  "companyName": "<clean company name, no taglines, no 'Home |'>",
  "website": "<the company URL given to you>",
  "detectedSignal": "<ONE sentence: the specific buying trigger from their actual site — what they build/manufacture and why they need our supply/service right now>",
  "targetRole": "<exact decision-maker title, e.g. 'VP of Hardware Engineering', 'Head of Global Procurement', 'Supply Chain Director'>",
  "estimatedAnnualValue": "<realistic B2B contract estimate in EUR, format '€45,000/yr'>",
  "fitScore": <integer between 86 and 98>,
  "sniperSubject": "<3-5 words, all lowercase, internal-sounding, e.g. 'pcb supply for [product]?' or 'component lead times'>",
  "sniperEmail": "<exactly 3 sentences, plain text>"
}]}

Rules for sniperEmail:
- Sentence 1 references a real, specific detail or product name found in THEIR scraped text. Never invent products.
- Sentence 2 states the seller's exact capability / speed / cost advantage from the offer.
- Sentence 3 is a low-friction ask, e.g. "Worth sending over a spec sheet?" or "Worth sending over a sample quote?".
- No greeting, no sign-off, no buzzwords (synergy, leverage, cutting-edge, innovative, solutions), never "Hope this finds you well".
- Under 75 words total. Write like an engineer, not a marketer.

If a site is clearly not a potential buyer (a directory, a news site, a listicle), still return it but score it 86 and say why in detectedSignal.`;

interface RawLead {
  index?: unknown;
  companyName?: unknown;
  website?: unknown;
  detectedSignal?: unknown;
  targetRole?: unknown;
  estimatedAnnualValue?: unknown;
  fitScore?: unknown;
  sniperSubject?: unknown;
  sniperEmail?: unknown;
}

function str(v: unknown, fallback: string): string {
  return typeof v === "string" && v.trim() ? v.trim() : fallback;
}

function normaliseValue(v: unknown): string {
  const s = str(v, "€40,000/yr");
  if (s.includes("€")) return s.includes("/yr") ? s : `${s}/yr`;
  const n = Number(String(s).replace(/[^\d.]/g, ""));
  return Number.isFinite(n) && n > 0 ? `€${Math.round(n).toLocaleString("en-US")}/yr` : "€40,000/yr";
}

function fallbackName(site: ScrapedSite): string {
  if (site.siteName) return site.siteName;
  const fromTitle = site.title.split(/[|\-–—:·]/)[0]?.trim();
  if (fromTitle && fromTitle.length > 2 && fromTitle.length < 50) return fromTitle;
  const base = site.hostname.split(".")[0];
  return base.charAt(0).toUpperCase() + base.slice(1);
}

export class GroqError extends Error {}

export async function analyseWithGroq(sites: ScrapedSite[], offer: string, apiKey: string): Promise<Lead[]> {
  const companiesBlock = sites
    .map(
      (s, i) => `### COMPANY ${i}
URL: ${s.url}
PAGE TITLE: ${s.title || s.searchTitle}
META DESCRIPTION: ${s.metaDescription || s.searchSnippet}
SITE TEXT: ${s.bodyText || s.searchSnippet}`,
    )
    .join("\n\n");

  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    signal: AbortSignal.timeout(45_000),
    cache: "no-store",
    body: JSON.stringify({
      model: GROQ_MODEL,
      temperature: 0.35,
      max_tokens: 4000,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: `SELLER OFFER & CAPABILITY:\n${offer}\n\nPROSPECT COMPANIES (${sites.length}):\n\n${companiesBlock}\n\nReturn the JSON object with one lead per company, in index order.`,
        },
      ],
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new GroqError(`Groq API HTTP ${res.status}${detail ? `: ${detail.slice(0, 180)}` : ""}`);
  }

  const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  const content = data.choices?.[0]?.message?.content;
  if (!content) throw new GroqError("Groq returned an empty completion");

  let parsed: { leads?: RawLead[] };
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new GroqError("Groq returned malformed JSON");
  }
  const raw = Array.isArray(parsed.leads) ? parsed.leads : [];
  if (raw.length === 0) throw new GroqError("Groq returned no leads");

  return raw
    .map((r, i): Lead | null => {
      const idx = typeof r.index === "number" && sites[r.index] ? r.index : i;
      const site = sites[idx];
      if (!site) return null;
      const score = Math.round(Number(r.fitScore));
      return {
        id: `live-${idx}-${site.hostname}`,
        companyName: str(r.companyName, fallbackName(site)),
        website: site.url,
        detectedSignal: str(r.detectedSignal, site.metaDescription || "Active hardware manufacturer."),
        targetRole: str(r.targetRole, "Head of Global Procurement"),
        estimatedAnnualValue: normaliseValue(r.estimatedAnnualValue),
        fitScore: Number.isFinite(score) ? Math.min(98, Math.max(86, score)) : 88,
        sniperSubject: str(r.sniperSubject, "supply question").toLowerCase(),
        sniperEmail: str(r.sniperEmail, "").replace(/\s+/g, " "),
        siteTitle: site.title,
        siteDescription: site.metaDescription || site.searchSnippet,
        source: "live",
      };
    })
    .filter((l): l is Lead => l !== null && l.sniperEmail.length > 0)
    .sort((a, b) => b.fitScore - a.fitScore);
}
