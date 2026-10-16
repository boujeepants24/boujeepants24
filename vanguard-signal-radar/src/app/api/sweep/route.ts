import { getCachedLeads } from "@/lib/demo-cache";
import { analyseWithGroq, GROQ_MODEL } from "@/lib/groq";
import { pickCompanyCandidates, scrapeHomepage, searchDuckDuckGo, type ScrapedSite } from "@/lib/scraper";
import type { Lead, LogTag, PresetId, SweepEvent, SweepMode } from "@/lib/types";

export const maxDuration = 60;

const TARGET_SITES = 5;
const CANDIDATE_POOL = 9;
const CACHE_DURATION_MS = 1500;
const PRESET_IDS: PresetId[] = ["pcb", "cnc", "logistics", "custom"];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function parseBody(body: unknown) {
  const b = (body ?? {}) as Record<string, unknown>;
  const query = typeof b.query === "string" ? b.query.trim().slice(0, 200) : "";
  const offer = typeof b.offer === "string" ? b.offer.trim().slice(0, 800) : "";
  const mode: SweepMode = b.mode === "live" ? "live" : "cache";
  const presetId: PresetId = PRESET_IDS.includes(b.presetId as PresetId) ? (b.presetId as PresetId) : "custom";
  return { query, offer, mode, presetId };
}

export async function POST(request: Request) {
  let input: ReturnType<typeof parseBody>;
  try {
    input = parseBody(await request.json());
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  if (!input.query || !input.offer) {
    return Response.json({ error: "Both a target niche and an offer are required." }, { status: 400 });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (event: SweepEvent) => controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
      const log = (tag: LogTag, message: string) => emit({ type: "log", tag, message, ts: Date.now() });
      const done = (leads: Lead[], source: "live" | "cache", fallbackReason?: string) =>
        emit({ type: "result", leads, source, fallbackReason, ts: Date.now() });

      const serveCache = async (reason?: string) => {
        const steps: [LogTag, string][] = [
          ["CACHE", "Mounting instant intelligence cache (8 pre-qualified OEM buyers)..."],
          ["CRAWLER", "Rehydrating DOM snapshots for EU/US hardware domains..."],
          ["GROQ-70B", "Re-ranking buyers against offer context & drafting sniper emails..."],
        ];
        const stepDelay = Math.floor(CACHE_DURATION_MS / (steps.length + 1));
        for (const [tag, msg] of steps) {
          log(tag, msg);
          await sleep(stepDelay);
        }
        const leads = getCachedLeads(input.presetId, input.offer);
        await sleep(stepDelay);
        log("DONE", `${leads.length} qualified buyers ready.`);
        done(leads, "cache", reason);
      };

      try {
        log("SYSTEM", `Signal sweep initialised · mode=${input.mode === "live" ? "LIVE" : "CACHE"}`);

        if (input.mode === "cache") {
          await serveCache();
          return;
        }

        const apiKey = process.env.GROQ_API_KEY;
        if (!apiKey || apiKey === "your_key_here") {
          log("WARN", "GROQ_API_KEY is not configured — failing over to instant cache.");
          await serveCache("GROQ_API_KEY missing. Add it to .env.local for live analysis.");
          return;
        }

        // 1. Search
        log("SYSTEM", `Querying search index: "${input.query}"`);
        let candidates;
        try {
          const hits = await searchDuckDuckGo(input.query);
          log("SEARCH", `${hits.length} organic results returned from DuckDuckGo.`);
          candidates = pickCompanyCandidates(hits, CANDIDATE_POOL);
          log("SEARCH", `${candidates.length} unique company domains after filtering directories & media.`);
        } catch (err) {
          const reason = err instanceof Error ? err.message : "search failed";
          log("WARN", `Search index unavailable (${reason}) — failing over to instant cache.`);
          await serveCache(`Live search unavailable: ${reason}`);
          return;
        }
        if (candidates.length === 0) {
          log("WARN", "No company domains found for this query — failing over to instant cache.");
          await serveCache("Search returned no usable company domains.");
          return;
        }

        // 2. Crawl — scrape the pool concurrently, keep the first TARGET_SITES successes in rank order.
        log("CRAWLER", `Extracting DOM from ${Math.min(TARGET_SITES, candidates.length)} target OEM domains...`);
        const settled = await Promise.allSettled(
          candidates.map(async (hit) => {
            const site = await scrapeHomepage(hit, 4000);
            log("CRAWLER", `✓ ${site.hostname} · ${site.bodyText.length} chars · "${(site.title || hit.title).slice(0, 60)}"`);
            return site;
          }),
        );
        const sites: ScrapedSite[] = [];
        settled.forEach((r, i) => {
          if (r.status === "fulfilled") {
            if (r.value.bodyText.length > 80 || r.value.metaDescription) sites.push(r.value);
          } else {
            const host = new URL(candidates[i].url).hostname;
            const reason = r.reason instanceof Error ? (r.reason.name === "AbortError" ? "timeout 4s" : r.reason.message) : "failed";
            log("CRAWLER", `✗ ${host} · ${reason}`);
          }
        });
        const targets = sites.slice(0, TARGET_SITES);
        if (targets.length === 0) {
          log("WARN", "All target domains blocked or timed out — failing over to instant cache.");
          await serveCache("Every scraped domain timed out or blocked the crawler.");
          return;
        }
        log("CRAWLER", `${targets.length} domains parsed: ${targets.map((s) => s.hostname).join(", ")}`);

        // 3. Analyse
        log("GROQ-70B", `Analyzing hardware specs & drafting procurement outreach (${GROQ_MODEL})...`);
        let leads: Lead[];
        try {
          const t0 = Date.now();
          leads = await analyseWithGroq(targets, input.offer, apiKey);
          const seen = new Set<string>();
          leads = leads.filter((l) => (seen.has(l.id) ? false : (seen.add(l.id), true)));
          log("GROQ-70B", `Inference complete in ${((Date.now() - t0) / 1000).toFixed(1)}s · ${leads.length} buyer profiles.`);
        } catch (err) {
          const reason = err instanceof Error ? err.message : "inference failed";
          log("ERROR", `Groq analysis failed (${reason}) — failing over to instant cache.`);
          await serveCache(`Groq analysis failed: ${reason}`);
          return;
        }

        log("DONE", `${leads.length} live buyers qualified with sniper emails drafted.`);
        done(leads, "live");
      } catch (err) {
        emit({ type: "error", message: err instanceof Error ? err.message : "Unexpected sweep failure", ts: Date.now() });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
