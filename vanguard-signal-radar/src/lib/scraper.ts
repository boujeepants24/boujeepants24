import * as cheerio from "cheerio";

export const BROWSER_HEADERS: Record<string, string> = {
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
  Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
  "Accept-Language": "en-GB,en;q=0.9,de;q=0.8",
  "Cache-Control": "no-cache",
  Pragma: "no-cache",
  "Upgrade-Insecure-Requests": "1",
};

export interface SearchHit {
  title: string;
  url: string;
  snippet: string;
}

export interface ScrapedSite {
  url: string;
  hostname: string;
  title: string;
  metaDescription: string;
  siteName: string;
  bodyText: string;
  searchTitle: string;
  searchSnippet: string;
}

/** Domains that are directories, media, social or marketplaces — never a buyer. */
const BLOCKED_DOMAINS = [
  "duckduckgo.com", "wikipedia.org", "linkedin.com", "youtube.com", "facebook.com", "twitter.com",
  "x.com", "instagram.com", "reddit.com", "quora.com", "medium.com", "amazon.", "ebay.", "alibaba.com",
  "aliexpress.com", "made-in-china.com", "indiamart.com", "glassdoor.", "indeed.", "crunchbase.com",
  "zoominfo.com", "dnb.com", "bloomberg.com", "forbes.com", "statista.com", "kompass.com",
  "europages.", "thomasnet.com", "wlw.de", "yelp.", "tracxn.com", "f6s.com", "ensun.io",
  "seedtable.com", "builtin.com", "clutch.co", "g2.com", "capterra.", "gartner.com",
  "mordorintelligence.com", "marketsandmarkets.com", "grandviewresearch.com", "researchandmarkets.com",
  "github.com", "pinterest.", "tiktok.com", "gov.uk", "europa.eu", "iot-analytics.com",
  "iotworldtoday.com", "techcrunch.com", "reuters.com", "ft.com", "wsj.com", "bbc.", "google.",
  "bing.com", "yahoo.", "apple.com", "microsoft.com",
];

function isBlocked(hostname: string): boolean {
  const h = hostname.toLowerCase();
  return BLOCKED_DOMAINS.some((d) => (d.endsWith(".") ? h.includes(d) : h === d || h.endsWith(`.${d}`)));
}

/** Basic SSRF guard: only public http(s) hostnames. */
function isSafePublicUrl(u: URL): boolean {
  if (u.protocol !== "http:" && u.protocol !== "https:") return false;
  const h = u.hostname.toLowerCase();
  if (h === "localhost" || h.endsWith(".local") || h.endsWith(".internal") || !h.includes(".")) return false;
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(h) || h.startsWith("[")) return false;
  return true;
}

/** DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded target>. */
function decodeDdgHref(href: string): string | null {
  try {
    const abs = href.startsWith("//") ? `https:${href}` : href;
    const u = new URL(abs, "https://duckduckgo.com");
    if (u.hostname.endsWith("duckduckgo.com")) {
      const target = u.searchParams.get("uddg");
      return target ? decodeURIComponent(target) : null;
    }
    return u.toString();
  } catch {
    return null;
  }
}

export class SearchBlockedError extends Error {}

export async function searchDuckDuckGo(query: string, timeoutMs = 8000): Promise<SearchHit[]> {
  const res = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}&kl=wt-wt`, {
    headers: { ...BROWSER_HEADERS, Referer: "https://html.duckduckgo.com/" },
    signal: AbortSignal.timeout(timeoutMs),
    cache: "no-store",
  });
  if (!res.ok) throw new SearchBlockedError(`DuckDuckGo responded HTTP ${res.status}`);
  const html = await res.text();
  const $ = cheerio.load(html);

  if ($(".anomaly-modal__modal, #challenge-form").length > 0 || /anomaly|unusual traffic/i.test($("title").text())) {
    throw new SearchBlockedError("DuckDuckGo served a bot challenge (rate-limited)");
  }

  const hits: SearchHit[] = [];
  $(".result").each((_, el) => {
    const node = $(el);
    if (node.hasClass("result--ad") || node.find(".badge--ad").length) return;
    const a = node.find("a.result__a").first();
    const href = a.attr("href");
    if (!href) return;
    const url = decodeDdgHref(href);
    if (!url) return;
    hits.push({
      title: a.text().trim(),
      url,
      snippet: node.find(".result__snippet").text().trim(),
    });
  });
  return hits;
}

/** Keeps one homepage per company domain, dropping directories/media/social. */
export function pickCompanyCandidates(hits: SearchHit[], limit: number): SearchHit[] {
  const seen = new Set<string>();
  const out: SearchHit[] = [];
  for (const hit of hits) {
    let u: URL;
    try {
      u = new URL(hit.url);
    } catch {
      continue;
    }
    if (!isSafePublicUrl(u)) continue;
    const host = u.hostname.replace(/^www\./, "").toLowerCase();
    if (isBlocked(host) || seen.has(host)) continue;
    seen.add(host);
    out.push({ ...hit, url: `${u.protocol}//${u.hostname}/` });
    if (out.length >= limit) break;
  }
  return out;
}

const MAX_HTML_BYTES = 1_500_000;

async function readCapped(res: Response, maxBytes: number): Promise<string> {
  if (!res.body) return "";
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let received = 0;
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    received += value.byteLength;
    text += decoder.decode(value, { stream: true });
    if (received >= maxBytes) {
      await reader.cancel().catch(() => undefined);
      break;
    }
  }
  return text + decoder.decode();
}

export async function scrapeHomepage(hit: SearchHit, timeoutMs = 4000, maxChars = 2200): Promise<ScrapedSite> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(hit.url, {
      headers: BROWSER_HEADERS,
      signal: controller.signal,
      redirect: "follow",
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const type = res.headers.get("content-type") ?? "";
    if (type && !type.includes("html")) throw new Error(`non-HTML content (${type})`);
    const finalUrl = new URL(res.url || hit.url);
    if (!isSafePublicUrl(finalUrl)) throw new Error("redirected to a non-public host");
    const html = await readCapped(res, MAX_HTML_BYTES);

    const $ = cheerio.load(html);
    const title = $("title").first().text().replace(/\s+/g, " ").trim();
    const metaDescription = (
      $('meta[name="description"]').attr("content") ??
      $('meta[property="og:description"]').attr("content") ??
      ""
    )
      .replace(/\s+/g, " ")
      .trim();
    const siteName = ($('meta[property="og:site_name"]').attr("content") ?? "").trim();

    $("script, style, noscript, svg, iframe, template, link, meta, form, [aria-hidden='true']").remove();
    $("[style*='display:none'], [style*='display: none'], [hidden]").remove();
    // Join text nodes with spaces so adjacent block elements don't run words together.
    const bodyText = $("body *")
      .contents()
      .filter((_, node) => node.type === "text")
      .map((_, node) => $(node).text())
      .get()
      .join(" ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, maxChars);

    return {
      url: `${finalUrl.protocol}//${finalUrl.hostname}/`,
      hostname: finalUrl.hostname.replace(/^www\./, ""),
      title,
      metaDescription,
      siteName,
      bodyText,
      searchTitle: hit.title,
      searchSnippet: hit.snippet,
    };
  } finally {
    clearTimeout(timer);
  }
}
