import { hostnameOf } from "./utils";

export interface EnrichmentResult {
  domain: string;
  /** Most likely corporate pattern, e.g. "{first}.{last}@acme.de". */
  primaryPattern: string;
  alternatePatterns: string[];
  linkedInPeopleUrl: string;
  linkedInXrayUrl: string;
}

/** Short initial-based patterns are more common at US/UK companies than in DACH/Nordics. */
const INITIAL_FIRST_TLDS = new Set(["com", "us", "io", "co"]);

export function enrichBuyer(companyName: string, website: string, role: string): EnrichmentResult {
  const domain = hostnameOf(website);
  const tld = domain.split(".").pop() ?? "com";

  const patterns = ["{first}.{last}", "{f}{last}", "{first}", "{first}{last}", "{f}.{last}"];
  if (INITIAL_FIRST_TLDS.has(tld)) patterns.splice(0, 2, "{f}{last}", "{first}.{last}");
  const withDomain = patterns.map((p) => `${p}@${domain}`);

  const company = cleanCompany(companyName);
  const keywords = `${role} ${company}`;
  const linkedInPeopleUrl = `https://www.linkedin.com/search/results/people/?keywords=${encodeURIComponent(keywords)}&origin=GLOBAL_SEARCH_HEADER`;
  const linkedInXrayUrl = `https://duckduckgo.com/?q=${encodeURIComponent(`site:linkedin.com/in "${company}" "${role}"`)}`;

  return {
    domain,
    primaryPattern: withDomain[0],
    alternatePatterns: withDomain.slice(1),
    linkedInPeopleUrl,
    linkedInXrayUrl,
  };
}

/** Strips legal suffixes so LinkedIn search matches the company page name. */
export function cleanCompany(name: string): string {
  return name
    .replace(/\b(GmbH|AG|Ltd\.?|Limited|Inc\.?|LLC|S\.p\.A\.?|SpA|SAS|S\.A\.?|SA|B\.V\.?|BV|AB|Oy|A\/S|Corp\.?|Corporation|plc|Co\.?)\b/gi, "")
    .replace(/[,.\s]+$/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function slug(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/ß/g, "ss")
    .toLowerCase()
    .replace(/[^a-z]/g, "");
}

/** Resolves a pattern like "{first}.{last}@acme.de" for a named contact. */
export function applyPattern(pattern: string, first: string, last: string): string {
  const f = slug(first);
  const l = slug(last);
  if (!f || !l) return "";
  return pattern.replace("{first}", f).replace("{last}", l).replace("{f}", f.charAt(0));
}
