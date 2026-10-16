import type { Lead } from "./types";
import { applyPattern, enrichBuyer } from "./enrich";

export interface LeadDraft {
  subject: string;
  body: string;
  firstName: string;
  lastName: string;
  pattern?: string;
  pushed: boolean;
}

/** Quotes a cell and neutralises spreadsheet formula injection. */
function cell(value: string | number): string {
  let s = String(value ?? "");
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  return `"${s.replace(/"/g, '""')}"`;
}

/**
 * Column names follow Instantly / Manyreach import conventions: `email`,
 * `first_name`, `last_name`, `company_name`, `website`, plus custom variables
 * usable as {{subject}} / {{personalization}} in sequences.
 */
export function buildCampaignCsv(leads: Lead[], drafts: Record<string, LeadDraft>): string {
  const header = [
    "email", "first_name", "last_name", "company_name", "website", "job_title",
    "subject", "personalization", "detected_signal", "fit_score", "est_annual_value",
    "email_pattern", "linkedin_search_url", "campaign_status",
  ];
  const rows = leads.map((lead) => {
    const d = drafts[lead.id];
    const enrich = enrichBuyer(lead.companyName, lead.website, lead.targetRole);
    const pattern = d?.pattern ?? enrich.primaryPattern;
    const email = d ? applyPattern(pattern, d.firstName, d.lastName) : "";
    return [
      email,
      d?.firstName ?? "",
      d?.lastName ?? "",
      lead.companyName,
      lead.website,
      lead.targetRole,
      d?.subject ?? lead.sniperSubject,
      d?.body ?? lead.sniperEmail,
      lead.detectedSignal,
      lead.fitScore,
      lead.estimatedAnnualValue,
      pattern,
      enrich.linkedInPeopleUrl,
      d?.pushed ? "queued" : "draft",
    ].map(cell).join(",");
  });
  return [header.map(cell).join(","), ...rows].join("\r\n");
}

export function downloadCsv(filename: string, csv: string) {
  const blob = new Blob(["﻿", csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
