"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, Check, ChevronLeft, ChevronRight, Copy, Mail, Search, Send, X } from "lucide-react";
import type { Lead } from "@/lib/types";
import type { LeadDraft } from "@/lib/csv";
import { applyPattern, cleanCompany, enrichBuyer } from "@/lib/enrich";
import { cn, hostnameOf } from "@/lib/utils";
import { FitMeter, SectionLabel } from "./ui";

interface Props {
  lead: Lead;
  rank: number;
  total: number;
  draft: LeadDraft;
  onChange: (patch: Partial<LeadDraft>) => void;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  onToast: (msg: string) => void;
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function InspectorDrawer({ lead, rank, total, draft, onChange, onClose, onPrev, onNext, onToast }: Props) {
  const [enriched, setEnriched] = useState(false);
  const [copied, setCopied] = useState<"email" | "address" | null>(null);
  const enrich = useMemo(
    () => enrichBuyer(lead.companyName, lead.website, lead.targetRole),
    [lead.companyName, lead.website, lead.targetRole],
  );
  const pattern = draft.pattern ?? enrich.primaryPattern;
  const resolvedEmail = applyPattern(pattern, draft.firstName, draft.lastName);
  const sentences = draft.body.split(/(?<=[.?!])\s+/).filter(Boolean).length;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
      if (e.key === "Escape") onClose();
      if (typing) return;
      if (e.key === "ArrowUp" || e.key === "k") onPrev?.();
      if (e.key === "ArrowDown" || e.key === "j") onNext?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onPrev, onNext]);

  const flash = (kind: "email" | "address") => {
    setCopied(kind);
    setTimeout(() => setCopied(null), 1400);
  };

  const handleCopyEmail = async () => {
    const text = `Subject: ${draft.subject}\n\n${draft.body}`;
    if (await copyText(text)) {
      flash("email");
      onToast("Sniper email copied to clipboard");
    } else onToast("Clipboard unavailable — select the text manually");
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label={`${lead.companyName} intelligence`}>
      <button type="button" aria-label="Close inspector" onClick={onClose} className="absolute inset-0 bg-black/60" />

      <aside className="animate-slide-in relative flex h-full w-full max-w-[640px] flex-col border-l border-line-strong bg-panel shadow-2xl shadow-black/60">
        {/* Header */}
        <header className="border-b border-line px-7 pb-5 pt-5">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.14em] text-faint">
              Account {String(rank).padStart(2, "0")} of {String(total).padStart(2, "0")}
              <span className="text-line-strong">|</span>
              <span>{lead.source === "live" ? "Live intelligence" : "Cached intelligence"}</span>
            </div>
            <div className="flex shrink-0 items-center gap-0.5">
              <IconButton label="Previous account (k)" onClick={onPrev} disabled={!onPrev}>
                <ChevronLeft className="size-4" />
              </IconButton>
              <IconButton label="Next account (j)" onClick={onNext} disabled={!onNext}>
                <ChevronRight className="size-4" />
              </IconButton>
              <IconButton label="Close (Esc)" onClick={onClose}>
                <X className="size-4" />
              </IconButton>
            </div>
          </div>
          <h2 className="mt-3 font-serif text-[26px] leading-tight text-ink">{lead.companyName}</h2>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-muted">
            <a
              href={lead.website}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-0.5 underline decoration-line-strong underline-offset-4 transition hover:text-ink hover:decoration-brass"
            >
              {hostnameOf(lead.website)}
              <ArrowUpRight className="size-3.5" />
            </a>
            {lead.hq && <span>{lead.hq}</span>}
            {lead.sector && <span className="text-faint">{lead.sector}</span>}
          </div>
        </header>

        <div className="flex-1 space-y-7 overflow-y-auto px-7 py-6">
          {/* Metrics */}
          <dl className="grid grid-cols-3 gap-px overflow-hidden rounded border border-line bg-line">
            <Metric label="Fit Score">
              <FitMeter score={lead.fitScore} wide />
            </Metric>
            <Metric label="Est. Deal Value">
              <span className="font-mono text-[13px] text-brass">{lead.estimatedAnnualValue}</span>
            </Metric>
            <Metric label="Target Role">
              <span className="block truncate text-[13px] text-ink" title={lead.targetRole}>
                {lead.targetRole}
              </span>
            </Metric>
          </dl>

          {/* Signal */}
          <div>
            <SectionLabel>Detected Buying Signal</SectionLabel>
            <blockquote className="mt-2.5 border-l-2 border-brass pl-4 font-serif text-[16px] leading-relaxed text-ink">
              {lead.detectedSignal}
            </blockquote>
          </div>

          {(lead.siteTitle || lead.siteDescription) && (
            <div>
              <SectionLabel>Website Extract</SectionLabel>
              <dl className="mt-2.5 space-y-2 text-[13px]">
                {lead.siteTitle && <IntelRow k="Title" v={lead.siteTitle} />}
                {lead.siteDescription && <IntelRow k="Description" v={lead.siteDescription} />}
              </dl>
            </div>
          )}

          {/* Buyer & enrichment */}
          <div>
            <SectionLabel>Decision Maker</SectionLabel>
            <div className="mt-2.5 rounded border border-line bg-panel-2">
              <div className="flex items-center justify-between gap-3 px-4 py-3">
                <div>
                  <div className="text-[14px] text-ink">{lead.targetRole}</div>
                  <div className="text-[12px] text-faint">{cleanCompany(lead.companyName)}</div>
                </div>
                {!enriched && (
                  <button
                    type="button"
                    onClick={() => setEnriched(true)}
                    className="inline-flex items-center gap-2 rounded border border-line-strong px-3 py-1.5 text-xs font-medium text-ink transition hover:border-brass-soft hover:bg-panel"
                  >
                    <Mail className="size-3.5 text-brass" /> Enrich Direct Buyer Email
                  </button>
                )}
              </div>

              {enriched && (
                <div className="animate-fade-in space-y-4 border-t border-line px-4 py-4">
                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-[0.12em] text-faint">Email pattern · {enrich.domain}</div>
                    <div className="flex flex-wrap gap-1.5">
                      {[enrich.primaryPattern, ...enrich.alternatePatterns].map((p, i) => (
                        <button
                          key={p}
                          type="button"
                          onClick={() => onChange({ pattern: p })}
                          className={cn(
                            "rounded-sm border px-2 py-1 font-mono text-[11px] transition",
                            p === pattern
                              ? "border-brass/60 bg-brass/10 text-ink"
                              : "border-line text-muted hover:border-line-strong hover:text-ink",
                          )}
                        >
                          {p}
                          {i === 0 && <span className="ml-1.5 font-sans text-[9px] uppercase tracking-wider text-brass">Likely</span>}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <a
                      href={enrich.linkedInPeopleUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center justify-center gap-2 rounded border border-line-strong bg-panel px-3 py-2 text-xs font-medium text-ink transition hover:border-steel/60"
                    >
                      <LinkedInGlyph /> Find on LinkedIn
                    </a>
                    <a
                      href={enrich.linkedInXrayUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center justify-center gap-2 rounded border border-line px-3 py-2 text-xs text-muted transition hover:border-line-strong hover:text-ink"
                    >
                      <Search className="size-3.5" /> X-ray profile search
                    </a>
                  </div>

                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-[0.12em] text-faint">Contact name (from LinkedIn)</div>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        value={draft.firstName}
                        onChange={(e) => onChange({ firstName: e.target.value })}
                        placeholder="First name"
                        className="rounded border border-line bg-obsidian px-2.5 py-1.5 text-[13px] text-ink placeholder:text-faint focus:border-brass focus:outline-none"
                      />
                      <input
                        value={draft.lastName}
                        onChange={(e) => onChange({ lastName: e.target.value })}
                        placeholder="Last name"
                        className="rounded border border-line bg-obsidian px-2.5 py-1.5 text-[13px] text-ink placeholder:text-faint focus:border-brass focus:outline-none"
                      />
                    </div>
                    <div className="mt-2 flex items-center justify-between gap-2 rounded border border-line bg-obsidian px-3 py-2 font-mono text-[12px]">
                      <span className={resolvedEmail ? "text-ink" : "text-faint"}>{resolvedEmail || pattern}</span>
                      {resolvedEmail && (
                        <button
                          type="button"
                          onClick={async () => {
                            if (await copyText(resolvedEmail)) flash("address");
                          }}
                          className="text-muted transition hover:text-ink"
                          aria-label="Copy email address"
                        >
                          {copied === "address" ? <Check className="size-3.5 text-sage" /> : <Copy className="size-3.5" />}
                        </button>
                      )}
                    </div>
                    <p className="mt-1.5 text-[11px] text-faint">Pattern-based estimate. Verify deliverability before sending.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sniper email */}
          <div>
            <div className="flex items-baseline justify-between">
              <SectionLabel>Sniper Email</SectionLabel>
              <span className={cn("text-[11px] tabular-nums", sentences === 3 ? "text-faint" : "text-ochre")}>
                {sentences} sentence{sentences === 1 ? "" : "s"} · {draft.body.split(/\s+/).filter(Boolean).length} words
              </span>
            </div>
            <div className="mt-2.5 overflow-hidden rounded border border-line bg-obsidian transition focus-within:border-brass-soft">
              <label className="flex items-center gap-3 border-b border-line px-4 py-2.5">
                <span className="text-[12px] text-faint">Subject</span>
                <input
                  value={draft.subject}
                  onChange={(e) => onChange({ subject: e.target.value })}
                  className="flex-1 bg-transparent text-[13px] text-ink focus:outline-none"
                />
              </label>
              <textarea
                value={draft.body}
                onChange={(e) => onChange({ body: e.target.value })}
                rows={7}
                className="block w-full resize-y bg-transparent px-4 py-3 text-[14px] leading-relaxed text-ink focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Footer actions */}
        <footer className="grid grid-cols-2 gap-2 border-t border-line px-7 py-4">
          <button
            type="button"
            onClick={handleCopyEmail}
            className="inline-flex items-center justify-center gap-2 rounded border border-line-strong px-4 py-2.5 text-[13px] font-medium text-ink transition hover:bg-panel-2"
          >
            {copied === "email" ? <Check className="size-4 text-sage" /> : <Copy className="size-4" />}
            {copied === "email" ? "Copied" : "Copy Email"}
          </button>
          <button
            type="button"
            onClick={() => {
              onChange({ pushed: !draft.pushed });
              onToast(draft.pushed ? `${lead.companyName} removed from campaign` : `${lead.companyName} pushed to outbound campaign`);
            }}
            className={cn(
              "inline-flex items-center justify-center gap-2 rounded px-4 py-2.5 text-[13px] font-semibold transition",
              draft.pushed
                ? "border border-sage/50 text-sage hover:bg-sage/10"
                : "bg-ivory text-obsidian hover:bg-white",
            )}
          >
            {draft.pushed ? <Check className="size-4" /> : <Send className="size-4" />}
            {draft.pushed ? "Queued in Campaign" : "Push to Outbound Campaign"}
          </button>
        </footer>
      </aside>
    </div>
  );
}

function IconButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      disabled={disabled}
      className="rounded p-1.5 text-faint transition hover:bg-panel-2 hover:text-ink disabled:opacity-30"
    >
      {children}
    </button>
  );
}

function Metric({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 bg-panel-2 px-3.5 py-3">
      <dt className="text-[10px] uppercase tracking-[0.14em] text-faint">{label}</dt>
      <dd className="mt-1.5">{children}</dd>
    </div>
  );
}

function IntelRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex gap-3">
      <dt className="w-24 shrink-0 text-[12px] text-faint">{k}</dt>
      <dd className="min-w-0 text-muted">{v}</dd>
    </div>
  );
}

/** lucide-react v1 dropped brand icons, so the LinkedIn mark is drawn inline. */
function LinkedInGlyph() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="size-3.5 fill-current">
      <path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z" />
    </svg>
  );
}
