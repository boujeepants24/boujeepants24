"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  ExternalLink,
  Globe,
  Mail,
  Radar,
  Rocket,
  Search,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import type { Lead } from "@/lib/types";
import type { LeadDraft } from "@/lib/csv";
import { applyPattern, enrichBuyer } from "@/lib/enrich";
import { cn, hostnameOf } from "@/lib/utils";
import { FitBadge, SectionLabel } from "./ui";

interface Props {
  lead: Lead;
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

export function InspectorDrawer({ lead, draft, onChange, onClose, onPrev, onNext, onToast }: Props) {
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
      <button type="button" aria-label="Close inspector" onClick={onClose} className="absolute inset-0 bg-black/55 backdrop-blur-[2px]" />

      <aside className="animate-slide-in relative flex h-full w-full max-w-[620px] flex-col border-l border-line bg-panel shadow-2xl shadow-black/60">
        {/* Header */}
        <header className="flex items-start justify-between gap-4 border-b border-line px-6 py-5">
          <div className="min-w-0">
            <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
              <Radar className="size-3 text-emerald-400" /> Target Intelligence
              <span className="rounded border border-line px-1 text-[9px]">{lead.source === "live" ? "LIVE" : "CACHE"}</span>
            </div>
            <h2 className="mt-1.5 truncate text-xl font-semibold text-ink">{lead.companyName}</h2>
            <a
              href={lead.website}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 inline-flex items-center gap-1.5 font-mono text-xs text-cyan-300 hover:text-cyan-200"
            >
              <Globe className="size-3" /> {hostnameOf(lead.website)} <ExternalLink className="size-3 opacity-60" />
            </a>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <IconButton label="Previous lead (k)" onClick={onPrev} disabled={!onPrev}>
              <ChevronLeft className="size-4" />
            </IconButton>
            <IconButton label="Next lead (j)" onClick={onNext} disabled={!onNext}>
              <ChevronRight className="size-4" />
            </IconButton>
            <IconButton label="Close (Esc)" onClick={onClose}>
              <X className="size-4" />
            </IconButton>
          </div>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto px-6 py-5">
          {/* Metrics */}
          <div className="grid grid-cols-3 gap-px overflow-hidden rounded-md border border-line bg-line">
            <Metric label="Fit Score">
              <FitBadge score={lead.fitScore} className="text-sm" />
            </Metric>
            <Metric label="Est. Deal Value">
              <span className="font-mono text-sm text-emerald-300">{lead.estimatedAnnualValue}</span>
            </Metric>
            <Metric label="HQ / Sector">
              <span className="block truncate text-xs text-muted" title={[lead.hq, lead.sector].filter(Boolean).join(" · ")}>
                {lead.hq ?? hostnameOf(lead.website).split(".").pop()?.toUpperCase()}
              </span>
            </Metric>
          </div>

          {/* Signal */}
          <div>
            <SectionLabel>Detected Buying Signal</SectionLabel>
            <p className="mt-2 rounded-md border border-emerald-500/20 bg-emerald-500/[0.04] p-3 text-sm leading-relaxed text-ink/90">
              <Sparkles className="mr-1.5 inline size-3.5 -translate-y-px text-emerald-400" />
              {lead.detectedSignal}
            </p>
          </div>

          {(lead.siteTitle || lead.siteDescription || lead.sector) && (
            <div>
              <SectionLabel>Scraped Site Intel</SectionLabel>
              <dl className="mt-2 space-y-1.5 text-[13px]">
                {lead.sector && <IntelRow k="Sector" v={lead.sector} />}
                {lead.siteTitle && <IntelRow k="<title>" v={lead.siteTitle} />}
                {lead.siteDescription && <IntelRow k="meta" v={lead.siteDescription} />}
              </dl>
            </div>
          )}

          {/* Buyer & enrichment */}
          <div>
            <SectionLabel>Decision Maker</SectionLabel>
            <div className="mt-2 rounded-md border border-line bg-panel-2 p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-sm text-ink">
                  <Target className="size-4 text-cyan-400" />
                  {lead.targetRole}
                </div>
                {!enriched && (
                  <button
                    type="button"
                    onClick={() => setEnriched(true)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1.5 text-xs font-medium text-cyan-200 transition hover:bg-cyan-500/20"
                  >
                    <Mail className="size-3.5" /> Enrich Direct Buyer Email
                  </button>
                )}
              </div>

              {enriched && (
                <div className="animate-fade-in mt-3 space-y-3 border-t border-line pt-3">
                  <div>
                    <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-faint">Corporate email pattern · {enrich.domain}</div>
                    <div className="flex flex-wrap gap-1.5">
                      {[enrich.primaryPattern, ...enrich.alternatePatterns].map((p, i) => (
                        <button
                          key={p}
                          type="button"
                          onClick={() => onChange({ pattern: p })}
                          className={cn(
                            "rounded border px-2 py-1 font-mono text-[11px] transition",
                            p === pattern
                              ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-200"
                              : "border-line text-muted hover:border-line-strong hover:text-ink",
                          )}
                        >
                          {p}
                          {i === 0 && <span className="ml-1.5 text-[9px] text-emerald-400/80">LIKELY</span>}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <a
                      href={enrich.linkedInPeopleUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center justify-center gap-1.5 rounded-md bg-[#0a66c2] px-3 py-2 text-xs font-medium text-white transition hover:bg-[#0b75de]"
                    >
                      <LinkedInGlyph /> Find on LinkedIn
                    </a>
                    <a
                      href={enrich.linkedInXrayUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center justify-center gap-1.5 rounded-md border border-line-strong px-3 py-2 text-xs text-muted transition hover:text-ink"
                    >
                      <Search className="size-3.5" /> X-ray profile search
                    </a>
                  </div>

                  <div>
                    <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-faint">Contact name (from LinkedIn)</div>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        value={draft.firstName}
                        onChange={(e) => onChange({ firstName: e.target.value })}
                        placeholder="First name"
                        className="rounded-md border border-line bg-obsidian px-2.5 py-1.5 text-sm text-ink placeholder:text-faint focus:border-emerald-500/50 focus:outline-none"
                      />
                      <input
                        value={draft.lastName}
                        onChange={(e) => onChange({ lastName: e.target.value })}
                        placeholder="Last name"
                        className="rounded-md border border-line bg-obsidian px-2.5 py-1.5 text-sm text-ink placeholder:text-faint focus:border-emerald-500/50 focus:outline-none"
                      />
                    </div>
                    <div className="mt-2 flex items-center justify-between gap-2 rounded-md border border-line bg-obsidian px-2.5 py-2 font-mono text-[12px]">
                      <span className={resolvedEmail ? "text-emerald-300" : "text-faint"}>
                        {resolvedEmail || pattern}
                      </span>
                      {resolvedEmail && (
                        <button
                          type="button"
                          onClick={async () => {
                            if (await copyText(resolvedEmail)) flash("address");
                          }}
                          className="text-muted hover:text-ink"
                          aria-label="Copy email address"
                        >
                          {copied === "address" ? <Check className="size-3.5 text-emerald-400" /> : <Copy className="size-3.5" />}
                        </button>
                      )}
                    </div>
                    <p className="mt-1.5 text-[11px] text-faint">
                      Pattern-based guess — verify deliverability before sending.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sniper email */}
          <div>
            <div className="flex items-center justify-between">
              <SectionLabel>Sniper Email</SectionLabel>
              <span className={cn("font-mono text-[10px]", sentences === 3 ? "text-emerald-400/80" : "text-amber-400/80")}>
                {sentences} sentence{sentences === 1 ? "" : "s"} · {draft.body.split(/\s+/).filter(Boolean).length} words
              </span>
            </div>
            <div className="mt-2 overflow-hidden rounded-md border border-line bg-obsidian focus-within:border-emerald-500/40">
              <label className="flex items-center gap-2 border-b border-line px-3 py-2">
                <span className="font-mono text-[11px] text-faint">Subject:</span>
                <input
                  value={draft.subject}
                  onChange={(e) => onChange({ subject: e.target.value })}
                  className="flex-1 bg-transparent font-mono text-[13px] text-ink focus:outline-none"
                />
              </label>
              <textarea
                value={draft.body}
                onChange={(e) => onChange({ body: e.target.value })}
                rows={7}
                className="block w-full resize-y bg-transparent px-3 py-2.5 text-sm leading-relaxed text-ink focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Footer actions */}
        <footer className="grid grid-cols-2 gap-2 border-t border-line bg-panel px-6 py-4">
          <button
            type="button"
            onClick={handleCopyEmail}
            className="inline-flex items-center justify-center gap-2 rounded-md border border-line-strong bg-panel-2 px-4 py-2.5 text-sm font-medium text-ink transition hover:border-line-strong hover:bg-white/[0.04]"
          >
            {copied === "email" ? <Check className="size-4 text-emerald-400" /> : <Copy className="size-4" />}
            {copied === "email" ? "Copied" : "Copy Email"}
          </button>
          <button
            type="button"
            onClick={() => {
              onChange({ pushed: !draft.pushed });
              onToast(draft.pushed ? `${lead.companyName} removed from campaign` : `${lead.companyName} pushed to outbound campaign`);
            }}
            className={cn(
              "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2.5 text-sm font-semibold transition",
              draft.pushed
                ? "border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/15"
                : "bg-emerald-500 text-[#04120c] hover:bg-emerald-400",
            )}
          >
            {draft.pushed ? <Check className="size-4" /> : <Rocket className="size-4" />}
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
      className="rounded-md border border-transparent p-1.5 text-muted transition hover:border-line hover:text-ink disabled:opacity-30"
    >
      {children}
    </button>
  );
}

function Metric({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 bg-panel-2 px-3 py-2.5">
      <div className="font-mono text-[9px] uppercase tracking-[0.18em] text-faint">{label}</div>
      <div className="mt-1">{children}</div>
    </div>
  );
}

function IntelRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex gap-3">
      <dt className="w-16 shrink-0 font-mono text-[11px] text-faint">{k}</dt>
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
