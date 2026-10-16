"use client";

import { ArrowUpRight, Download } from "lucide-react";
import type { Lead } from "@/lib/types";
import type { LeadDraft } from "@/lib/csv";
import { cn, hostnameOf } from "@/lib/utils";
import { FitMeter } from "./ui";

interface Props {
  leads: Lead[];
  drafts: Record<string, LeadDraft>;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onExport: () => void;
  source: "live" | "cache" | null;
  fallbackReason?: string;
}

export function ResultsTable({ leads, drafts, selectedId, onSelect, onExport, source, fallbackReason }: Props) {
  const sourceLabel = source === "live" ? "Live web · Groq 70B" : fallbackReason ? "Failsafe cache" : "Instant demo cache";

  return (
    <section className="overflow-hidden rounded-md border border-line bg-panel">
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-line px-5 py-4">
        <div>
          <h2 className="font-serif text-lg text-ink">Qualified Buyers</h2>
          <p className="mt-0.5 text-xs text-faint">
            {leads.length} accounts ranked by fit
            {source && (
              <>
                <span className="mx-1.5 text-line-strong">|</span>
                <span className={cn(fallbackReason ? "text-ochre" : "text-muted")} title={fallbackReason}>
                  Source: {sourceLabel}
                </span>
              </>
            )}
          </p>
        </div>
        <button
          type="button"
          onClick={onExport}
          disabled={leads.length === 0}
          className="inline-flex items-center gap-2 rounded border border-line-strong px-3 py-1.5 text-xs font-medium text-ink transition hover:border-brass-soft hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Download className="size-3.5 text-brass" />
          Export Campaign CSV
        </button>
      </header>

      {fallbackReason && (
        <p className="border-b border-line bg-ochre/[0.06] px-5 py-2 text-xs text-ochre">
          The live pipeline was unavailable and the instant cache was served instead. Reason: {fallbackReason}
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[980px] border-collapse text-left text-[13px]">
          <thead>
            <tr className="border-b border-line text-[11px] uppercase tracking-[0.12em] text-faint">
              <th className="w-10 py-2.5 pl-5 pr-2 text-right font-medium">#</th>
              <th className="px-3 py-2.5 font-medium">Company</th>
              <th className="px-3 py-2.5 font-medium">Website</th>
              <th className="px-3 py-2.5 font-medium">Detected Buying Signal</th>
              <th className="px-3 py-2.5 font-medium">Target Buyer Role</th>
              <th className="px-3 py-2.5 font-medium">Fit</th>
              <th className="py-2.5 pl-3 pr-5 text-right font-medium">Est. Deal Value</th>
            </tr>
          </thead>
          <tbody>
            {leads.map((lead, i) => {
              const pushed = drafts[lead.id]?.pushed;
              const selected = selectedId === lead.id;
              return (
                <tr
                  key={lead.id}
                  onClick={() => onSelect(lead.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(lead.id);
                    }
                  }}
                  tabIndex={0}
                  style={{ animationDelay: `${i * 55}ms` }}
                  className={cn(
                    "group animate-fade-in cursor-pointer border-b border-line/70 align-top outline-none transition-colors last:border-b-0 hover:bg-panel-2 focus-visible:bg-panel-2",
                    selected && "bg-panel-2 shadow-[inset_2px_0_0_0_var(--color-brass)]",
                  )}
                >
                  <td className="py-3.5 pl-5 pr-2 text-right font-mono text-[11px] tabular-nums text-faint">
                    {String(i + 1).padStart(2, "0")}
                  </td>
                  <td className="px-3 py-3.5">
                    <div className="flex items-center gap-2 font-medium text-ink">
                      {lead.companyName}
                      {pushed && (
                        <span className="rounded-sm border border-sage/40 px-1 py-px text-[10px] font-normal uppercase tracking-wider text-sage">
                          Queued
                        </span>
                      )}
                    </div>
                    {(lead.hq || lead.sector) && (
                      <div className="mt-0.5 text-[11px] text-faint">{[lead.hq, lead.sector].filter(Boolean).join(" · ")}</div>
                    )}
                  </td>
                  <td className="px-3 py-3.5">
                    <a
                      href={lead.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="inline-flex items-center gap-0.5 text-[12px] text-muted underline decoration-line-strong underline-offset-4 transition hover:text-ink hover:decoration-brass"
                    >
                      {hostnameOf(lead.website)}
                      <ArrowUpRight className="size-3" />
                    </a>
                  </td>
                  <td className="max-w-[380px] px-3 py-3.5 text-muted">
                    <p className="line-clamp-2 transition-colors group-hover:text-ink/80">{lead.detectedSignal}</p>
                  </td>
                  <td className="px-3 py-3.5 text-ink/90">{lead.targetRole}</td>
                  <td className="px-3 py-3.5">
                    <FitMeter score={lead.fitScore} />
                  </td>
                  <td className="py-3.5 pl-3 pr-5 text-right font-mono text-[12px] tabular-nums text-ink">
                    {lead.estimatedAnnualValue}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
