"use client";

import { CheckCircle2, Download, ExternalLink } from "lucide-react";
import type { Lead } from "@/lib/types";
import type { LeadDraft } from "@/lib/csv";
import { cn, hostnameOf } from "@/lib/utils";
import { FitBadge } from "./ui";

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
  return (
    <section className="overflow-hidden rounded-lg border border-line bg-panel">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <h2 className="text-sm font-medium text-ink">Qualified Buyer Signals</h2>
          <span className="rounded border border-line px-1.5 py-0.5 font-mono text-[10px] text-muted">{leads.length} rows</span>
          {source && (
            <span
              title={fallbackReason}
              className={cn(
                "rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                source === "live" && "border-emerald-500/40 text-emerald-300",
                source === "cache" && !fallbackReason && "border-cyan-500/40 text-cyan-300",
                source === "cache" && fallbackReason && "border-amber-500/40 text-amber-300",
              )}
            >
              {source === "live" ? "Live · Groq 70B" : fallbackReason ? "Failsafe cache" : "Instant demo cache"}
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={onExport}
          disabled={leads.length === 0}
          className="inline-flex items-center gap-2 rounded-md border border-line-strong bg-panel-2 px-3 py-1.5 text-xs font-medium text-ink transition hover:border-emerald-500/50 hover:text-emerald-300 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Download className="size-3.5" />
          Export Campaign CSV
        </button>
      </header>

      {fallbackReason && (
        <p className="border-b border-line bg-amber-500/5 px-4 py-2 font-mono text-[11px] text-amber-300/90">
          Live pipeline fell back to the instant cache — {fallbackReason}
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[960px] border-collapse text-left text-[13px]">
          <thead>
            <tr className="border-b border-line font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
              <th className="px-4 py-2.5 font-normal">Company</th>
              <th className="px-3 py-2.5 font-normal">Website</th>
              <th className="px-3 py-2.5 font-normal">Detected Buying Signal</th>
              <th className="px-3 py-2.5 font-normal">Target Buyer Role</th>
              <th className="px-3 py-2.5 text-center font-normal">Fit</th>
              <th className="px-4 py-2.5 text-right font-normal">Est. Deal Value</th>
            </tr>
          </thead>
          <tbody>
            {leads.map((lead, i) => {
              const pushed = drafts[lead.id]?.pushed;
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
                  style={{ animationDelay: `${i * 45}ms` }}
                  className={cn(
                    "animate-fade-in cursor-pointer border-b border-line/70 align-top outline-none transition-colors last:border-b-0 hover:bg-white/[0.025] focus-visible:bg-white/[0.04]",
                    selectedId === lead.id && "bg-emerald-400/[0.05] shadow-[inset_2px_0_0_0_#34d399]",
                  )}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1.5 font-medium text-ink">
                      {lead.companyName}
                      {pushed && <CheckCircle2 className="size-3.5 shrink-0 text-emerald-400" aria-label="Queued in campaign" />}
                    </div>
                    {(lead.hq || lead.sector) && (
                      <div className="mt-0.5 text-[11px] text-faint">{[lead.hq, lead.sector].filter(Boolean).join(" · ")}</div>
                    )}
                  </td>
                  <td className="px-3 py-3">
                    <a
                      href={lead.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="inline-flex items-center gap-1 font-mono text-[12px] text-cyan-300/90 hover:text-cyan-200"
                    >
                      {hostnameOf(lead.website)}
                      <ExternalLink className="size-3 opacity-60" />
                    </a>
                  </td>
                  <td className="max-w-[380px] px-3 py-3 text-muted">
                    <p className="line-clamp-2">{lead.detectedSignal}</p>
                  </td>
                  <td className="px-3 py-3 text-ink/90">{lead.targetRole}</td>
                  <td className="px-3 py-3 text-center">
                    <FitBadge score={lead.fitScore} />
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-[12px] tabular-nums text-emerald-300">
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
