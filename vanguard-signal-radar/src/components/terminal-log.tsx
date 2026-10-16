"use client";

import { useEffect, useRef } from "react";
import { Check, ChevronDown } from "lucide-react";
import type { LogTag } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface LogLine {
  id: number;
  tag: LogTag;
  message: string;
  ts: number;
}

const TAG_STYLES: Record<LogTag, string> = {
  SYSTEM: "text-muted",
  SEARCH: "text-steel",
  CRAWLER: "text-steel",
  "GROQ-70B": "text-brass",
  CACHE: "text-steel",
  WARN: "text-ochre",
  ERROR: "text-claret",
  DONE: "text-sage",
};

export const STAGES = ["Search", "Crawl", "Analyse", "Complete"] as const;

/** Maps the latest log tag onto the 4-stage sweep pipeline (index into STAGES, -1 = idle). */
export function stageFromLogs(lines: LogLine[]): number {
  let stage = lines.length ? 0 : -1;
  for (const l of lines) {
    if (l.tag === "CRAWLER") stage = Math.max(stage, 1);
    if (l.tag === "GROQ-70B") stage = Math.max(stage, 2);
    if (l.tag === "DONE") stage = 3;
  }
  return stage;
}

interface Props {
  lines: LogLine[];
  running: boolean;
  open: boolean;
  onToggle: () => void;
}

export function TerminalLog({ lines, running, open, onToggle }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const startTs = lines[0]?.ts ?? 0;
  const stage = stageFromLogs(lines);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length, open]);

  return (
    <section className="overflow-hidden rounded-md border border-line bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 border-b border-line px-5 py-3">
        <div className="flex items-center gap-3">
          <h2 className="font-serif text-[15px] text-ink">Sweep Log</h2>
          <span className="flex items-center gap-1.5 text-[11px] text-faint">
            <span
              className={cn(
                "size-1.5 rounded-full",
                running ? "animate-pulse bg-brass" : stage === 3 ? "bg-sage" : "bg-line-strong",
              )}
            />
            {running ? "In progress" : stage === 3 ? "Completed" : lines.length ? "Stopped" : "Standing by"}
          </span>
        </div>

        <div className="flex items-center gap-4">
          <ol className="hidden items-center sm:flex" aria-label="Sweep stages">
            {STAGES.map((label, i) => {
              const done = stage > i || (stage === 3 && i === 3);
              const active = running && stage === i;
              return (
                <li key={label} className="flex items-center">
                  {i > 0 && (
                    <span className="relative mx-2 h-px w-8 overflow-hidden bg-line-strong">
                      <span
                        className={cn(
                          "absolute inset-y-0 left-0 bg-brass transition-[width] duration-500",
                          stage >= i ? "w-full" : "w-0",
                        )}
                      />
                    </span>
                  )}
                  <span
                    className={cn(
                      "flex items-center gap-1.5 text-[11px] transition-colors",
                      done ? "text-ink" : active ? "text-brass" : "text-faint",
                    )}
                  >
                    <span
                      className={cn(
                        "grid size-4 place-items-center rounded-full border text-[9px] transition-colors",
                        done ? "border-brass bg-brass text-obsidian" : active ? "border-brass" : "border-line-strong",
                      )}
                    >
                      {done ? <Check className="size-2.5" strokeWidth={3} /> : i + 1}
                    </span>
                    {label}
                  </span>
                </li>
              );
            })}
          </ol>
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={open}
            aria-label={open ? "Collapse log" : "Expand log"}
            className="rounded p-1 text-faint transition hover:bg-panel-2 hover:text-ink"
          >
            <ChevronDown className={cn("size-4 transition-transform", open && "rotate-180")} />
          </button>
        </div>
      </div>

      {/* Indeterminate progress rail while a sweep is streaming. */}
      <div className="relative h-px overflow-hidden bg-transparent">
        {running && <span className="animate-sheen absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-brass to-transparent" />}
      </div>

      {open && (
        <div ref={scrollRef} className="max-h-56 overflow-y-auto bg-obsidian/60 px-5 py-3 font-mono text-[12px] leading-[1.75]">
          {lines.length === 0 ? (
            <p className="text-faint">
              Awaiting sweep command
              <span className="animate-caret ml-1 inline-block h-3 w-[6px] translate-y-0.5 bg-muted" />
            </p>
          ) : (
            lines.map((line) => (
              <div key={line.id} className="animate-fade-in flex gap-4 whitespace-pre-wrap break-words">
                <span className="w-14 shrink-0 text-right tabular-nums text-faint">
                  {((line.ts - startTs) / 1000).toFixed(2)}s
                </span>
                <span className={cn("w-20 shrink-0", TAG_STYLES[line.tag])}>{line.tag}</span>
                <span className={cn("min-w-0", line.tag === "ERROR" ? "text-claret" : "text-ink/85")}>{line.message}</span>
              </div>
            ))
          )}
          {running && (
            <div className="flex gap-4">
              <span className="w-14" />
              <span className="animate-caret mt-1.5 inline-block h-3 w-[6px] bg-brass" />
            </div>
          )}
        </div>
      )}
    </section>
  );
}
