import { cn } from "@/lib/utils";

export function FitBadge({ score, className }: { score: number; className?: string }) {
  const tone =
    score >= 95
      ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-300"
      : score >= 90
        ? "border-cyan-400/40 bg-cyan-400/10 text-cyan-300"
        : "border-sky-400/30 bg-sky-400/5 text-sky-300";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[11px] font-semibold tabular-nums",
        tone,
        className,
      )}
    >
      {score}
    </span>
  );
}

export function SectionLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <h3 className={cn("font-mono text-[10px] uppercase tracking-[0.2em] text-faint", className)}>{children}</h3>
  );
}

/** Parses "€145,000/yr" → 145000. */
export function parseEur(value: string): number {
  const m = value.replace(/[^\d.,kKmM]/g, "");
  const digits = m.replace(/[kKmM]$/, "");
  // "45.000" / "1.250.000" use dots as thousands separators (EU style).
  const normalised = /^\d{1,3}(\.\d{3})+$/.test(digits) ? digits.replace(/\./g, "") : digits.replace(/,/g, "");
  let n = Number(normalised);
  if (/k$/i.test(m)) n *= 1_000;
  if (/m$/i.test(m)) n *= 1_000_000;
  return Number.isFinite(n) ? n : 0;
}

export function formatEur(n: number): string {
  if (n >= 1_000_000) return `€${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `€${Math.round(n / 1_000)}k`;
  return `€${n}`;
}
