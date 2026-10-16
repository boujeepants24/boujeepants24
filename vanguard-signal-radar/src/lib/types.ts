export type SweepMode = "live" | "cache";

export type PresetId = "pcb" | "cnc" | "logistics" | "custom";

export interface Lead {
  id: string;
  companyName: string;
  website: string;
  detectedSignal: string;
  targetRole: string;
  estimatedAnnualValue: string;
  fitScore: number;
  sniperSubject: string;
  sniperEmail: string;
  /** Extra context shown in the inspector. */
  hq?: string;
  sector?: string;
  siteTitle?: string;
  siteDescription?: string;
  source: "live" | "cache";
}

export interface SweepRequest {
  query: string;
  offer: string;
  mode: SweepMode;
  presetId: PresetId;
}

export type LogTag = "SYSTEM" | "SEARCH" | "CRAWLER" | "GROQ-70B" | "CACHE" | "WARN" | "ERROR" | "DONE";

export type SweepEvent =
  | { type: "log"; tag: LogTag; message: string; ts: number }
  | { type: "result"; leads: Lead[]; source: "live" | "cache"; fallbackReason?: string; ts: number }
  | { type: "error"; message: string; ts: number };
