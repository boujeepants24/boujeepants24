## Aran Chundru

**AI-native product engineer.** I take a business problem, design the system, build it, ship it, and measure whether it helped.

Self-taught. I work across full-stack web, AI integrations, business automation, smart contracts and quantitative research. I'd rather show working, tested software than list skills, so everything below links to code you can run.

### What I'm building now

| Project | What it is | Status |
|---|---|---|
| **[BusinessOS](https://github.com/boujeepants24/businessos)** | Runs the back office of a local service business while the owner does the work: booking site, chat receptionist, missed-call text-back (Twilio), reminders, quotes, invoices, payment chasing, reviews, rebooking and reports. | Working prototype with automated tests (Apps Script back office, Python comms gateway). Moving to Next.js + Postgres. |
| **[domain-rag](https://github.com/boujeepants24/businessos/tree/main/domain-rag)** | Retrieval-augmented answers over one folder of documents, exposed as an MCP server and a multi-client web API. Hybrid search, cited answers from Claude, prompt-injection defences, evaluation harness. | Pipeline complete and tested; lives inside BusinessOS, where it powers the chat receptionist's answers. |
| **[Onchain Escrow](https://github.com/boujeepants24/businessos/tree/main/onchain-escrow)** | Milestone-based escrow for freelance and agency work: deposit, approve, release, refund, dispute. Solidity + Foundry. | Contract with unit, fuzz, adversarial and invariant tests. Not audited; testnet only. Frontend next. |
| **Quant Research Lab** (private while being rebuilt) | My four earlier backtests (funding-rate arbitrage, Kalman market-neutral, HMM regimes, FinBERT event study) in one repo, being rebuilt to one standard: pinned data, no look-ahead, out-of-sample tests, cost modelling. | Rebuild planned. Every project lists the issues found in my own audit; treat current results as unverified. |

### How I work

- **Start from the problem.** Software earns its place by saving someone time or money.
- **Tests before claims.** If a number is in a README, the command that produced it is next to it.
- **Say what's wrong.** Every project lists its known limitations.
- **Ship small, then improve.** A working core workflow beats a long feature list.
- **Security is part of the build,** not a step at the end. Contracts get adversarial tests; nothing touches real funds without an independent review.

### Stack

**Used in my public code:** Python (pandas, NumPy, SciPy, scikit-learn, Streamlit), JavaScript, HTML/CSS, Google Apps Script, PHP with MySQL, SQLite, HuggingFace Transformers, Solidity with Hardhat, OpenZeppelin and ethers.js, Git.

**Used in projects above:** Claude API (structured outputs), MCP servers, retrieval (BM25 + vectors), Starlette, Twilio, Foundry fuzz and invariant testing, GitHub Actions.

**Learning by building now:** TypeScript, Next.js, PostgreSQL/Supabase, Docker.

### Contact

aranthebotguy@gmail.com
