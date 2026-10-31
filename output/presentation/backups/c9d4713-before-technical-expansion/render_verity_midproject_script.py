"""Render the timed five-speaker script from the current mid-project slide content."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clock(seconds):
    return f"{seconds//60:02d}:{seconds%60:02d}"


def render():
    content = json.loads((ROOT / "docs/presentation/verity-midproject-content.json").read_text(encoding="utf-8"))
    rows = content["slides"]
    if sum(row["seconds"] for row in rows) != 600:
        raise ValueError("The presentation must total exactly ten minutes")
    for speaker in range(1, 6):
        if sum(row["seconds"] for row in rows if row["speaker"] == speaker) != 120:
            raise ValueError("Each speaker must have a two-minute slot")
    out = ["# Verity mid-project review: timed five-speaker script", "",
           "**Total: 10:00. Five speakers, 2:00 each. Thirteen slides, no extra introduction or closing slide.**", "",
           "Read only the quoted speech. The remaining text is rehearsal guidance. The schedule includes visual reading, pointing and handovers. Rehearse at about 125 words/minute and keep a visible timer. Q&A follows the ten-minute presentation. The website section uses genuine recorded captures; fresh GPU generation is an optional demonstration afterward.", "",
           "Replace Member 1–5 with names. Presentation assignments need not match contribution ownership. Identify the reused checkpoint once in configuration/Q&A. The main story is our approved-memory application and measured comparisons.", "",
           "## Fixed speaking order", "", "| Speaker | Clock | Slides | Main subject |", "| --- | --- | --- | --- |",
           "| Member 1 | 00:00–02:00 | 1–2 | Objective and sourced Air India motivation |",
           "| Member 2 | 02:00–04:00 | 3–4 | Implemented architecture and retrieval |",
           "| Member 3 | 04:00–06:00 | 5–8 | Governed corrections and website walkthrough |",
           "| Member 4 | 06:00–08:00 | 9–11 | Experimental design, findings and latency |",
           "| Member 5 | 08:00–10:00 | 12–13 | Engineering conclusion and next research milestone |", ""]
    elapsed = 0
    words_total = 0
    for row in rows:
        words = len(row["speech"].split())
        speech_seconds = words * 60 / 125
        if speech_seconds > row["seconds"] - 1:
            raise ValueError(f"Slide {row['n']} needs a shorter speech")
        if row["n"] == 1 or row["speaker"] != rows[row["n"]-2]["speaker"]:
            out += [f"## Member {row['speaker']}", ""]
        out += [f"### Slide {row['n']}: {row['title']}", "",
                f"**Clock: {clock(elapsed)}–{clock(elapsed + row['seconds'])}. Slot: {row['seconds']} seconds. Speech: {words} words.**", "",
                "> " + row["speech"], "",
                f"**Visual action and pacing:** {row['cue']} At 125 words/minute, leave approximately {row['seconds']-speech_seconds:.0f} seconds for visual reading/pointing and any handover. Do not fill that time with extra unscripted material.", "",
                "**Rehearsal detail:** " + row["detail"], "", "**Evidence:**", ""]
        out += [f"- [{source}]({source if source.startswith('https:') else '../../'+source})" for source in row["sources"]]
        out.append("")
        elapsed += row["seconds"]
        words_total += words
    out += ["## Clock discipline", "",
            "- At 01:55, 03:55, 05:55 and 07:55, the next speaker gets ready. Hand over on the two-minute boundary.",
            "- Advance on the listed clock marks. The final milestone slide ends at 10:00.",
            "- Use the captured website views during the timed presentation. Do not launch an unbounded live generation or perform a fresh model download in the slot.",
            "- If the panel interrupts, ask whether to pause the presentation clock. Do not silently add Q&A time to a strict ten-minute slot.", "",
            "## Short technical Q&A", "",
            "**Is the current implementation RAG?** Yes. It combines approved external memory, bounded lexical retrieval and fixed generators. Our implemented contribution is governance, context/history handling, the application, independent controlled evaluation and evidence engineering.", "",
            "**Which model is reused?** `YukinoKaren/ReasonKE_plus`, pinned at `ceaedc2ae7c8fed6257953587d991ce550db6106`. The comparison is `Qwen/Qwen2.5-7B-Instruct` at its pinned revision. The detailed upstream training tutorial is outside this review.", "",
            "**Where is our new model training?** SFT/PPO adaptation is the next milestone. The slide describes a matched study, beginning with reviewed data and profiling, then a fresh frozen broader evaluation.", "",
            "**Is the Air India example current ownership?** It is explicitly a dated historical snapshot around the 27 January 2022 handover. Later arrangements changed. The corporate-contact fact comes from the official page checked on 31 October 2026.", "",
            "**Does 256/256 guarantee arbitrary questions?** It is the observed final-span score on a controlled packet with 32 final lineages and small worlds. The broader next study addresses generalisation and larger memory.", "",
            "**Why does full memory fail?** All 25 original ReasonKE_plus wrong finals match other supplied values despite complete required support. That describes visible wrong associations; it does not prove an internal causal mechanism.", "",
            "**Does rollback change neural weights?** It atomically restores approved memory and events. Fresh requests use that state. Saved messages retain their original revision and output.", "",
            "**Will training beat the baseline?** That is the research question. We will measure accuracy, coverage, locality, latency and resources under matched conditions rather than promise a gain.", "",
            "## Evidence-based contribution allocation", "",
            "Attach names only to implementation, reproduction, review, analysis or documentation actually performed. These work areas can overlap and are independent of speaking roles.", "",
            "Proposed lead allocation for implementation/review/reproduction. Confirm actual work before treating this as historical credit; members can collaborate across rows.", "",
            "| Lead | Work area | Concrete contribution | Reviewable evidence |", "| --- | --- | --- | --- |",
            "| Member 1 | Data and sources | Controlled worlds, questions, splits, label audits and the sourced Air India fixture | data modules, dataset cards, data/demos/air-india-handover.json |",
            "| Member 2 | Retrieval and model integration | Pinned GPU worker, bounded context selection, iterative expansion and prompt/history interfaces | runtime, reasonke_backend.py, retrieval modules |",
            "| Member 3 | Backend reliability | Canonical identity, explicit approval, supersession/conflicts, events and atomic rollback | store.py, workspace.py, server.py, regression tests |",
            "| Member 4 | Evaluation and analysis | Fixed matched protocols, ledgers/resume, scoring, uncertainty, latency and failure diagnosis | benchmark/depth packages, raw receipts and audit scripts |",
            "| Member 5 | Product and reproducibility | Verity chat/editor/explorer, real-source walkthrough, containers/notebooks, CI and presentation material | static/, results.py, sourced-demo scripts and docs |", ""]
    target = ROOT / "docs/presentation/VERITY_SPEAKER_SCRIPT.md"
    target.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(json.dumps({"slides": len(rows), "seconds": elapsed, "speakers": 5, "spoken_words": words_total,
                      "script": str(target)}, indent=2))


if __name__ == "__main__":
    render()
