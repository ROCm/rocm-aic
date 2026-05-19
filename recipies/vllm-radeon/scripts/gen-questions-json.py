#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Write a questions JSON file for long-context QA from generic reading prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Placeholders: {title}, {author}
QUESTION_TEMPLATES: list[str] = [
    "What is the central conflict introduced early in {title}, and how is it framed?",
    "Who is the protagonist of {title}, and what traits define them in the opening chapters?",
    "Describe the social world of {title} as the author establishes it in the first major scenes.",
    "What role does setting play in shaping mood and expectations in {title}?",
    "How does {author} introduce major themes in {title} without stating them directly?",
    "Which secondary character in {title} most influences the protagonist's choices, and how?",
    "What turning point in {title} changes the direction of the plot, and why does it matter?",
    "How is family or community portrayed in {title}, and what tensions appear within it?",
    "What does {title} suggest about honor, duty, or reputation in its historical context?",
    "Compare two major characters in {title} in terms of ambition and moral limits.",
    "How does dialogue in {title} reveal class, education, or regional identity?",
    "What symbols or recurring images appear in {title}, and what might they represent?",
    "How does {title} handle love, marriage, or courtship, and what constraints shape those bonds?",
    "What is a pivotal decision in {title} that the protagonist later regrets or reconsiders?",
    "How does {author} use irony or contrast between appearance and reality in {title}?",
    "What does a major battle, journey, or migration episode in {title} reveal about leadership?",
    "How are servants, laborers, or lower-status figures depicted in {title}?",
    "What religious, philosophical, or political ideas surface in {title}, and who voices them?",
    "How does {title} portray wealth, debt, or inheritance as forces on character fate?",
    "What childhood or formative memory in {title} explains an adult character's behavior?",
    "How does {title} depict friendship versus rivalry among peers?",
    "What scandal, rumor, or public shame drives plot movement in {title}?",
    "How does {author} balance panoramic description with intimate character perspective in {title}?",
    "What does {title} imply about fate, chance, and individual agency?",
    "How is war or violence described in {title}, and whose viewpoint dominates those scenes?",
    "What domestic scene in {title} is most revealing of a marriage or household dynamic?",
    "How does {title} treat outsiders, foreigners, or people from another region or class?",
    "What moral test does the protagonist face in {title}, and how do they respond?",
    "How does illness, injury, or mortality affect relationships in {title}?",
    "What comic or satirical moments in {title} undercut solemn themes, and to what effect?",
    "How does {title} show the gap between official history and lived experience?",
    "What prophecy, omen, or foreshadowing device appears in {title}, and is it fulfilled?",
    "How does a mentor or elder guide—or misguide—a younger character in {title}?",
    "What does {title} suggest about education, reading, or self-improvement?",
    "How are women in {title} granted or denied agency within social rules of the era?",
    "What rivalry between institutions (church, state, army, commerce) appears in {title}?",
    "How does {title} use letters, diaries, or delayed news to advance the plot?",
    "What feast, ball, salon, or public gathering in {title} crystallizes social hierarchy?",
    "How does the antagonist or opposing force in {title} justify their actions?",
    "What act of forgiveness or reconciliation in {title} feels earned rather than convenient?",
    "How does landscape, weather, or season mirror emotional states in {title}?",
    "What legal, bureaucratic, or contractual detail in {title} traps or frees a character?",
    "How does {title} portray courage under fear in a crisis scene?",
    "What generational conflict between parent and child appears in {title}?",
    "How does {title} explore loyalty when it conflicts with self-interest?",
    "What disguise, secret identity, or hidden relationship complicates {title}?",
    "How does {author} handle time jumps or ellipses in {title}, and what do we lose or gain?",
    "What does a trial, duel, election, or contest scene in {title} expose about society?",
    "How is nostalgia or regret expressed in {title} through memory or return visits?",
    "What economic hardship in {title} forces characters to compromise their values?",
    "How does {title} depict hospitality, gift-giving, or obligation between hosts and guests?",
    "What role do children or heirs play in inheritance plots within {title}?",
    "How does music, art, or performance function in a key scene of {title}?",
    "What does {title} say about patriotism versus self-preservation during national crisis?",
    "How does exile, imprisonment, or captivity change a character in {title}?",
    "What parallel or foil pairing in {title} clarifies the author's moral argument?",
    "How does {title} treat superstition alongside rational explanation?",
    "What banquet or meal scene in {title} is politically or emotionally charged?",
    "How does gossip shape outcomes in {title}?",
    "What does {title} reveal about medical care or bodily suffering in its period?",
    "How does a character's name, title, or rank alter how others treat them in {title}?",
    "What desertion, betrayal, or broken oath in {title} has the widest consequences?",
    "How does {title} contrast city life with rural or provincial life?",
    "What dream, vision, or interior monologue in {title} reframes earlier events?",
    "How does {author} depict bureaucracy slowing or distorting urgent action in {title}?",
    "What friendship across class lines in {title} is tested by social pressure?",
    "How does {title} portray gambling, risk, or speculation?",
    "What religious ritual or moral sermon in {title} influences a character's path?",
    "How does {title} show the cost of ambition on private happiness?",
    "What rescue or escape sequence in {title} is most suspenseful, and why?",
    "How does {title} treat animals, hunting, or agriculture as part of character identity?",
    "What diplomatic or political negotiation in {title} fails or succeeds unexpectedly?",
    "How does shame versus guilt motivate different characters in {title}?",
    "What does {title} imply about the reliability of narrators or eyewitnesses?",
    "How does fashion, housing, or material objects signal status in {title}?",
    "What mentor's advice in {title} is later proven wrong by events?",
    "How does {title} handle jealousy between siblings or close friends?",
    "What public speech or sermon in {title} moves a crowd or changes policy?",
    "How does {title} depict the aftermath of a disaster (fire, flood, invasion)?",
    "What secret document, will, or map in {title} alters inheritance or strategy?",
    "How does {author} use repetition of a phrase, place, or image across {title}?",
    "What courtship mistake in {title} has long-term social repercussions?",
    "How does {title} portray veterans or survivors re-entering civilian life?",
    "What charity, reform, or utopian project in {title} reflects contemporary debates?",
    "How does {title} contrast youth and age in outlook and opportunity?",
    "What moment in {title} best illustrates the author's view of historical causation?",
    "How does {title} depict language barriers or translation between cultures?",
    "What role does chance encounter play in reuniting or separating characters in {title}?",
    "How does {title} treat suicide, despair, or spiritual crisis?",
    "What comic character in {title} nonetheless voices a serious truth?",
    "How does {title} show supply, hunger, or logistics during a campaign or siege?",
    "What inheritance dispute in {title} pits relatives against one another?",
    "How does {title} portray teachers, tutors, or schools shaping young minds?",
    "What oath or vow in {title} binds a character beyond their initial intention?",
    "How does {author} shift sympathy toward a previously unsympathetic figure in {title}?",
    "What festival, holiday, or sacred day in {title} structures the narrative calendar?",
    "How does {title} explore truth-telling versus polite deception in society?",
    "What scene of reconciliation between enemies in {title} feels plausible or forced?",
    "How does {title} depict craftsmen, merchants, or professionals distinct from the gentry?",
    "What does the ending of {title} resolve, and what questions remain deliberately open?",
    "How does {title} use a journey motif to test and transform character?",
    "What does {author} suggest about the limits of planning in {title}'s largest events?",
    "How does {title} portray idealism colliding with practical compromise?",
    "What single chapter or episode in {title} would you assign to illustrate the author's style?",
    "How does {title} treat memory distortion when characters recount the same event differently?",
    "What civic duty in {title} conflicts with personal desire?",
    "How does {title} depict corruption or abuse of power, and who resists it?",
    "What romantic triangle in {title} drives the most consequential choices?",
    "How does {title} use silence, pause, or omission as a narrative technique?",
    "What does {title} teach about patience versus impulsive action?",
    "How does {author} close a major arc in {title} while leaving subsidiary threads active?",
    "What historical figure or event in {title} is reinterpreted from a private angle?",
    "How does {title} portray grief and mourning rituals?",
    "What act of generosity in {title} is misunderstood or repaid badly?",
    "How does {title} frame justice: legal, divine, social, or poetic?",
    "What technological or modern intrusion (mail, rail, press) changes tempo in {title}?",
    "How does {title} depict peer pressure among soldiers, students, or courtiers?",
    "What does {title} suggest about the relationship between art and moral instruction?",
    "How does a secondary plot in {title} comment on the main plot's theme?",
    "What moment in {title} would you cite to defend the book's lasting reputation?",
]


def format_questions(
    templates: list[str],
    title: str,
    author: str,
    count: int,
) -> list[str]:
    if count > len(templates):
        raise ValueError(
            f"requested {count} questions but only {len(templates)} templates; "
            "add templates or lower --count"
        )
    out: list[str] = []
    for tpl in templates[:count]:
        out.append(tpl.format(title=title, author=author))
    return out


def load_extra_questions(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            return [str(x) for x in data["questions"]]
        raise ValueError(f"{path}: expected JSON array or {{questions: [...]}}")
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return lines


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slug",
        required=True,
        help="Book slug (hyphenated), e.g. war-and-peace; used in default output path.",
    )
    p.add_argument(
        "--title",
        required=True,
        help="Full book title for question text, e.g. 'War and Peace'.",
    )
    p.add_argument(
        "--author",
        default="the author",
        help="Author name inserted into prompts (default: 'the author').",
    )
    p.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of questions to emit (default: 100).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/<slug>/<slug>.questions.json).",
    )
    p.add_argument(
        "--extra-questions",
        type=Path,
        default=None,
        help="Optional .json or line-based file; if set, use these instead of templates.",
    )
    p.add_argument(
        "--pg-id",
        type=int,
        default=None,
        help="Project Gutenberg id (stored in metadata only).",
    )
    args = p.parse_args()

    slug = args.slug.strip()
    if not slug:
        print("error: --slug must be non-empty", file=sys.stderr)
        return 1

    out = args.output or (root / "data" / slug / f"{slug}.questions.json")

    try:
        if args.extra_questions is not None:
            questions = load_extra_questions(args.extra_questions)
            if len(questions) < args.count:
                print(
                    f"error: --extra-questions has {len(questions)} items, need {args.count}",
                    file=sys.stderr,
                )
                return 1
            questions = questions[: args.count]
        else:
            questions = format_questions(
                QUESTION_TEMPLATES, args.title, args.author, args.count
            )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    payload: dict[str, object] = {
        "slug": slug,
        "title": args.title,
        "author": args.author,
        "question_count": len(questions),
        "questions": questions,
        "note": (
            "Reader-study prompts for long-context QA. Pair with text chunks from "
            "split-gutenberg-random-chunks.py."
        ),
    }
    if args.pg_id is not None:
        payload["gutenberg_id"] = args.pg_id

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
