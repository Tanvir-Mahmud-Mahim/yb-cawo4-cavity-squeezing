"""Cross-file consistency checks, run as part of every build.

Catches the class of mistake that is easy to make and hard to see: a title, a
version or a DOI updated in one file and left stale in another, a citation key
used but not defined, an unresolved [[TOKEN]], a figure panel referred to that
does not exist, or a cross-reference to a section or equation number that has
moved.  Exits non-zero and prints every failure.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")


def read(*parts):
    path = os.path.join(ROOT, *parts)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


def main():
    fail = []
    main_tex = read("paper", "main.tex")
    supp_tex = read("paper", "supplement.tex")
    readme = read("README.md")
    cff = read("CITATION.cff")
    bib = read("paper", "refs.bib")
    if main_tex is None or supp_tex is None:
        print("check_consistency: paper sources not found, nothing to check")
        return 0

    # --- title: one source, used everywhere -------------------------------
    m = re.search(r"\\title\{(.+?)\}\s*\n", main_tex, re.S)
    title = " ".join(m.group(1).split()) if m else None
    if title is None:
        fail.append("could not read \\title{...} from paper/main.tex")
    else:
        if "[[TITLE]]" not in supp_tex:
            fail.append("paper/supplement.tex must carry [[TITLE]], not a copy of the title")
        if readme and title not in " ".join(readme.split()):
            fail.append("README.md does not carry the current article title")

    # --- version and DOI agree across README, CITATION and the paper ------
    ver_cff = re.search(r"^version:\s*(\S+)", cff or "", re.M)
    doi_readme = re.findall(r"v(\d+\.\d+\.\d+) is https://doi\.org/(10\.5281/zenodo\.\d+)", readme or "")
    doi_paper = set(re.findall(r"10\.5281/zenodo\.(\d+)", main_tex))
    if ver_cff and doi_readme:
        if ver_cff.group(1) != doi_readme[0][0]:
            fail.append(f"CITATION.cff version {ver_cff.group(1)} != README version {doi_readme[0][0]}")
    if doi_readme and doi_paper:
        want = doi_readme[0][1].split(".")[-1]
        if doi_paper != {want}:
            fail.append(f"main.tex cites zenodo {sorted(doi_paper)} but README names {want}")
    if len(doi_paper) > 1:
        fail.append(f"main.tex cites more than one Zenodo record: {sorted(doi_paper)}")

    # --- citations: every key used is defined, every key defined is used ---
    if bib:
        defined = set(re.findall(r"@\w+\{([^,]+),", bib))
        used = set()
        for t in (main_tex, supp_tex):
            for grp in re.findall(r"\\cite\{([^}]*)\}", t):
                used |= {k.strip() for k in grp.split(",")}
        for k in sorted(used - defined):
            fail.append(f"cited but not in refs.bib: {k}")
        for k in sorted(defined - used):
            fail.append(f"in refs.bib but never cited: {k}")

    # --- no unresolved tokens in the filled files --------------------------
    for name in ("main_filled.tex", "supplement_filled.tex"):
        t = read("paper", name)
        if t and "[[" in t:
            left = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", t)))
            fail.append("%s still contains unresolved tokens: %s" % (name, left))

    # --- panel references point at panels that exist ----------------------
    panels = {}
    for tex, prefix in ((main_tex, ""), (supp_tex, "")):
        for blk in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
            lab = re.search(r"\\label\{(fig:[^}]+)\}", blk.group(1))
            cap = re.search(r"\\caption\{(.*)", blk.group(1), re.S)
            if lab and cap:
                letters = set(re.findall(r"\(([a-z])\)", cap.group(1)[:2500]))
                if letters:
                    panels[lab.group(1)] = max(letters)
    for tex, where in ((main_tex, "main"), (supp_tex, "supplement")):
        for ref, pans in re.findall(r"\\ref\{(fig:[a-z_]+)\}\(([a-z, ]+)\)", tex):
            if ref not in panels:
                continue
            for ch in re.findall(r"[a-z]", pans):
                if ch > panels[ref]:
                    fail.append(f"{where}: {ref} has no panel ({ch}); highest is ({panels[ref]})")

    # --- supplement section numbers referred to from the main text exist ---
    secs = set(re.findall(r"\\section\*\{S(\d+)\.", supp_tex))
    for n in sorted(set(re.findall(r"Sec\.~S(\d+)", main_tex))):
        if n not in secs:
            fail.append(f"main.tex refers to Sec. S{n}, which the supplement does not have")

    # --- the supplement must not hard-code a main-text equation number -----
    for n in sorted(set(re.findall(r"Eq\.~\((\d+)\)", supp_tex))):
        fail.append(f"supplement hard-codes Eq. ({n}); use the [[EQ_*]] tokens instead")

    # --- the supplement must not hard-code a supplement figure number ------
    for n in sorted(set(re.findall(r"Fig\.~S(\d+)", supp_tex))):
        fail.append(f"supplement hard-codes Fig. S{n}; use \\ref{{fig:...}} instead")

    # --- nor a main-text figure number; those come from the [[FIG_*]] tokens
    for n in sorted(set(re.findall(r"Fig(?:ure|s)?\.?~(\d+)", supp_tex))):
        fail.append(f"supplement hard-codes Fig. {n} of the main text; "
                    "use the [[FIG_*]] tokens instead")

    # --- the main text must not hard-code a supplement float number --------
    for n in sorted(set(re.findall(r"(?:Figs?\.|Figure|Tables?|Table)~S(\d+)", main_tex))):
        fail.append(f"main.tex hard-codes S{n} of the supplement; "
                    "use the [[SFIG_*]] / [[STAB_*]] tokens instead")

    # --- nor a main-text section number; those come from [[SEC_*]] ---------
    for n in sorted(set(re.findall(r"Sec(?:tion)?\.?~([IVX]+)\\,?[A-Z]?", supp_tex))):
        fail.append(f"supplement hard-codes Sec. {n} of the main text; "
                    "use the [[SEC_*]] tokens instead")

    # --- the token numbers must match what LaTeX actually printed ----------
    aux = read("paper", "main_filled.aux")
    nums_raw = read("data", "numbers.json")
    if aux and nums_raw:
        nums = json.loads(nums_raw)
        aux_num = {lab: int(n) for lab, n in
                   re.findall(r"\\newlabel\{([^}]+)\}\{\{(\d+)\}", aux)}
        for tok, lab in (("EQ_TC", "eq:TC"), ("EQ_RATES", "eq:rates"),
                         ("EQ_LAWHOMO", "eq:lawhomo"), ("EQ_PENDULUM", "eq:pendulum"),
                         ("EQ_ZDOT", "eq:zdot"), ("EQ_LOCKING", "eq:locking"),
                         ("EQ_TWOLIMITS", "eq:twolimits"), ("EQ_RULE", "eq:rule"),
                         ("EQ_GM", "eq:gm_main"), ("FIG_DEVICE", "fig:device"),
                         ("FIG_BENCHMARK", "fig:benchmark"),
                         ("FIG_DESIGNMAP", "fig:designmap"), ("FIG_ECHO", "fig:echo")):
            if tok in nums and lab in aux_num and int(nums[tok]) != aux_num[lab]:
                fail.append("token %s is %s but LaTeX numbers %s as %d"
                            % (tok, nums[tok], lab, aux_num[lab]))

    if fail:
        print("consistency check FAILED:")
        for f in fail:
            print("  -", f)
        return 1
    print("consistency check passed"
          + (f" (title: {title[:60]}...)" if title else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
