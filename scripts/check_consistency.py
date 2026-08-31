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
    # the title carries a \\ so that it breaks in two balanced lines; the plain
    # text is what README and CITATION.cff have to agree with
    title = " ".join(m.group(1).replace(r"\\", " ").split()) if m else None
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
    ver_toml = re.search(r'^version\s*=\s*"(\S+?)"', read("pyproject.toml") or "", re.M)
    if ver_cff and doi_readme:
        if ver_cff.group(1) != doi_readme[0][0]:
            fail.append(f"CITATION.cff version {ver_cff.group(1)} != README version {doi_readme[0][0]}")
    if ver_cff and ver_toml and ver_cff.group(1) != ver_toml.group(1):
        fail.append(f"CITATION.cff version {ver_cff.group(1)} != pyproject.toml version {ver_toml.group(1)}")
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
    for n in sorted(set(re.findall(r"Secs?(?:tion)?\.?~([IVX]+)\\,?[A-Z]?", supp_tex))):
        fail.append(f"supplement hard-codes Sec. {n} of the main text; "
                    "use the [[SEC_*]] tokens instead")

    # --- the figure and table numbers the README quotes must be real -------
    saux = read("paper", "supplement_filled.aux")
    maux = read("paper", "main_filled.aux")
    if readme and saux and maux:
        mfig = set(re.findall(r"\\newlabel\{fig:[^}]+\}\{\{(\d+)\}", maux))
        meq = set(re.findall(r"\\newlabel\{eq:[^}]+\}\{\{(\d+)\}", maux))
        sfig = set(re.findall(r"\\newlabel\{fig:[^}]+\}\{\{(S\d+)\}", saux))
        stab = set(re.findall(r"\\newlabel\{tab:[^}]+\}\{\{(S\d+)\}", saux))
        for n in sorted(set(re.findall(r"Figs?\. (S\d+)", readme))):
            if n not in sfig:
                fail.append(f"README names Fig. {n}, which the supplement does not have")
        for n in sorted(set(re.findall(r"Table (S\d+)", readme))):
            if n not in stab:
                fail.append(f"README names Table {n}, which the supplement does not have")
        for n in sorted(set(re.findall(r"Fig\. (\d+)", readme))):
            if n not in mfig:
                fail.append(f"README names main Fig. {n}, which the main text does not have")
        for n in sorted(set(re.findall(r"Eq\. \((\d+)\)", readme))):
            if n not in meq:
                fail.append(f"README names Eq. ({n}), which the main text does not have")

        # Stronger than the above: where the README names a float it also names
        # the LaTeX label, so the number can be checked against the label rather
        # than merely against the set of numbers that exist.  This is what
        # catches a README pointing at the right kind of float but the wrong one.
        bylabel = {}
        for aux_text in (maux, saux):
            for lab, n in re.findall(r"\\newlabel\{((?:fig|tab|eq):[^}]+)\}\{\{(S?\d+)\}", aux_text):
                bylabel.setdefault(lab, n)
        for kind, n, lab in re.findall(
                r"(Figs?\.|Table|Eq\.) \(?(S?\d+)\)?[^(\n]{0,40}\((?:fig|tab|eq):([^)]+)\)", readme):
            full = ("fig:" if kind.startswith("Fig") else
                    "tab:" if kind == "Table" else "eq:") + lab
            if full not in bylabel:
                fail.append(f"README names label {full}, which neither document defines")
            elif bylabel[full] != n:
                fail.append(f"README calls {full} '{n}' but LaTeX numbers it {bylabel[full]}")

    # --- a claim that rests on a boolean must be true ----------------------
    # T1E_NOKAPPA_SAFE records whether the loss-free mean-field contrast really
    # stays above 1/e over the whole window.  The main text asserts that it
    # does, so the build must fail if a rerun ever contradicts it.
    nums_for_flags = read("data", "numbers.json")
    if nums_for_flags:
        flags = json.loads(nums_for_flags)
        if flags.get("T1E_NOKAPPA_SAFE") is False:
            fail.append("the loss-free contrast now falls to 1/e inside the window; "
                        "the sentence about the non-monotonic 1/e time in main.tex "
                        "is no longer supported by data/benchmark.npz")

    # --- a relation must not be closed just before a token ------------------
    # "$x=$ [[TOK]]" renders as a broken space, and when the token itself is a
    # math group it renders as two adjacent groups.  Write "$x$ is [[TOK]]".
    for tex, where in ((main_tex, "main.tex"), (supp_tex, "supplement.tex")):
        for tok in sorted(set(re.findall(
                r"(?:=|\\approx|\\simeq|<|>|\\le|\\ge|\\lesssim|\\gtrsim)\$[\s~]*\[\[([A-Z0-9_]+)\]\]", tex))):
            fail.append(f"{where}: a relation is closed just before [[{tok}]]; "
                        "move the symbol out of the relation instead")

    # --- source whitespace: no trailing spaces, no tabs, no double spaces ---
    for tex, where in ((main_tex, "main.tex"), (supp_tex, "supplement.tex")):
        for i, line in enumerate(tex.split("\n"), 1):
            if line != line.rstrip():
                fail.append(f"{where}: trailing whitespace on line {i}")
            if "\t" in line:
                fail.append(f"{where}: tab character on line {i}")
            if re.search(r"(?<![.!?:}\)])  +(?=\S)", line[1:] if line[:1] == " " else line):
                fail.append(f"{where}: repeated space inside a sentence on line {i}")

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

    # --- LaTeX must not have left a reference or a citation unresolved -----
    # A mistyped label prints "??" in the PDF and is easy to miss in a long
    # document, so the build logs are read back and any complaint is a failure.
    for stem in ("main_filled", "supplement_filled"):
        # LaTeX logs are not UTF-8 (they carry the font encoding's own bytes),
        # so they are read with the undecodable bytes replaced.
        log_path = os.path.join(PAPER, stem + ".log")
        if not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log = fh.read()
        seen = set()
        for m in re.finditer(r"(?:Reference|Citation) `([^']+)' on page (\d+) "
                             r"undefined", log):
            key = (stem, m.group(1))
            if key in seen:
                continue
            seen.add(key)
            fail.append("%s.tex: `%s' undefined (page %s), so the PDF prints ??"
                        % (stem, m.group(1), m.group(2)))
        if re.search(r"There were undefined references", log) and not seen:
            fail.append("%s.tex: LaTeX reports undefined references" % stem)
        if re.search(r"^! ", log, re.M):
            fail.append("%s.tex: LaTeX reported an error" % stem)

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
