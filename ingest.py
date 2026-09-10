"""Parse the Warhammer 40k core rules PDF into one chunk per numbered rule."""

import re

import pymupdf

CORE_RULES = "data/eng_01-06_warhammer40k_new40k_core_rules-was6fbu1ix-hfewhmxyiy.pdf"

# A header line looks like: MORTAL WOUNDS 06.02
# Group 1 is the title, group 2 is the rule number, and the number must end the line.
header_pattern = r"([A-Z \[\]-]+) (\d{2}\.\d{2})$"


def clean(line):
    """Normalise one line of extracted PDF text."""
    # the PDF uses a non-breaking hyphen that won't match a normal one
    line = line.replace("\u2011", "-")

    # stray control characters that pymupdf hands back as bullet glyphs
    line = line.replace("\x07", "")
    line = line.replace("\x08", "")
    line = line.replace("\x16", "")

    return line.strip()


def parse_number(number):
    """'06.02' -> (6, 2), so the parts can be compared as numbers."""
    parts = number.split(".")
    return (int(parts[0]), int(parts[1]))


def is_next(last, candidate):
    """True if `candidate` is the header that legally follows `last`."""
    last_section, last_sub = last
    section, sub = candidate

    # same section, next sub-number: 06.02 -> 06.03
    if section == last_section and sub == last_sub + 1:
        return True

    # next section, restarting at .01: 06.09 -> 07.01
    if section == last_section + 1 and sub == 1:
        return True

    return False


def parse_pdf(path=CORE_RULES):
    """Return (chunks, rejected).

    chunks is a list of dicts with keys: number, title, page, text.
    rejected holds the (page, line) pairs that looked like headers but broke
    the numbering sequence, which is how cross-references get filtered out.
    """
    doc = pymupdf.open(path)

    chunks = []
    current = None  # the chunk being filled, None until the first header
    last = (0, 0)  # number of the last ACCEPTED header
    rejected = []

    for page_number in range(len(doc)):
        text = doc[page_number].get_text()

        for line in text.split("\n"):
            line = clean(line)
            if not line:
                continue

            m = re.search(header_pattern, line)
            if m:
                candidate = parse_number(m.group(2))
                if is_next(last, candidate):
                    # the chunk we were filling is finished now
                    if current is not None:
                        chunks.append(current)

                    # start a fresh chunk for the header we just hit
                    current = {
                        "number": m.group(2),
                        "title": m.group(1).strip(),
                        "page": page_number,
                        "text": "",
                    }
                    last = candidate
                    continue  # the header line itself is not body text
                else:
                    # a cross-reference, not a real header: keep it as body text
                    rejected.append((page_number, line))

            if current is not None:
                current["text"] += line + "\n"

    # the loops ended while still filling the last chunk
    if current is not None:
        chunks.append(current)

    return chunks, rejected


if __name__ == "__main__":
    chunks, rejected = parse_pdf()
    print(len(chunks), "chunks")
    print("range:", chunks[0]["number"], "to", chunks[-1]["number"])
    for page, line in rejected:
        print("  rejected on page", page, ":", line)
