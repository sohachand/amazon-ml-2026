"""Text normalisation for business names and addresses.

Everything is deterministic and uses only the input strings (no external lookup).
Key idea: transliterate every script to Latin (unidecode), then reduce each token to a
"consonant skeleton" that is robust to vowel typos, doubled letters and the typical
artefacts of Indic -> Latin transliteration (ph/f, d/t, c/k/s, ...).
"""
import re
from unidecode import unidecode

_non_alnum = re.compile(r"[^a-z0-9 ]+")
_ws = re.compile(r"\s+")
_digits = re.compile(r"\d+")
_url = re.compile(r"(https?://)?(www\.)?([a-z0-9\-]+)\.(com|net|org|in|co|fr|biz|info|us|io)\b")

# legal / generic tokens (after abbreviation expansion) removed to build the "core" name
LEGAL = {
    "private", "limited", "llp", "llc", "incorporated", "corporation", "company", "pc", "pllc",
    "lp", "plc", "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "scop", "gie", "the", "and",
    "of", "dba", "public", "opc", "ltd", "pvt", "inc", "corp", "co", "l", "p", "s", "a", "c", "et",
    "fils", "freres", "smt", "shri", "shree", "sri", "m", "s", "de", "du", "des", "la", "le", "les",
    "group", "groupe", "elelpi", "elelsi", "praaivett", "limittedd", "piraiveett", "limittett",
}
ABBR = {
    "pvt": "private", "prvt": "private", "pte": "private", "ltd": "limited", "lmt": "limited",
    "ltdd": "limited", "inc": "incorporated", "corp": "corporation", "co": "company",
    "cos": "company", "intl": "international", "mfg": "manufacturing", "bros": "brothers",
    "assoc": "associates", "svcs": "services", "svc": "services", "mgmt": "management",
    "tech": "technologies", "&": "and", "n": "and",
}
ADDR_ABBR = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "lane": "ln", "boulevard": "blvd", "bd": "blvd", "court": "ct", "place": "pl",
    "terrace": "ter", "circle": "cir", "highway": "hwy", "parkway": "pkwy", "trail": "trl",
    "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w", "suite": "ste",
    "apartment": "apt", "floor": "fl", "building": "bldg", "number": "no", "nagar": "ngr",
    "sector": "sec", "near": "nr", "opposite": "opp", "rue": "r", "allee": "all", "chemin": "ch",
    "route": "rte", "impasse": "imp", "saint": "st", "sainte": "ste", "city": "",
    "township": "", "townshiip": "", "null": "", "po": "", "box": "", "unit": "", "fl": "",
    "no": "", "h": "", "d": "", "door": "", "plot": "", "flat": "", "shop": "", "house": "",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
_us_state_re = re.compile(r"\b(" + "|".join(sorted(US_STATES, key=len, reverse=True)) + r")\b")


def to_latin(s: str) -> str:
    if not s:
        return ""
    s = unidecode(s).lower()
    return s


def clean(s: str) -> str:
    s = s.replace("&", " and ").replace("+", " and ")
    s = _non_alnum.sub(" ", s)
    return _ws.sub(" ", s).strip()


_sk_sub = [
    ("ph", "f"), ("th", "t"), ("sh", "s"), ("ch", "k"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
    ("dh", "d"), ("ck", "k"), ("x", "ks"), ("q", "k"), ("w", "v"), ("z", "s"), ("j", "s"),
]
_c_soft = re.compile(r"c(?=[eiy])")
_vowel = re.compile(r"[aeiouy]")
_rep = re.compile(r"(.)\1+")
_tr = str.maketrans({"d": "t", "b": "p", "g": "k", "c": "k"})


def skel(tok: str) -> str:
    """Consonant skeleton of a latin lowercase token. Digits are kept verbatim."""
    if tok.isdigit():
        return tok.lstrip("0") or "0"
    t = _c_soft.sub("s", tok)
    for a, b in _sk_sub:
        t = t.replace(a, b)
    first = t[0]
    t = _vowel.sub("", t).translate(_tr)
    t = _rep.sub(r"\1", t)
    if not t:  # all-vowel token
        t = first
    return t


def norm_name(raw: str):
    """Return (latin_clean_name, core_skeleton_tokens list, full_skeleton_tokens list, domain_flag)."""
    s = to_latin(raw)
    dom = 0
    m = _url.search(s)
    if m:
        dom = 1
        s = _url.sub(lambda mm: " " + mm.group(3) + " ", s)
    s = s.replace("d.b.a.", " dba ").replace("d/b/a", " dba ")
    s = clean(s)
    toks = [ABBR.get(t, t) for t in s.split()]
    lat = " ".join(toks)
    full = []
    core = []
    for t in toks:
        k = skel(t)
        full.append(k)
        if t not in LEGAL and len(t) > 1:
            core.append(k)
    # de-duplicate while keeping order
    core = list(dict.fromkeys(core))
    return lat, core, full, dom


INDIA_STATE_SK = None  # skeletons of Indian state names / codes (filled lazily)
_IN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "tamilnadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "jammu", "kashmir", "chandigarh", "puducherry",
    "pondicherry", "india", "usa", "france", "united states",
    # French regions
    "ile de france", "hauts de france", "nouvelle aquitaine", "occitanie", "bretagne",
    "normandie", "grand est", "pays de la loire", "centre val de loire", "bourgogne franche comte",
    "auvergne rhone alpes", "provence alpes cote d azur", "corse",
]


def _state_skels():
    """Latin tokens that only denote a state/region/country (dropped from address tokens)."""
    global INDIA_STATE_SK
    if INDIA_STATE_SK is None:
        st = set()
        for n in _IN_STATES:
            for t in n.split():
                if len(t) > 2:
                    st.add(t)
        st.update(US_STATES.values())
        st.update(["mhaaraassttr", "krnaattk", "gujraat", "tmilllnaattu", "uttr", "prdesh",
                   "dilli", "hriyaannaa", "raajsthaan", "telNgaanaa".lower(), "kerlm", "pNjaab".lower(),
                   "pshcim", "bNgaal".lower(), "bihaar", "mdhy", "oddishaa", "aaNdhr".lower(),
                   "tn", "ka", "mh", "dl", "up", "hr", "tg", "rj", "gj", "wb", "kl", "ap", "mp",
                   "pradesh", "nadu", "bengal"])
        INDIA_STATE_SK = st
    return INDIA_STATE_SK


def norm_addr(raw: str, country: str):
    """Return (latin_clean_addr, token skeleton list (no state/generic tokens), numbers list)."""
    s = to_latin(raw)
    if not s:
        return "", [], []
    s = clean(s)
    if country == "US":
        # only replace a trailing / leading state name (avoid 'Kansas City')
        s = _us_state_re.sub(lambda m: US_STATES[m.group(1)] if not s[m.end():].startswith(" city") else m.group(1), s)
    nums = [n.lstrip("0") or "0" for n in _digits.findall(s)]
    toks = []
    stsk = _state_skels()
    for t in s.split():
        t = ADDR_ABBR.get(t, t)
        if not t or t.isdigit():
            continue
        # tokens like 45th / 45nd / 1056c -> keep digit part only (already in nums)
        if t[0].isdigit():
            continue
        if len(t) < 2 or t in stsk:
            continue
        toks.append(skel(t))
    toks = list(dict.fromkeys(toks))
    return " ".join(ADDR_ABBR.get(t, t) for t in s.split()), toks, list(dict.fromkeys(nums))
