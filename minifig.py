#!/usr/bin/env python3
"""minifig.py — photo -> LEGO minifigure parts recommendation (Pick a Brick links).

Four methods in agreement, cheapest first:
  1. Vision LLM (oMLX/Qwen): structured description of the person (age/hair/garments).
  2. Pixel color analysis (NO LLM): dominant garment color in the photo, matched to
     the nearest official LEGO color via CIE Lab distance.
  3. Element + image matching (NO LLM): for each candidate DESIGN we resolve the
     element VARIANT whose actual product image color is closest to the observed
     photo color (part *names* list every color and lie; rendered pixels don't).
     Candidates are then re-ranked by Lab distance and only the best 3 advance.
  4. Image-grounded verification loop (LLM): photo + numbered candidate images; the
     model must pick a number or abstain (0). Fallback = best image match.

Catalog: Rebrickable CSVs (https://rebrickable.com/downloads/) — run download_data.sh first.
Env: reuse ~/.venvs/brickify (Pillow, httpx, numpy) per python-env consolidation rule.

Usage:
  ~/.venvs/brickify/bin/python minifig.py PHOTO [-o OUTDIR] [--no-images] [--no-accessories]
"""
import argparse, base64, collections, csv, html, json, re, sys, time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OMLX_URL = 'http://127.0.0.1:8010/v1/chat/completions'
MODEL = 'Qwen3.8-Flash-Next-heretic-2-oQ4e-mtp'
CORE_SLOTS = ['hair', 'head', 'torso', 'legs']
EXTRA_SLOTS = ['headwear', 'held']

# ---------------------------------------------------------------- catalog ----
def load_csv(name):
    p = DATA / name
    if not p.exists():
        sys.exit(f'missing {p} — run {HERE}/download_data.sh first')
    with open(p, newline='', encoding='utf-8', errors='replace') as f:
        return list(csv.DictReader(f))

def build_catalog():
    parts = load_csv('parts.csv')
    names = {r['part_num']: r['name'] for r in parts}
    colors = {r['id']: {'name': r['name'], 'rgb': r.get('rgb', '')} for r in load_csv('colors.csv')}
    by_part = collections.defaultdict(list)
    for r in load_csv('elements.csv'):
        by_part[r['part_num']].append((r['color_id'], r['element_id']))

    hair, head, torso, legs = [], [], [], []
    plain_hues = {}  # pn -> set of hue families available among its elements
    for r in parts:
        pn, nm = r['part_num'], r['name']
        if nm.startswith('Hair') and 'Pattern' not in nm and 'Hair and' not in nm and 'Head' not in nm:
            hair.append(pn)
        elif nm.startswith('Minifig Head') and 'Print' in nm and not any(
                x in nm for x in ('Special', 'Alien', 'Droid', 'Robot', 'Fish', 'Wolf', 'Ogre',
                                   'Dinosaur', 'Shark', 'Skeleton', 'Zombie', 'Vampire', 'Gorilla',
                                   'Panda', 'Palpatine', 'Garazeb', 'Shrek', 'Zeb', 'Tongue')) \
                and not re.match(r'^Minifig Head \d', nm):
            head.append(pn)
        elif nm in ('Torso [Plain]',) or (nm.startswith('Torso') and 'Print' in nm and not any(
                x in nm for x in ('Mechanical', 'Wings', 'Droid', 'Robot', 'Bionicle',
                                   'Exo-Force', 'Baby', 'Minidoll')) and 'Dress' not in nm[:30]):
            torso.append(pn)  # includes plain torso — the "orange something" fallback
        elif nm.startswith('Legs') and 'Pattern' in nm and not any(
                x in nm for x in ('Mechanical', 'Martian', 'Enderman', 'Skeleton')):
            legs.append(pn)
        elif re.match(r'^Hips (and|with) .*Legs', nm) and not any(
                x in nm for x in ('Slope', 'Droid', 'Mechanical', 'Dress', 'Robe', 'Creature',
                                   'Minidoll', 'Belville', 'Duplo', 'Odd', 'Claws', 'Toes')):
            legs.append(pn)  # prints AND plain combos like 'Hips and Black Legs'

    headwear, held = [], []
    for r in parts:
        pn, nm = r['part_num'], r['name']
        if re.match(r'^(Cap[ ,]|Hat |Helmet |Beret|Crown |Bonnet)', nm) and 'Hair and' not in nm \
                and 'Duplo' not in nm and 'Creature' not in nm and 'Minidoll' not in nm:
            headwear.append(pn)
        if (nm.startswith('Equipment') or nm.startswith('Sports') or nm.startswith('Minifig Accessory')
                or nm.startswith('Tool')) and 'Print' not in nm and 'Duplo' not in nm \
                and 'Harry Potter' not in nm and 'Quidditch' not in nm and 'Trophy' not in nm:
            held.append(pn)
    return {'hair': hair, 'head': head, 'torso': torso, 'legs': legs,
            'headwear': headwear, 'held': held}, names, colors, dict(by_part)

# ---------------------------------------------------------------- omlx api ---
def _omlx_key():
    found = ''
    def fk(o):
        nonlocal found
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'api_key' and isinstance(v, str) and len(v) > 10:
                    found = v
                else:
                    fk(v)
    try:
        fk(json.loads((Path.home() / '.omlx' / 'settings.json').read_text()))
    except Exception:
        pass
    return found

def omlx(messages, max_tokens=900):
    import httpx
    body = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}}
    key = _omlx_key()
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    r = httpx.post(OMLX_URL, json=body, timeout=300, headers=headers)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

def image_data_url(path, size=768):
    from PIL import Image
    import io
    im = Image.open(path).convert('RGB')
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

# --------------------------------------- pixel analysis (no LLM anywhere) -----
def srgb_to_lab(rgb):
    def f(u):
        return ((u + 0.055) / 1.055) ** 2.4 if u > 0.04045 else u / 12.92
    r, g, b = [f(v) for v in rgb]
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    def g2(t):
        return max(t, 1e-9) ** (1 / 3)
    fx, fy, fz = g2(x), g2(y), g2(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

def lab_dist(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(srgb_to_lab(c1), srgb_to_lab(c2))) ** 0.5

def nearest_lego_color(rgb01, colors):
    lab = srgb_to_lab(rgb01)
    best, bd = None, 1e9
    for cid, c in colors.items():
        nm = c['name']
        if nm == '[Unknown]' or 'Trans' in nm or 'Glitter' in nm or not c['rgb']:
            continue
        try:
            rr, gg, bb = int(c['rgb'][0:2], 16), int(c['rgb'][2:4], 16), int(c['rgb'][4:6], 16)
        except ValueError:
            continue
        d = lab_dist(lab, srgb_to_lab((rr / 255, gg / 255, bb / 255)))
        if d < bd:
            bd, best = d, (cid, nm)
    return best

def region_color(path, box):
    """Dominant color of a relative region: cluster saturated pixels, pick the
    brightest well-populated cluster so shadows don't pull toward brown/rust."""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert('RGB')
    w, h = im.size
    crop = np.asarray(im.crop((int(w * box[0]), int(h * box[1]),
                               max(int(w * box[2]), 2), max(int(h * box[3]), 2))),
                    dtype=float) / 255.0
    mx = crop.max(axis=2)
    mn = crop.min(axis=2)
    if (mx < 0.24).mean() > 0.5:
        return (0.03, 0.03, 0.03)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    mask = (sat > 0.25) & (mx > 0.15)
    px = crop.reshape(-1, 3)[mask.reshape(-1)]
    if len(px) < 60:
        px = crop.reshape(-1, 3)
    keys = (px * 7).astype(int)
    keys = keys[:, 0] * 64 + keys[:, 1] * 8 + keys[:, 2]
    cnt = collections.Counter(keys.tolist())
    best, bl = None, -1.0
    for k, n in cnt.most_common(8):
        if n < 40:
            continue
        m = px[keys == k].mean(axis=0)
        score = float(m.sum()) + n / 10000.0
        if score > bl:
            bl, best = score, m
    return tuple(best) if best is not None else tuple(px.mean(axis=0))

def region_hue_coverage(path, box, family):
    """Fraction of non-white, sufficiently-saturated pixels in the region whose
    hue falls in the target family (vectorized HSV, no LLM, no per-pixel search)."""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert('RGB')
    w, h = im.size
    a = np.asarray(im.crop((int(w * box[0]), int(h * box[1]),
                            max(int(w * box[2]), 2), max(int(h * box[3]), 2))),
                  dtype=float) / 255.0
    flat = a.reshape(-1, 3)
    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    sel = (sat > 0.30) & (mx > 0.18)
    px = flat[sel]
    if len(px) < 50:
        return 0.0
    import colorsys
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px[:: max(1, len(px) // 400)]])
    hue = hsv[:, 0] * 360.0
    val = hsv[:, 2]
    s2 = hsv[:, 1]
    if family == 'black':
        rng = val < 0.28
    elif family == 'white':
        rng = (val > 0.85) & (s2 < 0.15)
    elif family == 'gray':
        rng = (val >= 0.28) & (val <= 0.75) & (s2 < 0.2)
    elif family == 'brown':
        rng = ((hue >= 10) & (hue < 45) & (val >= 0.12) & (val <= 0.55)) | \
              ((hue >= 0) & (hue < 10) & (val >= 0.15) & (val <= 0.6) & (s2 > 0.3))
    else:
        rng = {
            'red': ((hue < 12) | (hue >= 345)),
            'orange': ((hue >= 12) & (hue < 42)),
            'yellow': ((hue >= 42) & (hue < 70)),
            'green': ((hue >= 70) & (hue < 170)),
            'blue': ((hue >= 170) & (hue < 260)),
            'purple': ((hue >= 260) & (hue < 290)),
            'magenta': ((hue >= 290) & (hue < 345)),
        }.get(family)
    if rng is None:
        return 0.0
    return float(rng.mean())

HUE_WORDS = ('orange', 'rust', 'red', 'brick', 'salmon', 'coral', 'maroon', 'blue', 'azure',
             'teal', 'turquoise', 'green', 'yellow', 'gold', 'lemon', 'cream', 'brown', 'tan',
             'nougat', 'black', 'white', 'ivory', 'gray', 'grey', 'silver', 'purple', 'violet',
             'magenta', 'pink', 'lilac')

def hue_family(name):
    s = (name or '').lower()
    for w in HUE_WORDS:
        if w in s:
            if w in ('rust',):
                return 'orange'
            if w in ('brick', 'salmon', 'coral', 'maroon'):
                return 'red'
            if w in ('azure', 'teal', 'turquoise'):
                return 'blue'
            if w in ('gold', 'lemon', 'cream'):
                return 'yellow'
            if w in ('tan', 'nougat'):
                return 'brown'
            if w in ('ivory',):
                return 'white'
            if w in ('grey', 'silver'):
                return 'gray'
            if w in ('violet', 'magenta', 'pink', 'lilac'):
                return 'purple'
            return w
    return None

BODY_BOXES = {'torso': (0.30, 0.42, 0.70, 0.72), 'legs': (0.34, 0.74, 0.66, 0.86)}

def observed_colors(path, colors):
    out = {}
    for slot, box in BODY_BOXES.items():
        try:
            rgb = region_color(path, box)
            out[slot] = {'rgb': rgb, 'lego': nearest_lego_color(rgb, colors)}
        except Exception as e:
            print(f'  pixel analysis failed for {slot}: {e}')
    return out

# ------------------------------------------------------------- stage 1: see --
DESCRIBE_PROMPT = """Look at this photo of a real person. Answer ONLY with a JSON object, no prose:
{"age": "child/young adult/middle-aged/elderly", "sex": "male/female",
 "skin_tone": "one of: white, light tan, tan, brown, dark brown",
 "hair_color": "black/dark brown/medium brown/light brown/reddish brown/auburn/gray/white/blond/bald",
 "hair_style": "short/short curly/medium bob/long straight/long curly/ponytail/braided/pulled back messy/bald",
 "glasses": true, "facial_hair": "none/moustache/beard/beard and moustache/goatee/stubble",
 "top_description": "brief description of shirt/jacket incl the dominant color",
 "bottom_description": "brief description of pants/shorts incl color",
 "headwear": "none, or the exact headwear incl color",
 "held_item": "none unless they are CLEARLY gripping an object; jewelry held up for display still counts as held if gripped",
 "notable": "any other distinguishing feature or accessory"}"""

def describe_person(img_path):
    txt = omlx([{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_data_url(img_path)}},
        {'type': 'text', 'text': DESCRIBE_PROMPT}]}])
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group(0))

# ------------------------------------------------------ ranking (no LLM) ----
STOP = {'with', 'and', 'the', 'of', 'on', 'in', 'a', 'an', 'to', 'for', 'print'}

def tokens(text):
    return set(w for w in re.findall(r'[a-z]+', text.lower()) if w not in STOP)

def first_hue_index(text):
    """Position of the first hue-family word in a part name (dominance proxy)."""
    best = None
    low = text.lower()
    for w in HUE_WORDS:
        i = low.find(w)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, hue_family(w))
    return best  # (pos, family) or None

def build_queries(desc):
    q = {}
    if desc.get('hair_color') == 'bald':
        q['hair'] = 'bald hair'
    else:
        q['hair'] = f"{desc.get('hair_color','')} {desc.get('hair_style','')} hair"
    face_bits = []
    if desc.get('age') == 'child':
        face_bits.append('boy' if desc.get('sex') != 'female' else 'girl')
    if desc.get('glasses'):
        face_bits.append('glasses')
    fh = desc.get('facial_hair', 'none')
    if fh and fh != 'none':
        face_bits.append(fh.replace(' and ', ' '))
    if desc.get('age') == 'elderly':
        face_bits.append('old man')
    face_bits.append('smile')
    q['head'] = 'Minifig Head ' + ' '.join(face_bits) + ' Print'
    q['torso'] = f"Torso {desc.get('top_description','')} Print"
    bd = (desc.get('bottom_description') or '').lower()
    if bd and bd != 'not visible':
        q['legs'] = f"Hips Legs {bd.split(',')[0]} {'Shorts' if 'short' in bd else 'Pants'} Print"
    else:
        q['legs'] = 'Hips Legs Blue Jeans Print'
    hw = desc.get('headwear', 'none')
    if hw and hw != 'none':
        q['headwear'] = f"Cap Hat {hw}"
    held = desc.get('held_item', 'none')
    if held and held != 'none':
        q['held'] = f"Equipment {held}"
    return q

def target_hue(slot, desc, pixel):
    """Hue family the part's main plastic MUST be, per the best available evidence.
    Heads excluded: LEGO heads are canonically Yellow regardless of photo skin tone."""
    if slot == 'head':
        return None
    if slot == 'torso' and pixel.get('torso'):
        return hue_family(pixel['torso']['lego'][1])
    if slot == 'legs':
        bd = (desc.get('bottom_description') or '').lower()
        fam2 = hue_family(bd)
        fam = hue_family(pixel['legs']['lego'][1]) if pixel.get('legs') else None
        return fam2 or fam
    if slot == 'hair' and desc.get('hair_color') and desc['hair_color'] != 'bald':
        return hue_family(desc['hair_color'])
    return None

SKIN_IDS = {'Nougat': None}

def pick_element(pn, by_part, colors, want_fam, cache_dir=None):
    """Choose the element variant whose plastic color matches the target hue.
    When cache_dir is given, prefer a variant whose product image actually
    exists on the CDN (proves the element exists and lets us show a real
    picture + working link). Falls back to the first variant if nothing matches."""
    plist = by_part.get(pn, [])
    if not plist:
        return '', ''
    if want_fam and want_fam in ('nougat', 'tan', 'dark nougat'):
        for cid, eid in plist:
            nm = colors.get(cid, {}).get('name', '').lower()
            if nm == want_fam or (want_fam == 'nougat' and nm in ('light nougat', 'medium nougat')):
                return cid, eid
        for cid, eid in plist:
            nm = colors.get(cid, {}).get('name', '').lower()
            if 'nougat' in nm or nm in ('tan', 'fair', 'yellow'):
                return cid, eid
    elif want_fam:
        cands = [(cid, eid) for cid, eid in plist
                 if hue_family(colors.get(cid, {}).get('name', '')) == want_fam]
        if cands:
            if cache_dir is not None:
                for cid, eid in cands:
                    if fetch_element_image(eid, cache_dir):
                        return cid, eid
            return cands[0]
    return plist[0]

def retrieve(catalog, names, desc, pixel, by_part=None, colors=None, topk=10):
    out = {}
    for slot, qtext in build_queries(desc).items():
        if slot not in catalog:
            continue
        qt = tokens(qtext)
        want = target_hue(slot, desc, pixel) if slot != 'head' else None
        scored = []
        for pn in catalog[slot]:
            nm = names[pn]
            nt = tokens(nm)
            sc = len(nt & qt)
            if want:
                # the plastic itself: does ANY element ship in the target hue?
                elem_fam = False
                if by_part is not None and colors is not None:
                    for cid, _ in by_part.get(pn, []):
                        if hue_family(colors.get(cid, {}).get('name', '')) == want:
                            elem_fam = True
                            break
                plain = 'Print' not in nm and 'Pattern' not in nm
                if elem_fam:
                    sc += 14 if plain else 4
                    if not plain:
                        pos = first_hue_index(nm)
                        if pos and pos[1] == want:
                            sc += 4
                else:
                    sc -= 6
            if slot == 'legs':
                bd = qtext.lower()
                if 'short' in bd:
                    if re.search(r'(Nougat|Fair|Warm Tan|Tan) Legs', nm):
                        sc += 6
                    elif re.search(r'(Magenta|Bright Green|Purple|Neon|Pyjama|Pajama)', nm):
                        sc -= 10
                    if re.search(r'Skirt|Dress|Robe|Leotard', nm):
                        sc -= 12
                    if 'Boots' in nm and 'Shorts' not in nm:
                        sc -= 2
                    if re.match(r'^Hips and (Black|Dark Bluish Gray|Blue|Tan|Dark Tan|Brown|Nougat) Legs$', nm):
                        sc += 10
            if slot == 'head' and desc.get('skin_tone') in (None, 'white', 'light tan'):
                if any(x in nm for x in ('Jungle', 'Black ', 'Dark Nougat', 'Alien')):
                    sc -= 8
            if slot == 'head' and desc.get('facial_hair') in (None, 'none'):
                if re.search(r'(Beard|Moustache|Mustache|Stubble)', nm):
                    sc -= 6
            scored.append((sc, pn))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if slot in ('headwear', 'held') and (not scored or scored[0][0] < 4):
            continue
        out[slot] = [pn for _, pn in scored[:topk]]
    return out

# --------------------------- element/image matching (no LLM) ------------------
def fetch_element_image(eid, outdir):
    if not eid:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f'{eid}.jpg'
    if dst.exists() and dst.stat().st_size > 100:
        return dst
    url = f'https://cdn.rebrickable.com/media/parts/elements/{eid}.jpg'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as r:
            dst.write_bytes(r.read())
        return dst
    except Exception as e:
        print(f'  image fetch failed {eid}: {e}')
        return None

def part_image_color(img_path):
    return region_color(img_path, (0.12, 0.12, 0.88, 0.88))

# ----------------------------------------------------- verify+pick (LLM) -----
def compose_montage(cands, imgs, path, cell=230, pad=18):
    from PIL import Image, ImageDraw
    W = pad * (len(cands) + 1) + cell * len(cands)
    im = Image.new('RGB', (W, cell + 40), 'white')
    d = ImageDraw.Draw(im)
    for i, pn in enumerate(cands):
        if pn not in imgs:
            continue
        src = Image.open(imgs[pn]).convert('RGB')
        src.thumbnail((cell - 8, cell - 8))
        x = pad + i * (cell + pad) + 4
        im.paste(src, (x, 26))
        d.text((x + 4, 6), f'OPTION {i + 1}', fill='black')
    im.save(path)

def verify_slot(photo, slot, cands, imgs, desc):
    with_img = [p for p in cands if p in imgs]
    if not with_img:
        return None
    mpath = str(Path(photo).parent / f'_montage_{slot}.png')
    try:
        compose_montage(cands, imgs, mpath)
    except Exception:
        return None
    hint = ''
    if slot == 'head':
        hint = f"\nThe person is a {desc.get('age','')} {desc.get('sex','')} with skin tone {desc.get('skin_tone','')}."
    elif slot == 'torso':
        hint = f"\nTarget garment: {desc.get('top_description')}"
    elif slot == 'legs':
        hint = f"\nTarget garment: {desc.get('bottom_description')}"
    prompt = (
        f'IMAGE 1 is a photo of a real person. IMAGE 2 shows numbered LEGO {slot} options (OPTION 1..{len(cands)}).{hint}\n'
        f"These are the top {len(cands)} matches already ranked by color similarity. "
        "Pick the option whose " + slot + " most resembles this person's " + slot + ". "
        "Always pick the BEST of the options; only answer 0 if every option is clearly wrong "
        "(e.g. a dinosaur print for a normal child).\n"
        'Answer ONLY JSON: {"pick": <number>}')
    try:
        txt = omlx([{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': image_data_url(photo)}},
            {'type': 'image_url', 'image_url': {'url': image_data_url(mpath)}},
            {'type': 'text', 'text': prompt}]}], max_tokens=120)
        n = None
        m = re.search(r'"pick"\s*:\s*(\d+)', txt)
        if m:
            n = int(m.group(1))
        else:
            m2 = re.search(r'OPTION\s+(\d+)', txt, re.I)
            if m2:
                n = int(m2.group(1))
            else:
                m3 = re.search(r'\b(\d+)\b', txt)
                if m3:
                    n = int(m3.group(1))
        if n is not None and 1 <= n <= len(cands) and cands[n - 1] in imgs:
            return cands[n - 1]
        print(f'  verify {slot}: model answered {txt[:80]!r}')
    except Exception as e:
        print(f'  verify failed for {slot}: {e}')
    return None

# -------------------------------------------------------------- compose ------
def compose_sheet(photo, slots, result, outdir):
    from PIL import Image, ImageDraw
    cell, pad = 240, 30
    imgs = result.get('_images', {})
    shown = [s for s in slots if imgs.get(s)]
    n = len(shown) + 1
    W = pad * (n + 1) + cell * n
    sheet = Image.new('RGB', (W, cell + 60), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), 'Minifigure match', fill='black')
    src = Image.open(photo).convert('RGB')
    src.thumbnail((cell, cell))
    sheet.paste(src, (pad, 40))
    d.text((pad + 4, cell + 46), 'PHOTO', fill='black')
    for i, slot in enumerate(shown):
        im = Image.open(imgs[slot]).convert('RGB')
        im.thumbnail((cell, cell))
        x = pad + (i + 1) * (cell + pad)
        sheet.paste(im, (x, 40))
        d.text((x + 4, cell + 46), slot.upper(), fill='black')
    sheet.save(outdir / 'sheet.png')

def compose_html(photo, slots, result, outdir):
    def esc(s):
        return html.escape(str(s))
    rows = []
    for slot in slots:
        r = result['picks'].get(slot)
        if not r:
            rows.append(f'<div class="card none"><h3>{esc(slot)}</h3><p>NONE</p></div>')
            continue
        img = ''
        ip = outdir / f"{r['element_id']}.jpg"
        if r['element_id'] and ip.exists():
            b64 = base64.b64encode(ip.read_bytes()).decode()
            img = f'<img src="data:image/jpeg;base64,{b64}" alt="{esc(r["name"])}">'
        rows.append(f'''<div class="card">
          <h3>{esc(slot)}</h3>
          <div class="img">{img}</div>
          <p class="name">{esc(r['name'])}</p>
          <p class="meta">part <code>{esc(r['part_num'])}</code> · element <code>{esc(r['element_id'])}</code> · {esc(r['color'])}</p>
          <p><a href="{esc(r['pick_a_brick'])}" target="_blank">Buy on LEGO.com Pick a Brick →</a></p>
          <p><a href="https://rebrickable.com/parts/{esc(r['part_num'])}/" target="_blank">View on Rebrickable</a></p>
        </div>''')
    photo_b64 = base64.b64encode(Path(photo).read_bytes()).decode()
    doc = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Minifigure recommendation</title>
<style>
 body {{ font-family: -apple-system, sans-serif; background:#f5f5f5; margin:0; padding:24px; color:#222; }}
 h1 {{ font-size:20px; }}
 .strip {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }}
 .photo {{ background:#fff; border-radius:12px; padding:12px; box-shadow:0 1px 4px rgba(0,0,0,.15); }}
 .photo img {{ max-width:360px; border-radius:8px; }}
 .card {{ background:#fff; border-radius:12px; padding:16px; width:280px; box-shadow:0 1px 4px rgba(0,0,0,.15); }}
 .card h3 {{ margin:0 0 8px; text-transform:capitalize; }}
 .card .img img {{ width:100%; border-radius:8px; background:#fafafa; }}
 .card .name {{ font-size:13px; font-weight:600; }}
 .card .meta {{ font-size:12px; color:#666; }}
 .none {{ opacity:.6; }}
 .reason {{ margin-top:16px; padding:12px 16px; background:#fff; border-radius:12px; font-size:13px; }}
</style></head><body>
<h1>Minifigure recommendation</h1>
<div class="strip">
  <div class="photo"><h3>photo</h3><img src="data:image/jpeg;base64,{photo_b64}"></div>
  {''.join(rows)}
</div>
<div class="reason"><b>pixel color analysis (no LLM):</b> {esc(json.dumps(result.get('pixel_colors', {})))}</div>
<div class="reason"><b>verification:</b> {esc(json.dumps(result.get('verification', {})))}</div>
</body></html>'''
    (outdir / 'report.html').write_text(doc)

# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--no-images', action='store_true')
    ap.add_argument('--no-accessories', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    outdir = Path(a.out) if a.out else HERE / 'out' / Path(a.photo).stem
    outdir.mkdir(parents=True, exist_ok=True)

    catalog, names, colors, by_part = build_catalog()
    print('catalog: ' + ' '.join(f'{k}={len(v)}' for k, v in catalog.items()))

    print('stage 1: describing photo (vision LLM)...')
    desc = describe_person(a.photo)
    print(json.dumps(desc, indent=2))

    print('stage 2: pixel color analysis (no LLM)...')
    pixel = observed_colors(a.photo, colors)
    for k, v in pixel.items():
        print(f'  {k}: rgb={[round(float(c),2) for c in v["rgb"]]} -> LEGO "{v["lego"][1]}" '
              f'(family {hue_family(v["lego"][1])})')

    print('stage 3: retrieving candidates (name match)...')
    shortlist = retrieve(catalog, names, desc, pixel, by_part=by_part, colors=colors)
    if a.no_accessories:
        for s in EXTRA_SLOTS:
            shortlist.pop(s, None)
    slots = CORE_SLOTS + [s for s in EXTRA_SLOTS if s in shortlist]

    # resolve elements (hue-matched variant) + download images + coverage/Lab re-rank
    print('stage 4: element + image-color matching (no LLM)...')
    imgs, final_cands, deltas = {}, {}, {}
    PART_BOX = (0.12, 0.12, 0.88, 0.88)
    for s in slots:
        want = target_hue(s, desc, pixel)
        cands = shortlist[s]
        scored = []
        for pn in cands:
            cid, eid = pick_element(pn, by_part, colors, want, cache_dir=HERE / 'out' / '_cache')
            if not eid:
                continue  # no element in the required plastic color — skip design
            ip = fetch_element_image(eid, HERE / 'out' / '_cache')
            if ip:
                imgs[pn] = str(ip)
            d = 9999
            cov = 1.0  # unknown/na => assume fine, preserve retrieval order
            if ip and want and s in ('hair', 'torso', 'legs') and pixel.get(s):
                try:
                    pic = part_image_color(ip)
                    d = lab_dist(pixel[s]['rgb'], pic)
                    if hue_family(nearest_lego_color(pic, colors)[1]) != hue_family(pixel[s]['lego'][1]):
                        d += 40
                    cov = region_hue_coverage(ip, PART_BOX, want)
                except Exception:
                    pass
            scored.append((cov, d, pn))
        # keep only candidates whose actual part image really shows the target hue
        good = [t for t in scored if t[0] >= 0.30]
        if not good:
            good = sorted(scored, key=lambda t: (t[1], t[2]))
        good.sort(key=lambda t: (-t[0], t[1]))
        final_cands[s] = [pn for _, _, pn in good[:3]]
        for cov, d, pn in scored:
            deltas[pn] = d
        for pn in final_cands[s]:
            cid, eid = pick_element(pn, by_part, colors, want, cache_dir=HERE / 'out' / '_cache')
            print(f'  {s}: {eid} {names[pn][:55]} [{colors.get(cid,{}).get("name","?")}]')

    print('stage 5: image-grounded verification (LLM)...')
    picks, verification = {}, {}
    for s in slots:
        v = verify_slot(a.photo, s, final_cands[s], imgs, desc)
        verification[s] = f'verified pick: {v}' if v else 'abstained -> best image match'
        picks[s] = v if v else (final_cands[s][0] if final_cands[s] else None)
        print(f'  {s}: {picks[s] or "NONE"} — {names.get(picks[s], "")[:60]}')

    result = {'description': desc,
              'pixel_colors': {k: v['lego'][1] for k, v in pixel.items()},
              'verification': verification, 'picks': {}, 'shortlist': final_cands}
    img_paths = {}
    for slot in slots:
        pn = picks.get(slot)
        if not pn:
            continue
        want = target_hue(slot, desc, pixel) if slot != 'head' else None
        cid, eid = pick_element(pn, by_part, colors, want, cache_dir=HERE / 'out' / '_cache')
        url = (f'https://www.lego.com/en-us/pick-and-build/pick-a-brick?query={eid}' if eid
               else 'https://www.lego.com/en-us/pick-and-build/pick-a-brick?query='
                    + urllib.parse.quote(names.get(pn, pn).split(' [')[0]))
        result['picks'][slot] = {'part_num': pn, 'name': names.get(pn, '?'),
                                  'element_id': eid, 'color': colors.get(cid, {}).get('name', ''),
                                  'pick_a_brick': url}
        if not a.no_images:
            fetch_element_image(eid, outdir)
            ip = outdir / f'{eid}.jpg'
            if ip.exists():
                img_paths[slot] = str(ip)
    (outdir / 'recommendation.json').write_text(json.dumps(result, indent=2))
    result['_images'] = img_paths
    compose_sheet(a.photo, slots, result, outdir)
    compose_html(a.photo, slots, result, outdir)
    print(f'\ndone in {time.time()-t0:.0f}s -> {outdir}/report.html')

if __name__ == '__main__':
    main()
