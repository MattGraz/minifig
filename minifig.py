#!/usr/bin/env python3
"""minifig.py — photo -> LEGO minifigure parts recommendation (Pick a Brick codes).

Replicates brickify.fun's core workflow locally: analyze a photo of a person with a
local vision LLM (oMLX), then recommend official LEGO minifigure parts (wig, head
face print, torso, legs) with element IDs and Pick a Brick links.

Catalog: Rebrickable CSVs (https://rebrickable.com/downloads/) — run download_data.sh first.
Env: reuse ~/.venvs/brickify (Pillow, httpx) per python-env consolidation rule.

Usage:
  ~/.venvs/brickify/bin/python minifig.py PHOTO [-o OUTDIR] [--no-images]
"""
import argparse, base64, csv, json, os, re, sys, time
import urllib.request
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OMLX_URL = 'http://127.0.0.1:8010/v1/chat/completions'
MODEL = 'Qwen3.8-Flash-Next-heretic-2-oQ4e-mtp'

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
    # element_id by (part_num, color_id)
    elem = {}
    for r in load_csv('elements.csv'):
        elem[(r['part_num'], r['color_id'])] = r['element_id']

    def ok(n, *bad):
        return not any(b.lower() in n.lower() for b in bad)

    hair, head, torso, legs = [], [], [], []
    for r in parts:
        pn, nm = r['part_num'], r['name']
        if nm.startswith('Hair') and 'Pattern' not in nm and 'Hair and' not in nm and 'Head' not in nm:
            hair.append(pn)
        elif nm.startswith('Minifig Head') and 'Print' in nm and not any(x in nm for x in ('Special', 'Alien', 'Droid', 'Robot', 'Fish', 'Wolf', 'Ogre', 'Dinosaur', 'Shark', 'Skeleton', 'Zombie', 'Vampire', 'Gorilla', 'Panda', 'Palpatine', 'Garazeb', 'Shrek', 'Zeb', 'Tongue')) and not re.match(r'^Minifig Head \d', nm):
            head.append(pn)
        elif nm.startswith('Torso') and 'Print' in nm and not any(x in nm for x in ('Mechanical', 'Wings', 'Droid', 'Robot', 'Bionicle', 'Exo-Force', 'Baby', 'Minidoll')) and 'Dress' not in nm[:30]:
            torso.append(pn)
        elif nm.startswith('Legs') and 'Pattern' in nm and not any(x in nm for x in ('Mechanical', 'Martian', 'Enderman', 'Skeleton')):
            legs.append(pn)
        elif re.match(r'^Hips (and|with) .*Legs.*Print', nm) and not any(x in nm for x in ('Slope', 'Droid', 'Mechanical', 'Dress', 'Robe')):
            legs.append(pn)
        elif 'Shorts' in nm and 'Print' in nm and 'Hips' in nm:
            legs.append(pn)
    return {'hair': hair, 'head': head, 'torso': torso, 'legs': legs}, names, colors, elem

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

def omlx(prompt_messages, max_tokens=900):
    import httpx
    body = {'model': MODEL, 'messages': prompt_messages,
            'max_tokens': max_tokens, 'temperature': 0.2,
            'chat_template_kwargs': {'enable_thinking': False}}
    key = _omlx_key()
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    r = httpx.post(OMLX_URL, json=body, timeout=300, headers=headers)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

def image_data_url(path):
    from PIL import Image
    import io
    im = Image.open(path).convert('RGB')
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85)
    return f'data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'

# ------------------------------------------------------------- stage 1: see --
DESCRIBE_PROMPT = """Look at this photo of a real person. Answer ONLY with a JSON object, no prose:
{"age": "child/young adult/middle-aged/elderly", "sex": "male/female",
 "skin_tone": "one of: white, light tan, tan, brown, dark brown",
 "hair_color": "black/dark brown/medium brown/light brown/reddish brown/auburn/gray/white/blond/bald",
 "hair_style": "short/short curly/medium bob/long straight/long curly/ponytail/braided/pulled back messy/bald",
 "glasses": true or false, "facial_hair": "none/moustache/beard/beard and moustache/goatee/stubble",
 "top_description": "brief description of shirt/jacket incl colors, e.g. blue denim shirt with collar",
 "bottom_description": "brief description of pants incl color",
 "headwear": "none or description",
 "notable": "any other distinguishing feature"}"""

def describe_person(img_path):
    txt = omlx([{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_data_url(img_path)}},
        {'type': 'text', 'text': DESCRIBE_PROMPT}]}])
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group(0))

# ----------------------------------------------------------- stage 2: rank ---
COLOR_SYNONYMS = {
    'black': ['black', 'dark'], 'dark brown': ['dark brown', 'brown'],
    'medium brown': ['brown', 'medium'], 'light brown': ['tan', 'light brown'],
    'reddish brown': ['reddish', 'red', 'auburn', 'orange'],
    'auburn': ['reddish', 'auburn'], 'gray': ['gray', 'grey', 'light gray'],
    'white': ['white'], 'blond': ['yellow', 'blond', 'light yellow'],
    'blue': ['blue'], 'denim': ['blue', 'denim'], 'red': ['red'],
    'green': ['green'], 'brown': ['brown', 'nougat', 'tan'],
    'grayish': ['gray', 'grey'], 'orange': ['orange'],
}

def tokens(text):
    return set(re.findall(r'[a-z]+', text.lower()))

def score_name(name, query_tokens, synonym_tokens):
    nt = tokens(name)
    s = len(nt & query_tokens)
    s += 2 * len(nt & synonym_tokens)
    return s

def build_queries(desc):
    """For each slot, (primary_text, alt_text) used to score catalog names."""
    q = {}
    hair_q = f"{desc.get('hair_color','')} {desc.get('hair_style','')} hair"
    if desc.get('hair_color') == 'bald':
        hair_q = 'bald hair'
    q['hair'] = hair_q
    face_bits = []
    if desc.get('age') == 'child':
        face_bits.append('boy' if desc.get('sex') != 'female' else 'girl')
    if desc.get('glasses'): face_bits.append('glasses')
    fh = desc.get('facial_hair', 'none')
    if fh and fh != 'none': face_bits.append(fh.replace(' and ', ' '))
    if desc.get('age') == 'elderly': face_bits.append('old man')
    face_bits.append('smile')
    q['head'] = 'Minifig Head ' + ' '.join(face_bits) + ' Print'
    q['torso'] = f"Torso {desc.get('top_description','')} Print"
    if desc.get('bottom_description'):
        bd = desc['bottom_description'].lower()
        shorts_hint = 'Shorts' if 'short' in bd else 'Pants'
        q['legs'] = f"Hips Legs {bd.split(',')[0]} {shorts_hint} Print"
    else:
        q['legs'] = "Hips Legs Pants Print"
    return q

def retrieve(catalog, names, desc, topk=6):
    out = {}
    for slot, qtext in build_queries(desc).items():
        qt = tokens(qtext)
        syn = set()
        for word in list(qt):
            syn.update(COLOR_SYNONYMS.get(word, []))
        scored = []
        for pn in catalog[slot]:
            nm = names[pn]
            sc = score_name(nm, qt, syn)
            # shorts on a real child = bare legs; skin-tone leg color is mandatory.
            # Deprioritize costume-colored legs (magenta/green/etc) and
            # boost "Hips and <Skin> ... Shorts" structures.
            if slot == 'legs' and 'short' in qtext.lower():
                if re.search(r'(Nougat|Fair|Warm Tan|Tan) Legs', nm):
                    sc += 8
                elif re.search(r'(Magenta|Bright Green|Dark Red|Purple|Orange|Neon) Legs', nm):
                    sc -= 6
                if 'Knee' in nm:
                    sc -= 2  # knee-pad prints are a costume, not plain shorts
            scored.append((sc, pn))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[slot] = [pn for _, pn in scored[:topk]]
    return out

# --------------------------------------------------------- stage 3: choose ---
def final_pick(img_path, desc, shortlist, names, colors, elem):
    lines = []
    for slot, pns in shortlist.items():
        lines.append(f'\n{slot.upper()} candidates:')
        for pn in pns:
            e = elem.get((pn, ''), elem.get(pn, ''))
            # find any element for this printed part
            eid = ''
            if slot in ('head', 'torso'):
                for (p, c), v in elem.items():
                    if p == pn:
                        eid = v; break
            colname = ''
            for (p, c), v in elem.items():
                if p == pn and c in colors:
                    colname = colors[c]['name']; break
            lines.append(f'  {pn}: {names[pn]} [element {eid}]')
    pick_prompt = (
        'You recommend official LEGO minifigure parts that best match the person in the photo.\n'
        f'Observed description: {json.dumps(desc)}\n'
        'Pick EXACTLY ONE part per slot from the candidate lists. '
        'Prefer the candidate whose printed design most resembles the person. '
        'IMPORTANT accuracy rules: if the person is COMPLETELY bald (clean scalp), output "NONE" for hair — '
        'do not pick a bald-top-with-hair-on-sides piece unless they truly have long hair on the sides; '
        'if they have only light stubble, pick a stubble or clean-shaven head, NOT a full-beard head; '
        'if they are a child, pick a child/neutral face, NOT an animal or alien print; '
        'for LEGS with shorts, the legs must be a SKIN color (nougat/tan/white), never magenta/green/costume colors; '
        'match garment colors as literally as possible. '
        'Answer ONLY JSON: {"hair": "<part_num or NONE>", "head": "<part_num>", "torso": "<part_num>", "legs": "<part_num>", "reasoning": "<one line>"}'
        + '\n'.join(lines))
    txt = omlx([{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_data_url(img_path)}},
        {'type': 'text', 'text': pick_prompt}]}], max_tokens=500)
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group(0))

# ------------------------------------------------------------ render sheet --
def fetch_element_image(eid, outdir):
    if not eid: return None
    url = f'https://cdn.rebrickable.com/media/parts/elements/{eid}.jpg'
    dst = outdir / f'{eid}.jpg'
    if dst.exists(): return dst
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as r, open(dst, 'wb') as f:
            f.write(r.read())
        return dst
    except Exception as e:
        print(f'  image fetch failed {eid}: {e}')
        return None

def compose_sheet(photo, picks, outdir):
    from PIL import Image, ImageDraw
    slots = ['hair', 'head', 'torso', 'legs']
    cell, pad = 240, 30
    W = cell * 2 + pad * 3
    sheet = Image.new('RGB', (W, cell + 60), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), 'Minifigure match', fill='black')
    src = Image.open(photo).convert('RGB')
    src.thumbnail((cell, cell))
    sheet.paste(src, (pad, 40))
    for i, slot in enumerate(slots):
        p = picks.get('_images', {}).get(slot)
        if p and Path(p).exists():
            im = Image.open(p).convert('RGB')
            im.thumbnail((cell, cell))
            x = pad + (i + 1) * (cell + pad)
            sheet.paste(im, (x, 40))
            d.text((x + 4, cell + 46), slot, fill='black')
    sheet.save(outdir / 'sheet.png')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--no-images', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    outdir = Path(a.out) if a.out else HERE / 'out' / Path(a.photo).stem
    outdir.mkdir(parents=True, exist_ok=True)

    catalog, names, colors, elem = build_catalog()
    print(f'catalog: hair={len(catalog["hair"])} head={len(catalog["head"])} torso={len(catalog["torso"])} legs={len(catalog["legs"])}')

    print('analyzing photo (vision)...')
    desc = describe_person(a.photo)
    print(json.dumps(desc, indent=2))

    print('retrieving candidates...')
    shortlist = retrieve(catalog, names, desc)
    # legs: if the photo doesn't show legs, default to classic blue — don't guess
    if desc.get('bottom_description') in (None, 'not visible', ''):
        jeans = [pn for pn in catalog['legs'] if re.search(r'jeans|denim', names[pn], re.I)]
        plain = [pn for pn in catalog['legs'] if 'Blue' in names[pn]]
        shortlist['legs'] = (jeans + plain)[:4]
    for s, pns in shortlist.items():
        print(f'  {s}: ' + ' | '.join(f'{p} ({names[p][:50]})' for p in pns[:3]))

    print('final pick (vision + candidates)...')
    picks = final_pick(a.photo, desc, shortlist, names, colors, elem)

    # resolve element ids + Pick a Brick URLs (hair needs color selection by name)
    result = {'description': desc, 'picks': {}, 'shortlist': shortlist}
    hair_color_match = None
    if desc.get('hair_color') and desc['hair_color'] != 'bald':
        # find nearest LEGO color name for hair
        want = tokens(desc['hair_color'])
        best, bs = None, -1
        for cid, c in colors.items():
            ct = tokens(c['name'])
            s = len(want & ct)
            if s > bs and c['name'] not in ('[Unknown]',):
                bs, best = s, cid
        if bs <= 0: best = None
        hair_color_match = best
    img_paths = {}
    for slot in ('hair', 'head', 'torso', 'legs'):
        pn = picks.get(slot)
        if not pn or pn == 'NONE': continue
        cid = ''
        if slot == 'hair' and hair_color_match:
            # does this part exist in that color?
            for (p, c), v in elem.items():
                if p == pn:
                    if c == hair_color_match:
                        cid = c; break
                    if not cid: cid = c
        else:
            for (p, c), v in elem.items():
                if p == pn: cid = c; break
        eid = elem.get((pn, cid), '')
        if not eid:
            # design-only parts (e.g. 16709pat28) have no element row; use name search
            url = 'https://www.lego.com/en-us/pick-and-build/pick-a-brick?query=' + \
                  urllib.parse.quote(names.get(pn, pn).split(' [')[0])
        else:
            url = f'https://www.lego.com/en-us/pick-and-build/pick-a-brick?query={eid}'
        result['picks'][slot] = {'part_num': pn, 'name': names.get(pn, '?'),
                                  'element_id': eid, 'color': colors.get(cid, {}).get('name', ''),
                                  'pick_a_brick': url}
        if not a.no_images and eid:
            ip = fetch_element_image(eid, outdir)
            if ip: img_paths[slot] = str(ip)
    result['reasoning'] = picks.get('reasoning', '')

    (outdir / 'recommendation.json').write_text(json.dumps(result, indent=2))
    md = [f'# Minifigure recommendation\n\n- photo: {a.photo}\n']
    for slot in ('hair', 'head', 'torso', 'legs'):
        r = result['picks'].get(slot)
        if not r:
            md.append(f'## {slot}\n- NONE (clean-shaven / no hair piece needed)\n')
            continue
        md.append(f'## {slot}\n- {r.get("name","?")}\n- part `{r.get("part_num")}` element `{r.get("element_id")}` color {r.get("color")}\n- {r.get("pick_a_brick")}\n')
    md.append(f'\nreasoning: {result["reasoning"]}\n')
    (outdir / 'recommendation.md').write_text('\n'.join(md))
    result['picks']['_images'] = img_paths
    compose_sheet(a.photo, result['picks'], outdir)
    print(f'\ndone in {time.time()-t0:.0f}s -> {outdir}/recommendation.md')

if __name__ == '__main__':
    main()
