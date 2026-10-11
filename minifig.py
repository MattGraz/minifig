#!/usr/bin/env python3
"""minifig.py — photo -> LEGO minifigure parts recommendation (Pick a Brick links).

Replicates brickify.fun's core workflow locally: analyze a photo of a person with a
local vision LLM (oMLX), then recommend official LEGO minifigure parts — hair, head
print, torso, legs, plus headwear and held accessories — with element IDs, purchase
links, and an HTML report.

Catalog: Rebrickable CSVs (https://rebrickable.com/downloads/) — run download_data.sh first.
Env: reuse ~/.venvs/brickify (Pillow, httpx) per python-env consolidation rule.

Usage:
  ~/.venvs/brickify/bin/python minifig.py PHOTO [-o OUTDIR] [--no-images] [--no-accessories]
"""
import argparse, base64, csv, html, json, os, re, sys, time
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
    elem = {}
    for r in load_csv('elements.csv'):
        elem[(r['part_num'], r['color_id'])] = r['element_id']

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
        elif nm.startswith('Hips') and 'Shorts' in nm and 'Print' in nm and 'Minidoll' not in nm:
            legs.append(pn)

    headwear, held = [], []
    for r in parts:
        pn, nm = r['part_num'], r['name']
        if re.match(r'^(Cap[ ,]|Hat |Helmet |Beret|Crown |Bonnet)', nm) and 'Hair and' not in nm and 'Duplo' not in nm and 'Creature' not in nm and 'Minidoll' not in nm:
            headwear.append(pn)
        if (nm.startswith('Equipment') or nm.startswith('Sports') or nm.startswith('Minifig Accessory') or nm.startswith('Tool')) and 'Print' not in nm and 'Duplo' not in nm and 'Harry Potter' not in nm and 'Quidditch' not in nm and 'Trophy' not in nm:
            held.append(pn)
    return {'hair': hair, 'head': head, 'torso': torso, 'legs': legs,
            'headwear': headwear, 'held': held}, names, colors, elem

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
 "glasses": true, "facial_hair": "none/moustache/beard/beard and moustache/goatee/stubble",
 "top_description": "brief description of shirt/jacket incl colors, e.g. blue denim shirt with collar",
 "bottom_description": "brief description of pants/shorts incl color",
 "headwear": "none, or the exact headwear incl color",
 "held_item": "none unless they are CLEARLY gripping an object in their hand; if they are wearing or displaying jewelry (a chain, a necklace), put that in notable and keep held_item none",
 "notable": "any other distinguishing feature or accessory (chains, badges, etc)"}"""

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
    'grayish': ['gray', 'grey'], 'orange': ['orange'], 'gold': ['gold', 'yellow'],
}

def tokens(text):
    return set(re.findall(r'[a-z]+', text.lower()))

def score_name(name, query_tokens, synonym_tokens):
    nt = tokens(name)
    return len(nt & query_tokens) + 2 * len(nt & synonym_tokens)

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
        hint = 'Shorts' if 'short' in bd else 'Pants'
        q['legs'] = f"Hips Legs {bd.split(',')[0]} {hint} Print"
    else:
        q['legs'] = 'Hips Legs Blue Jeans Print'
    hw = desc.get('headwear', 'none')
    if hw and hw != 'none':
        q['headwear'] = f"Cap Hat {hw}"
    held = desc.get('held_item', 'none')
    if held and held != 'none':
        q['held'] = f"Equipment {held}"
    return q

def retrieve(catalog, names, desc, topk=6):
    out = {}
    for slot, qtext in build_queries(desc).items():
        if slot not in catalog:
            continue
        qt = tokens(qtext)
        syn = set()
        for word in list(qt):
            syn.update(COLOR_SYNONYMS.get(word, []))
        scored = []
        for pn in catalog[slot]:
            nm = names[pn]
            sc = score_name(nm, qt, syn)
            if slot == 'legs' and 'short' in qtext.lower():
                if re.search(r'(Nougat|Fair|Warm Tan|Tan) Legs', nm):
                    sc += 8
                elif re.search(r'(Magenta|Bright Green|Dark Red|Purple|Orange|Neon) Legs', nm):
                    sc -= 6
                if 'Knee' in nm:
                    sc -= 2
            scored.append((sc, pn))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if slot in ('headwear', 'held'):
            # accessory slots only fire on strong matches — no weak guesses
            if scored and scored[0][0] >= 4:
                out[slot] = [pn for _, pn in scored[:topk]]
            continue
        out[slot] = [pn for _, pn in scored[:topk]]
    return out

# --------------------------------------------------------- stage 3: choose ---
def final_pick(img_path, desc, shortlist, names, elem):
    lines = []
    for slot, pns in shortlist.items():
        lines.append(f'\n{slot.upper()} candidates:')
        for pn in pns:
            eid = ''
            for (p, c), v in elem.items():
                if p == pn:
                    eid = v
                    break
            lines.append(f'  {pn}: {names[pn]} [element {eid}]')
    slot_list = list(shortlist.keys())
    answer_keys = ', '.join(f'"{s}": "<part_num or NONE>"' for s in slot_list)
    pick_prompt = (
        'You recommend official LEGO minifigure parts that best match the person in the photo.\n'
        f'Observed description: {json.dumps(desc)}\n'
        'Pick EXACTLY ONE part per slot from the candidate lists, or "NONE" if none of the candidates is a good match.\n'
        'IMPORTANT accuracy rules:\n'
        '- If the person is COMPLETELY bald, output NONE for hair.\n'
        '- If they have only light stubble, pick a stubble or clean-shaven head, NOT a full-beard head.\n'
        '- If they are a child, pick a child/neutral face, NOT an animal or alien print.\n'
        '- For LEGS with shorts, the legs must be a SKIN color (nougat/tan/white), never magenta/green/costume colors.\n'
        '- Only pick HEADWEAR if they are actually wearing something on their head; otherwise NONE.\n'
        '- Only pick a HELD item if the photo clearly shows them holding something purchasable; when in doubt NONE.\n'
        '- Match garment colors as literally as possible.\n'
        f'Answer ONLY JSON: {{{answer_keys}, "reasoning": "<one line>"}}'
        + '\n'.join(lines))
    txt = omlx([{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_data_url(img_path)}},
        {'type': 'text', 'text': pick_prompt}]}], max_tokens=600)
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group(0))

# ------------------------------------------------------------ render sheet --
def fetch_element_image(eid, outdir):
    if not eid:
        return None
    url = f'https://cdn.rebrickable.com/media/parts/elements/{eid}.jpg'
    dst = outdir / f'{eid}.jpg'
    if dst.exists():
        return dst
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as r, open(dst, 'wb') as f:
            f.write(r.read())
        return dst
    except Exception as e:
        print(f'  image fetch failed {eid}: {e}')
        return None

def compose_sheet(photo, slots, picks, outdir):
    from PIL import Image, ImageDraw
    cell, pad = 240, 30
    shown = [s for s in slots if picks.get('_images', {}).get(s)]
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
        im = Image.open(picks['_images'][slot]).convert('RGB')
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
 .reason {{ margin-top:16px; padding:12px 16px; background:#fff; border-radius:12px; }}
</style></head><body>
<h1>Minifigure recommendation</h1>
<div class="strip">
  <div class="photo"><h3>photo</h3><img src="data:image/jpeg;base64,{photo_b64}"></div>
  {''.join(rows)}
</div>
<div class="reason"><b>reasoning:</b> {esc(result.get('reasoning',''))}</div>
</body></html>'''
    (outdir / 'report.html').write_text(doc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--no-images', action='store_true')
    ap.add_argument('--no-accessories', action='store_true',
                    help='skip headwear/held accessory slots')
    a = ap.parse_args()
    t0 = time.time()
    outdir = Path(a.out) if a.out else HERE / 'out' / Path(a.photo).stem
    outdir.mkdir(parents=True, exist_ok=True)

    catalog, names, colors, elem = build_catalog()
    print('catalog: ' + ' '.join(f'{k}={len(v)}' for k, v in catalog.items()))

    print('analyzing photo (vision)...')
    desc = describe_person(a.photo)
    print(json.dumps(desc, indent=2))

    print('retrieving candidates...')
    shortlist = retrieve(catalog, names, desc)
    if a.no_accessories:
        for s in EXTRA_SLOTS:
            shortlist.pop(s, None)
    # legs: if the photo doesn't show legs, default to classic blue jeans
    if desc.get('bottom_description') in (None, 'not visible', ''):
        jeans = [pn for pn in catalog['legs'] if re.search(r'jeans|denim', names[pn], re.I)]
        plain = [pn for pn in catalog['legs'] if 'Blue' in names[pn]]
        shortlist['legs'] = (jeans + plain)[:4]
    slots = CORE_SLOTS + [s for s in EXTRA_SLOTS if s in shortlist]
    for s in slots:
        print(f'  {s}: ' + ' | '.join(f'{p} ({names[p][:50]})' for p in shortlist[s][:3]))

    print('final pick (vision + candidates)...')
    picks = final_pick(a.photo, desc, shortlist, names, elem)

    result = {'description': desc, 'picks': {}, 'shortlist': {s: shortlist[s] for s in slots}}
    hair_color_match = None
    if desc.get('hair_color') and desc['hair_color'] != 'bald':
        want = tokens(desc['hair_color'])
        best, bs = None, -1
        for cid, c in colors.items():
            s = len(want & tokens(c['name']))
            if s > bs and c['name'] != '[Unknown]':
                bs, best = s, cid
        if bs <= 0:
            best = None
        hair_color_match = best
    img_paths = {}
    for slot in slots:
        pn = picks.get(slot)
        if not pn or pn == 'NONE':
            continue
        cid = ''
        want_c = hair_color_match if slot == 'hair' else None
        for (p, c) in elem:
            if p != pn:
                continue
            if want_c and c == want_c:
                cid = c
                break
            if not cid:
                cid = c
        eid = elem.get((pn, cid), '')
        if eid:
            url = f'https://www.lego.com/en-us/pick-and-build/pick-a-brick?query={eid}'
        else:
            url = 'https://www.lego.com/en-us/pick-and-build/pick-a-brick?query=' + \
                  urllib.parse.quote(names.get(pn, pn).split(' [')[0])
        result['picks'][slot] = {'part_num': pn, 'name': names.get(pn, '?'),
                                  'element_id': eid, 'color': colors.get(cid, {}).get('name', ''),
                                  'pick_a_brick': url}
        if not a.no_images and eid:
            ip = fetch_element_image(eid, outdir)
            if ip:
                img_paths[slot] = str(ip)
    result['reasoning'] = picks.get('reasoning', '')

    (outdir / 'recommendation.json').write_text(json.dumps(result, indent=2))
    result['picks']['_images'] = img_paths
    compose_sheet(a.photo, slots, result, outdir)
    compose_html(a.photo, slots, {'picks': result['picks'], 'reasoning': result['reasoning']}, outdir)
    print(f'\ndone in {time.time()-t0:.0f}s -> {outdir}/report.html')

if __name__ == '__main__':
    main()
