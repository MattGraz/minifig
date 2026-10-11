import csv, re, sys, urllib.request
byp = [r for r in csv.DictReader(open('data/elements.csv')) if r['part_num'] == '3814']
cols = {r['id']: r['name'] for r in csv.DictReader(open('data/colors.csv'))}
import collections
seen = []
for r in byp:
    nm = cols.get(r['color_id'], '?')
    if 'Orange' in nm:
        seen.append((r['color_id'], nm, r['element_id']))
print('orange-ish elements for 3814:', len(seen))
for cid, nm, eid in seen[:10]:
    req = urllib.request.Request(f'https://cdn.rebrickable.com/media/parts/elements/{eid}.jpg',
                                 headers={'User-Agent': 'Mozilla/5.0'})
    try:
        r = urllib.request.urlopen(req, timeout=15)
        print(eid, nm, 'IMG OK', len(r.read()))
    except Exception as e:
        print(eid, nm, 'IMG FAIL', e)
