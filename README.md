
## Usage

```
cd ~/repos/minifig
./download_data.sh          # Rebrickable CSV catalog (updated daily upstream)
~/.venvs/brickify/bin/python minifig.py PHOTO.jpg
```

Outputs `out/<name>/recommendation.md` (+ JSON, element photos, comparison sheet).
Runs fully local: oMLX (Qwen vision, thinking OFF) + Rebrickable catalog. ~10 s end-to-end.

## How it works
1. Vision pass: structured description of the person (age, hair, facial hair, clothing).
2. Retrieval: name-token + color-synonym scoring over catalog candidates per slot
   (hair/head/torso/legs), top ~6 per slot.
3. Second vision pass: model picks exactly one part per slot from the shortlist,
   shown the photo + candidate names + element IDs.
4. Resolve element IDs; emit Pick a Brick links (`?query=<elementId>`).

Data: Rebrickable (rebrickable.com/downloads) — attribution required as their source.
LEGO® is a trademark of The LEGO Group; this is not affiliated with or endorsed by them.
