"""Re-record assets/openclatura-demo.gif against the live site.

    python website/scripts/record_demo.py <out-dir>

Drives https://openclatura.org with Playwright (dark colour scheme, 1200x660),
loads four example molecules, opens the explanation for the last one and writes
<out-dir>/openclatura-demo.gif. Copy it over assets/openclatura-demo.gif.
Needs the dev venv plus `playwright install chromium`.
"""
import sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
URL = "https://openclatura.org"
W, H, BAR = 1200, 660, 39            # viewport + fake browser bar = 1200x699 like the old gif
frames = []

def shot(page, n=1):
    for _ in range(n):
        frames.append(Image.open(__import__("io").BytesIO(page.screenshot())).convert("RGB"))

MOLS = [
    ("CC(=O)Oc1ccccc1C(=O)O", False),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", False),
    ("CCN(CC)CC(=O)Nc1c(C)cccc1C", False),
    ("CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O", True),   # testosterone: explain at the end
]

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="dark")
    page.goto(URL, wait_until="networkidle")
    page.wait_for_function("() => { try { return !!document.getElementById('ketcher-frame').contentWindow.ketcher; } catch (e) { return false; } }", timeout=60000)
    time.sleep(3)
    shot(page, 10)
    prev = ""
    for smi, explain in MOLS:
        page.evaluate("""s => { const f=document.getElementById('ketcher-frame');
            return f.contentWindow.ketcher.setMolecule(s); }""", smi)
        time.sleep(0.5); shot(page, 6)   # structure drawn, panel still busy
        # setMolecule fires Ketcher's change event, which names the structure;
        # the old name stays until the new one lands and the status line clears
        page.wait_for_function("p => { const t=document.getElementById('result-name').textContent.trim(); return t && t !== p && document.getElementById('status').textContent.trim()==='' && !document.getElementById('btn-explain').disabled; }", arg=prev, timeout=180000)
        prev = page.evaluate("() => document.getElementById('result-name').textContent.trim()")
        print("named:", prev)
        shot(page, 30)
        if explain:
            page.click("#btn-explain")
            page.wait_for_function("() => document.getElementById('explain').textContent.trim().length > 200", timeout=120000)
            shot(page, 14)
            for y in (160, 320, 420):
                page.evaluate("y => window.scrollTo({top: y, behavior: 'instant'})", y)
                time.sleep(0.2); shot(page, 12)
            shot(page, 30)
    b.close()

# --- compose: browser bar on top, then a title card ---------------------------
BG = (14, 19, 32)
def font(sz, bold=False):
    for f in (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]):
        try: return ImageFont.truetype(f, sz)
        except OSError: pass
    return ImageFont.load_default()

def with_bar(im):
    out = Image.new("RGB", (W, H + BAR), BG)
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((463, 6, 737, 29), radius=12, fill=(26, 33, 48), outline=(52, 62, 84))
    d.text((600, 17), "openclatura.org", fill=(220, 226, 240), font=font(13), anchor="mm")
    out.paste(im, (0, BAR)); return out

def title_card():
    im = Image.new("RGB", (W, H + BAR), BG); d = ImageDraw.Draw(im)
    f1, f2, f3 = font(44, True), font(20), font(14)
    a, bb = "openclatura", ".org"
    wa, wb = d.textlength(a, font=f1), d.textlength(bb, font=f1)
    x = (W - wa - wb) / 2
    d.text((x, 325), a, fill=(105, 140, 255), font=f1, anchor="lm")
    d.text((x + wa, 325), bb, fill=(230, 234, 245), font=f1, anchor="lm")
    d.text((W/2, 378), "draw a molecule → get its IUPAC name", fill=(230, 234, 245), font=f2, anchor="mm")
    d.text((W/2, 411), "deterministic Blue Book rules · OPSIN-verified", fill=(140, 155, 190), font=f3, anchor="mm")
    return im

seq = [with_bar(f) for f in frames] + [title_card()] * 40
# quantize with one shared palette so colours don't flicker between frames
pal = seq[len(seq)//3].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in seq]
gif = OUT / "openclatura-demo.gif"
q[0].save(gif, save_all=True, append_images=q[1:], duration=70, loop=0, optimize=True)
print("frames", len(seq), "->", gif, gif.stat().st_size // 1024, "KB")
