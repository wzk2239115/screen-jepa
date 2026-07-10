import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def geom_augment(img, bg_color=(255, 255, 255), max_angle=10, scale_lo=0.88,
                 scale_hi=1.15, max_shear=6):
    """Random scale + rotation + shear on an HxWx3 uint8 image, filling exposed
    corners with bg_color. Prevents the model from keying on exact glyph layout."""
    import random

    pil = Image.fromarray(img)
    S = pil.size[0]
    # scale (zoom in/out then center crop/pad back)
    s = random.uniform(scale_lo, scale_hi)
    ns = max(8, int(round(S * s)))
    pil = pil.resize((ns, ns), Image.BICUBIC)
    if ns >= S:
        off = (ns - S) // 2
        pil = pil.crop((off, off, off + S, off + S))
    else:
        canvas = Image.new("RGB", (S, S), bg_color)
        canvas.paste(pil, ((S - ns) // 2, (S - ns) // 2))
        pil = canvas
    # shear (horizontal tilt) via affine, then rotation
    sh = random.uniform(-max_shear, max_shear)
    cx = cy = S / 2
    pil = pil.transform((S, S), Image.AFFINE,
                        (1, sh, -sh * cy, 0, 1, 0), resample=Image.BICUBIC,
                        fillcolor=bg_color)
    ang = random.uniform(-max_angle, max_angle)
    pil = pil.rotate(ang, resample=Image.BICUBIC, fillcolor=bg_color)
    return np.array(pil)



def _layout(words, font, line_h, space_w, S, margin):
    """Greedy word-wrap. Returns (placements, fits).
    placements: list of (word, x0, y0, x1, y1). fits: all words placed inside canvas."""
    placements = []
    x, y = margin, margin
    usable = S - margin
    for w in words:
        ww = font.getlength(w)
        if x + ww > usable and x > margin:
            x = margin
            y += line_h
        if y + line_h > usable:
            return placements, False
        placements.append((w, x, y, x + ww, y + line_h))
        x += ww + space_w
    return placements, True


class TextRenderer:
    """Render a sentence to an RGB image (black text on white), recording per-word
    pixel bboxes. Font size is auto-fitted so the text fills the canvas (dense),
    which makes word masking a meaningful perturbation. Supports erasing words."""

    def __init__(self, img_size=224, font_size=40, font_path=DEFAULT_FONT, margin=8):
        self.img_size = img_size
        self.max_font_size = font_size
        self.margin = margin
        self.font_path = font_path
        self._font_cache = {}

    def _font(self, size):
        if size not in self._font_cache:
            self._font_cache[size] = ImageFont.truetype(self.font_path, size)
        return self._font_cache[size]

    def _fit(self, words):
        """Binary search the largest font size whose wrapped layout fits the canvas."""
        S, margin = self.img_size, self.margin
        lo, hi, best = 8, self.max_font_size, 8
        best_font, best_lh, best_pl = None, 0, None
        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._font(mid)
            lh = int(round(mid * 1.3))
            sp = font.getlength(" ")
            pl, fits = _layout(words, font, lh, sp, S, margin)
            if fits:
                best, best_font, best_lh, best_pl = mid, font, lh, pl
                lo = mid + 1
            else:
                hi = mid - 1
        if best_pl is None:
            font = self._font(best)
            best_lh = int(round(best * 1.3))
            best_pl, _ = _layout(words, font, best_lh, font.getlength(" "), S, margin)
            best_font = font
        return best_font, best_lh, best_pl

    def render(self, sentence, bg_color=(255, 255, 255)):
        """Return (img: HxWx3 uint8, word_boxes). bg_color sets background;
        text color auto-contrasts. Stores self.last_font for compositing."""
        S = self.img_size
        img = Image.new("RGB", (S, S), bg_color)
        draw = ImageDraw.Draw(img)
        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        text_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
        words = sentence.split()
        font, lh, placements = self._fit(words)
        self.last_font = font
        self.last_bg = bg_color
        boxes = []
        for w, x0, y0, x1, y1 in placements:
            draw.text((x0, y0), w, fill=text_color, font=font)
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
        return np.array(img), boxes

    def render_centered(self, text, font_size=40):
        """Render a single word/short text centered at a fixed font size (for probes)."""
        S = self.img_size
        img = Image.new("RGB", (S, S), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = self._font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w >= S - 2 * self.margin:
            font_size = max(10, int(font_size * (S - 2 * self.margin) / w))
            font = self._font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (S - w) / 2 - bbox[0]
        y = (S - h) / 2 - bbox[1]
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
        return np.array(img)

    def mask_words(self, img, boxes, indices, bg_color=(255, 255, 255)):
        """Erase selected words by filling their bboxes with the background."""
        pil = Image.fromarray(img.copy())
        d = ImageDraw.Draw(pil)
        for i in indices:
            x0, y0, x1, y1 = boxes[i]
            d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], fill=bg_color)
        return np.array(pil)


if __name__ == "__main__":
    r = TextRenderer(font_size=64)
    full, boxes = r.render("the quick brown fox jumps over the lazy dog")
    idx = [1, 4]
    masked = r.mask_words(full, boxes, idx)
    dark = (full.sum(2) < 300).mean()
    print(f"words={len(boxes)} masked={idx} text_coverage={dark:.3f}")
    Image.fromarray(full).save("/tmp/opencode/full.png")
    Image.fromarray(masked).save("/tmp/opencode/masked.png")
    print("saved /tmp/opencode/full.png and /tmp/opencode/masked.png")
