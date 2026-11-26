"""Output marking (``CLAUDE.md`` §11, PR 4.4): every enhanced image says that it is AI-enhanced.

The marks written into the file:

- **PNG:**
  - text chunks: ``AI-enhanced`` (``true``), ``Description`` (``NOTICE``), ``Software`` and
    ``facehall`` (the record as JSON);
  - XMP (iTXt ``XML:com.adobe.xmp``);
  - EXIF.
- **JPEG and WebP:** XMP and EXIF.
- **Video** (MP4, MOV, MKV; ``facehall.video``): container tags ``ai_enhanced``, ``comment``
  and ``description`` (``NOTICE``), ``digital_source_type`` and ``facehall`` (the record, with
  face counts over all frames).

The XMP packet carries:

- the IPTC digital source type ``compositeWithTrainedAlgorithmicMedia`` ("Edited using Generative
  AI": "Augmentation, correction or enhancement using a Generative AI model, such as with
  inpainting or outpainting operations", cv.iptc.org);
- ``dc:description`` = ``NOTICE`` and ``xmp:CreatorTool``;
- ``facehall:AIEnhanced``, ``facehall:Model``, ``facehall:ModelSHA256``,
  ``facehall:InputSHA256`` and ``facehall:Record`` (the JSON record).

EXIF ``ImageDescription`` holds ``NOTICE`` and ``Software`` the package version.

**The record** (``record``) holds:
- the notice;
- the package version and the time;
- the model: run, checkpoint, weights sha256, architecture, scale;
- the pipeline parameters, without local paths;
- the sha256 and size of the input;
- the output size;
- per face: how it was found, the score, whether it was restored and the changed region.

It holds no file names, no landmarks and no image content.

A visible mark is optional (``watermark``). C2PA is not written (``docs/DECISIONS.md``).
"""

import html
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, PngImagePlugin

from facehall import __version__

NOTICE = (
    "AI-enhanced: the faces in this image were restored by a super-resolution model "
    "(facehall). Their details are a plausible reconstruction, not a recovery of the real "
    "ones. Do not use this image to identify a person or as evidence."
)
DIGITAL_SOURCE_TYPE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
)
IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
FACEHALL_NS = "https://github.com/dronja/robodream_cv/xmp/1.0/"
XMP_KEY = "XML:com.adobe.xmp"  # the PNG iTXt keyword for XMP
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
EXIF_DESCRIPTION, EXIF_SOFTWARE = 0x010E, 0x0131
QUOTE = {'"': "&quot;"}  # XML attributes always in double quotes

Image8 = npt.NDArray[np.uint8]  # (H, W, 3) RGB


def record(
    faces: list[dict[str, Any]],
    *,
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    input_sha256: str,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    faces_from: str,
    watermark: bool,
    created: str | None = None,
) -> dict[str, Any]:
    """The provenance record of one output (sizes are (width, height))."""
    return {
        "ai_enhanced": True,
        "notice": NOTICE,
        "software": {"name": "facehall", "version": __version__},
        "created": created or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": dict(model),
        "parameters": dict(parameters),
        "input": {
            "sha256": input_sha256,
            "width": input_size[0],
            "height": input_size[1],
            "faces_from": faces_from,
        },
        "output": {"width": output_size[0], "height": output_size[1]},
        "faces": [
            {k: f[k] for k in ("source", "score", "restored", "reason", "region")} for f in faces
        ],
        "watermark": watermark,
    }


def video_record(
    *,
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    input_sha256: str,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    video: Mapping[str, Any],
    faces: Mapping[str, int],
    watermark: bool,
    created: str | None = None,
) -> dict[str, Any]:
    """The provenance record of an output video: as ``record``, with face counts over all
    frames instead of one entry per face."""
    rec = record(
        [],
        model=model,
        parameters=parameters,
        input_sha256=input_sha256,
        input_size=input_size,
        output_size=output_size,
        faces_from="detector",
        watermark=watermark,
        created=created,
    )
    rec["video"] = dict(video)
    rec["faces"] = dict(faces)
    return rec


def video_tags(rec: Mapping[str, Any]) -> dict[str, str]:
    """Container tags of a marked video (MP4, MOV, MKV)."""
    return {
        "ai_enhanced": "true",
        "comment": NOTICE,
        "description": NOTICE,
        "digital_source_type": DIGITAL_SOURCE_TYPE,
        "facehall": json.dumps(rec, sort_keys=True),
    }


def xmp_packet(rec: Mapping[str, Any]) -> str:
    """An XMP packet with the IPTC digital source type, the notice and the record."""
    model = rec["model"]
    fields = {
        "Iptc4xmpExt:DigitalSourceType": DIGITAL_SOURCE_TYPE,
        "xmp:CreatorTool": f"facehall {rec['software']['version']}",
        "facehall:AIEnhanced": "True",
        "facehall:Model": str(model.get("name", "")),
        "facehall:ModelSHA256": str(model.get("sha256", "")),
        "facehall:InputSHA256": str(rec["input"]["sha256"]),
    }
    attributes = "".join(f"\n    {key}={quoteattr(value, QUOTE)}" for key, value in fields.items())
    payload = escape(json.dumps(rec, sort_keys=True))
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        f'    xmlns:Iptc4xmpExt="{IPTC_EXT_NS}"\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        f'    xmlns:facehall="{FACEHALL_NS}"{attributes}>\n'
        "   <dc:description><rdf:Alt>"
        f'<rdf:li xml:lang="x-default">{escape(NOTICE)}</rdf:li>'
        "</rdf:Alt></dc:description>\n"
        f"   <facehall:Record>{payload}</facehall:Record>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )


def exif(rec: Mapping[str, Any]) -> Image.Exif:
    data = Image.Exif()
    data[EXIF_DESCRIPTION] = NOTICE
    data[EXIF_SOFTWARE] = f"facehall {rec['software']['version']}"
    return data


def output_format(path: Path) -> str:
    try:
        return FORMATS[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"{path}: unsupported output format; use one of {sorted(FORMATS)}"
        ) from None


def save(path: Path, image: Image8, rec: Mapping[str, Any], *, quality: int = 95) -> None:
    """Write ``image`` with its marks. JPEG and WebP use ``quality``; PNG is lossless."""
    kind = output_format(path)
    picture = Image.fromarray(image, "RGB")
    packet = xmp_packet(rec)
    if kind == "PNG":
        info = PngImagePlugin.PngInfo()
        info.add_text("AI-enhanced", "true")
        info.add_itxt("Description", NOTICE)
        info.add_text("Software", f"facehall {rec['software']['version']}")
        info.add_itxt("facehall", json.dumps(rec, sort_keys=True))
        info.add_itxt(XMP_KEY, packet)
        picture.save(path, "PNG", pnginfo=info, exif=exif(rec))
    else:
        picture.save(path, kind, quality=quality, xmp=packet.encode(), exif=exif(rec))


def read_record(path: Path) -> dict[str, Any] | None:
    """The facehall record of a marked image, or None for an unmarked one."""
    with Image.open(path) as picture:
        info = dict(picture.info)
    if "facehall" in info:
        value: dict[str, Any] = json.loads(info["facehall"])
        return value
    packet = info.get("xmp") or info.get(XMP_KEY)
    if packet is None:
        return None
    text = packet.decode() if isinstance(packet, bytes) else str(packet)
    found = re.search(r"<facehall:Record>(.*?)</facehall:Record>", text, re.DOTALL)
    if found is None:
        return None
    value = json.loads(html.unescape(found.group(1)))
    return value


def watermark(image: Image8, text: str = "AI-enhanced") -> Image8:
    """A copy of ``image`` with ``text`` in its bottom-right corner on a dark band."""
    out = image.copy()
    height, width = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    target = max(10.0, 0.03 * min(height, width))  # text height in pixels
    (_, h0), _ = cv2.getTextSize(text, font, 1.0, 1)
    font_scale = target / h0
    thickness = max(1, round(font_scale * 1.5))
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    pad = max(2, round(0.3 * th))
    x1, y1 = width - pad, height - pad
    x0, y0 = max(0, x1 - tw - 2 * pad), max(0, y1 - th - base - 2 * pad)
    band = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.rint(band * 0.4).astype(np.uint8)
    origin = (x0 + pad, y1 - pad - base)
    cv2.putText(out, text, origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out
