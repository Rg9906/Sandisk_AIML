"""
Build WaferLens_Deck.pptx from the rendered slide PNGs.

Written against the OOXML spec directly rather than python-pptx, because PyPI is not
reachable from this machine and the project rule is to add no dependency we do not need.
A PPTX is a zip of XML parts; a deck of full-bleed images needs only the minimum set.

    python build_pptx.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
OUT = HERE / "WaferLens_Deck.pptx"

# 16:9 at 13.333in x 7.5in, in English Metric Units (914400 EMU per inch)
W, H = 12192000, 6858000

REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PML = "http://schemas.openxmlformats.org/presentationml/2006/main"
DML = "http://schemas.openxmlformats.org/drawingml/2006/main"


def content_types(n: int) -> str:
    slides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.'
        f'openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, n + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.'
        'openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.extended-properties+xml"/>'
        f'{slides}</Types>')


ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{REL}/officeDocument" Target="ppt/presentation.xml"/>'
    f'<Relationship Id="rId2" Type="{REL}/metadata/core-properties" Target="docProps/core.xml"/>'
    f'<Relationship Id="rId3" Type="{REL}/extended-properties" Target="docProps/app.xml"/>'
    '</Relationships>')

CORE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<cp:coreProperties '
    'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    '<dc:title>WaferLens - SanDisk Hackathon</dc:title>'
    '<dc:creator>Team WaferLens</dc:creator>'
    '<cp:lastModifiedBy>Team WaferLens</cp:lastModifiedBy>'
    '</cp:coreProperties>')

APP = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
    'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
    '<Application>Microsoft Office PowerPoint</Application>'
    '<PresentationFormat>Widescreen</PresentationFormat>'
    '</Properties>')


def presentation(n: int) -> str:
    ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, n + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:presentation xmlns:a="{DML}" xmlns:r="{REL}" xmlns:p="{PML}" saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{ids}</p:sldIdLst>'
        f'<p:sldSz cx="{W}" cy="{H}"/><p:notesSz cx="{H}" cy="{W}"/>'
        '</p:presentation>')


def presentation_rels(n: int) -> str:
    rels = [f'<Relationship Id="rId1" Type="{REL}/slideMaster" '
            f'Target="slideMasters/slideMaster1.xml"/>']
    for i in range(1, n + 1):
        rels.append(f'<Relationship Id="rId{i + 1}" Type="{REL}/slide" '
                    f'Target="slides/slide{i}.xml"/>')
    rels.append(f'<Relationship Id="rId{n + 2}" Type="{REL}/theme" Target="theme/theme1.xml"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels) + '</Relationships>')


def _font_scheme() -> str:
    def block(tag, face):
        return (f'<a:{tag}><a:latin typeface="{face}"/><a:ea typeface=""/><a:cs typeface=""/></a:{tag}>')
    return ('<a:fontScheme name="WaferLens">'
            + block("majorFont", "Calibri Light") + block("minorFont", "Calibri")
            + '</a:fontScheme>')


THEME = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<a:theme xmlns:a="{DML}" name="WaferLens"><a:themeElements>'
    '<a:clrScheme name="WaferLens">'
    '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
    '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="0F1720"/></a:dk2><a:lt2><a:srgbClr val="F7F9FB"/></a:lt2>'
    '<a:accent1><a:srgbClr val="0E6F7C"/></a:accent1><a:accent2><a:srgbClr val="167A4C"/></a:accent2>'
    '<a:accent3><a:srgbClr val="B07500"/></a:accent3><a:accent4><a:srgbClr val="B02135"/></a:accent4>'
    '<a:accent5><a:srgbClr val="6B4EA8"/></a:accent5><a:accent6><a:srgbClr val="63727F"/></a:accent6>'
    '<a:hlink><a:srgbClr val="0E6F7C"/></a:hlink><a:folHlink><a:srgbClr val="6B4EA8"/></a:folHlink>'
    '</a:clrScheme>'
    + _font_scheme() +
    '<a:fmtScheme name="WaferLens">'
    '<a:fillStyleLst>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '</a:fillStyleLst>'
    '<a:lnStyleLst>'
    '<a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '</a:lnStyleLst>'
    '<a:effectStyleLst>'
    '<a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle>'
    '</a:effectStyleLst>'
    '<a:bgFillStyleLst>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '</a:bgFillStyleLst>'
    '</a:fmtScheme></a:themeElements></a:theme>')

_EMPTY_TREE = (
    '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree>')

SLIDE_MASTER = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<p:sldMaster xmlns:a="{DML}" xmlns:r="{REL}" xmlns:p="{PML}">'
    '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
    '<a:effectLst/></p:bgPr></p:bg>' + _EMPTY_TREE + '</p:cSld>'
    '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
    'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" '
    'folHlink="folHlink"/>'
    '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
    '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>'
    '</p:sldMaster>')

SLIDE_MASTER_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{REL}/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    f'<Relationship Id="rId2" Type="{REL}/theme" Target="../theme/theme1.xml"/>'
    '</Relationships>')

SLIDE_LAYOUT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<p:sldLayout xmlns:a="{DML}" xmlns:r="{REL}" xmlns:p="{PML}" type="blank" preserve="1">'
    '<p:cSld name="Blank">' + _EMPTY_TREE + '</p:cSld>'
    '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')

SLIDE_LAYOUT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{REL}/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
    '</Relationships>')


def slide(i: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sld xmlns:a="{DML}" xmlns:r="{REL}" xmlns:p="{PML}">'
        '<p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '<p:pic><p:nvPicPr>'
        f'<p:cNvPr id="2" name="Slide {i}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        '<p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{W}" cy="{H}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def slide_rels(i: int) -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{REL}/image" Target="../media/image{i}.png"/>'
            f'<Relationship Id="rId2" Type="{REL}/slideLayout" '
            f'Target="../slideLayouts/slideLayout1.xml"/>'
            '</Relationships>')


def main() -> None:
    images = sorted(PNG.glob("slide*.png"))
    if not images:
        raise SystemExit(f"no slide PNGs in {PNG}")
    n = len(images)

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(n))
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("docProps/core.xml", CORE)
        z.writestr("docProps/app.xml", APP)
        z.writestr("ppt/presentation.xml", presentation(n))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(n))
        z.writestr("ppt/theme/theme1.xml", THEME)
        z.writestr("ppt/slideMasters/slideMaster1.xml", SLIDE_MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", SLIDE_MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", SLIDE_LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", SLIDE_LAYOUT_RELS)
        for i, img in enumerate(images, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide(i))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels(i))
            z.write(img, f"ppt/media/image{i}.png")

    print(f"wrote {OUT.name}: {n} slides, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
