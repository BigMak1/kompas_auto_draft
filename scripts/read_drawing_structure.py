"""Read the structure of drawings (.cdw) already open in the running KOMPAS.

Read-only: attaches with GetKompas(), never opens, rebuilds, updates or saves
anything. Prints one JSON report per drawing: sheets with format and title block
cells, views with layers and object counts, dimensions, leaders, technical
requirements and the link to the 3D model.

Usage (x64 Python on the KOMPAS machine):
    python read_drawing_structure.py [name-substring ...]
Without arguments every open drawing is reported.
"""

import json
import sys

KOMPAS_BIN = r"C:\Program Files\ASCON\KOMPAS-3D v25 Study\Bin"

sys.path.insert(0, KOMPAS_BIN)
import ksapi  # noqa: E402  (loads ksAPICLink.dll from KOMPAS_BIN on import)
from constants import constants as K  # noqa: E402

FORMATS = {
    K.ksFormatA0: "A0",
    K.ksFormatA1: "A1",
    K.ksFormatA2: "A2",
    K.ksFormatA3: "A3",
    K.ksFormatA4: "A4",
    K.ksFormatA5: "A5",
    K.ksFormatUser: "user",
}
VIEW_TYPES = {
    K.vt_System: "system",
    K.vt_Normal: "normal",
    K.vt_Arbitrary: "arbitrary",
    K.vt_Standart: "standard",
    K.vt_Projected: "projected",
    K.vt_Arrow: "arrow",
    K.vt_Remote: "remote",
    K.vt_Section: "section",
    K.vt_Remote2D: "remote2d",
}

DRAWING_GETTERS = [
    "GetLineSegments",
    "GetCircles",
    "GetArcs",
    "GetEllipses",
    "GetEllipseArcs",
    "GetPolyLines2D",
    "GetBeziers",
    "GetNurbses",
    "GetRectangles",
    "GetRegularPolygons",
    "GetDrawingContours",
    "GetHatches",
    "GetColourings",
    "GetDrawingTexts",
    "GetPoints",
    "GetLines",
    "GetMacroObjects",
    "GetInsertionObjects",
    "GetRasters",
    "GetEquidistants",
]
SYMBOL_GETTERS = [
    "GetLineDimensions",
    "GetRadialDimensions",
    "GetDiametralDimensions",
    "GetAngleDimensions",
    "GetArcDimensions",
    "GetHeightDimensions",
    "GetBreakLineDimensions",
    "GetBreakRadialDimensions",
    "GetRoughs",
    "GetLeaders",
    "GetBases",
    "GetTolerances",
    "GetDrawingTables",
    "GetCutLines",
    "GetViewPointers",
    "GetCentreMarkers",
    "GetAxisLines",
    "GetWaveLines",
    "GetBrokenLines",
    "GetRemoteElements",
    "GetCircularsCentres",
    "GetLinearsCentres",
]
DIMENSION_GETTERS = [g for g in SYMBOL_GETTERS if g.endswith("Dimensions")]


def safe(fn, *args, default=None):
    """Call a KsAPI getter; turn failures into a short marker instead of aborting the report."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001  (report must survive any single failing getter)
        return default if default is not None else f"<error: {type(exc).__name__}: {exc}>"


def text_str(text):
    """IText -> plain string, line by line."""
    if not text:
        return None
    lines = [safe(text.GetTextLine(i).GetStr) for i in range(safe(text.GetCount, default=0))]
    return "\n".join(str(line) for line in lines if line)


def collection_count(container, getter):
    coll = safe(getattr(container, getter))
    return safe(coll.GetCount, default=0) if coll and not isinstance(coll, str) else 0


def read_stamp(stamp):
    cells, prev, seen = {}, 0, set()
    for _ in range(1000):
        cid = safe(stamp.GetNextColumnId, prev, default=0)
        if cid == prev:  # tolerate an inclusive search start
            cid = safe(stamp.GetNextColumnId, prev + 1, default=0)
        if not cid or cid in seen:
            break
        seen.add(cid)
        cells[str(cid)] = text_str(safe(stamp.GetText, cid))
        prev = cid
    return cells


def read_sheets(doc):
    sheets = []
    layout = safe(doc.GetLayoutSheets)
    for i in range(safe(layout.GetCount, default=0) if layout else 0):
        sheet = layout.GetItem(i)
        fmt = safe(sheet.GetFormat)
        stamp = safe(sheet.GetStamp)
        sheets.append(
            {
                "index": i,
                "format": FORMATS.get(safe(fmt.GetFormat), safe(fmt.GetFormat)) if fmt else None,
                "multiplicity": safe(fmt.GetFormatMultiplicity) if fmt else None,
                "vertical": safe(fmt.IsVerticalOrientation) if fmt else None,
                "size_mm": [safe(fmt.GetFormatWidth), safe(fmt.GetFormatHeight)] if fmt else None,
                "layout_style": safe(sheet.GetLayoutStyleNumber),
                "layout_library": safe(sheet.GetLayoutLibraryFileName),
                "title_block": read_stamp(stamp) if stamp else None,
            }
        )
    return sheets


def read_dimensions(symbols):
    dims = []
    for getter in DIMENSION_GETTERS:
        coll = safe(getattr(symbols, getter))
        if not coll or isinstance(coll, str):
            continue
        for obj in safe(coll.GetObjects, default=[]):
            dt = ksapi.IDimensionText(obj)
            dims.append(
                {
                    "kind": getter[3 : -len("Dimensions")] or getter,
                    "nominal": safe(dt.GetNominalValue) if dt else None,
                    "sign": safe(dt.GetSign) if dt else None,
                    "has_tolerance": safe(dt.IsHasTolerance) if dt else None,
                }
            )
    return dims


def read_leaders(symbols):
    leaders = []
    coll = safe(symbols.GetLeaders)
    if not coll or isinstance(coll, str):
        return leaders
    for obj in safe(coll.GetObjects, default=[]):
        ld = ksapi.ILeader(obj)
        if ld:
            leaders.append(
                {
                    "on_shelf": text_str(safe(ld.GetTextOnShelf)),
                    "under_shelf": text_str(safe(ld.GetTextUnderShelf)),
                }
            )
    return leaders


def read_view(view):
    info = {
        "number": safe(view.GetNumber),
        "name": safe(view.GetName),
        "type": VIEW_TYPES.get(safe(view.GetViewType), safe(view.GetViewType)),
        "scale": safe(view.GetScale),
        "origin": [safe(view.GetX), safe(view.GetY)],
        "angle": safe(view.GetAngle),
        "current": safe(view.IsCurrent),
        "visible": safe(view.IsVisible),
        "object_count": safe(view.GetObjectCount),
    }
    assoc = ksapi.IAssociationView(view)
    if assoc:
        info["association"] = {
            "source_file": safe(assoc.GetSourceFileName),
            "projection": safe(assoc.GetProjectionName),
            "section": safe(assoc.IsSection),
            "projection_link": safe(assoc.IsProjectionLink),
            "hidden_lines_visible": safe(assoc.IsHiddenLinesVisible),
        }
    layers = safe(view.GetLayers)
    info["layers"] = (
        [
            {
                "name": safe(layer.GetName),
                "objects": safe(layer.GetObjectCount),
                "visible": safe(layer.IsVisible),
            }
            for layer in (layers.GetLayer(i) for i in range(safe(layers.GetCount, default=0)))
            if layer
        ]
        if layers and not isinstance(layers, str)
        else []
    )
    drawing = ksapi.IDrawingContainer(view)
    info["geometry"] = (
        {g[3:]: n for g in DRAWING_GETTERS if (n := collection_count(drawing, g))} if drawing else {}
    )
    symbols = ksapi.ISymbols2DContainer(view)
    if symbols:
        info["annotations"] = {g[3:]: n for g in SYMBOL_GETTERS if (n := collection_count(symbols, g))}
        info["dimensions"] = read_dimensions(symbols)
        info["leaders"] = read_leaders(symbols)
    return info


def read_drawing(doc):
    doc2d = ksapi.IKompasDocument2D(doc)
    drawing = ksapi.IDrawingDocument(doc)
    report = {
        "name": safe(doc.GetName),
        "full_path": safe(doc.GetFullPath),
        "read_only": safe(doc.IsReadOnly),
        "changed": safe(doc.IsChanged),
        "author": safe(doc.GetAuthor),
        "organization": safe(doc.GetOrganization),
        "comment": safe(doc.GetComment),
        "sheets": read_sheets(doc),
    }
    if drawing:
        model_view = safe(drawing.GetAssociationViewForCurrentModel)
        report["model_link"] = {
            "linked": safe(drawing.IsLinkWithModel),
            "spec_data_from_model": safe(drawing.IsSpecificationDataFromModel),
            "model_file": safe(model_view.GetSourceFileName)
            if model_view and not isinstance(model_view, str)
            else None,
        }
        demand = safe(drawing.GetTechnicalDemand)
        created = demand and not isinstance(demand, str) and safe(demand.IsCreated)
        report["technical_requirements"] = text_str(safe(demand.GetText)) if created else None
    views = safe(doc2d.GetViewsAndLayersManager().GetViews)
    count = safe(views.GetCount, default=0) if views and not isinstance(views, str) else 0
    report["views"] = [read_view(v) for v in (views.GetView(i) for i in range(count)) if v]
    return report


def main():
    filters = [a.lower() for a in sys.argv[1:]]
    app = ksapi.GetKompas()
    if app is None:
        print("GetKompas: no running KOMPAS visible from this session", file=sys.stderr)
        return 2
    previous = app.GetHideMessage()
    app.SetHideMessage(K.ksHideMessageNo)  # no modal dialog may block an unattended read
    try:
        seen = set()
        for obj in app.GetDocuments().GetObjects():
            doc = ksapi.IKompasDocument(obj)
            if safe(doc.GetDocumentType) != K.ksDocumentDrawing:
                continue
            path = safe(doc.GetFullPath)
            name = str(safe(doc.GetName))
            if filters and not any(f in name.lower() for f in filters):
                continue
            report = read_drawing(doc)
            report["duplicate_of_already_listed"] = path in seen
            seen.add(path)
            print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    finally:
        app.SetHideMessage(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
