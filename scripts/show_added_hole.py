"""Reopen and display the saved task result, checking the new cylindrical hole."""

import json
import sys
from pathlib import Path

KOMPAS_BIN = r"C:\Program Files\ASCON\KOMPAS-3D v25 Study\Bin"
sys.path.insert(0, KOMPAS_BIN)
import ksapi  # noqa: E402
from constants import constants as K  # noqa: E402
from constants3d import constants as K3  # noqa: E402

app = ksapi.GetKompas()
if not app:
    raise RuntimeError("KOMPAS is not running")
folder = Path(
    r"C:\Mac\Home\Documents\VS_code_projects\ITMO\ASCON\kompas_auto_draft"
    r"\kompas_data\1.Уголок_мебельный"
)
docs = app.GetDocuments()
for name in (
    "Уголок_мебельный_копия_центральные_отверстия.m3d",
    "Уголок_мебельный_копия_отверстие_4мм.m3d",
):
    existing = docs.GetItemByFilePath(str(folder / name))
    if existing and not existing.Close(K.kdDoNotSaveChanges):
        raise RuntimeError(f"Cannot close task copy {name}")
doc = docs.Open(str(folder / "Уголок_мебельный_копия_отверстие_4мм.m3d"), True, False)
if not doc:
    raise RuntimeError("Cannot reopen saved result")
doc3d = ksapi.IKompasDocument3D(doc)
part = doc3d.GetTopPart()
holes = []
for obj in ksapi.IModelContainer(part).GetObjects(K3.o3d_face):
    face = ksapi.IFace(obj)
    if face and face.IsCylinder() and abs(face.GetRadius() - 2.0) < 1e-6:
        origin = face.GetMathSurface().GetPlacement().GetOrigin(0.0, 0.0, 0.0)
        holes.append({"diameter_mm": 4.0, "axis_origin_mm": origin})
if len(holes) != 1:
    raise RuntimeError(f"Expected one new diameter-4 hole, got {holes}")
doc.SetActive()
print(json.dumps({"active": doc.IsActive(), "path": doc.GetFullPath(), "holes": holes}, ensure_ascii=False))
