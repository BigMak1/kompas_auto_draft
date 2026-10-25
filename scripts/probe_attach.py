"""Attach-only probe: connect to the already running KOMPAS and report who we see.

Never starts a new KOMPAS (GetKompas only attaches). Run with an x64 Python,
because ksapi loads the x64 ksAPICLink.dll from the KOMPAS Bin folder.
"""

import os
import platform
import sys

KOMPAS_BIN = r"C:\Program Files\ASCON\KOMPAS-3D v25 Study\Bin"

sys.path.insert(0, KOMPAS_BIN)
import ksapi  # noqa: E402  (loads ksAPICLink.dll from KOMPAS_BIN on import)

print(f"python {platform.python_version()} {platform.machine()} pid={os.getpid()}")
app = ksapi.GetKompas()
if app is None:
    print("GetKompas: None (no running KOMPAS visible from this session)")
    sys.exit(2)

print("GetKompas: attached")
print("application:", app.GetApplicationName(True))
print("build:", app.GetBuildNumber())
print("visible:", app.IsVisible())
docs = app.GetDocuments()
print("open documents:", docs.GetCount())
for doc in docs.GetObjects():
    d = ksapi.IKompasDocument(doc)
    print("  -", d.GetName(), "| active" if d.IsActive() else "")
