from fastapi import APIRouter, Request, HTTPException
from starlette.responses import StreamingResponse

from .nbio import read_nb, write_nb, find_cell_index
from .kernel import KernelPool

router = APIRouter()
kernels = KernelPool()


@router.get("/nb/{path:path}")
async def view_nb(request: Request, path: str):
    nb = read_nb(path)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "notebook.html",
        {"request": request, "nb_path": path, "cells": nb.get("cells", [])},
    )


@router.post("/nb/{path:path}/cells/{cell_id}/save")
async def save_cell(request: Request, path: str, cell_id: str):
    form = await request.form()
    source = form.get("source", "")
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None:
        raise HTTPException(404, "cell not found")
    nb["cells"][i]["source"] = source.split("\n")
    if nb["cells"][i]["cell_type"] == "code":
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
    write_nb(path, nb)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "_cell.html",
        {"request": request, "nb_path": path, "cell": nb["cells"][i], "run_now": False},
    )


@router.post("/nb/{path:path}/cells/{cell_id}/run")
async def run_cell(request: Request, path: str, cell_id: str):
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None:
        raise HTTPException(404, "cell not found")
    # Return fresh cell fragment with SSE connection activated
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "_cell.html",
        {"request": request, "nb_path": path, "cell": nb["cells"][i], "run_now": True},
    )


@router.get("/nb/{path:path}/cells/{cell_id}/stream")
async def stream_cell(path: str, cell_id: str):
    nb = read_nb(path)
    i = find_cell_index(nb, cell_id)
    if i is None:
        raise HTTPException(404, "cell not found")
    cell = nb["cells"][i]
    if cell.get("cell_type") != "code":
        raise HTTPException(400, "only code cells are executable")
    code = "".join(cell.get("source", []))

    async def gen():
        async for chunk in kernels.execute_stream(path, code):
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


