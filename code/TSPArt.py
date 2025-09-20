import os
import math
import copy
import random
import itertools
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set

from PIL import Image, ImageStat, ImageDraw
import matplotlib.pyplot as plt

import VoronoiDiagram
from CleanUp import correct
import NN
from Seg import Seg


# --- Utilidades de imagen / puntos -------------------------------------------------

def readImage(filename: str) -> Image.Image:
    """Abre imagen como escala de grises (PIL L)."""
    return Image.open(filename).convert('L')

def drawCirc(draw: ImageDraw.ImageDraw, pt: Tuple[int, int], r: int, color: int) -> None:
    """Dibuja un círculo sólido en PIL."""
    pt0 = (pt[0] - r, pt[1] - r)
    pt1 = (pt[0] + r, pt[1] + r)
    draw.ellipse([pt0, pt1], fill=color)

def computeCellSizeForTarget(im_size: Tuple[int, int], target: int) -> int:
    """Calcula tamaño de celda inicial para aproximar target puntos."""
    big = max(im_size)
    guess = int(max(2, math.sqrt((big * big) / max(target, 1)) // 2) or 2)
    return max(2, guess)

def stipple(im: Image.Image, box_size: int, iters: int) -> List[Tuple[int, int]]:
    """Genera puntos ponderando por oscuridad y los relaja iters veces (Lloyd-lite)."""
    x_size, y_size = im.size
    gen_pts: List[Tuple[int, int]] = []
    for x in itertools.product(range(0, x_size - int(box_size / 2), box_size),
                               range(0, y_size - int(box_size / 2), box_size)):
        box = (x[0], x[1], x[0] + box_size, x[1] + box_size)
        region = im.crop(box)
        if (ImageStat.Stat(region).mean[0] / 255 < random.random()):
            gen_pts.append((x[0] + int(box_size / 2), x[1] + int(box_size / 2)))

    for _ in range(iters):
        im2 = copy.deepcopy(im)
        m = VoronoiDiagram.getVoronoi(gen_pts, (x_size, y_size))
        draw = ImageDraw.Draw(im2)
        for pt in gen_pts:
            drawCirc(draw, (pt[0], pt[1]), 1, 0)
        centroids = VoronoiDiagram.findCentroids(
            m, (x_size, y_size), len(gen_pts),
            lambda xx, yy: 1 - im.getpixel((xx, yy)) / 255
        )
        gen_pts = [(round(pt[0]), round(pt[1])) for pt in centroids]
    return gen_pts

def createSegSet(order_pts: List[Tuple[int, int]]) -> Set[Seg]:
    """Crea un conjunto de segmentos enlazados que forman el ciclo del tour."""
    seg_list = [Seg(order_pts[i], order_pts[i + 1]) for i in range(len(order_pts) - 1)] + [Seg(order_pts[0], order_pts[-1])]
    for i in range(len(seg_list)):
        seg_list[i].prevSeg = seg_list[i - 1]
        seg_list[i].nextSeg = seg_list[(i + 1) % len(seg_list)]
    return set(seg_list)

def segSetToOrder(seg_set: Set[Seg]) -> List[Tuple[int, int]]:
    """Convierte un conjunto de segmentos cerrados en una lista ordenada de puntos."""
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for s in seg_set:
        p1 = (int(s.pt1[0]), int(s.pt1[1]))
        p2 = (int(s.pt2[0]), int(s.pt2[1]))
        adj.setdefault(p1, []).append(p2)
        adj.setdefault(p2, []).append(p1)
    start = min(adj.keys())
    order = [start]
    prev = None
    cur = start
    while True:
        neigh = adj[cur]
        nxt = neigh[0] if neigh[0] != prev else neigh[1]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order


# --- Exportación TSPLIB ------------------------------------------------------------

def exportTsplib(coords: List[Tuple[int, int]], path: str, name: str, comment: str) -> Dict[Tuple[int, int], int]:
    """Escribe .tsp EUC_2D y devuelve el mapa punto→índice (1..n)."""
    idx_map: Dict[Tuple[int, int], int] = {}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: " + comment + "\n")
        f.write(f"DIMENSION: {len(coords)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            xi, yi = int(x), int(y)
            idx_map[(xi, yi)] = i
            f.write(f"{i} {xi} {yi}\n")
        f.write("EOF\n")
    return idx_map

def exportTour(order_pts: List[Tuple[int, int]], path: str, idx_map: Dict[Tuple[int, int], int], name: str) -> None:
    """Escribe .tour usando los índices del .tsp ya exportado."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {len(order_pts)}\n")
        f.write("TOUR_SECTION\n")
        for p in order_pts:
            xi, yi = int(p[0]), int(p[1])
            f.write(f"{idx_map[(xi, yi)]}\n")
        f.write("-1\nEOF\n")


# --- Preview con matplotlib (misma orientación que PIL) ----------------------------

def drawPreviewMatplotlib(order_pts, im_size, out_path, show_points=False, pil_orientation=False):
    """Preview; si pil_orientation=True invierte Y, si no, dibuja cartesiano."""
    xs = [p[0] for p in order_pts] + [order_pts[0][0]]
    ys = [p[1] for p in order_pts] + [order_pts[0][1]]
    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, ys, '-', linewidth=1)
    if show_points:
        plt.plot(xs, ys, linestyle='None', marker='o', markersize=1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    if pil_orientation:
        ax.invert_yaxis()
        ax.set_ylim(im_size[1], 0)
    else:
        ax.set_ylim(0, im_size[1])
    ax.set_xlim(0, im_size[0])
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# --- Nombres de salida y Voltear puntos ----------------------------------------------

def buildOutputPaths(image_path: str, target: int) -> Tuple[str, str, str]:
    """Construye rutas: preview, tsp y tour en la carpeta de la imagen."""
    p = Path(image_path).resolve()
    folder = p.parent
    stem = p.stem
    preview = str(folder / f"{stem}_pv.png")
    tsp = str(folder / f"{stem}_{target}.tsp")
    tour = str(folder / f"{stem}_{target}.nn_clean.tour")
    return preview, tsp, tour

def flipY(points, height):
    """Convierte coordenadas PIL (Y-down) a cartesianas (Y-up)."""
    return [(int(x), int(height - y)) for (x, y) in points]



# --- Programa principal ------------------------------------------------------------

def main() -> None:
    """CLI: lee imagen, crea puntos, limpia tour, exporta .tsp/.tour y preview."""
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="ruta de la imagen (png/jpg)")
    ap.add_argument("--target", type=int, default=150, help="número objetivo de nodos (50–300)")
    ap.add_argument("--iters", type=int, default=8, help="iteraciones de relajación (Lloyd)")
    ap.add_argument("--maxdim", type=int, default=600, help="redimensionar lado mayor a este tamaño")
    ap.add_argument("--showPoints", action="store_true", help="muestra puntos en el preview")
    args = ap.parse_args()

    im = readImage(args.image)
    if max(im.size) > args.maxdim:
        scale = args.maxdim / float(max(im.size))
        im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)))

    cell = computeCellSizeForTarget(im.size, args.target)
    pts = stipple(im, cell, 0)
    tries = 0
    while not (args.target * 0.9 <= len(pts) <= args.target * 1.1) and tries < 12:
        cell = int(cell * 1.3) + 1 if len(pts) > args.target else max(2, int(cell * 0.8))
        pts = stipple(im, cell, 0)
        tries += 1
    pts = stipple(im, cell, args.iters)
    pts = [(int(x), int(y)) for (x, y) in pts]

    H = im.size[1]
    
    # Tour NN + cleanup para obtener el orden "bonito"
    nn_order = NN.tsp(pts[:])
    segs = createSegSet(nn_order)
    segs = correct(segs, im)
    order_clean = segSetToOrder(segs)

    # ---- VOLTEO a cartesianas para exportar y previsualizar coherente con Matplotlib
    pts_cart   = flipY(pts, H)
    order_cart = flipY(order_clean, H)

    # Salidas automáticas
    preview_path, tsp_path, tour_path = buildOutputPaths(args.image, len(pts))

    # Preview en cartesianas (sin invertir eje)
    drawPreviewMatplotlib(order_cart, im.size, preview_path, show_points=args.showPoints, pil_orientation=False)

    # Exportar .tsp (indices 1..n fijos) y .tour con esos índices
    idx_map = exportTsplib(pts_cart, tsp_path, name=Path(args.image).stem, comment=f"{len(pts_cart)} puntos desde {Path(args.image).name}; COORD_SYSTEM=CARTESIAN_Y_UP")
    exportTour(order_cart, tour_path, idx_map, name=f"{Path(args.image).stem}_opt")

    print("Archivos generados:")
    print(f"  Preview: {preview_path}")
    print(f"  TSPLIB : {tsp_path}")
    print(f"  TOUR   : {tour_path}")


if __name__ == "__main__":
    main()
