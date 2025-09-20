from PIL import Image, ImageStat, ImageDraw
import copy, random, sys, itertools, argparse, math
import VoronoiDiagram
from CleanUp import correct
import NN
from Seg import Seg

def readImage(filename):
    return Image.open(filename).convert('L')

def drawCirc(draw, pt, r, color):
    pt0 = (pt[0]-r, pt[1]-r)
    pt1 = (pt[0]+r, pt[1]+r)
    draw.ellipse([pt0, pt1], fill=color)

def stipple(im, bxSz, itr):
    xSz, ySz = im.size
    genPts = []
    for x in itertools.product(range(0, xSz-int(bxSz/2), bxSz),
                               range(0, ySz-int(bxSz/2), bxSz)):
        box = (x[0], x[1], x[0]+bxSz, x[1]+bxSz)
        region = im.crop(box)
        # prob. por oscuridad
        if (ImageStat.Stat(region).mean[0]/255 < random.random()):
            genPts.append((x[0]+int(bxSz/2), x[1]+int(bxSz/2)))
    # Iterar para centrar puntos (Lloyd)
    for i in range(itr):
        im2 = copy.deepcopy(im)
        m = VoronoiDiagram.getVoronoi(genPts, (xSz, ySz))
        draw = ImageDraw.Draw(im2)
        for pt in genPts:
            drawCirc(draw, (pt[0], pt[1]), 1, 0)
        centroids = VoronoiDiagram.findCentroids(
            m, (xSz, ySz), len(genPts),
            lambda x, y: 1 - im.getpixel((x, y))/255
        )
        genPts = [(round(pt[0]), round(pt[1])) for pt in centroids]
        # im2.save(f'./stip_{i}.jpg')  # opcional
    return genPts

def createSegSet(lst):
    segList = [Seg(lst[i], lst[i+1]) for i in range(len(lst)-1)] + [Seg(lst[0], lst[-1])]
    for i in range(len(segList)):
        segList[i].prevSeg = segList[i-1]
        segList[i].nextSeg = segList[(i+1) % len(segList)]
    return set(segList)

def drawSegSet(segSet, sz, fname):
    im = Image.new('RGB', sz, (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for seg in segSet:
        draw.line(seg.toList(), fill=(127, 127, 127), width=1)
    im.save(fname)

def export_tsplib(coords, filename, name="custom", comment="tsp-art"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: {comment}\n")
        f.write(f"DIMENSION: {len(coords)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            # TSPLIB acepta floats o enteros; aquí enteros (pixeles)
            f.write(f"{i} {int(x)} {int(y)}\n")
        f.write("EOF\n")

def export_tour(order, filename, name="custom_tour"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {len(order)}\n")
        f.write("TOUR_SECTION\n")
        # order es la lista de puntos en orden; mapea a índices 1..n
        idx = {pt:i+1 for i, pt in enumerate(order)}
        for pt in order:
            f.write(f"{idx[pt]}\n")
        f.write("-1\nEOF\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="ruta de la imagen (png/jpg)")
    ap.add_argument("--target", type=int, default=150, help="número objetivo de nodos (50–300)")
    ap.add_argument("--iters", type=int, default=8, help="iteraciones Lloyd (suaviza puntos)")
    ap.add_argument("--maxdim", type=int, default=600, help="redimensionar lado mayor a este tamaño")
    ap.add_argument("--out", default="imagen_custom.tsp", help="salida .tsp")
    ap.add_argument("--name", default="custom_art", help="NAME TSPLIB")
    args = ap.parse_args()

    im = readImage(args.image)
    if max(im.size) > args.maxdim:
        scale = args.maxdim / float(max(im.size))
        im = im.resize((int(im.size[0]*scale), int(im.size[1]*scale)))

    # Buscar cellSize que aproxime al target
    # heurística: más grande -> menos puntos
    cell = max(2, int(math.sqrt(max(im.size) * max(im.size) / max(args.target, 1)) // 2) or 2)
    pts = stipple(im, cell, 0)
    # afinar
    tries = 0
    while not (args.target*0.9 <= len(pts) <= args.target*1.1) and tries < 12:
        if len(pts) > args.target:
            cell = int(cell * 1.3) + 1
        else:
            cell = max(2, int(cell * 0.8))
        pts = stipple(im, cell, 0)
        tries += 1
    # iteraciones de relajación
    pts = stipple(im, cell, args.iters)
    print(f"Puntos finales: {len(pts)} (cellSize={cell})")

    # (opcional) visualizar NN + cleanup como preview (no afecta .tsp)
    nn_order = NN.tsp(pts[:])
    segs = createSegSet(nn_order)
    segs = correct(segs, im)
    drawSegSet(segs, im.size, "preview.jpg")
    print("Preview guardado en preview.jpg")

    # EXPORTAR .TSP desde los PUNTOS (sin ordenar)
    export_tsplib(pts, args.out, name=args.name, comment=f"{len(pts)} puntos; from image {args.image}")
    nn_order = NN.tsp(pts[:])           # lista de puntos en orden
    export_tour(nn_order, "imagen_custom.tour", name="custom_tour")

    print(f".tsp escrito en {args.out}")

if __name__ == "__main__":
    main()
