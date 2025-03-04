# HASZNÁLAT:
# python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
# Például, egyetlen kép szkenneléséhez interaktív módban:
# python scan.py --image sample_images/desk.JPG -i
# Az összes kép automatikus szkenneléséhez egy mappában:
# python scan.py --images sample_images

# A szkennelt képek a 'output' nevű könyvtárba kerülnek

from pyimagesearch import transform
from pyimagesearch import imutils
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon
import polygon_interacter as poly_i
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
from pylsd.lsd import lsd

import argparse
import os

class DocScanner(object):
    """Kép szkenner"""

    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        """
        Paraméterek:
            interactive (bool): Ha True, a felhasználó képes manuálisan módosítani a kontúrt a
                transzformáció előtt egy interaktív pyplot ablakban.
            MIN_QUAD_AREA_RATIO (float): Egy kontúrt elutasítunk, ha annak sarkaiból
                alkotott négyzet területe nem éri el a képernyő területének MIN_QUAD_AREA_RATIO arányát.
                Alapértelmezett érték: 0.25.
            MAX_QUAD_ANGLE_RANGE (int): A kontúr belső szögeinek eltérése nem haladhatja meg
                a MAX_QUAD_ANGLE_RANGE értéket. Alapértelmezett érték: 40.
        """        
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE        


    def filter_corners(self, corners, min_dist=20):
        """Szűri a sarkokat, amelyek másoktól legfeljebb min_dist távolságra vannak"""
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                       for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def angle_between_vectors_degrees(self, u, v):
        """Visszaadja a két vektor közötti szöget fokokban"""
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        """
        Visszaadja a p2-től p1-ig és p2-től p3-ig tartó vonal szöget fokokban
        """
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        """
        Visszaadja a négyzet belső szögeinek max és min eltérése közötti különbséget.
        A bemeneti négyzetnek numpy tömb formájában kell lennie, amely az óramutató járásával megegyező irányban
        tartalmazza a csúcsokat, kezdve a bal felső sarokkal.
        """
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)          

    def get_corners(self, img):
        """
        Visszaad egy listát a képen talált sarkok koordinátáiról ((x, y) tuple).
        A megfelelő előfeldolgozás és szűrés után legfeljebb 10 potenciális sarkot kell visszaadnia.
        Ez egy segédfunkció, amelyet a get_contours használ. A bemeneti kép várhatóan 
        először átméretezett és Canny-szűrővel van ellátva.
        """
        lines = lsd(img)

        # Feldolgozza az LSD kimenetét
        # Az LSD élekre működik. Egy "vonal" két élt tartalmaz, így az éleket vissza kell kombinálni vonalakká
        # 1. Elválasztja a vonalakat vízszintes és függőleges vonalakra.
        # 2. A vízszintes vonalakat vissza kell rajzolni egy vászonra, de kissé vastagabbaknak és hosszabbaknak.
        # 3. Végrehajt egy összekapcsolt komponens vizsgálatot az új vásznon
        # 4. Minden komponenshez egy bounding box-ot keres, és az a végső vonal.
        # 5. A vonal végpontjai a sarkok
        # 6. Ismételjük meg függőleges vonalakkal
        # 7. Minden végső vonalat egy új vászonra rajzolunk. Azokon a helyeken, ahol a vonalak átfedik egymást, szintén sarkok

        corners = []
        if lines is not None:
            # Elválasztja a vízszintes és függőleges vonalakat, és külön vásznakra rajzolja őket
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            # Keresd meg a vízszintes vonalakat (összekapcsolt komponensek -> bounding box-ok -> végső vonalak)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # Keresd meg a függőleges vonalakat (összekapcsolt komponensek -> bounding box-ok -> végső vonalak)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # Keresd meg a sarkokat
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        # Távolítsd el a közeli sarkokat
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """Visszaadja True értéket, ha a kontúr megfelel az összes, a példányosításkor beállított követelménynek"""

        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
            and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)


    def get_contour(self, rescaled_image):
        """
        Visszaad egy numpy tömböt (4, 2) formátumban, amely a dokumentum négy sarkát tartalmazza
        a képen. A get_corners() által visszaadott sarkokat figyelembe véve 
        heurisztikákat használ a négy sarkot kiválasztani, amelyek legvalószínűbb, hogy a dokumentum sarkai.
        Ha nem talált sarkokat, vagy a négy sarok túl kicsi vagy konvex, visszaadja az eredeti négy sarkot.
        """        

        # Ezen konstansokat gondosan választották ki
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        # A kép átalakítása szürkeárnyalatúvá és kissé elmosása
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        # A dilatáció segít eltávolítani a potenciális hézagokat az él szegmensek között
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Keresd az éleket és jelöld őket a kimeneti térképen a Canny algoritmus segítségével
        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)

            # Az első öt legnagyobb területű négyzet
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            # A potenciális négyzetek szögeltérése alapján történő rendezés, ami segít eltávolítani az anomáliákat
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

            # Hibakereséshez: a következő kód törlésével rajzolhatók a sarkok és a kontúrok,
            # amelyeket a get_corners és a kontúr talált a képre

            # cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
            # plt.scatter(*zip(*test_corners))
            # plt.imshow(rescaled_image)
            # plt.show()

        # Próbálja meg közvetlenül a kontúrokat találni az éles képből, ami időnként jobb eredményeket ad
        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # Végigmegyünk a kontúrokon
        for c in cnts:
            # Közelítjük a kontúrt
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        # Ha nem találtunk érvényes kontúrokat, akkor használjuk az egész képet
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            
        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title(('Húzd a doboz sarkait a dokumentum sarkaihoz. \n'
            'Zárd be az ablakot, amikor befejezted.'))
        p = poly_i.PolygonInteractor(ax, poly)
        plt.imshow(rescaled_image)
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype = "int32")
        return new_points.reshape(4, 2)

    def scan(self, image_path):

        RESCALED_HEIGHT = 500.0

        OUTPUT_DIR = os.path.join(os.getcwd(), "output")

        # A kép betöltése és a régi magasság és az új magasság arányának kiszámítása,
        # majd az átméretezett kép másolása és átméretezése
        image = cv2.imread(image_path)

        assert(image is not None)

        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height = int(RESCALED_HEIGHT))

        # A dokumentum kontúrjának lekérése
        screenCnt = self.get_contour(rescaled_image)

        if self.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        # A perspektív transzformáció alkalmazása
        warped = transform.four_point_transform(orig, screenCnt * ratio)

        # A torzított kép konvertálása szürkeárnyalatúvá
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Az élesítés alkalmazása
        sharpen = cv2.GaussianBlur(gray, (0,0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # Alkalmazzunk adaptív küszöbölést, hogy fekete-fehér hatást érjünk el
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        # Mentse el a transzformált képet
        basename = os.path.basename(image_path)

        if os.path.exists(OUTPUT_DIR):
            cv2.imwrite(OUTPUT_DIR + '/' + basename, thresh)
            print("Feldolgozva: " + basename)
        else:
            os.mkdir(OUTPUT_DIR)
            cv2.imwrite(OUTPUT_DIR + '/' + basename, thresh)
            print("Feldolgozva: " + basename)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="A beolvasandó képek könyvtára")
    group.add_argument("--image", help="Egyetlen képfájl elérési útja a beolvasáshoz")
    ap.add_argument("-i", action='store_true',
        help="Jelző, hogy manuálisan ellenőrizze és/vagy beállítsa a dokumentum sarkait")

    args = vars(ap.parse_args())
    im_dir = args["images"]
    im_file_path = args["image"]
    interactive_mode = args["i"]

    scanner = DocScanner(interactive_mode)

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Egyetlen kép beolvasása a parancssori argumentum --image <IMAGE_PATH> segítségével
    if im_file_path:
        scanner.scan(im_file_path)

    # Minden érvényes képet beolvas a könyvtárból, amelyet a --images <IMAGE_DIR> parancssori argumentumban adtak meg
    else:
        im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
        for im in im_files:
            scanner.scan(im_dir + '/' + im)

