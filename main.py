
import pygame as p
from copy import deepcopy
import numpy as np
from numba import jit

# things for pygame
H = 600
W = 800
disp = p.display.set_mode((W, H))
FPS = 60

white = (255, 255, 255)
black = (0, 0, 0)

# for fps printing
p.font.init()
clock = p.time.Clock()
p.display.set_caption('3d demo') 
font = p.font.SysFont('ubuntu mono', 20)

def createMat():
    return np.zeros((4, 4), dtype=np.float32)


def createVector(x=0, y=0, z=0, w=1):
    return np.array([x, y, z, w], dtype=np.float32)


def createTriangle():
    return np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float64)


def createIdentity():
    return np.eye(4, 4)


def matrix_makeProj():
    a = H / W
    far = 1000  # stuff for 3D
    near = 0.1  # like most of the code)
    FieldOfView = 90
    angle = (FieldOfView * 0.5) / 180 * np.pi  # 180 times pi is to convert to radians
    FovRad = 1 / np.tan(angle)

    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = a * FovRad
    mat[1][1] = FovRad
    mat[2][2] = far / (far - near)
    mat[3][2] = (-far * near) / (far - near)
    mat[2][3] = 1
    return mat


def matrix_makeTranslation(x, y, z):
    mat = np.eye(4, 4)
    mat[3][0] = x
    mat[3][1] = y
    mat[3][2] = z
    return mat


def FillTriangle(tri, dp, color):
    dp = min(dp, 1)
    color = [color[0] * dp, color[1] * dp, color[2] * dp]
    points = [(tri[0][0], tri[0][1]), (tri[1][0], tri[1][1]), (tri[2][0], tri[2][1])]
    p.draw.polygon(disp, color, points)


@jit(nopython=True)
def matrixVector_Mul(mat, v):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v[0] * mat[0][0] + v[1] * mat[1][0] + v[2] * mat[2][0] + v[3] * mat[3][0]
    vect[1] = v[0] * mat[0][1] + v[1] * mat[1][1] + v[2] * mat[2][1] + v[3] * mat[3][1]
    vect[2] = v[0] * mat[0][2] + v[1] * mat[1][2] + v[2] * mat[2][2] + v[3] * mat[3][2]
    vect[3] = v[0] * mat[0][3] + v[1] * mat[1][3] + v[2] * mat[2][3] + v[3] * mat[3][3]
    return vect


def matrixMatrix_Mul(mat1, mat2):
    return np.matmul(mat1, mat2)


def get_mesh(filename):
    #filename = input('input obj name:')
    try:
        f = open(f'{filename}.obj', 'r')
    except:
        print("obj wasn't converted, perhaps name error")
        exit()
    else:
        mesh = list()
        lines = np.array(f.readlines())
        vert = list()

        for i in lines:
            i = i.split()
            try:
                if i[0] == 'v':
                    i = i[1:]
                    for g in range(len(i)):
                        i[g] = float(i[g])
                    vert.append(i)

                if i[0] == 'f':
                    i = i[1:]
                    for g in range(len(i)):
                        i[g] = int(i[g])
                    mesh.append([vert[i[0] - 1], vert[i[1] - 1], vert[i[2] - 1]])
            except IndexError:
                pass
        del vert
        del lines
        return np.array(mesh, dtype=np.float64)

@jit(nopython=True)
def Sort(tri):
    return (tri[0][2] + tri[1][2] + tri[2][2]) / 3


@jit(nopython=True)
def vector_add(v1, v2):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v1[0] + v2[0]
    vect[1] = v1[1] + v2[1]
    vect[2] = v1[2] + v2[2]
    vect[3] = 1
    return vect


@jit(nopython=True)
def vector_sub(v1, v2):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v1[0] - v2[0]
    vect[1] = v1[1] - v2[1]
    vect[2] = v1[2] - v2[2]
    vect[3] = 1
    return vect


@jit(nopython=True)
def vector_mul(v, k):
    v[0] = v[0] * k
    v[1] = v[1] * k
    v[2] = v[2] * k
    return v


@jit(nopython=True)
def vector_div(v, k):
    if k != 0:
        v[0] = v[0] / k
        v[1] = v[1] / k
        v[2] = v[2] / k
    return v


@jit(nopython=True)
def vector_dotProduct(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@jit(nopython=True)
def Vector_len(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@jit(nopython=True)
def vector_norm(v):
    length = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length != 0:
        v[0] = v[0] / length
        v[1] = v[1] / length
        v[2] = v[2] / length
    return v


@jit(nopython=True)
def vector_crossProd(v1, v2):
    vect = np.zeros(4, dtype=np.float32)
    vect[0] = v1[1] * v2[2] - v1[2] * v2[1]
    vect[1] = v1[2] * v2[0] - v1[0] * v2[2]
    vect[2] = v1[0] * v2[1] - v1[1] * v2[0]
    vect[3] = 1
    return vect


@jit(nopython=True)
def matrixRotationZ(angle):
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = np.cos(angle)
    mat[0][1] = np.sin(angle)
    mat[1][0] = -np.sin(angle)
    mat[1][1] = np.cos(angle)
    mat[2][2] = 1
    mat[3][3] = 1
    return mat


@jit(nopython=True)
def matrixRotationX(angle):
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = 1
    mat[1][1] = np.cos(angle)
    mat[1][2] = np.sin(angle)
    mat[2][1] = -np.sin(angle)
    mat[2][2] = np.cos(angle)
    mat[3][3] = 1
    return mat


@jit(nopython=True)
def matrixRotationY(angle):
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0][0] = np.cos(angle)
    mat[0][2] = np.sin(angle)
    mat[2][0] = -np.sin(angle)
    mat[1][1] = 1
    mat[2][2] = np.cos(angle)
    mat[3][3] = 1
    return mat


def main():

    mesh = get_mesh('videoship') # <- name of the obj file to load mesh

    # constants and variables
    fTheta = 0

    vCamera = createVector()
    triProj = createTriangle()
    matProj = matrix_makeProj()

    trianglesToRaster = list()
    append = trianglesToRaster.append

    while True:
        for event in p.event.get():
            if event.type == p.QUIT:
                exit()

        disp.fill(black)

        # rotating over z and x
        fTheta += 0.001 * clock.get_time()
        matRotZ = matrixRotationZ(fTheta)
        matRotX = matrixRotationX(fTheta * 0.5)

        # adding offset to z
        matTrans = matrix_makeTranslation(0, 0, 8)# <- 15 is just enough for videoship and axis(for teapot 7 is fine)

        # creating worls matrix
        matWorld = matrixMatrix_Mul(matRotZ, matRotX)
        matWorld = matrixMatrix_Mul(matWorld, matTrans)

        # calculating triangles
        for tri in mesh:

            tri0 = createVector(x=tri[0][0], y=tri[0][1], z=tri[0][2])
            tri1 = createVector(x=tri[1][0], y=tri[1][1], z=tri[1][2])
            tri2 = createVector(x=tri[2][0], y=tri[2][1], z=tri[2][2])

            triTransformed = createTriangle()
            
            triTransformed[0] = matrixVector_Mul(matWorld, tri0)
            triTransformed[1] = matrixVector_Mul(matWorld, tri1)
            triTransformed[2] = matrixVector_Mul(matWorld, tri2)

            line1 = vector_sub(triTransformed[1], triTransformed[0])
            line2 = vector_sub(triTransformed[2], triTransformed[0])

            normal = vector_norm(vector_crossProd(line1, line2))

            vCameraRay = vector_sub(triTransformed[0], vCamera)

            # testing to draw triangle
            if vector_dotProduct(vCameraRay, normal) < 0:

                # placing the light
                light_direction = createVector(x=0, y=1, z=-1)
                light_direction = vector_norm(light_direction)

                dp = max(0.1, vector_dotProduct(light_direction, normal))

                triProj[0] = matrixVector_Mul(matProj, triTransformed[0])
                triProj[1] = matrixVector_Mul(matProj, triTransformed[1])
                triProj[2] = matrixVector_Mul(matProj, triTransformed[2])

                triProj[0] = vector_div(triProj[0], triProj[0][3])
                triProj[1] = vector_div(triProj[1], triProj[1][3])
                triProj[2] = vector_div(triProj[2], triProj[2][3])

                vOffset = createVector(x=1, y=1, z=0)

                triProj[0] = vector_add(triProj[0], vOffset)
                triProj[1] = vector_add(triProj[1], vOffset)
                triProj[2] = vector_add(triProj[2], vOffset)

                triProj[0][0] *= 0.5 * W
                triProj[0][1] *= 0.5 * H
                triProj[1][0] *= 0.5 * W
                triProj[1][1] *= 0.5 * H
                triProj[2][0] *= 0.5 * W
                triProj[2][1] *= 0.5 * H

                triProj[3][0] = dp
                append(deepcopy(triProj))

        # sorting triangles from bottom to top
        trianglesToRaster.sort(reverse=True, key=Sort)

        # drawing sorted triangles
        for tr in trianglesToRaster:
            color = [255, 255, 255]
            FillTriangle(tr, tr[3][0], color)

        trianglesToRaster.clear()


        text = font.render(str(int(clock.get_fps())), True, white)
        place = text.get_rect(center=(20, 20))
        disp.blit(text, place)

        p.display.update()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
